#!/usr/bin/env python3
"""
Debug training script for single model runs without HPO.
Use this to quickly test individual models and debug inf/nan issues.

Usage:
    # Test data loading and feature computation only
    python debug_train.py --check-data-only

    # Train a single model with default config
    python debug_train.py --model voc_nn --epochs 10

    # Train all models with minimal epochs for quick testing
    python debug_train.py --all --epochs 5

    # Use a subset of data for faster iteration
    python debug_train.py --model jsc_lgbm --n-samples 1000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_PARAMS_FILE, DEFAULT_IV_FILE, RANDOM_SEED
from data import load_raw_data, prepare_tensors, extract_targets_gpu, split_indices
from features import (
    compute_all_physics_features, get_feature_names,
    compute_jsc_ceiling, compute_voc_ceiling, validate_features
)
from models.voc_nn import VocNNConfig, VocNN, VocTrainer
from models.jsc_lgbm import JscLGBMConfig, JscLGBM
from models.vmpp_lgbm import VmppLGBMConfig, VmppLGBM, JmppLGBM, FFLGBM


def check_for_invalid_values(tensor: torch.Tensor, name: str) -> bool:
    """Check if tensor contains inf or nan values."""
    n_inf = torch.isinf(tensor).sum().item()
    n_nan = torch.isnan(tensor).sum().item()
    if n_inf > 0 or n_nan > 0:
        print(f"  WARNING: {name} has {n_inf} inf, {n_nan} nan values")
        return True
    return False


def print_tensor_stats(tensor: torch.Tensor, name: str):
    """Print statistics for a tensor."""
    valid = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]
    if len(valid) > 0:
        print(f"  {name}: min={valid.min().item():.6f}, max={valid.max().item():.6f}, "
              f"mean={valid.mean().item():.6f}, std={valid.std().item():.6f}")
    else:
        print(f"  {name}: ALL VALUES INVALID")


def load_and_validate_data(params_file: str, iv_file: str, device: torch.device,
                           n_samples: int = None, verbose: bool = True):
    """Load data and validate for issues."""
    print("\n" + "=" * 60)
    print("Loading and Validating Data")
    print("=" * 60)

    # Load raw data
    params_df, iv_data = load_raw_data(params_file, iv_file)
    n_total = len(params_df)
    print(f"Loaded {n_total} samples")

    # Subsample if requested
    if n_samples is not None and n_samples < n_total:
        np.random.seed(RANDOM_SEED)
        idx = np.random.choice(n_total, n_samples, replace=False)
        params_df = params_df.iloc[idx].reset_index(drop=True)
        iv_data = iv_data[idx]
        print(f"Subsampled to {n_samples} samples")

    # Check raw params for issues
    print("\n--- Raw Parameters Check ---")
    params_np = params_df.values.astype(np.float32)
    n_inf = np.isinf(params_np).sum()
    n_nan = np.isnan(params_np).sum()
    print(f"  Raw params: {n_inf} inf, {n_nan} nan values")

    if verbose:
        for i, col in enumerate(params_df.columns):
            vals = params_np[:, i]
            print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

    # Convert to tensors
    params_tensor, iv_tensor, v_grid = prepare_tensors(params_df, iv_data, device)

    # Extract targets
    print("\n--- Target Extraction Check ---")
    targets = extract_targets_gpu(iv_tensor, v_grid)
    targets_np = {k: v.cpu().numpy() for k, v in targets.items()}

    has_issues = False
    for name, vals in targets.items():
        if check_for_invalid_values(vals, name):
            has_issues = True
        print_tensor_stats(vals, name)

    # Compute physics features
    print("\n--- Physics Features Check ---")
    features = compute_all_physics_features(params_tensor)
    feature_stats = validate_features(features, verbose=True)

    if feature_stats['has_issues']:
        has_issues = True
        print(f"\nFeatures have {feature_stats['n_inf']} inf and {feature_stats['n_nan']} nan values")

    features_np = features.cpu().numpy()
    print(f"Features shape: {features_np.shape}")
    print(f"Features min: {features_np.min():.4f}, max: {features_np.max():.4f}")

    # Compute ceilings
    print("\n--- Ceiling Values Check ---")
    jsc_ceiling = compute_jsc_ceiling(params_tensor)
    voc_ceiling = compute_voc_ceiling(params_tensor)

    check_for_invalid_values(jsc_ceiling, "jsc_ceiling")
    check_for_invalid_values(voc_ceiling, "voc_ceiling")
    print_tensor_stats(jsc_ceiling, "jsc_ceiling")
    print_tensor_stats(voc_ceiling, "voc_ceiling")

    if has_issues:
        print("\n" + "!" * 60)
        print("DATA HAS ISSUES - Review warnings above")
        print("!" * 60)
    else:
        print("\n" + "=" * 60)
        print("DATA VALIDATION PASSED")
        print("=" * 60)

    return {
        'params_df': params_df,
        'params_tensor': params_tensor,
        'iv_tensor': iv_tensor,
        'v_grid': v_grid,
        'targets': targets,
        'targets_np': targets_np,
        'features': features,
        'features_np': features_np,
        'jsc_ceiling': jsc_ceiling.cpu().numpy(),
        'voc_ceiling': voc_ceiling.cpu().numpy(),
        'has_issues': has_issues
    }


def train_voc_nn(data: dict, device: torch.device, epochs: int = 50,
                 batch_size: int = 2048, verbose: bool = True):
    """Train Voc neural network with default config."""
    print("\n" + "=" * 60)
    print("Training Voc Neural Network")
    print("=" * 60)

    # Prepare data
    X_raw = data['params_df'].values.astype(np.float32)
    X_physics = data['features_np']
    y = data['targets_np']['Voc']

    X_full = np.hstack([X_raw, X_physics])

    # Split
    n_samples = len(y)
    train_idx, val_idx, test_idx = split_indices(n_samples, 0.1, 0.1)

    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # CRITICAL: Standardize features to prevent gradient explosion
    # Features have huge range (-77 to 1591) which causes inf/nan without normalization
    feature_mean = X_train.mean(axis=0, keepdims=True)
    feature_std = X_train.std(axis=0, keepdims=True) + 1e-8  # Avoid division by zero
    X_train = (X_train - feature_mean) / feature_std
    X_val = (X_val - feature_mean) / feature_std

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Features normalized: mean={X_train.mean():.6f}, std={X_train.std():.6f}")

    # Create model
    config = VocNNConfig(
        input_dim=X_full.shape[1],
        hidden_dims=[256, 256, 128],
        dropout=0.1,
        epochs=epochs,
        patience=10,
        lr=1e-3
    )

    model = VocNN(config).to(device)
    trainer = VocTrainer(model, config, device)

    # Data loaders
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            trainer.optimizer.zero_grad()
            pred = model(batch_x)

            # Check for inf/nan in predictions
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"  Epoch {epoch}: INVALID PREDICTIONS DETECTED")
                return None

            loss = torch.nn.functional.mse_loss(pred, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Epoch {epoch}: INVALID LOSS DETECTED (loss={loss.item()})")
                return None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            trainer.optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                val_loss = torch.nn.functional.mse_loss(pred, batch_y)
                val_losses.append(val_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        if verbose or epoch % 5 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    print(f"Best val loss: {best_val_loss:.6f}")
    return model


def train_jsc_lgbm(data: dict, n_estimators: int = 100, verbose: bool = True):
    """Train Jsc LightGBM with default config."""
    print("\n" + "=" * 60)
    print("Training Jsc LightGBM")
    print("=" * 60)

    # Prepare data
    X_raw = data['params_df'].values.astype(np.float32)
    X_physics = data['features_np']
    y = data['targets_np']['Jsc']
    jsc_ceiling = data['jsc_ceiling']

    # Split
    n_samples = len(y)
    train_idx, val_idx, test_idx = split_indices(n_samples, 0.1, 0.1)

    config = JscLGBMConfig(
        n_estimators=n_estimators,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=10,
        device='cpu'  # CPU for debugging
    )

    model = JscLGBM(config)

    model.fit(
        X_raw[train_idx], X_physics[train_idx],
        y[train_idx], jsc_ceiling[train_idx],
        X_raw[val_idx], X_physics[val_idx],
        y[val_idx], jsc_ceiling[val_idx]
    )

    # Evaluate
    pred = model.predict(X_raw[test_idx], X_physics[test_idx], jsc_ceiling[test_idx])
    rmse = np.sqrt(np.mean((pred - y[test_idx]) ** 2))
    print(f"Test RMSE: {rmse:.6f}")

    return model


def train_vmpp_lgbm(data: dict, n_estimators: int = 100, verbose: bool = True):
    """Train Vmpp LightGBM with default config."""
    print("\n" + "=" * 60)
    print("Training Vmpp LightGBM")
    print("=" * 60)

    X_raw = data['params_df'].values.astype(np.float32)
    X_physics = data['features_np']
    y_vmpp = data['targets_np']['Vmpp']
    y_voc = data['targets_np']['Voc']

    n_samples = len(y_vmpp)
    train_idx, val_idx, test_idx = split_indices(n_samples, 0.1, 0.1)

    config = VmppLGBMConfig(
        n_estimators=n_estimators,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=10,
        device='cpu'  # CPU for debugging
    )

    model = VmppLGBM(config)

    model.fit(
        X_raw[train_idx], X_physics[train_idx],
        y_vmpp[train_idx], y_voc[train_idx],
        X_raw[val_idx], X_physics[val_idx],
        y_vmpp[val_idx], y_voc[val_idx]
    )

    pred = model.predict(X_raw[test_idx], X_physics[test_idx], y_voc[test_idx])
    rmse = np.sqrt(np.mean((pred - y_vmpp[test_idx]) ** 2))
    print(f"Test RMSE: {rmse:.6f}")

    return model


def train_ff_lgbm(data: dict, n_estimators: int = 100, verbose: bool = True):
    """Train FF LightGBM with default config."""
    print("\n" + "=" * 60)
    print("Training FF LightGBM")
    print("=" * 60)

    X_raw = data['params_df'].values.astype(np.float32)
    X_physics = data['features_np']
    y_ff = data['targets_np']['FF']
    y_voc = data['targets_np']['Voc']
    y_jsc = data['targets_np']['Jsc']

    n_samples = len(y_ff)
    train_idx, val_idx, test_idx = split_indices(n_samples, 0.1, 0.1)

    config = VmppLGBMConfig(
        n_estimators=n_estimators,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=10,
        device='cpu'  # CPU for debugging
    )

    model = FFLGBM(config)

    model.fit(
        X_raw[train_idx], X_physics[train_idx],
        y_ff[train_idx], y_voc[train_idx], y_jsc[train_idx],
        X_raw[val_idx], X_physics[val_idx],
        y_ff[val_idx], y_voc[val_idx], y_jsc[val_idx]
    )

    pred = model.predict(X_raw[test_idx], X_physics[test_idx], y_voc[test_idx], y_jsc[test_idx])
    rmse = np.sqrt(np.mean((pred - y_ff[test_idx]) ** 2))
    print(f"Test RMSE: {rmse:.6f}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Debug training for scalar predictors')
    parser.add_argument('--params', type=str, default=DEFAULT_PARAMS_FILE,
                        help='Path to parameters file')
    parser.add_argument('--iv', type=str, default=DEFAULT_IV_FILE,
                        help='Path to IV curves file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of samples to use (for quick testing)')
    parser.add_argument('--check-data-only', action='store_true',
                        help='Only check data, do not train')
    parser.add_argument('--model', type=str, default=None,
                        choices=['voc_nn', 'jsc_lgbm', 'vmpp_lgbm', 'ff_lgbm'],
                        help='Single model to train')
    parser.add_argument('--all', action='store_true',
                        help='Train all models')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for NN training')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of estimators for LGBM')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and validate data
    data = load_and_validate_data(
        args.params, args.iv, device,
        n_samples=args.n_samples,
        verbose=args.verbose
    )

    if args.check_data_only:
        print("\nData check complete.")
        sys.exit(0 if not data['has_issues'] else 1)

    if data['has_issues']:
        print("\nWARNING: Data has issues. Training may produce inf/nan values.")
        # In batch/non-interactive runs, fail fast to avoid hangs.
        sys.exit(1)

    # Train models
    if args.model == 'voc_nn' or args.all:
        train_voc_nn(data, device, epochs=args.epochs, verbose=args.verbose)

    if args.model == 'jsc_lgbm' or args.all:
        train_jsc_lgbm(data, n_estimators=args.n_estimators, verbose=args.verbose)

    if args.model == 'vmpp_lgbm' or args.all:
        train_vmpp_lgbm(data, n_estimators=args.n_estimators, verbose=args.verbose)

    if args.model == 'ff_lgbm' or args.all:
        train_ff_lgbm(data, n_estimators=args.n_estimators, verbose=args.verbose)

    if not args.model and not args.all:
        print("\nNo model specified. Use --model <name> or --all to train.")
        print("Available models: voc_nn, jsc_lgbm, vmpp_lgbm, ff_lgbm")


if __name__ == '__main__':
    main()
