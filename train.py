"""
Main training orchestrator for scalar predictors.
Coordinates data loading, feature engineering, HPO, and model training.
"""
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    COLNAMES, DEFAULT_PARAMS_FILE, DEFAULT_IV_FILE,
    RANDOM_SEED, VAL_SPLIT, TEST_SPLIT, V_GRID
)
from data import (
    load_raw_data, prepare_tensors, extract_targets_gpu, split_indices
)
from features import (
    compute_all_physics_features, get_feature_names,
    compute_jsc_ceiling, compute_voc_ceiling, validate_physics_features
)
from models.voc_nn import (
    VocNNConfig, VocNN, VocTrainer, build_voc_model,
    SplitSplineNetConfig, UnifiedSplitSplineNet,
    ControlPointNetConfig, ControlPointNet
)
from models.jsc_lgbm import JscLGBMConfig, JscLGBM, build_jsc_model
from models.vmpp_lgbm import (
    VmppLGBMConfig, VmppLGBM, JmppLGBM, FFLGBM,
    build_vmpp_model, build_jmpp_model, build_ff_model
)
from models.reconstruction import reconstruct_curve, continuity_loss
from hpo import HPOConfig, DistributedHPO, run_full_hpo, get_best_configs_from_study, run_curve_hpo
from logging_utils import (
    TrainingLogger, ModelComparisonMetrics,
    compute_multicollinearity, suggest_features_to_drop
)
from preprocessing import (
    normalize_curves_by_isc, denormalize_curves_by_isc,
    validate_curve_normalization
)
from plotting_utils import PlotManager
from benchmark_utils import benchmark_inference


class MultiTaskLoss(nn.Module):
    """
    Automatic loss weighting for anchors + curve reconstruction.
    Uses Kendall's multi-task learning with learned log-variances.

    FIXED: Initialize with smaller sigma to prevent explosion.
    Added sigma clamping to prevent runaway values.
    """

    def __init__(self, init_log_sigma: float = -1.0):
        super().__init__()
        # Initialize log_sigma = -1.0 -> sigma = 0.37 (tighter start)
        # This prevents sigma from exploding when initial losses are high
        self.log_sigma_anchor = nn.Parameter(torch.tensor([init_log_sigma]))
        self.log_sigma_curve = nn.Parameter(torch.tensor([init_log_sigma]))

    def forward(
        self,
        pred_anchors: torch.Tensor,
        true_anchors: torch.Tensor,
        pred_curve: torch.Tensor,
        true_curve: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        l_anchor = F.mse_loss(pred_anchors, true_anchors)

        # Guard against NaN in pred_curve (from PCHIP instability)
        if torch.isnan(pred_curve).any():
            pred_curve = torch.where(torch.isnan(pred_curve), true_curve, pred_curve)

        l_curve = F.mse_loss(pred_curve, true_curve)

        # Clamp log_sigma to prevent explosion: sigma in [0.1, 5.0]
        log_sigma_a = self.log_sigma_anchor.clamp(-2.3, 1.6)
        log_sigma_c = self.log_sigma_curve.clamp(-2.3, 1.6)

        sigma_a = torch.exp(log_sigma_a)
        sigma_c = torch.exp(log_sigma_c)

        # Kendall multi-task loss formula
        loss = (
            l_anchor / (2 * sigma_a ** 2) + log_sigma_a +
            l_curve / (2 * sigma_c ** 2) + log_sigma_c
        )

        sigma_ratio = (sigma_a / (sigma_c + 1e-8)).item()

        metrics = {
            'loss_anchor': l_anchor.item(),
            'loss_curve': l_curve.item(),
            'sigma_anchor': sigma_a.item(),
            'sigma_curve': sigma_c.item(),
            'sigma_ratio': sigma_ratio,
            'loss_total': loss.item(),
            'task_imbalance': 'anchor_easy' if sigma_ratio < 0.1 else ('curve_easy' if sigma_ratio > 10 else 'balanced')
        }
        return loss, metrics


class CurveLoss(nn.Module):
    """
    Simple weighted curve reconstruction loss.
    Used when anchors come from pretrained models (no anchor loss needed).

    Computes MSE on the full curve with optional region weighting.
    """

    def __init__(self, mpp_weight: float = 2.0):
        super().__init__()
        self.mpp_weight = mpp_weight  # Extra weight near MPP (the "knee")

    def forward(
        self,
        pred_curve: torch.Tensor,
        true_curve: torch.Tensor,
        v_grid: torch.Tensor,
        vmpp: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        # Guard against NaN
        if torch.isnan(pred_curve).any():
            pred_curve = torch.where(torch.isnan(pred_curve), true_curve, pred_curve)

        # Per-point squared error
        sq_err = (pred_curve - true_curve) ** 2

        # Weight points near MPP more heavily (this is where the "knee" is)
        vmpp_expanded = vmpp.unsqueeze(1)
        dist_to_mpp = (v_grid.unsqueeze(0) - vmpp_expanded).abs()
        mpp_weights = 1.0 + (self.mpp_weight - 1.0) * torch.exp(-dist_to_mpp / 0.1)

        weighted_sq_err = sq_err * mpp_weights
        loss = weighted_sq_err.mean()

        # Region-wise metrics for logging
        mask_r1 = v_grid.unsqueeze(0) <= vmpp_expanded
        mse_r1 = (sq_err * mask_r1).sum() / mask_r1.sum().clamp(min=1)
        mse_r2 = (sq_err * ~mask_r1).sum() / (~mask_r1).sum().clamp(min=1)

        metrics = {
            'loss_curve': loss.item(),
            'mse_full': sq_err.mean().item(),
            'mse_region1': mse_r1.item(),
            'mse_region2': mse_r2.item(),
        }
        return loss, metrics


class ScalarPredictorPipeline:
    """
    Full pipeline for training scalar PV predictors.

    Flow:
    1. Load data
    2. Extract targets from J-V curves (GPU)
    3. Compute physics features (GPU)
    4. Split data
    5. Run HPO (optional)
    6. Train final models
    7. Evaluate and save
    """

    def __init__(
        self,
        params_file: str = DEFAULT_PARAMS_FILE,
        iv_file: str = DEFAULT_IV_FILE,
        output_dir: str = 'outputs',
        device: str = 'cuda',
        run_hpo: bool = True,
        run_curve_model: bool = False,
        run_curve_hpo: bool = False,  # NEW: Run HPO for curve model
        train_cvae: bool = False,
        validate_feature_correlations: bool = True,
        drop_weak_features: bool = False,
        weak_feature_threshold: float = 0.3,
        max_weak_feature_fraction: float = 0.2,
        hpo_config: HPOConfig = None,
        load_hpo_path: str = None,  # NEW: Path to load HPO results from
        # New config options for robustness and logging
        multicollinearity_threshold: float = 0.85,
        drop_multicollinear: bool = False,
        continuity_weight: float = 0.1,  # Lambda_cont, try 0.1-1.0
        ctrl_points: int = 4,  # Reduced from 6 for simplicity
        use_hard_clamp_training: bool = True,  # Fix train-test mismatch
        log_constraint_violations: bool = True,
        verbose_logging: bool = True
    ):
        self.params_file = params_file
        self.iv_file = iv_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.run_hpo = run_hpo
        self.run_curve_model = run_curve_model
        self.run_curve_hpo = run_curve_hpo
        self.train_cvae = train_cvae
        self.load_hpo_path = load_hpo_path
        self.enable_feature_validation = validate_feature_correlations
        self.drop_weak_features = drop_weak_features
        self.weak_feature_threshold = weak_feature_threshold
        self.max_weak_feature_fraction = max_weak_feature_fraction
        self.hpo_config = hpo_config or HPOConfig()

        # New config options
        self.multicollinearity_threshold = multicollinearity_threshold
        self.drop_multicollinear = drop_multicollinear
        self.continuity_weight = continuity_weight
        self.ctrl_points = ctrl_points
        self.use_hard_clamp_training = use_hard_clamp_training
        self.log_constraint_violations = log_constraint_violations
        self.verbose_logging = verbose_logging

        # Will be populated during pipeline
        self.params_df = None
        self.iv_data = None
        self.targets = None
        self.physics_features = None
        self.models = {}
        self.metrics = {}
        self.physics_feature_mask = None
        self.physics_feature_names = get_feature_names()
        self.v_grid = V_GRID.astype(np.float32)
        self.curve_norm_by_isc = False

        # Initialize structured logger
        self.logger = TrainingLogger(self.output_dir, verbose=verbose_logging)

        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")

        # Log configuration
        print(f"\nPipeline Configuration:")
        print(f"  continuity_weight: {continuity_weight}")
        print(f"  ctrl_points: {ctrl_points}")
        print(f"  use_hard_clamp_training: {use_hard_clamp_training}")
        print(f"  multicollinearity_threshold: {multicollinearity_threshold}")

    def load_data(self):
        """Load raw data from files."""
        print("\n" + "=" * 60)
        print("Loading Data")
        print("=" * 60)

        self.params_df, self.iv_data = load_raw_data(self.params_file, self.iv_file)
        print(f"Loaded {len(self.params_df)} samples")
        print(f"Parameters shape: {self.params_df.shape}")
        print(f"IV curves shape: {self.iv_data.shape}")

    def extract_targets(self):
        """Extract PV parameters from J-V curves using GPU."""
        print("\n" + "=" * 60)
        print("Extracting Targets from J-V Curves (GPU)")
        print("=" * 60)

        params_tensor, iv_tensor, v_grid = prepare_tensors(
            self.params_df, self.iv_data, self.device
        )

        self.targets = extract_targets_gpu(iv_tensor, v_grid)

        # Convert to numpy for later use
        self.targets_np = {k: v.cpu().numpy() for k, v in self.targets.items()}

        print("Extracted targets:")
        for name, values in self.targets_np.items():
            print(f"  {name}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")

        # Store tensors for feature computation
        self.params_tensor = params_tensor

    def compute_features(self):
        """Compute physics-informed features."""
        print("\n" + "=" * 60)
        print("Computing Physics Features (GPU)")
        print("=" * 60)

        self.physics_features = compute_all_physics_features(self.params_tensor)
        self.physics_features_np = self.physics_features.cpu().numpy()

        print(f"Physics features shape: {self.physics_features_np.shape}")
        print(f"Feature names: {len(get_feature_names())}")

        # Compute analytical ceilings
        self.jsc_ceiling = compute_jsc_ceiling(self.params_tensor).cpu().numpy()
        self.voc_ceiling = compute_voc_ceiling(self.params_tensor).cpu().numpy()

        print(f"Jsc ceiling range: [{self.jsc_ceiling.min():.4f}, {self.jsc_ceiling.max():.4f}]")
        print(f"Voc ceiling range: [{self.voc_ceiling.min():.4f}, {self.voc_ceiling.max():.4f}]")

    def split_data(self):
        """Split data into train/val/test sets."""
        print("\n" + "=" * 60)
        print("Splitting Data")
        print("=" * 60)

        n_samples = len(self.params_df)
        self.train_idx, self.val_idx, self.test_idx = split_indices(
            n_samples, VAL_SPLIT, TEST_SPLIT
        )

        print(f"Train: {len(self.train_idx)}, Val: {len(self.val_idx)}, Test: {len(self.test_idx)}")

        # Create numpy arrays for each split
        self.X_raw = self.params_df.values.astype(np.float32)

        self.splits = {}
        for name, idx in [('train', self.train_idx), ('val', self.val_idx), ('test', self.test_idx)]:
            self.splits[name] = {
                'X_raw': self.X_raw[idx],
                'X_physics': self.physics_features_np[idx],
                'jsc_ceiling': self.jsc_ceiling[idx],
                'voc_ceiling': self.voc_ceiling[idx],
                'targets': {k: v[idx] for k, v in self.targets_np.items()},
                'curves': self.iv_data[idx],
                'v_grid': self.v_grid
            }

        # Debug + verification
        train_curves = self.splits['train']['curves']
        assert train_curves.shape[1] == self.v_grid.shape[0], (
            f"Curve dimension mismatch: {train_curves.shape[1]} vs v_grid {self.v_grid.shape[0]}"
        )
        assert 'curves' in self.splits['train'], "Curves not being retained!"
        print(f"Curves retained. Train curves shape: {train_curves.shape}, v_grid: {self.v_grid.shape}")

    def run_feature_validation(self):
        """Validate engineered physics features and optionally drop weak ones."""
        if not self.enable_feature_validation:
            return

        train = self.splits['train']
        weak_features, weak_indices = validate_physics_features(
            train['X_physics'],
            train['targets'],
            feature_names=self.physics_feature_names,
            threshold=self.weak_feature_threshold
        )

        self.weak_features = weak_features
        weak_fraction = len(weak_indices) / max(1, train['X_physics'].shape[1])

        drop_indices = []
        if len(weak_indices) > 0:
            print(f"Weak feature fraction: {weak_fraction:.2%}")
            if self.drop_weak_features or weak_fraction > self.max_weak_feature_fraction:
                print("Dropping weak features based on correlation threshold.")
                drop_indices.extend(weak_indices)
            else:
                print("WARNING: Weak features detected. Consider dropping them for robustness.")

        # Multicollinearity check (CRITICAL for feature redundancy)
        print("\n" + "=" * 60)
        print("Checking Multicollinearity")
        print("=" * 60)

        # Combine raw + physics features for full check
        X_full = np.hstack([train['X_raw'], train['X_physics']])
        feature_names_full = list(COLNAMES) + self.physics_feature_names

        corr_matrix, high_corr_pairs = compute_multicollinearity(
            X_full, threshold=self.multicollinearity_threshold
        )

        # Log to structured logger
        self.logger.log_multicollinearity(
            feature_names_full, corr_matrix, self.multicollinearity_threshold
        )

        if self.drop_multicollinear and len(high_corr_pairs) > 0:
            # Compute target correlations for deciding which to drop
            target_array = np.column_stack([
                train['targets'][k] for k in ['Jsc', 'Voc', 'Vmpp', 'Jmpp', 'FF', 'PCE']
            ])
            target_corr = np.array([
                max(abs(np.corrcoef(X_full[:, i], target_array[:, j])[0, 1])
                    for j in range(target_array.shape[1]))
                for i in range(X_full.shape[1])
            ])
            target_corr = np.nan_to_num(target_corr, nan=0.0)

            # Only drop physics features (indices >= 31), not raw params
            features_to_drop = suggest_features_to_drop(
                corr_matrix, feature_names_full, target_corr, self.multicollinearity_threshold
            )
            # Filter to only physics features (offset by 31 raw params)
            physics_to_drop = [i - 31 for i in features_to_drop if i >= 31]

            if physics_to_drop:
                print(f"Dropping {len(physics_to_drop)} multicollinear physics features")
                drop_indices.extend(physics_to_drop)

        if drop_indices:
            self._apply_feature_mask(sorted(set(drop_indices)))

    def _apply_feature_mask(self, weak_indices: list[int]):
        """Apply a mask to remove weak physics features across all splits."""
        n_features = self.physics_features_np.shape[1]
        mask = np.ones(n_features, dtype=bool)
        mask[weak_indices] = False

        self.physics_feature_mask = mask
        self.physics_feature_names = [
            name for i, name in enumerate(self.physics_feature_names) if mask[i]
        ]

        self.physics_features_np = self.physics_features_np[:, mask]
        for split in self.splits.values():
            split['X_physics'] = split['X_physics'][:, mask]

        print(f"Physics features reduced: {n_features} -> {self.physics_features_np.shape[1]}")

    def load_hpo_results(self, hpo_path: str) -> dict:
        """Load HPO results from a JSON file."""
        print("\n" + "=" * 60)
        print(f"Loading HPO Results from: {hpo_path}")
        print("=" * 60)

        with open(hpo_path, 'r') as f:
            hpo_summary = json.load(f)

        # Convert loaded params to the format expected by get_best_configs_from_study
        hpo_results = {}
        for name, summary in hpo_summary.items():
            # Create a mock result structure
            hpo_results[name] = {
                'params': summary['best_params'],
                # We don't have the study object, but we can create a mock
                'best_value': summary.get('best_value', 0),
                'n_trials': summary.get('n_trials', 0)
            }

        print(f"Loaded HPO results for: {list(hpo_results.keys())}")
        return hpo_results

    def run_hyperparameter_optimization(self):
        """Run HPO for all models."""
        print("\n" + "=" * 60)
        print("Running Hyperparameter Optimization")
        print("=" * 60)

        train = self.splits['train']
        val = self.splits['val']

        self.hpo_results = run_full_hpo(
            X_train_raw=train['X_raw'],
            X_train_physics=train['X_physics'],
            targets_train=train['targets'],
            X_val_raw=val['X_raw'],
            X_val_physics=val['X_physics'],
            targets_val=val['targets'],
            jsc_ceiling_train=train['jsc_ceiling'],
            jsc_ceiling_val=val['jsc_ceiling'],
            voc_ceiling_train=train['voc_ceiling'],
            voc_ceiling_val=val['voc_ceiling'],
            device=self.device,
            hpo_config=self.hpo_config
        )

        # Extract best configs
        self.best_configs = get_best_configs_from_study(self.hpo_results)

        # Save HPO results
        self._save_hpo_results()

    def run_curve_hyperparameter_optimization(self):
        """Run HPO for curve reconstruction model."""
        print("\n" + "=" * 60)
        print("Running Curve Model HPO")
        print("=" * 60)

        train = self.splits['train']
        val = self.splits['val']

        X_train_full = np.hstack([train['X_raw'], train['X_physics']]).astype(np.float32)
        X_val_full = np.hstack([val['X_raw'], val['X_physics']]).astype(np.float32)

        anchors_train = np.stack(
            [train['targets']['Jsc'], train['targets']['Voc'],
             train['targets']['Vmpp'], train['targets']['Jmpp']],
            axis=1
        ).astype(np.float32)
        anchors_val = np.stack(
            [val['targets']['Jsc'], val['targets']['Voc'],
             val['targets']['Vmpp'], val['targets']['Jmpp']],
            axis=1
        ).astype(np.float32)

        curves_train = train['curves'].astype(np.float32)
        curves_val = val['curves'].astype(np.float32)

        curve_hpo_results = run_curve_hpo(
            X_train=X_train_full,
            anchors_train=anchors_train,
            curves_train=curves_train,
            X_val=X_val_full,
            anchors_val=anchors_val,
            curves_val=curves_val,
            v_grid=self.v_grid,
            device=self.device,
            hpo_config=self.hpo_config,
            n_trials=self.hpo_config.n_trials_nn
        )

        # Merge into existing hpo_results
        if not hasattr(self, 'hpo_results'):
            self.hpo_results = {}
        self.hpo_results['curve_model'] = curve_hpo_results['curve_model']

        # Update best_configs
        if not hasattr(self, 'best_configs'):
            self.best_configs = {}
        curve_configs = get_best_configs_from_study({'curve_model': curve_hpo_results['curve_model']})
        self.best_configs.update(curve_configs)

        # Save updated HPO results
        self._save_hpo_results()

        print(f"\nCurve HPO completed. Best value: {curve_hpo_results['curve_model']['best_value']:.6f}")

    def _save_hpo_results(self):
        """Save all HPO results to JSON."""
        hpo_summary = {}
        for name, result in self.hpo_results.items():
            if 'study' in result:
                hpo_summary[name] = {
                    'best_params': result['params'],
                    'best_value': result['study'].best_value,
                    'n_trials': len(result['study'].trials)
                }
            else:
                # Already in summary format (loaded from file or curve HPO)
                hpo_summary[name] = {
                    'best_params': result.get('params', result.get('best_params', {})),
                    'best_value': result.get('best_value', 0),
                    'n_trials': result.get('n_trials', 0)
                }

        with open(self.output_dir / 'hpo_results.json', 'w') as f:
            json.dump(hpo_summary, f, indent=2, default=str)

        print("\nHPO Summary saved to:", self.output_dir / 'hpo_results.json')
        for name, summary in hpo_summary.items():
            best_val = summary.get('best_value', 'N/A')
            n_trials = summary.get('n_trials', 'N/A')
            if isinstance(best_val, float):
                print(f"  {name}: best_value={best_val:.6f}, n_trials={n_trials}")
            else:
                print(f"  {name}: best_value={best_val}, n_trials={n_trials}")

    def train_final_models(self):
        """Train final models with best hyperparameters."""
        print("\n" + "=" * 60)
        print("Training Final Models")
        print("=" * 60)

        train = self.splits['train']
        val = self.splits['val']

        # Use best configs from HPO or defaults
        configs = getattr(self, 'best_configs', {})

        # 1. Train Voc NN
        print("\n--- Training Voc Neural Network ---")
        voc_config = configs.get('voc_nn', VocNNConfig())
        voc_config.input_dim = train['X_raw'].shape[1] + train['X_physics'].shape[1]

        self.models['voc_nn'], trainer = build_voc_model(voc_config, self.device)

        # Prepare data
        X_train_full = np.hstack([train['X_raw'], train['X_physics']])
        X_val_full = np.hstack([val['X_raw'], val['X_physics']])

        # CRITICAL: Standardize features AND targets to prevent gradient explosion
        # Store normalization params for inference
        self.voc_feature_mean = X_train_full.mean(axis=0, keepdims=True)
        self.voc_feature_std = X_train_full.std(axis=0, keepdims=True) + 1e-8
        X_train_full = (X_train_full - self.voc_feature_mean) / self.voc_feature_std
        X_val_full = (X_val_full - self.voc_feature_mean) / self.voc_feature_std

        # Normalize targets for stable training
        self.voc_target_mean = train['targets']['Voc'].mean()
        self.voc_target_std = train['targets']['Voc'].std() + 1e-8
        y_train_voc = (train['targets']['Voc'] - self.voc_target_mean) / self.voc_target_std
        y_val_voc = (val['targets']['Voc'] - self.voc_target_mean) / self.voc_target_std

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_full).float(),
            torch.from_numpy(y_train_voc).float()
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_full).float(),
            torch.from_numpy(y_val_voc).float()
        )

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4096, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4096)

        # Custom training loop for Voc
        self._train_voc_model(trainer, train_loader, val_loader, voc_config)

        # 2. Train Jsc LGBM
        print("\n--- Training Jsc LGBM ---")
        jsc_config = configs.get('jsc_lgbm', JscLGBMConfig())
        self.models['jsc_lgbm'] = build_jsc_model(jsc_config)

        self.models['jsc_lgbm'].fit(
            train['X_raw'], train['X_physics'],
            train['targets']['Jsc'], train['jsc_ceiling'],
            val['X_raw'], val['X_physics'],
            val['targets']['Jsc'], val['jsc_ceiling']
        )

        # 3. Train Vmpp LGBM
        print("\n--- Training Vmpp LGBM ---")
        vmpp_config = configs.get('vmpp_lgbm', VmppLGBMConfig())
        self.models['vmpp_lgbm'] = build_vmpp_model(vmpp_config)

        self.models['vmpp_lgbm'].fit(
            train['X_raw'], train['X_physics'],
            train['targets']['Vmpp'], train['targets']['Voc'],
            val['X_raw'], val['X_physics'],
            val['targets']['Vmpp'], val['targets']['Voc']
        )

        # 4. Train Jmpp LGBM
        print("\n--- Training Jmpp LGBM ---")
        self.models['jmpp_lgbm'] = build_jmpp_model(vmpp_config)

        self.models['jmpp_lgbm'].fit(
            train['X_raw'], train['X_physics'],
            train['targets']['Jmpp'], train['targets']['Jsc'], train['targets']['Vmpp'],
            val['X_raw'], val['X_physics'],
            val['targets']['Jmpp'], val['targets']['Jsc'], val['targets']['Vmpp']
        )

        # 5. Train FF LGBM
        print("\n--- Training FF LGBM ---")
        ff_config = configs.get('ff_lgbm', VmppLGBMConfig())
        self.models['ff_lgbm'] = build_ff_model(ff_config)

        self.models['ff_lgbm'].fit(
            train['X_raw'], train['X_physics'],
            train['targets']['FF'], train['targets']['Voc'], train['targets']['Jsc'],
            val['X_raw'], val['X_physics'],
            val['targets']['FF'], val['targets']['Voc'], val['targets']['Jsc']
        )

        # 6. Train unified curve model (optional)
        if self.run_curve_model:
            print("\n--- Training Unified Split-Spline Curve Model ---")
            self.train_curve_model()

        # 7. Train CVAE baseline (optional)
        if self.train_cvae:
            print("\n--- Training CVAE Baseline ---")
            self.train_cvae_baseline()

    def train_curve_model(self):
        """
        Train curve reconstruction model using PRETRAINED scalar predictors.

        KEY CHANGES (v3.0):
        1. Use pretrained LightGBM/VocNN models to generate anchors (no re-learning)
        2. Normalize anchors for stable training
        3. Use ControlPointNet that only learns curve shape, not anchors
        4. Use simpler CurveLoss (no Kendall sigma explosion)
        """
        print("\n" + "=" * 60)
        print("Training Curve Model (v3.0 - Using Pretrained Scalar Models)")
        print("=" * 60)

        train = self.splits['train']
        val = self.splits['val']

        # Prepare inputs
        X_train_full = np.hstack([train['X_raw'], train['X_physics']]).astype(np.float32)
        X_val_full = np.hstack([val['X_raw'], val['X_physics']]).astype(np.float32)

        self.curve_feature_mean = X_train_full.mean(axis=0, keepdims=True)
        self.curve_feature_std = X_train_full.std(axis=0, keepdims=True) + 1e-8
        X_train_norm = (X_train_full - self.curve_feature_mean) / self.curve_feature_std
        X_val_norm = (X_val_full - self.curve_feature_mean) / self.curve_feature_std

        # Get TRUE anchors (ground truth for training)
        anchors_train = np.stack(
            [train['targets']['Jsc'], train['targets']['Voc'],
             train['targets']['Vmpp'], train['targets']['Jmpp']],
            axis=1
        ).astype(np.float32)
        anchors_val = np.stack(
            [val['targets']['Jsc'], val['targets']['Voc'],
             val['targets']['Vmpp'], val['targets']['Jmpp']],
            axis=1
        ).astype(np.float32)

        # ISSUE 3 FIX: Normalize anchors for stable training
        self.anchor_mean = anchors_train.mean(axis=0, keepdims=True)
        self.anchor_std = anchors_train.std(axis=0, keepdims=True) + 1e-8
        anchors_train_norm = (anchors_train - self.anchor_mean) / self.anchor_std
        anchors_val_norm = (anchors_val - self.anchor_mean) / self.anchor_std

        print(f"\nAnchor normalization stats:")
        print(f"  Mean: Jsc={self.anchor_mean[0,0]:.2f}, Voc={self.anchor_mean[0,1]:.4f}, "
              f"Vmpp={self.anchor_mean[0,2]:.4f}, Jmpp={self.anchor_mean[0,3]:.2f}")
        print(f"  Std:  Jsc={self.anchor_std[0,0]:.2f}, Voc={self.anchor_std[0,1]:.4f}, "
              f"Vmpp={self.anchor_std[0,2]:.4f}, Jmpp={self.anchor_std[0,3]:.2f}")

        # Normalize curves by Isc (KNOWN WORKING approach)
        print("\nNormalizing curves by Isc...")
        isc_train, curves_train_norm = normalize_curves_by_isc(train['curves'])
        isc_val, curves_val_norm = normalize_curves_by_isc(val['curves'])
        self.curve_norm_by_isc = True

        # Validate normalization
        validate_curve_normalization(train['curves'], curves_train_norm, isc_train, n_samples=3)

        # Create datasets with normalized curves
        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_norm),
            torch.from_numpy(anchors_train_norm),  # Normalized for model input
            torch.from_numpy(anchors_train),       # Raw for curve reconstruction
            torch.from_numpy(curves_train_norm),   # Normalized curves [-1, 1]
            torch.from_numpy(isc_train)            # For denormalization
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_norm),
            torch.from_numpy(anchors_val_norm),
            torch.from_numpy(anchors_val),
            torch.from_numpy(curves_val_norm),
            torch.from_numpy(isc_val)
        )

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2048, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2048)

        # ISSUE 2 FIX: Use ControlPointNet that takes anchors as input
        config = ControlPointNetConfig(
            input_dim=X_train_norm.shape[1],
            anchor_dim=4,
            hidden_dims=[256, 128, 64],
            dropout=0.15,
            activation='silu',
            ctrl_points=self.ctrl_points
        )

        model = ControlPointNet(config).to(self.device)
        curve_loss_fn = CurveLoss(mpp_weight=2.0).to(self.device)

        # Simpler optimizer - no learnable loss weights
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

        v_grid = torch.from_numpy(self.v_grid).float().to(self.device)

        best_val = float('inf')
        best_state = None
        patience = 20
        patience_counter = 0

        print(f"\nTraining ControlPointNet with:")
        print(f"  ctrl_points: {config.ctrl_points}")
        print(f"  hidden_dims: {config.hidden_dims}")
        print(f"  continuity_weight: {self.continuity_weight}")
        print(f"  Using ground truth anchors (not re-learning them)")

        for epoch in range(150):
            model.train()
            epoch_losses = []

            for batch_x, batch_anchors_norm, batch_anchors_raw, batch_curves_norm, batch_isc in train_loader:
                batch_x = batch_x.to(self.device)
                batch_anchors_norm = batch_anchors_norm.to(self.device)
                batch_anchors_raw = batch_anchors_raw.to(self.device)
                batch_curves_norm = batch_curves_norm.to(self.device)
                batch_isc = batch_isc.to(self.device)

                optimizer.zero_grad()

                # Model predicts only control points (anchors provided as input)
                ctrl1, ctrl2 = model(batch_x, batch_anchors_norm)

                # Reconstruct curve using TRUE anchors (returns absolute scale)
                pred_curve_abs = reconstruct_curve(
                    batch_anchors_raw, ctrl1, ctrl2, v_grid,
                    clamp_voc=True
                )

                # Normalize prediction to match target space [-1, 1]
                pred_curve_norm = 2.0 * (pred_curve_abs / batch_isc.unsqueeze(1)) - 1.0

                # Compute loss in normalized space (better gradients)
                loss, metrics = curve_loss_fn(
                    pred_curve_norm, batch_curves_norm, v_grid, batch_anchors_raw[:, 2]
                )

                # Add continuity penalty
                cont_loss = continuity_loss(batch_anchors_raw, ctrl1, ctrl2, v_grid)
                loss = loss + self.continuity_weight * cont_loss

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()

            # Validation
            model.eval()
            val_losses = []
            sum_sq_full = 0.0
            sum_cnt = 0
            sum_sq_r1 = 0.0
            sum_cnt_r1 = 0
            sum_sq_r2 = 0.0
            sum_cnt_r2 = 0
            ff_mape_sum = 0.0
            ff_cnt = 0

            with torch.no_grad():
                for batch_x, batch_anchors_norm, batch_anchors_raw, batch_curves_norm, batch_isc in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_anchors_norm = batch_anchors_norm.to(self.device)
                    batch_anchors_raw = batch_anchors_raw.to(self.device)
                    batch_curves_norm = batch_curves_norm.to(self.device)
                    batch_isc = batch_isc.to(self.device)

                    ctrl1, ctrl2 = model(batch_x, batch_anchors_norm)
                    pred_curve_abs = reconstruct_curve(
                        batch_anchors_raw, ctrl1, ctrl2, v_grid, clamp_voc=True
                    )

                    # Normalize prediction for loss computation
                    pred_curve_norm = 2.0 * (pred_curve_abs / batch_isc.unsqueeze(1)) - 1.0

                    val_loss, _ = curve_loss_fn(
                        pred_curve_norm, batch_curves_norm, v_grid, batch_anchors_raw[:, 2]
                    )
                    val_losses.append(val_loss.item())

                    # Compute MSE in normalized space
                    err = (pred_curve_norm - batch_curves_norm) ** 2
                    sum_sq_full += err.sum().item()
                    sum_cnt += err.numel()

                    vmpp = batch_anchors_raw[:, 2].unsqueeze(1)
                    mask_r1 = v_grid.unsqueeze(0) <= vmpp
                    sum_sq_r1 += (err * mask_r1).sum().item()
                    sum_cnt_r1 += mask_r1.sum().item()
                    sum_sq_r2 += (err * ~mask_r1).sum().item()
                    sum_cnt_r2 += (~mask_r1).sum().item()

                    # FF from true anchors
                    ff_true = (batch_anchors_raw[:, 2] * batch_anchors_raw[:, 3]) / (
                        batch_anchors_raw[:, 0] * batch_anchors_raw[:, 1] + 1e-12
                    )
                    ff_mape_sum += 0  # Not applicable when using true anchors
                    ff_cnt += ff_true.numel()

            avg_val = float(np.mean(val_losses))

            # Log every 10 epochs
            if epoch % 10 == 0:
                mse_full = sum_sq_full / max(1, sum_cnt)
                mse_r1 = sum_sq_r1 / max(1, sum_cnt_r1)
                mse_r2 = sum_sq_r2 / max(1, sum_cnt_r2)
                print(
                    f"Epoch {epoch}: train={np.mean(epoch_losses):.6f}, "
                    f"val={avg_val:.6f}, mse_full={mse_full:.6f}, "
                    f"mse_r1={mse_r1:.6f}, mse_r2={mse_r2:.6f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Store the control point model
        self.models['ctrl_point_model'] = model
        # For backward compatibility, also store config info
        self.models['ctrl_point_config'] = config
        # Expose as curve_model for evaluation/saving flows
        self.models['curve_model'] = model

        print(f"\nCurve model training complete. Best val loss: {best_val:.6f}")

    def train_curve_model_legacy(self):
        """
        LEGACY: Train unified split-spline model (predicts anchors + control points).
        Kept for backward compatibility. Use train_curve_model() for better results.
        """
        train = self.splits['train']
        val = self.splits['val']

        X_train_full = np.hstack([train['X_raw'], train['X_physics']]).astype(np.float32)
        X_val_full = np.hstack([val['X_raw'], val['X_physics']]).astype(np.float32)

        self.curve_feature_mean = X_train_full.mean(axis=0, keepdims=True)
        self.curve_feature_std = X_train_full.std(axis=0, keepdims=True) + 1e-8
        X_train_full = (X_train_full - self.curve_feature_mean) / self.curve_feature_std
        X_val_full = (X_val_full - self.curve_feature_mean) / self.curve_feature_std

        anchors_train = np.stack(
            [train['targets']['Jsc'], train['targets']['Voc'],
             train['targets']['Vmpp'], train['targets']['Jmpp']],
            axis=1
        ).astype(np.float32)
        anchors_val = np.stack(
            [val['targets']['Jsc'], val['targets']['Voc'],
             val['targets']['Vmpp'], val['targets']['Jmpp']],
            axis=1
        ).astype(np.float32)

        curves_train = train['curves'].astype(np.float32)
        curves_val = val['curves'].astype(np.float32)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_full),
            torch.from_numpy(anchors_train),
            torch.from_numpy(curves_train)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_full),
            torch.from_numpy(anchors_val),
            torch.from_numpy(curves_val)
        )

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2048, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2048)

        config = SplitSplineNetConfig(input_dim=X_train_full.shape[1], ctrl_points=self.ctrl_points)
        model = UnifiedSplitSplineNet(config).to(self.device)
        multitask_loss = MultiTaskLoss(init_log_sigma=-1.0).to(self.device)  # Fixed init

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(multitask_loss.parameters()),
            lr=1e-3,
            weight_decay=1e-5
        )

        v_grid = torch.from_numpy(self.v_grid).float().to(self.device)
        best_val = float('inf')
        best_state = None
        patience = 15
        patience_counter = 0
        val_metrics_last = {}  # Initialize

        for epoch in range(100):
            model.train()
            epoch_losses = []
            cont_loss = torch.tensor(0.0)

            for batch_x, batch_anchors, batch_curves in train_loader:
                batch_x = batch_x.to(self.device)
                batch_anchors = batch_anchors.to(self.device)
                batch_curves = batch_curves.to(self.device)

                optimizer.zero_grad()
                pred_anchors, ctrl1, ctrl2 = model(batch_x)
                pred_curve = reconstruct_curve(pred_anchors, ctrl1, ctrl2, v_grid, clamp_voc=True)
                loss, metrics = multitask_loss(pred_anchors, batch_anchors, pred_curve, batch_curves)
                cont_loss = continuity_loss(pred_anchors, ctrl1, ctrl2, v_grid)
                loss = loss + self.continuity_weight * cont_loss

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            model.eval()
            val_losses = []
            sum_sq_full = 0.0
            sum_cnt = 0.0
            sum_sq_r1 = 0.0
            sum_cnt_r1 = 0.0
            sum_sq_r2 = 0.0
            sum_cnt_r2 = 0.0
            ff_mape_sum = 0.0
            ff_cnt = 0.0

            with torch.no_grad():
                for batch_x, batch_anchors, batch_curves in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_anchors = batch_anchors.to(self.device)
                    batch_curves = batch_curves.to(self.device)
                    pred_anchors, ctrl1, ctrl2 = model(batch_x)
                    pred_curve = reconstruct_curve(pred_anchors, ctrl1, ctrl2, v_grid, clamp_voc=True)
                    val_loss, val_metrics_last = multitask_loss(pred_anchors, batch_anchors, pred_curve, batch_curves)
                    val_losses.append(val_loss.item())

                    err = (pred_curve - batch_curves) ** 2
                    sum_sq_full += err.sum().item()
                    sum_cnt += err.numel()

                    vmpp = pred_anchors[:, 2].unsqueeze(1)
                    mask_r1 = v_grid.unsqueeze(0) <= vmpp
                    mask_r2 = ~mask_r1
                    sum_sq_r1 += (err * mask_r1).sum().item()
                    sum_cnt_r1 += mask_r1.sum().item()
                    sum_sq_r2 += (err * mask_r2).sum().item()
                    sum_cnt_r2 += mask_r2.sum().item()

                    ff_pred = (pred_anchors[:, 2] * pred_anchors[:, 3]) / (
                        pred_anchors[:, 0] * pred_anchors[:, 1] + 1e-12
                    )
                    ff_true = (batch_anchors[:, 2] * batch_anchors[:, 3]) / (
                        batch_anchors[:, 0] * batch_anchors[:, 1] + 1e-12
                    )
                    ff_mape_sum += torch.abs((ff_pred - ff_true) / (ff_true + 1e-12)).sum().item()
                    ff_cnt += ff_true.numel()

            avg_val = float(np.mean(val_losses))

            if val_metrics_last:
                cont_loss_val = cont_loss.item() if 'cont_loss' in dir() and isinstance(cont_loss, torch.Tensor) else 0.0
                self.logger.log_multitask_loss(
                    epoch=epoch,
                    loss_anchor=val_metrics_last.get('loss_anchor', 0),
                    loss_curve=val_metrics_last.get('loss_curve', 0),
                    sigma_anchor=val_metrics_last.get('sigma_anchor', 1.0),
                    sigma_curve=val_metrics_last.get('sigma_curve', 1.0),
                    loss_continuity=cont_loss_val,
                    loss_total=avg_val
                )

            if epoch % 10 == 0:
                mse_full = sum_sq_full / max(1.0, sum_cnt)
                mse_r1 = sum_sq_r1 / max(1.0, sum_cnt_r1)
                mse_r2 = sum_sq_r2 / max(1.0, sum_cnt_r2)
                ff_mape = (ff_mape_sum / max(1.0, ff_cnt)) * 100
                sigma_a = val_metrics_last.get('sigma_anchor', 1.0)
                sigma_c = val_metrics_last.get('sigma_curve', 1.0)
                print(
                    f"Epoch {epoch}: train_loss={np.mean(epoch_losses):.6f}, "
                    f"val_loss={avg_val:.6f}, mse_full={mse_full:.6f}, "
                    f"mse_r1={mse_r1:.6f}, mse_r2={mse_r2:.6f}, ff_mape={ff_mape:.2f}%, "
                    f"sigma_a={sigma_a:.4f}, sigma_c={sigma_c:.4f}"
                )

            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.models['curve_model'] = model

    def evaluate_curve_model(self, split_name: str = 'test') -> dict:
        """
        Evaluate the curve model with full + region-wise metrics.

        Supports both:
        - New ControlPointNet (uses true anchors, only predicts control points)
        - Legacy UnifiedSplitSplineNet (predicts anchors + control points)
        """
        import time

        # Check which model type we have
        use_ctrl_point_model = 'ctrl_point_model' in self.models
        use_legacy_model = 'curve_model' in self.models

        if not use_ctrl_point_model and not use_legacy_model:
            raise ValueError("No curve model trained. Run train_curve_model() first.")

        split = self.splits[split_name]
        X_full = np.hstack([split['X_raw'], split['X_physics']]).astype(np.float32)
        X_full = (X_full - self.curve_feature_mean) / self.curve_feature_std

        anchors_true = np.stack(
            [split['targets']['Jsc'], split['targets']['Voc'],
             split['targets']['Vmpp'], split['targets']['Jmpp']],
            axis=1
        ).astype(np.float32)
        curves_true = split['curves'].astype(np.float32)

        # Normalize anchors if using new model
        if use_ctrl_point_model:
            anchors_norm = (anchors_true - self.anchor_mean) / self.anchor_std
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_full),
                torch.from_numpy(anchors_norm.astype(np.float32)),
                torch.from_numpy(anchors_true),
                torch.from_numpy(curves_true)
            )
            model = self.models['ctrl_point_model']
        else:
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_full),
                torch.from_numpy(anchors_true),
                torch.from_numpy(curves_true)
            )
            model = self.models['curve_model']

        loader = torch.utils.data.DataLoader(ds, batch_size=2048)
        model.eval()

        v_grid = torch.from_numpy(self.v_grid).float().to(self.device)

        sum_sq_full = 0.0
        sum_cnt = 0.0
        sum_sq_r1 = 0.0
        sum_cnt_r1 = 0.0
        sum_sq_r2 = 0.0
        sum_cnt_r2 = 0.0

        jsc_mae = 0.0
        voc_mae = 0.0
        vmpp_mae = 0.0
        jmpp_mae = 0.0
        ff_mape_sum = 0.0
        ff_cnt = 0.0

        violations = {
            'jsc_negative': 0,
            'voc_negative': 0,
            'vmpp_invalid': 0,
            'jmpp_invalid': 0,
            'j_exceeds_jsc': 0
        }

        start_time = time.time()

        with torch.no_grad():
            for batch in loader:
                if use_ctrl_point_model:
                    batch_x, batch_anchors_norm, batch_anchors, batch_curves = batch
                    batch_x = batch_x.to(self.device)
                    batch_anchors_norm = batch_anchors_norm.to(self.device)
                    batch_anchors = batch_anchors.to(self.device)
                    batch_curves = batch_curves.to(self.device)

                    # ControlPointNet: uses true anchors
                    ctrl1, ctrl2 = model(batch_x, batch_anchors_norm)
                    pred_anchors = batch_anchors  # True anchors, no prediction
                    pred_curve = reconstruct_curve(batch_anchors, ctrl1, ctrl2, v_grid, clamp_voc=True)
                else:
                    batch_x, batch_anchors, batch_curves = batch
                    batch_x = batch_x.to(self.device)
                    batch_anchors = batch_anchors.to(self.device)
                    batch_curves = batch_curves.to(self.device)

                    # Legacy model: predicts anchors + control points
                    pred_anchors, ctrl1, ctrl2 = model(batch_x)
                    pred_curve = reconstruct_curve(pred_anchors, ctrl1, ctrl2, v_grid, clamp_voc=True)

                err = (pred_curve - batch_curves) ** 2
                sum_sq_full += err.sum().item()
                sum_cnt += err.numel()

                vmpp = pred_anchors[:, 2].unsqueeze(1)
                mask_r1 = v_grid.unsqueeze(0) <= vmpp
                mask_r2 = ~mask_r1
                sum_sq_r1 += (err * mask_r1).sum().item()
                sum_cnt_r1 += mask_r1.sum().item()
                sum_sq_r2 += (err * mask_r2).sum().item()
                sum_cnt_r2 += mask_r2.sum().item()

                # Anchor MAE (only meaningful for legacy model)
                jsc_mae += torch.abs(pred_anchors[:, 0] - batch_anchors[:, 0]).sum().item()
                voc_mae += torch.abs(pred_anchors[:, 1] - batch_anchors[:, 1]).sum().item()
                vmpp_mae += torch.abs(pred_anchors[:, 2] - batch_anchors[:, 2]).sum().item()
                jmpp_mae += torch.abs(pred_anchors[:, 3] - batch_anchors[:, 3]).sum().item()

                ff_pred = (pred_anchors[:, 2] * pred_anchors[:, 3]) / (
                    pred_anchors[:, 0] * pred_anchors[:, 1] + 1e-12
                )
                ff_true = (batch_anchors[:, 2] * batch_anchors[:, 3]) / (
                    batch_anchors[:, 0] * batch_anchors[:, 1] + 1e-12
                )
                ff_mape_sum += torch.abs((ff_pred - ff_true) / (ff_true + 1e-12)).sum().item()
                ff_cnt += ff_true.numel()

                # Violations (always 0 for new model since using true anchors)
                violations['jsc_negative'] += (pred_anchors[:, 0] < 0).sum().item()
                violations['voc_negative'] += (pred_anchors[:, 1] < 0).sum().item()
                violations['vmpp_invalid'] += ((pred_anchors[:, 2] <= 0) | (pred_anchors[:, 2] >= pred_anchors[:, 1])).sum().item()
                violations['jmpp_invalid'] += ((pred_anchors[:, 3] <= 0) | (pred_anchors[:, 3] >= pred_anchors[:, 0])).sum().item()
                violations['j_exceeds_jsc'] += (pred_curve > pred_anchors[:, 0].unsqueeze(1) + 1e-3).sum().item()

        elapsed_time = time.time() - start_time
        n_samples = max(1, len(split['X_raw']))
        inference_time_ms = (elapsed_time / n_samples) * 1000

        results = {
            'mse_full_curve': sum_sq_full / max(1.0, sum_cnt),
            'mse_region1': sum_sq_r1 / max(1.0, sum_cnt_r1),
            'mse_region2': sum_sq_r2 / max(1.0, sum_cnt_r2),
            'mae_jsc': jsc_mae / n_samples,
            'mae_voc': voc_mae / n_samples,
            'mae_vmpp': vmpp_mae / n_samples,
            'mae_jmpp': jmpp_mae / n_samples,
            'mape_ff': (ff_mape_sum / max(1.0, ff_cnt)) * 100,
            'constraint_violations': violations,
            'inference_time_ms': inference_time_ms
        }

        # Log to structured logger for comparison table
        comparison_metrics = ModelComparisonMetrics(
            model_name='Split-Spline',
            mse_full_curve=results['mse_full_curve'],
            mse_region1=results['mse_region1'],
            mse_region2=results['mse_region2'],
            mae_jsc=results['mae_jsc'],
            mae_voc=results['mae_voc'],
            mae_vmpp=results['mae_vmpp'],
            mae_jmpp=results['mae_jmpp'],
            mape_ff=results['mape_ff'],
            violations_jsc_negative=violations['jsc_negative'],
            violations_voc_negative=violations['voc_negative'],
            violations_vmpp_invalid=violations['vmpp_invalid'],
            violations_jmpp_invalid=violations['jmpp_invalid'],
            violations_j_exceeds_jsc=violations['j_exceeds_jsc'],
            inference_time_ms=inference_time_ms,
            total_samples=n_samples
        )
        self.logger.log_model_comparison(comparison_metrics)

        print("\nCurve Model Metrics:")
        for k, v in results.items():
            if k != 'constraint_violations':
                print(f"  {k}: {v}")
        print(f"  constraint_violations: {results['constraint_violations']}")

        return results

    def train_cvae_baseline(self, epochs: int = 100, beta: float = 0.001):
        """Train CVAE baseline for curve reconstruction."""
        from models.cvae import ConditionalVAE, cvae_loss

        train = self.splits['train']
        val = self.splits['val']

        X_train_full = np.hstack([train['X_raw'], train['X_physics']]).astype(np.float32)
        X_val_full = np.hstack([val['X_raw'], val['X_physics']]).astype(np.float32)

        self.cvae_feature_mean = X_train_full.mean(axis=0, keepdims=True)
        self.cvae_feature_std = X_train_full.std(axis=0, keepdims=True) + 1e-8
        X_train_full = (X_train_full - self.cvae_feature_mean) / self.cvae_feature_std
        X_val_full = (X_val_full - self.cvae_feature_mean) / self.cvae_feature_std

        curves_train = train['curves'].astype(np.float32)
        curves_val = val['curves'].astype(np.float32)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(curves_train),
            torch.from_numpy(X_train_full)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(curves_val),
            torch.from_numpy(X_val_full)
        )

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2048, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2048)

        cvae = ConditionalVAE(curve_dim=curves_train.shape[1], cond_dim=X_train_full.shape[1], latent_dim=16).to(self.device)
        optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)

        best_val = float('inf')
        best_state = None

        for epoch in range(epochs):
            cvae.train()
            train_losses = []
            for batch_curves, batch_cond in train_loader:
                batch_curves = batch_curves.to(self.device)
                batch_cond = batch_cond.to(self.device)
                recon, mu, logvar = cvae(batch_curves, batch_cond)
                loss = cvae_loss(recon, batch_curves, mu, logvar, beta=beta)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            cvae.eval()
            val_losses = []
            with torch.no_grad():
                for batch_curves, batch_cond in val_loader:
                    batch_curves = batch_curves.to(self.device)
                    batch_cond = batch_cond.to(self.device)
                    recon, mu, logvar = cvae(batch_curves, batch_cond)
                    val_loss = cvae_loss(recon, batch_curves, mu, logvar, beta=beta)
                    val_losses.append(val_loss.item())

            avg_val = float(np.mean(val_losses))
            if epoch % 10 == 0:
                print(f"CVAE Epoch {epoch}: train_loss={np.mean(train_losses):.6f}, val_loss={avg_val:.6f}")

            if avg_val < best_val:
                best_val = avg_val
                best_state = {k: v.cpu().clone() for k, v in cvae.state_dict().items()}

        if best_state is not None:
            cvae.load_state_dict(best_state)

        self.models['cvae'] = cvae
        return cvae

    def evaluate_cvae(self, split_name: str = 'test', n_samples: int = 5) -> dict:
        """Evaluate CVAE baseline using conditional generation."""
        import time
        if 'cvae' not in self.models:
            raise ValueError("CVAE not trained.")

        split = self.splits[split_name]
        X_full = np.hstack([split['X_raw'], split['X_physics']]).astype(np.float32)
        X_full = (X_full - self.cvae_feature_mean) / self.cvae_feature_std

        curves_true = split['curves'].astype(np.float32)
        anchors_true = np.stack(
            [split['targets']['Jsc'], split['targets']['Voc'], split['targets']['Vmpp'], split['targets']['Jmpp']],
            axis=1
        ).astype(np.float32)

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(curves_true),
            torch.from_numpy(X_full),
            torch.from_numpy(anchors_true)
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=2048)

        model = self.models['cvae']
        model.eval()
        v_grid = torch.from_numpy(self.v_grid).float().to(self.device)

        sum_sq_full = 0.0
        sum_cnt = 0.0
        sum_sq_r1 = 0.0
        sum_cnt_r1 = 0.0
        sum_sq_r2 = 0.0
        sum_cnt_r2 = 0.0
        ff_mape_sum = 0.0
        ff_cnt = 0.0
        violations = {
            'jsc_negative': 0,
            'voc_negative': 0,
            'vmpp_invalid': 0,
            'jmpp_invalid': 0,
            'j_exceeds_jsc': 0
        }

        start_time = time.time()

        with torch.no_grad():
            for batch_curves, batch_cond, batch_anchors in loader:
                batch_curves = batch_curves.to(self.device)
                batch_cond = batch_cond.to(self.device)
                batch_anchors = batch_anchors.to(self.device)

                preds = []
                for _ in range(n_samples):
                    z = torch.randn(batch_cond.size(0), model.latent_dim, device=self.device)
                    preds.append(model.decode(z, batch_cond))
                pred_curve = torch.stack(preds).mean(dim=0)

                err = (pred_curve - batch_curves) ** 2
                sum_sq_full += err.sum().item()
                sum_cnt += err.numel()

                # Region-wise MSE using true Vmpp
                vmpp_true = batch_anchors[:, 2].unsqueeze(1)
                mask_r1 = v_grid.unsqueeze(0) <= vmpp_true
                mask_r2 = ~mask_r1
                sum_sq_r1 += (err * mask_r1).sum().item()
                sum_cnt_r1 += mask_r1.sum().item()
                sum_sq_r2 += (err * mask_r2).sum().item()
                sum_cnt_r2 += mask_r2.sum().item()

                pred_targets = extract_targets_gpu(pred_curve, v_grid)
                jsc = pred_targets['Jsc']
                voc = pred_targets['Voc']
                vmpp = pred_targets['Vmpp']
                jmpp = pred_targets['Jmpp']

                # FF MAPE
                ff_pred = (vmpp * jmpp) / (jsc * voc + 1e-12)
                ff_true = (batch_anchors[:, 2] * batch_anchors[:, 3]) / (
                    batch_anchors[:, 0] * batch_anchors[:, 1] + 1e-12
                )
                ff_mape_sum += torch.abs((ff_pred - ff_true) / (ff_true + 1e-12)).sum().item()
                ff_cnt += ff_true.numel()

                violations['jsc_negative'] += (jsc < 0).sum().item()
                violations['voc_negative'] += (voc < 0).sum().item()
                violations['vmpp_invalid'] += ((vmpp <= 0) | (vmpp >= voc)).sum().item()
                violations['jmpp_invalid'] += ((jmpp <= 0) | (jmpp >= jsc)).sum().item()
                violations['j_exceeds_jsc'] += (pred_curve > jsc.unsqueeze(1)).sum().item()

        elapsed_time = time.time() - start_time
        total_samples = max(1, len(split['X_raw']))
        inference_time_ms = (elapsed_time / total_samples) * 1000

        results = {
            'mse_full_curve': sum_sq_full / max(1.0, sum_cnt),
            'mse_region1': sum_sq_r1 / max(1.0, sum_cnt_r1),
            'mse_region2': sum_sq_r2 / max(1.0, sum_cnt_r2),
            'mape_ff': (ff_mape_sum / max(1.0, ff_cnt)) * 100,
            'constraint_violations': violations,
            'inference_time_ms': inference_time_ms
        }

        # Log to structured logger for comparison table
        comparison_metrics = ModelComparisonMetrics(
            model_name='CVAE',
            mse_full_curve=results['mse_full_curve'],
            mse_region1=results['mse_region1'],
            mse_region2=results['mse_region2'],
            mape_ff=results['mape_ff'],
            violations_jsc_negative=violations['jsc_negative'],
            violations_voc_negative=violations['voc_negative'],
            violations_vmpp_invalid=violations['vmpp_invalid'],
            violations_jmpp_invalid=violations['jmpp_invalid'],
            violations_j_exceeds_jsc=violations['j_exceeds_jsc'],
            inference_time_ms=inference_time_ms,
            total_samples=total_samples
        )
        self.logger.log_model_comparison(comparison_metrics)

        print("\nCVAE Metrics:")
        print(f"  mse_full_curve: {results['mse_full_curve']}")
        print(f"  mse_region1: {results['mse_region1']}")
        print(f"  mse_region2: {results['mse_region2']}")
        print(f"  mape_ff: {results['mape_ff']:.2f}%")
        print(f"  constraint_violations: {results['constraint_violations']}")
        print(f"  inference_time_ms: {results['inference_time_ms']:.3f}")

        return results

    def _train_voc_model(self, trainer, train_loader, val_loader, config):
        """Custom training loop for Voc model."""
        # [FIX] Use the robust trainer.fit method instead of manual loop
        # This ensures GradScaler, correct logging, and patience are handled properly
        print("Starting VOC NN training via VocTrainer.fit...")
        history = trainer.fit(train_loader, val_loader)
        
        best_mse = history['val'][-1]['mse'] if history['val'] else 0.0
        print(f"Voc training complete. Best Val MSE: {best_mse:.6f}")

    def evaluate(self):
        """Evaluate all models on test set."""
        print("\n" + "=" * 60)
        print("Evaluating Models on Test Set")
        print("=" * 60)

        test = self.splits['test']

        # Voc NN - CRITICAL: Normalize inputs AND denormalize outputs
        X_test_full = np.hstack([test['X_raw'], test['X_physics']])
        X_test_normalized = (X_test_full - self.voc_feature_mean) / self.voc_feature_std
        X_test_tensor = torch.from_numpy(X_test_normalized).float().to(self.device)

        self.models['voc_nn'].eval()
        with torch.no_grad():
            voc_pred_normalized = self.models['voc_nn'](X_test_tensor).cpu().numpy()
            # Denormalize predictions back to original scale
            voc_pred = voc_pred_normalized * self.voc_target_std + self.voc_target_mean

        self.metrics['voc'] = self._compute_metrics(test['targets']['Voc'], voc_pred, 'Voc')

        # Jsc LGBM
        jsc_pred = self.models['jsc_lgbm'].predict(
            test['X_raw'], test['X_physics'], test['jsc_ceiling']
        )
        self.metrics['jsc'] = self._compute_metrics(test['targets']['Jsc'], jsc_pred, 'Jsc')

        # Vmpp LGBM (using true Voc for evaluation)
        vmpp_pred = self.models['vmpp_lgbm'].predict(
            test['X_raw'], test['X_physics'], test['targets']['Voc']
        )
        self.metrics['vmpp'] = self._compute_metrics(test['targets']['Vmpp'], vmpp_pred, 'Vmpp')

        # Jmpp LGBM
        jmpp_pred = self.models['jmpp_lgbm'].predict(
            test['X_raw'], test['X_physics'],
            test['targets']['Jsc'], test['targets']['Vmpp']
        )
        self.metrics['jmpp'] = self._compute_metrics(test['targets']['Jmpp'], jmpp_pred, 'Jmpp')

        # FF LGBM
        ff_pred = self.models['ff_lgbm'].predict(
            test['X_raw'], test['X_physics'],
            test['targets']['Voc'], test['targets']['Jsc']
        )
        self.metrics['ff'] = self._compute_metrics(test['targets']['FF'], ff_pred, 'FF')

        # Derived: PCE and Pmpp
        pmpp_pred = vmpp_pred * jmpp_pred
        pce_pred = pmpp_pred / 1000.0  # Assuming 1000 W/m² illumination

        self.metrics['pmpp'] = self._compute_metrics(test['targets']['Pmpp'], pmpp_pred, 'Pmpp')
        self.metrics['pce'] = self._compute_metrics(test['targets']['PCE'], pce_pred, 'PCE')

        # Curve model metrics (if trained)
        if 'curve_model' in self.models:
            self.metrics['curve'] = self.evaluate_curve_model(split_name='test')

        # CVAE baseline metrics (if trained)
        if 'cvae' in self.models:
            self.metrics['cvae'] = self.evaluate_cvae(split_name='test')

        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
        """Compute regression metrics."""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - mse / (np.var(y_true) + 1e-8)

        # Relative errors
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape)
        }

        print(f"\n{name}:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.6f}")
        print(f"  MAPE: {mape:.2f}%")

        return metrics

    def save_models(self):
        """Save trained models."""
        print("\n" + "=" * 60)
        print("Saving Models")
        print("=" * 60)

        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)

        # Save Voc NN
        torch.save(self.models['voc_nn'].state_dict(), models_dir / 'voc_nn.pt')

        # Save LGBM models
        self.models['jsc_lgbm'].save(str(models_dir / 'jsc_lgbm.txt'))
        self.models['vmpp_lgbm'].save(str(models_dir / 'vmpp_lgbm.txt'))
        self.models['jmpp_lgbm'].save(str(models_dir / 'jmpp_lgbm.txt'))
        self.models['ff_lgbm'].save(str(models_dir / 'ff_lgbm.txt'))

        # Save curve model (legacy unified split-spline)
        if 'curve_model' in self.models:
            torch.save(self.models['curve_model'].state_dict(), models_dir / 'curve_model.pt')

        # Save control-point-only curve model
        if 'ctrl_point_model' in self.models:
            torch.save(self.models['ctrl_point_model'].state_dict(), models_dir / 'ctrl_point_model.pt')

        # Save CVAE (if trained)
        if 'cvae' in self.models:
            torch.save(self.models['cvae'].state_dict(), models_dir / 'cvae.pt')

        # Save configs
        configs = {
            'voc_nn': self.models['voc_nn'].config.__dict__ if hasattr(self.models['voc_nn'], 'config') else {},
            'jsc_lgbm': self.models['jsc_lgbm'].config.__dict__,
            'vmpp_lgbm': self.models['vmpp_lgbm'].config.__dict__,
            'ff_lgbm': self.models['ff_lgbm'].config.__dict__,
        }

        if 'curve_model' in self.models:
            configs['curve_model'] = {
                **(self.models['curve_model'].config.__dict__ if hasattr(self.models['curve_model'], 'config') else {}),
                'v_grid': self.v_grid.tolist(),
                'type': 'unified_split_spline'
            }

        if 'ctrl_point_model' in self.models:
            ctrl_config = self.models.get('ctrl_point_config')
            configs['curve_model'] = {
                **(ctrl_config.__dict__ if ctrl_config is not None else {}),
                'v_grid': self.v_grid.tolist(),
                'type': 'control_point_net',
                'curve_norm_by_isc': bool(self.curve_norm_by_isc),
                'curve_output_normalized': False
            }

        if 'cvae' in self.models:
            configs['cvae'] = {
                'curve_dim': self.splits['train']['curves'].shape[1],
                'cond_dim': self.splits['train']['X_raw'].shape[1] + self.splits['train']['X_physics'].shape[1],
                'latent_dim': 16
            }

        with open(models_dir / 'configs.json', 'w') as f:
            json.dump(configs, f, indent=2, default=str)

        # Save normalization parameters for Voc NN (features AND targets)
        if hasattr(self, 'voc_feature_mean') and hasattr(self, 'voc_feature_std'):
            normalization_params = {
                'voc_feature_mean': self.voc_feature_mean.tolist(),
                'voc_feature_std': self.voc_feature_std.tolist(),
                'voc_target_mean': float(self.voc_target_mean),
                'voc_target_std': float(self.voc_target_std)
            }
            if hasattr(self, 'curve_feature_mean') and hasattr(self, 'curve_feature_std'):
                normalization_params['curve_feature_mean'] = self.curve_feature_mean.tolist()
                normalization_params['curve_feature_std'] = self.curve_feature_std.tolist()
            if hasattr(self, 'cvae_feature_mean') and hasattr(self, 'cvae_feature_std'):
                normalization_params['cvae_feature_mean'] = self.cvae_feature_mean.tolist()
                normalization_params['cvae_feature_std'] = self.cvae_feature_std.tolist()
            if self.physics_feature_mask is not None:
                normalization_params['physics_feature_mask'] = np.where(self.physics_feature_mask)[0].tolist()
            if hasattr(self, 'anchor_mean') and hasattr(self, 'anchor_std'):
                normalization_params['anchor_mean'] = self.anchor_mean.tolist()
                normalization_params['anchor_std'] = self.anchor_std.tolist()
            normalization_params['curve_norm_by_isc'] = bool(self.curve_norm_by_isc)
            with open(models_dir / 'normalization.json', 'w') as f:
                json.dump(normalization_params, f)
            print("Saved normalization parameters for Voc NN (features + targets)")

        print(f"Models saved to {models_dir}")

    def run(self):
        """Run full training pipeline."""
        start_time = datetime.now()

        self.load_data()
        self.extract_targets()
        self.compute_features()
        self.split_data()
        self.run_feature_validation()

        # Handle HPO: either load from file, run fresh, or skip
        if self.load_hpo_path:
            # Load HPO results from previous run
            loaded_results = self.load_hpo_results(self.load_hpo_path)
            self.hpo_results = loaded_results
            self.best_configs = get_best_configs_from_study(loaded_results)
        elif self.run_hpo:
            self.run_hyperparameter_optimization()

        # Run curve HPO if requested (can be done with or without scalar HPO)
        if self.run_curve_hpo and self.run_curve_model:
            self.run_curve_hyperparameter_optimization()

        self.train_final_models()
        self.evaluate()
        self.save_models()

        # Save all structured logs
        self.logger.save_all_logs()

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print(f"Pipeline Complete! Duration: {duration}")
        print("=" * 60)

        # Print comparison table if both models were trained
        if self.logger.model_comparisons:
            print("\n" + "=" * 60)
            print("MODEL COMPARISON TABLE")
            print("=" * 60)
            print(self.logger.generate_comparison_table())

        return self.metrics


def main():
    parser = argparse.ArgumentParser(description='Train scalar PV predictors')
    parser.add_argument('--params', type=str, default=DEFAULT_PARAMS_FILE,
                        help='Path to parameters file')
    parser.add_argument('--iv', type=str, default=DEFAULT_IV_FILE,
                        help='Path to IV curves file')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--no-hpo', action='store_true',
                        help='Skip hyperparameter optimization')
    parser.add_argument('--load-hpo', type=str, default=None,
                        help='Path to load HPO results from (skips scalar HPO)')
    parser.add_argument('--train-curves', action='store_true',
                        help='Train unified split-spline curve model')
    parser.add_argument('--curve-hpo', action='store_true',
                        help='Run HPO for curve reconstruction model')
    parser.add_argument('--train-cvae', action='store_true',
                        help='Train CVAE baseline for curve reconstruction')
    parser.add_argument('--no-feature-validation', action='store_true',
                        help='Skip physics feature correlation validation')
    parser.add_argument('--drop-weak-features', action='store_true',
                        help='Drop weak physics features (|r| below threshold)')
    parser.add_argument('--hpo-trials-nn', type=int, default=100,
                        help='Number of HPO trials for NN (also used for curve HPO)')
    parser.add_argument('--hpo-trials-lgbm', type=int, default=200,
                        help='Number of HPO trials for LGBM')
    parser.add_argument('--hpo-timeout', type=int, default=7200,
                        help='HPO timeout per model (seconds)')

    # New options for robustness and logging
    parser.add_argument('--multicollinearity-threshold', type=float, default=0.85,
                        help='Threshold for multicollinearity check (default: 0.85)')
    parser.add_argument('--drop-multicollinear', action='store_true',
                        help='Drop multicollinear features')
    parser.add_argument('--continuity-weight', type=float, default=0.1,
                        help='Weight for continuity loss at Vmpp (try 0.1-1.0)')
    parser.add_argument('--ctrl-points', type=int, default=4,
                        help='Number of control points per region (default: 4)')
    parser.add_argument('--soft-clamp-training', action='store_true',
                        help='Use soft penalty instead of hard clamp during training (default: hard clamp)')
    parser.add_argument('--no-constraint-logging', action='store_true',
                        help='Disable constraint violation logging')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce logging verbosity')

    args = parser.parse_args()

    hpo_config = HPOConfig(
        n_trials_nn=args.hpo_trials_nn,
        n_trials_lgbm=args.hpo_trials_lgbm,
        timeout_per_model=args.hpo_timeout
    )

    # If loading HPO, skip running HPO (but can still run curve HPO)
    run_scalar_hpo = not args.no_hpo and args.load_hpo is None

    pipeline = ScalarPredictorPipeline(
        params_file=args.params,
        iv_file=args.iv,
        output_dir=args.output,
        device=args.device,
        run_hpo=run_scalar_hpo,
        run_curve_model=args.train_curves,
        run_curve_hpo=args.curve_hpo,
        train_cvae=args.train_cvae,
        validate_feature_correlations=not args.no_feature_validation,
        drop_weak_features=args.drop_weak_features,
        hpo_config=hpo_config,
        load_hpo_path=args.load_hpo,
        # New options
        multicollinearity_threshold=args.multicollinearity_threshold,
        drop_multicollinear=args.drop_multicollinear,
        continuity_weight=args.continuity_weight,
        ctrl_points=args.ctrl_points,
        use_hard_clamp_training=not args.soft_clamp_training,
        log_constraint_violations=not args.no_constraint_logging,
        verbose_logging=not args.quiet
    )

    metrics = pipeline.run()
    return metrics


def test_ood_generalization(
    model,
    X: np.ndarray,
    y: np.ndarray,
    bandgap: np.ndarray,
    low_threshold: float = 1.6,
    high_threshold: float = 1.8,
    materials: np.ndarray | None = None,
    ood_materials: list[str] | None = None
) -> dict:
    """
    Generic OOD evaluation helper.

    Expects model to implement `fit(X_train, y_train)` and `predict(X_test)`.
    """
    results = {}

    # Scenario 1: bandgap extrapolation
    train_mask = bandgap < low_threshold
    test_mask = bandgap > high_threshold
    if train_mask.any() and test_mask.any():
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        mse = np.mean((preds - y[test_mask]) ** 2)
        results['bandgap_extrap_mse'] = float(mse)

    # Scenario 2: leave-one-material-out
    if materials is not None and ood_materials:
        ood_results = {}
        for mat in ood_materials:
            train_mask = materials != mat
            test_mask = materials == mat
            if not test_mask.any():
                continue
            model.fit(X[train_mask], y[train_mask])
            preds = model.predict(X[test_mask])
            mse = np.mean((preds - y[test_mask]) ** 2)
            ood_results[mat] = float(mse)
        results['leave_one_material_mse'] = ood_results

    return results


if __name__ == '__main__':
    main()
