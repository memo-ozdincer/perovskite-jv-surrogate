"""
Ultra-fast distributed Hyperparameter Optimization using Optuna.
Designed for maximum parallelization on H100 GPU + EPYC CPU clusters.
"""
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import HyperbandPruner, MedianPruner
import numpy as np
import torch
import lightgbm as lgb
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

from models.voc_nn import VocNNConfig, VocNN, VocTrainer, SplitSplineNetConfig, UnifiedSplitSplineNet
from models.jsc_lgbm import JscLGBMConfig, JscLGBM
from models.vmpp_lgbm import VmppLGBMConfig, VmppLGBM, JmppLGBM, FFLGBM
from models.reconstruction import reconstruct_curve, continuity_loss
from models.direct_curve import (
    DirectCurveShapeNet, DirectCurveShapeNetConfig, DirectCurveShapeLoss,
    reconstruct_curve_shape
)


@dataclass
class HPOConfig:
    """
    Global HPO configuration.

    UPDATED v2.0: Increased trial counts and relaxed pruning to allow
    more thorough exploration. Previous settings were too aggressive
    and led to meager HPO gains.
    """
    # Trial counts - increased for more thorough exploration
    n_trials_nn: int = 200       # WAS 100
    n_trials_lgbm: int = 300     # WAS 200

    # Parallelization
    n_parallel_trials: int = 24  # Match GPU node cores
    n_startup_trials: int = 75   # WAS 50 - more random exploration before TPE

    # Timeouts - increased for longer training runs
    timeout_per_model: int = 14400  # WAS 7200 (4h vs 2h)

    # Pruning - relaxed to allow more trials to complete
    use_pruning: bool = True
    pruning_warmup: int = 15          # WAS 10
    pruner_min_resource: int = 10     # NEW: min epochs before pruning
    pruner_reduction_factor: int = 4  # NEW: less aggressive (was 3)

    # Storage (for distributed)
    storage: str = None  # Use in-memory by default
    study_name_prefix: str = 'pv_scalar'


# ============================================================================
# NEURAL NETWORK HPO (for Voc)
# ============================================================================

def sample_voc_nn_config(trial: optuna.Trial, input_dim: int) -> VocNNConfig:
    """
    Sample hyperparameters for Voc neural network.

    UPDATED v2.0: Widened search space to allow HPO to find better configs.
    Previous ranges were too narrow and often converged to suboptimal solutions.
    """

    # Architecture - allow deeper networks with more capacity
    n_layers = trial.suggest_int('n_layers', 2, 5)  # WAS [2, 3]
    hidden_dims = []
    for i in range(n_layers):
        # Tapering architecture: wide -> narrow
        if i == 0:
            dim = trial.suggest_categorical(f'hidden_{i}', [128, 256, 384, 512, 768])  # Added 384, 768
        elif i == 1:
            dim = trial.suggest_categorical(f'hidden_{i}', [64, 128, 256, 384, 512])   # Added 384, 512
        else:
            dim = trial.suggest_categorical(f'hidden_{i}', [32, 64, 128, 256])         # Added 32
        hidden_dims.append(dim)

    # Jacobian regularization - expanded range
    jacobian_weight = trial.suggest_categorical(
        'jacobian_weight',
        [0.0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]  # Added 1e-6, 1e-1
    )

    return VocNNConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,

        # Regularization - wider range for dropout
        dropout=trial.suggest_float('dropout', 0.05, 0.35),  # WAS [0.1, 0.25]
        use_layer_norm=True,  # Always use for stability
        use_residual=trial.suggest_categorical('use_residual', [True, False]),

        # Activation - GELU is usually best for smooth problems
        activation=trial.suggest_categorical('activation', ['gelu', 'silu', 'relu']),  # Added relu

        # Regularizers
        jacobian_weight=jacobian_weight,
        physics_weight=trial.suggest_float('physics_weight', 0.0, 0.2),  # WAS [0.0, 0.1]

        # Optimizer - WIDER learning rate range
        lr=trial.suggest_float('lr', 1e-5, 5e-3, log=True),  # WAS [5e-5, 1e-3]
        weight_decay=trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True),  # Expanded

        # Training - wider epoch range, let HPO decide
        epochs=trial.suggest_int('epochs', 100, 400),  # WAS [150, 300]
        patience=trial.suggest_int('patience', 20, 50),  # WAS [25, 40]
        use_amp=True,
    )


class VocNNObjective:
    """Optuna objective for Voc neural network."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        voc_ceiling_train: np.ndarray,
        voc_ceiling_val: np.ndarray,
        device: torch.device,
        batch_size: int = 4096
    ):
        self.X_train = torch.from_numpy(X_train).float()
        self.X_val = torch.from_numpy(X_val).float()
        self.ceiling_train = torch.from_numpy(voc_ceiling_train).float()
        self.ceiling_val = torch.from_numpy(voc_ceiling_val).float()
        self.device = device
        self.batch_size = batch_size
        self.input_dim = X_train.shape[1]

        # Normalize targets for stable training
        self.y_mean = y_train.mean()
        self.y_std = y_train.std() + 1e-8
        self.y_train = torch.from_numpy((y_train - self.y_mean) / self.y_std).float()
        self.y_val = torch.from_numpy((y_val - self.y_mean) / self.y_std).float()

    def __call__(self, trial: optuna.Trial) -> float:
        config = sample_voc_nn_config(trial, self.input_dim)

        # Build model
        model = VocNN(config).to(self.device)
        trainer = VocTrainer(model, config, self.device)

        # UPDATED v2.0: Add LR scheduler to match final training
        # This ensures HPO evaluates configs under the same training regime
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            trainer.optimizer, T_0=10, T_mult=2
        )

        # Create data loaders
        train_ds = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        val_ds = torch.utils.data.TensorDataset(self.X_val, self.y_val)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        # Train with pruning callbacks
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(config.epochs):
            # Train epoch - MATCH ACTUAL TRAINING OBJECTIVE
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                trainer.optimizer.zero_grad()
                # Use same loss as actual training: MSE + Jacobian regularization
                pred, jac_norm = model.forward_with_jacobian(batch_x)
                loss = torch.nn.functional.mse_loss(pred, batch_y)
                loss = loss + config.jacobian_weight * jac_norm
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                trainer.optimizer.step()

            # Step the LR scheduler (matches final training)
            scheduler.step()

            # Validate
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(self.device)
                    pred = model(batch_x)
                    val_preds.append(pred)

                    # DEBUG: Check for identical predictions (remove after debugging)
                    if epoch == 0 and trial.number % 50 == 0:
                        print(f"\nTrial {trial.number}, Epoch {epoch}:")
                        print(f"  Pred range: [{pred.min().item():.6f}, {pred.max().item():.6f}]")
                        print(f"  Pred mean: {pred.mean().item():.6f}, std: {pred.std().item():.6f}")

            val_pred = torch.cat(val_preds)
            val_loss = torch.nn.functional.mse_loss(val_pred, self.y_val.to(self.device)).item()

            # Report for pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break

        return best_val_loss


# ============================================================================
# LGBM HPO (for Jsc, Vmpp, etc.)
# ============================================================================

def sample_lgbm_config(trial: optuna.Trial, model_type: str = 'jsc') -> dict:
    """
    Sample hyperparameters for LightGBM.

    SIMPLIFIED for robustness:
    - Use GBDT only (DART is slower and rarely better)
    - Narrower ranges based on what typically works
    - Lower learning rates for stability
    """
    params = {
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'boosting_type': 'gbdt',  # GBDT is faster and usually sufficient
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1,
        'force_col_wise': True,

        # Tree structure - moderate complexity
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1.0, log=True),

        # Learning rate and iterations - LOWER LR for stability
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),

        # Sampling - moderate ranges
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),

        # Regularization - moderate L1/L2
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),

        # Feature/bagging fraction
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),

        # Path smoothing
        'path_smooth': trial.suggest_float('path_smooth', 0.0, 0.5),
    }

    return params


def default_lgbm_config(model_type: str = 'jsc') -> dict:
    """Deterministic fallback config when no trials complete."""
    return {
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1,
        'force_col_wise': True,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 20,
        'min_child_weight': 1e-2,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 1e-3,
        'reg_lambda': 1e-3,
        'min_split_gain': 0.0,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'path_smooth': 0.0,
    }


class LGBMObjective:
    """Generic Optuna objective for LGBM models."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = 50
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.early_stopping_rounds = early_stopping_rounds

    def __call__(self, trial: optuna.Trial) -> float:
        params = sample_lgbm_config(trial)

        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)

        # Custom callback for Optuna pruning
        # Fix: Use just 'rmse' instead of 'valid_0 rmse' for newer LightGBM versions
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'rmse')

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds),
            pruning_callback
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=params.pop('n_estimators'),
            valid_sets=[val_data],
            valid_names=['valid_0'],
            callbacks=callbacks
        )

        # Return validation RMSE
        val_pred = model.predict(self.X_val)
        rmse = np.sqrt(np.mean((val_pred - self.y_val) ** 2))
        return rmse


# ============================================================================
# CURVE MODEL HPO (for Split-Spline reconstruction)
# ============================================================================

def sample_curve_config(trial: optuna.Trial, input_dim: int) -> SplitSplineNetConfig:
    """
    Sample hyperparameters for curve reconstruction model.

    Key hyperparameters:
    - Architecture: hidden dims, dropout
    - Control points: number per region
    - Training: lr, weight decay, continuity weight
    """
    # Architecture - moderate complexity
    n_layers = trial.suggest_int('n_layers', 2, 4)
    hidden_dims = []
    for i in range(n_layers):
        if i == 0:
            dim = trial.suggest_categorical(f'hidden_{i}', [256, 512])
        elif i == 1:
            dim = trial.suggest_categorical(f'hidden_{i}', [128, 256])
        else:
            dim = trial.suggest_categorical(f'hidden_{i}', [64, 128])
        hidden_dims.append(dim)

    return SplitSplineNetConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=trial.suggest_float('dropout', 0.1, 0.3),
        activation=trial.suggest_categorical('activation', ['silu', 'gelu']),
        ctrl_points=trial.suggest_int('ctrl_points', 3, 6),
    )


class CurveObjective:
    """Optuna objective for curve reconstruction model."""

    def __init__(
        self,
        X_train: np.ndarray,
        anchors_train: np.ndarray,
        curves_train: np.ndarray,
        X_val: np.ndarray,
        anchors_val: np.ndarray,
        curves_val: np.ndarray,
        v_grid: np.ndarray,
        device: torch.device,
        batch_size: int = 2048,
        max_epochs: int = 50,  # Reduced for HPO
        patience: int = 10
    ):
        self.X_train = torch.from_numpy(X_train).float()
        self.X_val = torch.from_numpy(X_val).float()
        self.anchors_train = torch.from_numpy(anchors_train).float()
        self.anchors_val = torch.from_numpy(anchors_val).float()
        self.curves_train = torch.from_numpy(curves_train).float()
        self.curves_val = torch.from_numpy(curves_val).float()
        self.v_grid = torch.from_numpy(v_grid).float()
        self.device = device
        self.batch_size = batch_size
        self.input_dim = X_train.shape[1]
        self.max_epochs = max_epochs
        self.patience = patience

    def __call__(self, trial: optuna.Trial) -> float:
        config = sample_curve_config(trial, self.input_dim)

        # Additional training hyperparameters
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        continuity_weight = trial.suggest_float('continuity_weight', 0.05, 0.5, log=True)

        model = UnifiedSplitSplineNet(config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_ds = torch.utils.data.TensorDataset(
            self.X_train, self.anchors_train, self.curves_train
        )
        val_ds = torch.utils.data.TensorDataset(
            self.X_val, self.anchors_val, self.curves_val
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        v_grid = self.v_grid.to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Train
            model.train()
            for batch_x, batch_anchors, batch_curves in train_loader:
                batch_x = batch_x.to(self.device)
                batch_anchors = batch_anchors.to(self.device)
                batch_curves = batch_curves.to(self.device)

                optimizer.zero_grad()
                pred_anchors, ctrl1, ctrl2 = model(batch_x)
                pred_curve = reconstruct_curve(pred_anchors, ctrl1, ctrl2, v_grid, clamp_voc=True)

                # Combined loss: anchor MSE + curve MSE + continuity
                loss_anchor = torch.nn.functional.mse_loss(pred_anchors, batch_anchors)
                loss_curve = torch.nn.functional.mse_loss(pred_curve, batch_curves)
                loss_cont = continuity_loss(pred_anchors, ctrl1, ctrl2, v_grid)

                loss = loss_anchor + loss_curve + continuity_weight * loss_cont

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_anchors, batch_curves in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_anchors = batch_anchors.to(self.device)
                    batch_curves = batch_curves.to(self.device)

                    pred_anchors, ctrl1, ctrl2 = model(batch_x)
                    pred_curve = reconstruct_curve(pred_anchors, ctrl1, ctrl2, v_grid, clamp_voc=True)

                    val_loss = torch.nn.functional.mse_loss(pred_curve, batch_curves)
                    val_losses.append(val_loss.item())

            avg_val_loss = float(np.mean(val_losses))

            # Report for pruning
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        return best_val_loss


# ============================================================================
# DIRECT CURVE SHAPE HPO (NEW v2.0)
# ============================================================================

def sample_direct_curve_shape_config(trial: optuna.Trial, input_dim: int) -> DirectCurveShapeNetConfig:
    """
    Sample hyperparameters for DirectCurveShapeNet.

    This is the shape-only curve model that uses pretrained Jsc/Voc.
    Key hyperparameters to tune:
    - Architecture (hidden_dims, dropout)
    - Control points for shape flexibility
    - Residual connections
    """
    # Architecture - allow varying depth and width
    n_layers = trial.suggest_int('n_layers', 2, 5)
    hidden_dims = []
    for i in range(n_layers):
        if i == 0:
            dim = trial.suggest_categorical(f'hidden_{i}', [256, 384, 512, 768])
        elif i == 1:
            dim = trial.suggest_categorical(f'hidden_{i}', [128, 256, 384, 512])
        else:
            dim = trial.suggest_categorical(f'hidden_{i}', [64, 128, 256])
        hidden_dims.append(dim)

    return DirectCurveShapeNetConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=trial.suggest_float('dropout', 0.05, 0.3),
        activation=trial.suggest_categorical('activation', ['silu', 'gelu', 'relu']),
        ctrl_points=trial.suggest_int('ctrl_points', 6, 12),  # More control points = more shape flexibility
        use_residual=trial.suggest_categorical('use_residual', [True, False])
    )


class DirectCurveShapeObjective:
    """
    Optuna objective for DirectCurveShapeNet (shape-only curve model).

    This model predicts curve shape using pretrained Jsc/Voc endpoints.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        jsc_train: np.ndarray,
        voc_train: np.ndarray,
        curves_train: np.ndarray,
        X_val: np.ndarray,
        jsc_val: np.ndarray,
        voc_val: np.ndarray,
        curves_val: np.ndarray,
        v_grid: np.ndarray,
        device: torch.device,
        batch_size: int = 2048,
        max_epochs: int = 100,
        patience: int = 20
    ):
        self.X_train = torch.from_numpy(X_train).float()
        self.X_val = torch.from_numpy(X_val).float()
        self.jsc_train = torch.from_numpy(jsc_train).float()
        self.jsc_val = torch.from_numpy(jsc_val).float()
        self.voc_train = torch.from_numpy(voc_train).float()
        self.voc_val = torch.from_numpy(voc_val).float()
        self.curves_train = torch.from_numpy(curves_train).float()
        self.curves_val = torch.from_numpy(curves_val).float()
        self.v_grid = torch.from_numpy(v_grid).float()
        self.device = device
        self.batch_size = batch_size
        self.input_dim = X_train.shape[1]
        self.max_epochs = max_epochs
        self.patience = patience

    def __call__(self, trial: optuna.Trial) -> float:
        config = sample_direct_curve_shape_config(trial, self.input_dim)

        # Training hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)

        # Loss hyperparameters
        weight_smooth = trial.suggest_float('weight_smooth', 0.01, 0.2, log=True)
        weight_mono = trial.suggest_float('weight_mono', 0.5, 2.0)
        knee_weight = trial.suggest_float('knee_weight', 1.0, 4.0)
        huber_delta = trial.suggest_float('huber_delta', 0.05, 0.3)

        model = DirectCurveShapeNet(config).to(self.device)
        loss_fn = DirectCurveShapeLoss(
            weight_smooth=weight_smooth,
            weight_mono=weight_mono,
            knee_weight=knee_weight,
            loss_type='huber',
            huber_delta=huber_delta,
            sample_weight_power=0.5
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )

        train_ds = torch.utils.data.TensorDataset(
            self.X_train, self.jsc_train, self.voc_train, self.curves_train
        )
        val_ds = torch.utils.data.TensorDataset(
            self.X_val, self.jsc_val, self.voc_val, self.curves_val
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        v_grid = self.v_grid.to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Train
            model.train()
            for batch_x, batch_jsc, batch_voc, batch_curves in train_loader:
                batch_x = batch_x.to(self.device)
                batch_jsc = batch_jsc.to(self.device)
                batch_voc = batch_voc.to(self.device)
                batch_curves = batch_curves.to(self.device)

                optimizer.zero_grad()
                ctrl = model(batch_x, batch_jsc, batch_voc)
                pred_curve = reconstruct_curve_shape(batch_voc, ctrl, v_grid, clamp_voc=True)

                loss, _ = loss_fn(pred_curve, batch_curves, batch_voc, v_grid)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_jsc, batch_voc, batch_curves in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_jsc = batch_jsc.to(self.device)
                    batch_voc = batch_voc.to(self.device)
                    batch_curves = batch_curves.to(self.device)

                    ctrl = model(batch_x, batch_jsc, batch_voc)
                    pred_curve = reconstruct_curve_shape(batch_voc, ctrl, v_grid, clamp_voc=True)

                    # Use MSE for validation metric (consistent across trials)
                    val_loss = torch.nn.functional.mse_loss(pred_curve, batch_curves)
                    val_losses.append(val_loss.item())

            avg_val_loss = float(np.mean(val_losses))

            # Report for pruning
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        return best_val_loss


# ============================================================================
# DISTRIBUTED HPO ENGINE
# ============================================================================

class DistributedHPO:
    """
    Distributed HPO engine using Optuna.

    Features:
    - Parallel trial execution across CPU cores
    - Smart pruning with Hyperband
    - TPE + CMA-ES ensemble sampling
    - Automatic study persistence
    """

    def __init__(self, config: HPOConfig):
        self.config = config

    def create_study(
        self,
        name: str,
        direction: str = 'minimize',
        sampler: str = 'tpe'
    ) -> optuna.Study:
        """Create an Optuna study with appropriate sampler and pruner."""

        if sampler == 'tpe':
            sampler_obj = TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                multivariate=True,
                group=True,
                seed=42
            )
        elif sampler == 'cmaes':
            sampler_obj = CmaEsSampler(seed=42)
        else:
            sampler_obj = TPESampler(seed=42)

        # UPDATED v2.0: Use config values for pruner settings
        # Relaxed pruning allows more trials to complete and find better configs
        pruner = HyperbandPruner(
            min_resource=self.config.pruner_min_resource,  # WAS hardcoded 1
            max_resource=400,                               # WAS 300
            reduction_factor=self.config.pruner_reduction_factor  # WAS hardcoded 3
        ) if self.config.use_pruning else MedianPruner()

        study = optuna.create_study(
            study_name=f"{self.config.study_name_prefix}_{name}",
            direction=direction,
            sampler=sampler_obj,
            pruner=pruner,
            storage=self.config.storage,
            load_if_exists=True
        )

        return study

    def optimize_voc_nn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        voc_ceiling_train: np.ndarray,
        voc_ceiling_val: np.ndarray,
        device: torch.device,
        n_trials: int = None
    ) -> tuple[dict, optuna.Study]:
        """Run HPO for Voc neural network."""

        n_trials = n_trials or self.config.n_trials_nn

        objective = VocNNObjective(
            X_train, y_train, X_val, y_val,
            voc_ceiling_train, voc_ceiling_val,
            device
        )

        study = self.create_study('voc_nn', sampler='tpe')

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_per_model,
            n_jobs=1,  # GPU models run serially but fast
            show_progress_bar=True,
            gc_after_trial=True
        )

        return study.best_params, study

    def optimize_lgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str = 'jsc',
        n_trials: int = None
    ) -> tuple[dict, optuna.Study]:
        """Run HPO for LGBM model."""

        n_trials = n_trials or self.config.n_trials_lgbm

        objective = LGBMObjective(X_train, y_train, X_val, y_val)
        study = self.create_study(f'{model_name}_lgbm', sampler='tpe')

        # LGBM trials can run in parallel since each uses GPU efficiently
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_per_model,
            n_jobs=1,  # Avoid GPU oversubscription
            show_progress_bar=True,
            gc_after_trial=True
        )

        completed_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            fallback_params = default_lgbm_config(model_name)
            state_counts = {}
            for trial in study.trials:
                state = trial.state.name
                state_counts[state] = state_counts.get(state, 0) + 1
            print(
                f"[WARN] No completed {model_name} LGBM trials. "
                f"Falling back to default params. Trial states: {state_counts}"
            )
            return fallback_params, study

        return study.best_params, study

    def optimize_curve(
        self,
        X_train: np.ndarray,
        anchors_train: np.ndarray,
        curves_train: np.ndarray,
        X_val: np.ndarray,
        anchors_val: np.ndarray,
        curves_val: np.ndarray,
        v_grid: np.ndarray,
        device: torch.device,
        n_trials: int = None
    ) -> tuple[dict, optuna.Study]:
        """Run HPO for curve reconstruction model."""

        n_trials = n_trials or self.config.n_trials_nn

        objective = CurveObjective(
            X_train, anchors_train, curves_train,
            X_val, anchors_val, curves_val,
            v_grid, device
        )

        study = self.create_study('curve_model', sampler='tpe')

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_per_model,
            n_jobs=1,  # GPU model
            show_progress_bar=True,
            gc_after_trial=True
        )

        return study.best_params, study

    def optimize_direct_curve_shape(
        self,
        X_train: np.ndarray,
        jsc_train: np.ndarray,
        voc_train: np.ndarray,
        curves_train: np.ndarray,
        X_val: np.ndarray,
        jsc_val: np.ndarray,
        voc_val: np.ndarray,
        curves_val: np.ndarray,
        v_grid: np.ndarray,
        device: torch.device,
        n_trials: int = None
    ) -> tuple[dict, optuna.Study]:
        """Run HPO for DirectCurveShapeNet (shape-only curve model)."""

        n_trials = n_trials or self.config.n_trials_nn

        objective = DirectCurveShapeObjective(
            X_train, jsc_train, voc_train, curves_train,
            X_val, jsc_val, voc_val, curves_val,
            v_grid, device
        )

        study = self.create_study('direct_curve_shape', sampler='tpe')

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_per_model,
            n_jobs=1,  # GPU model
            show_progress_bar=True,
            gc_after_trial=True
        )

        return study.best_params, study


def run_full_hpo(
    X_train_raw: np.ndarray,
    X_train_physics: np.ndarray,
    targets_train: dict,
    X_val_raw: np.ndarray,
    X_val_physics: np.ndarray,
    targets_val: dict,
    jsc_ceiling_train: np.ndarray,
    jsc_ceiling_val: np.ndarray,
    voc_ceiling_train: np.ndarray,
    voc_ceiling_val: np.ndarray,
    device: torch.device,
    hpo_config: HPOConfig = None,
    direct_curve_only: bool = False
) -> dict:
    """
    Run full HPO pipeline for all models.

    Args:
        direct_curve_only: If True, only run Jsc LGBM HPO (for direct curve model).
                          Skips Voc NN, Vmpp LGBM, Jmpp LGBM, FF LGBM.

    Returns dict with best params for each model.
    """
    hpo_config = hpo_config or HPOConfig()
    engine = DistributedHPO(hpo_config)
    results = {}

    # Combine features for NN
    X_train_full = np.hstack([X_train_raw, X_train_physics])
    X_val_full = np.hstack([X_val_raw, X_val_physics])

    # IMPORTANT: Use the same normalization regime as training.
    # Robust scaling (median/IQR) behaves better with long-tailed physics features.
    feature_mean = np.median(X_train_full, axis=0, keepdims=True)
    q75 = np.percentile(X_train_full, 75, axis=0, keepdims=True)
    q25 = np.percentile(X_train_full, 25, axis=0, keepdims=True)
    feature_std = (q75 - q25)
    feature_std[feature_std < 1e-8] = 1.0
    X_train_full = (X_train_full - feature_mean) / feature_std
    X_val_full = (X_val_full - feature_mean) / feature_std

    def _train_voc_for_features(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: VocNNConfig,
        device: torch.device
    ) -> tuple[VocNN, float, float]:
        """Train a VocNN model to generate predicted Voc features."""
        # Normalize targets
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8
        y_train_norm = (y_train - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std

        model = VocNN(config).to(device)
        trainer = VocTrainer(model, config, device)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train_norm).float()
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val_norm).float()
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4096, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4096)
        trainer.fit(train_loader, val_loader)

        return model, y_mean, y_std

    def _train_jsc_for_features(
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        y_jsc: np.ndarray,
        jsc_ceiling: np.ndarray,
        X_raw_val: np.ndarray,
        X_physics_val: np.ndarray,
        y_jsc_val: np.ndarray,
        jsc_ceiling_val: np.ndarray,
        config: JscLGBMConfig
    ) -> JscLGBM:
        """Train a Jsc LGBM model to generate predicted Jsc features."""
        model = JscLGBM(config)
        model.fit(
            X_raw, X_physics, y_jsc, jsc_ceiling,
            X_raw_val, X_physics_val, y_jsc_val, jsc_ceiling_val
        )
        return model

    # 1. Voc Neural Network (ALWAYS run - needed for direct curve shape model too)
    print("=" * 60)
    print("HPO: Voc Neural Network")
    print("=" * 60)
    voc_params, voc_study = engine.optimize_voc_nn(
        X_train_full, targets_train['Voc'],
        X_val_full, targets_val['Voc'],
        voc_ceiling_train, voc_ceiling_val,
        device
    )
    results['voc_nn'] = {'params': voc_params, 'study': voc_study}

    # Train Voc NN to generate predicted Voc for downstream HPO
    voc_config = get_best_configs_from_study({'voc_nn': {'params': voc_params}})['voc_nn']
    voc_config.input_dim = X_train_full.shape[1]
    voc_model, voc_mean, voc_std = _train_voc_for_features(
        X_train_full, targets_train['Voc'],
        X_val_full, targets_val['Voc'],
        voc_config, device
    )
    voc_model.eval()
    with torch.no_grad():
        voc_pred_train = voc_model(torch.from_numpy(X_train_full).float().to(device)).cpu().numpy()
        voc_pred_val = voc_model(torch.from_numpy(X_val_full).float().to(device)).cpu().numpy()
    voc_pred_train = voc_pred_train * voc_std + voc_mean
    voc_pred_val = voc_pred_val * voc_std + voc_mean

    # 2. Jsc LGBM (with ceiling feature)
    print("=" * 60)
    print("HPO: Jsc LGBM")
    print("=" * 60)
    # Prepare features with ceiling
    X_train_jsc = np.hstack([
        X_train_raw, X_train_physics,
        np.log10(jsc_ceiling_train + 1e-30).reshape(-1, 1)
    ])
    X_val_jsc = np.hstack([
        X_val_raw, X_val_physics,
        np.log10(jsc_ceiling_val + 1e-30).reshape(-1, 1)
    ])
    # Target as efficiency
    y_train_jsc = targets_train['Jsc'] / (jsc_ceiling_train + 1e-30)
    y_val_jsc = targets_val['Jsc'] / (jsc_ceiling_val + 1e-30)

    jsc_params, jsc_study = engine.optimize_lgbm(
        X_train_jsc, y_train_jsc, X_val_jsc, y_val_jsc, 'jsc'
    )
    results['jsc_lgbm'] = {'params': jsc_params, 'study': jsc_study}

    # Train Jsc LGBM to generate predicted Jsc for downstream HPO
    jsc_config = get_best_configs_from_study({'jsc_lgbm': {'params': jsc_params}})['jsc_lgbm']
    jsc_model = _train_jsc_for_features(
        X_train_raw, X_train_physics, targets_train['Jsc'], jsc_ceiling_train,
        X_val_raw, X_val_physics, targets_val['Jsc'], jsc_ceiling_val,
        jsc_config
    )
    jsc_pred_train = jsc_model.predict(X_train_raw, X_train_physics, jsc_ceiling_train)
    jsc_pred_val = jsc_model.predict(X_val_raw, X_val_physics, jsc_ceiling_val)

    # For direct curve mode, we only need Jsc LGBM - skip the rest
    if direct_curve_only:
        print("=" * 60)
        print("Skipping Vmpp, Jmpp, FF LGBM HPO (direct curve mode)")
        print("=" * 60)
        return results

    # 3. Vmpp LGBM (with Voc feature)
    print("=" * 60)
    print("HPO: Vmpp LGBM")
    print("=" * 60)
    X_train_vmpp = np.hstack([
        X_train_raw, X_train_physics,
        voc_pred_train.reshape(-1, 1),
        np.log10(voc_pred_train + 1e-30).reshape(-1, 1)
    ])
    X_val_vmpp = np.hstack([
        X_val_raw, X_val_physics,
        voc_pred_val.reshape(-1, 1),
        np.log10(voc_pred_val + 1e-30).reshape(-1, 1)
    ])
    # Target as Vmpp/Voc ratio
    y_train_vmpp = targets_train['Vmpp'] / (voc_pred_train + 1e-30)
    y_val_vmpp = targets_val['Vmpp'] / (voc_pred_val + 1e-30)

    vmpp_params, vmpp_study = engine.optimize_lgbm(
        X_train_vmpp, y_train_vmpp, X_val_vmpp, y_val_vmpp, 'vmpp'
    )
    results['vmpp_lgbm'] = {'params': vmpp_params, 'study': vmpp_study}

    # Train Vmpp LGBM to generate predicted Vmpp for downstream HPO
    vmpp_config = get_best_configs_from_study({'vmpp_lgbm': {'params': vmpp_params}})['vmpp_lgbm']
    vmpp_model = VmppLGBM(vmpp_config)
    vmpp_model.fit(
        X_train_raw, X_train_physics, targets_train['Vmpp'], voc_pred_train,
        X_val_raw, X_val_physics, targets_val['Vmpp'], voc_pred_val
    )
    vmpp_pred_train = vmpp_model.predict(X_train_raw, X_train_physics, voc_pred_train)
    vmpp_pred_val = vmpp_model.predict(X_val_raw, X_val_physics, voc_pred_val)

    # 4. Jmpp LGBM (was missing - now added)
    print("=" * 60)
    print("HPO: Jmpp LGBM")
    print("=" * 60)
    X_train_jmpp = np.hstack([
        X_train_raw, X_train_physics,
        jsc_pred_train.reshape(-1, 1),
        np.log10(np.abs(jsc_pred_train) + 1e-30).reshape(-1, 1),
        vmpp_pred_train.reshape(-1, 1),
        np.log10(vmpp_pred_train + 1e-30).reshape(-1, 1)
    ])
    X_val_jmpp = np.hstack([
        X_val_raw, X_val_physics,
        jsc_pred_val.reshape(-1, 1),
        np.log10(np.abs(jsc_pred_val) + 1e-30).reshape(-1, 1),
        vmpp_pred_val.reshape(-1, 1),
        np.log10(vmpp_pred_val + 1e-30).reshape(-1, 1)
    ])
    # Target as Jmpp/Jsc ratio
    y_train_jmpp = targets_train['Jmpp'] / (jsc_pred_train + 1e-30)
    y_val_jmpp = targets_val['Jmpp'] / (jsc_pred_val + 1e-30)

    jmpp_params, jmpp_study = engine.optimize_lgbm(
        X_train_jmpp, y_train_jmpp, X_val_jmpp, y_val_jmpp, 'jmpp'
    )
    results['jmpp_lgbm'] = {'params': jmpp_params, 'study': jmpp_study}

    # 5. FF LGBM
    print("=" * 60)
    print("HPO: FF LGBM")
    print("=" * 60)
    X_train_ff = np.hstack([
        X_train_raw, X_train_physics,
        voc_pred_train.reshape(-1, 1),
        jsc_pred_train.reshape(-1, 1),
        (voc_pred_train * jsc_pred_train).reshape(-1, 1)
    ])
    X_val_ff = np.hstack([
        X_val_raw, X_val_physics,
        voc_pred_val.reshape(-1, 1),
        jsc_pred_val.reshape(-1, 1),
        (voc_pred_val * jsc_pred_val).reshape(-1, 1)
    ])

    ff_params, ff_study = engine.optimize_lgbm(
        X_train_ff, targets_train['FF'], X_val_ff, targets_val['FF'], 'ff'
    )
    results['ff_lgbm'] = {'params': ff_params, 'study': ff_study}

    return results


def get_best_configs_from_study(results: dict) -> dict:
    """
    Extract best configurations from HPO results.

    FIXED: Default values now match the search space ranges.
    """
    configs = {}

    # Voc NN - defaults match search space: n_layers in [2,3], hidden in [128,256]/[64,128]
    if 'voc_nn' in results:
        params = results['voc_nn']['params']
        n_layers = params.get('n_layers', 2)  # FIXED: was 6, search space is 2-3
        hidden_dims = []
        for i in range(n_layers):
            if i == 0:
                hidden_dims.append(params.get(f'hidden_{i}', 256))  # First layer default
            else:
                hidden_dims.append(params.get(f'hidden_{i}', 128))  # Later layers default
        configs['voc_nn'] = VocNNConfig(
            hidden_dims=hidden_dims,
            dropout=params.get('dropout', 0.15),  # FIXED: was 0.1, search is 0.1-0.25
            use_layer_norm=True,  # Always True in search space
            use_residual=params.get('use_residual', True),
            activation=params.get('activation', 'gelu'),
            jacobian_weight=params.get('jacobian_weight', 0.01),
            physics_weight=params.get('physics_weight', 0.05),  # FIXED: middle of 0.01-0.1
            lr=params.get('lr', 5e-4),  # FIXED: middle of 5e-5 to 1e-3
            weight_decay=params.get('weight_decay', 1e-5),
        )

    # Jsc LGBM - defaults match search space
    if 'jsc_lgbm' in results:
        params = results['jsc_lgbm']['params']
        configs['jsc_lgbm'] = JscLGBMConfig(
            num_leaves=params.get('num_leaves', 127),  # FIXED: middle of 31-255
            max_depth=params.get('max_depth', 10),  # FIXED: middle of 6-15
            learning_rate=params.get('learning_rate', 0.03),  # FIXED: middle of 0.01-0.1
            n_estimators=params.get('n_estimators', 1000),  # FIXED: middle of 500-2000
            min_child_samples=params.get('min_child_samples', 25),  # FIXED: middle of 10-50
            subsample=params.get('subsample', 0.85),  # FIXED: middle of 0.7-0.95
            colsample_bytree=params.get('colsample_bytree', 0.85),
            reg_alpha=params.get('reg_alpha', 0.01),  # FIXED: middle of 1e-4 to 1.0 (log)
            reg_lambda=params.get('reg_lambda', 0.01),
        )

    # Vmpp LGBM
    if 'vmpp_lgbm' in results:
        params = results['vmpp_lgbm']['params']
        configs['vmpp_lgbm'] = VmppLGBMConfig(
            num_leaves=params.get('num_leaves', 127),
            max_depth=params.get('max_depth', 10),
            learning_rate=params.get('learning_rate', 0.03),
            n_estimators=params.get('n_estimators', 1000),
            min_child_samples=params.get('min_child_samples', 25),
            subsample=params.get('subsample', 0.85),
            colsample_bytree=params.get('colsample_bytree', 0.85),
            reg_alpha=params.get('reg_alpha', 0.01),
            reg_lambda=params.get('reg_lambda', 0.01),
        )

    # Jmpp LGBM
    if 'jmpp_lgbm' in results:
        params = results['jmpp_lgbm']['params']
        configs['jmpp_lgbm'] = VmppLGBMConfig(
            num_leaves=params.get('num_leaves', 127),
            max_depth=params.get('max_depth', 10),
            learning_rate=params.get('learning_rate', 0.03),
            n_estimators=params.get('n_estimators', 1000),
            min_child_samples=params.get('min_child_samples', 25),
            subsample=params.get('subsample', 0.85),
            colsample_bytree=params.get('colsample_bytree', 0.85),
            reg_alpha=params.get('reg_alpha', 0.01),
            reg_lambda=params.get('reg_lambda', 0.01),
        )

    # FF LGBM
    if 'ff_lgbm' in results:
        params = results['ff_lgbm']['params']
        configs['ff_lgbm'] = VmppLGBMConfig(
            num_leaves=params.get('num_leaves', 127),
            max_depth=params.get('max_depth', 10),
            learning_rate=params.get('learning_rate', 0.03),
            n_estimators=params.get('n_estimators', 1000),
            min_child_samples=params.get('min_child_samples', 25),
            subsample=params.get('subsample', 0.85),
            colsample_bytree=params.get('colsample_bytree', 0.85),
            reg_alpha=params.get('reg_alpha', 0.01),
            reg_lambda=params.get('reg_lambda', 0.01),
        )

    # Curve model - defaults match search space
    if 'curve_model' in results:
        params = results['curve_model']['params']
        n_layers = params.get('n_layers', 3)  # Search is 2-4
        hidden_dims = []
        for i in range(n_layers):
            if i == 0:
                hidden_dims.append(params.get(f'hidden_{i}', 256))
            elif i == 1:
                hidden_dims.append(params.get(f'hidden_{i}', 128))
            else:
                hidden_dims.append(params.get(f'hidden_{i}', 64))
        configs['curve_model'] = {
            'config': SplitSplineNetConfig(
                hidden_dims=hidden_dims,
                dropout=params.get('dropout', 0.2),  # FIXED: middle of 0.1-0.3
                activation=params.get('activation', 'silu'),
                ctrl_points=params.get('ctrl_points', 4),  # FIXED: middle of 3-6
            ),
            'lr': params.get('lr', 1e-3),
            'weight_decay': params.get('weight_decay', 1e-5),
            'continuity_weight': params.get('continuity_weight', 0.15),  # FIXED: middle of 0.05-0.5
        }

    return configs


def run_curve_hpo(
    X_train: np.ndarray,
    anchors_train: np.ndarray,
    curves_train: np.ndarray,
    X_val: np.ndarray,
    anchors_val: np.ndarray,
    curves_val: np.ndarray,
    v_grid: np.ndarray,
    device: torch.device,
    hpo_config: HPOConfig = None,
    n_trials: int = None
) -> dict:
    """
    Run HPO for curve reconstruction model separately.

    Returns dict with best params for curve model.
    """
    hpo_config = hpo_config or HPOConfig()
    engine = DistributedHPO(hpo_config)

    # Normalize features (match train.py: robust median/IQR scaling)
    feature_mean = np.median(X_train, axis=0, keepdims=True)
    q75 = np.percentile(X_train, 75, axis=0, keepdims=True)
    q25 = np.percentile(X_train, 25, axis=0, keepdims=True)
    feature_std = (q75 - q25)
    feature_std[feature_std < 1e-8] = 1.0
    X_train_norm = (X_train - feature_mean) / feature_std
    X_val_norm = (X_val - feature_mean) / feature_std

    print("=" * 60)
    print("HPO: Curve Reconstruction Model")
    print("=" * 60)

    n_trials = n_trials or hpo_config.n_trials_nn

    curve_params, curve_study = engine.optimize_curve(
        X_train_norm, anchors_train, curves_train,
        X_val_norm, anchors_val, curves_val,
        v_grid, device, n_trials
    )

    results = {
        'curve_model': {
            'params': curve_params,
            'study': curve_study,
            'best_value': curve_study.best_value,
            'n_trials': len(curve_study.trials)
        },
        'normalization': {
            'feature_mean': feature_mean.tolist(),
            'feature_std': feature_std.tolist()
        }
    }

    return results


def run_direct_curve_shape_hpo(
    X_train: np.ndarray,
    jsc_train: np.ndarray,
    voc_train: np.ndarray,
    curves_train: np.ndarray,
    X_val: np.ndarray,
    jsc_val: np.ndarray,
    voc_val: np.ndarray,
    curves_val: np.ndarray,
    v_grid: np.ndarray,
    device: torch.device,
    hpo_config: HPOConfig = None,
    n_trials: int = None
) -> dict:
    """
    Run HPO for DirectCurveShapeNet (shape-only curve model).

    This is used for the --direct-curve mode where Jsc and Voc come from
    pretrained models and only the curve shape is learned.

    Args:
        X_train: Normalized feature array for training
        jsc_train: Predicted Jsc values from pretrained LGBM
        voc_train: Predicted Voc values from pretrained NN
        curves_train: Normalized target curves
        X_val, jsc_val, voc_val, curves_val: Validation data
        v_grid: Voltage grid
        device: torch device
        hpo_config: HPO configuration
        n_trials: Override number of trials

    Returns:
        dict with best params and study results
    """
    hpo_config = hpo_config or HPOConfig()
    engine = DistributedHPO(hpo_config)

    print("=" * 60)
    print("HPO: DirectCurveShapeNet (Shape-Only Model)")
    print("=" * 60)

    n_trials = n_trials or hpo_config.n_trials_nn

    shape_params, shape_study = engine.optimize_direct_curve_shape(
        X_train, jsc_train, voc_train, curves_train,
        X_val, jsc_val, voc_val, curves_val,
        v_grid, device, n_trials
    )

    results = {
        'direct_curve_shape': {
            'params': shape_params,
            'study': shape_study,
            'best_value': shape_study.best_value,
            'n_trials': len(shape_study.trials)
        }
    }

    # Print summary
    print(f"\nBest DirectCurveShapeNet params (val_loss={shape_study.best_value:.6f}):")
    for k, v in shape_params.items():
        print(f"  {k}: {v}")

    return results
