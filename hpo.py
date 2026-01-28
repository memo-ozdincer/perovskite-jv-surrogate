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


@dataclass
class HPOConfig:
    """Global HPO configuration."""
    # Trial counts - reduced after architecture simplification
    # Simpler search space requires fewer trials
    n_trials_nn: int = 100
    n_trials_lgbm: int = 200

    # Parallelization
    n_parallel_trials: int = 24  # Match GPU node cores
    n_startup_trials: int = 50   # Random trials before TPE

    # Timeouts
    timeout_per_model: int = 7200  # 2 hours per model type

    # Pruning
    use_pruning: bool = True
    pruning_warmup: int = 10

    # Storage (for distributed)
    storage: str = None  # Use in-memory by default
    study_name_prefix: str = 'pv_scalar'


# ============================================================================
# NEURAL NETWORK HPO (for Voc)
# ============================================================================

def sample_voc_nn_config(trial: optuna.Trial, input_dim: int) -> VocNNConfig:
    """
    Sample hyperparameters for Voc neural network.

    SIMPLIFIED for robustness:
    - Narrower search ranges based on physics intuition
    - Favor simpler architectures that generalize
    - Lower learning rates for stability
    """

    # Architecture - SIMPLE and focused
    # 2-3 layers are usually sufficient for this problem
    n_layers = trial.suggest_int('n_layers', 2, 3)
    hidden_dims = []
    for i in range(n_layers):
        # Tapering architecture: wide -> narrow
        if i == 0:
            dim = trial.suggest_categorical(f'hidden_{i}', [128, 256])
        else:
            dim = trial.suggest_categorical(f'hidden_{i}', [64, 128])
        hidden_dims.append(dim)

    return VocNNConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,

        # Regularization - stronger to prevent overfitting
        dropout=trial.suggest_float('dropout', 0.1, 0.25),
        use_layer_norm=True,  # Always use for stability
        use_residual=trial.suggest_categorical('use_residual', [True, False]),

        # Activation - GELU is usually best for smooth problems
        activation=trial.suggest_categorical('activation', ['gelu', 'silu']),

        # Physics losses - narrower range, known to work
        jacobian_weight=trial.suggest_float('jacobian_weight', 1e-3, 0.05, log=True),
        physics_weight=trial.suggest_float('physics_weight', 0.01, 0.1, log=True),

        # Optimizer - LOWER learning rates for stability
        lr=trial.suggest_float('lr', 5e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),

        # Training - reasonable epochs with good patience
        epochs=trial.suggest_int('epochs', 150, 300),
        patience=trial.suggest_int('patience', 25, 40),
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

        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=300,
            reduction_factor=3
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
    hpo_config: HPOConfig = None
) -> dict:
    """
    Run full HPO pipeline for all models.

    Returns dict with best params for each model.
    """
    hpo_config = hpo_config or HPOConfig()
    engine = DistributedHPO(hpo_config)
    results = {}

    # Combine features for NN
    X_train_full = np.hstack([X_train_raw, X_train_physics])
    X_val_full = np.hstack([X_val_raw, X_val_physics])

    # CRITICAL: Standardize features for neural network to prevent gradient explosion
    feature_mean = X_train_full.mean(axis=0, keepdims=True)
    feature_std = X_train_full.std(axis=0, keepdims=True) + 1e-8
    X_train_full = (X_train_full - feature_mean) / feature_std
    X_val_full = (X_val_full - feature_mean) / feature_std

    # 1. Voc Neural Network
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

    # 3. Vmpp LGBM (with Voc feature)
    print("=" * 60)
    print("HPO: Vmpp LGBM")
    print("=" * 60)
    X_train_vmpp = np.hstack([
        X_train_raw, X_train_physics,
        targets_train['Voc'].reshape(-1, 1),
        np.log10(targets_train['Voc'] + 1e-30).reshape(-1, 1)
    ])
    X_val_vmpp = np.hstack([
        X_val_raw, X_val_physics,
        targets_val['Voc'].reshape(-1, 1),
        np.log10(targets_val['Voc'] + 1e-30).reshape(-1, 1)
    ])
    # Target as Vmpp/Voc ratio
    y_train_vmpp = targets_train['Vmpp'] / (targets_train['Voc'] + 1e-30)
    y_val_vmpp = targets_val['Vmpp'] / (targets_val['Voc'] + 1e-30)

    vmpp_params, vmpp_study = engine.optimize_lgbm(
        X_train_vmpp, y_train_vmpp, X_val_vmpp, y_val_vmpp, 'vmpp'
    )
    results['vmpp_lgbm'] = {'params': vmpp_params, 'study': vmpp_study}

    # 4. Jmpp LGBM (was missing - now added)
    print("=" * 60)
    print("HPO: Jmpp LGBM")
    print("=" * 60)
    X_train_jmpp = np.hstack([
        X_train_raw, X_train_physics,
        targets_train['Jsc'].reshape(-1, 1),
        np.log10(np.abs(targets_train['Jsc']) + 1e-30).reshape(-1, 1),
        targets_train['Vmpp'].reshape(-1, 1),
        np.log10(targets_train['Vmpp'] + 1e-30).reshape(-1, 1)
    ])
    X_val_jmpp = np.hstack([
        X_val_raw, X_val_physics,
        targets_val['Jsc'].reshape(-1, 1),
        np.log10(np.abs(targets_val['Jsc']) + 1e-30).reshape(-1, 1),
        targets_val['Vmpp'].reshape(-1, 1),
        np.log10(targets_val['Vmpp'] + 1e-30).reshape(-1, 1)
    ])
    # Target as Jmpp/Jsc ratio
    y_train_jmpp = targets_train['Jmpp'] / (targets_train['Jsc'] + 1e-30)
    y_val_jmpp = targets_val['Jmpp'] / (targets_val['Jsc'] + 1e-30)

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
        targets_train['Voc'].reshape(-1, 1),
        targets_train['Jsc'].reshape(-1, 1),
        (targets_train['Voc'] * targets_train['Jsc']).reshape(-1, 1)
    ])
    X_val_ff = np.hstack([
        X_val_raw, X_val_physics,
        targets_val['Voc'].reshape(-1, 1),
        targets_val['Jsc'].reshape(-1, 1),
        (targets_val['Voc'] * targets_val['Jsc']).reshape(-1, 1)
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

    # Normalize features
    feature_mean = X_train.mean(axis=0, keepdims=True)
    feature_std = X_train.std(axis=0, keepdims=True) + 1e-8
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
