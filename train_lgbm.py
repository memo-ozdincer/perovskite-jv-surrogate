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
    RANDOM_SEED, VAL_SPLIT, TEST_SPLIT
)
from data import (
    load_raw_data, prepare_tensors, extract_targets_gpu, split_indices
)
from features import (
    compute_all_physics_features, get_feature_names,
    compute_jsc_ceiling, compute_voc_ceiling
)
from models.voc_nn import (
    VocNNConfig, VocNN, VocTrainer, build_voc_model,
    SplitSplineNetConfig, UnifiedSplitSplineNet, SplitSplineCtrlNet
)
from models.jsc_lgbm import JscLGBMConfig, JscLGBM, build_jsc_model
from models.vmpp_lgbm import (
    VmppLGBMConfig, VmppLGBM, JmppLGBM, FFLGBM,
    build_vmpp_model, build_jmpp_model, build_ff_model
)
from models.reconstruction import reconstruct_curve, continuity_loss
from hpo import HPOConfig, DistributedHPO, run_full_hpo, get_best_configs_from_study


class MultiTaskLoss(nn.Module):
    """Automatic loss weighting for anchors + curve reconstruction."""

    def __init__(self):
        super().__init__()
        self.log_sigma_anchor = nn.Parameter(torch.zeros(1))
        self.log_sigma_curve = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pred_anchors: torch.Tensor,
        true_anchors: torch.Tensor,
        pred_curve: torch.Tensor,
        true_curve: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        l_anchor = F.mse_loss(pred_anchors, true_anchors)
        l_curve = F.mse_loss(pred_curve, true_curve)

        sigma_a = torch.exp(self.log_sigma_anchor)
        sigma_c = torch.exp(self.log_sigma_curve)

        loss = (
            l_anchor / (2 * sigma_a ** 2) + torch.log(sigma_a) +
            l_curve / (2 * sigma_c ** 2) + torch.log(sigma_c)
        )

        metrics = {
            'anchor': l_anchor.item(),
            'curve': l_curve.item(),
            'sigma_anchor': sigma_a.item(),
            'sigma_curve': sigma_c.item()
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
        hpo_config: HPOConfig = None
    ):
        self.params_file = params_file
        self.iv_file = iv_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.run_hpo = run_hpo
        self.run_curve_model = run_curve_model
        self.hpo_config = hpo_config or HPOConfig()

        # Will be populated during pipeline
        self.params_df = None
        self.iv_data = None
        self.targets = None
        self.physics_features = None
        self.models = {}
        self.metrics = {}

        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")

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
                'curves': self.iv_data[idx]
            }

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
        hpo_summary = {}
        for name, result in self.hpo_results.items():
            hpo_summary[name] = {
                'best_params': result['params'],
                'best_value': result['study'].best_value,
                'n_trials': len(result['study'].trials)
            }

        with open(self.output_dir / 'hpo_results.json', 'w') as f:
            json.dump(hpo_summary, f, indent=2, default=str)

        print("\nHPO Summary:")
        for name, summary in hpo_summary.items():
            print(f"  {name}: best_value={summary['best_value']:.6f}, n_trials={summary['n_trials']}")

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

    def train_curve_model(self):
        """Train unified split-spline model for full J-V reconstruction."""
        train = self.splits['train']
        val = self.splits['val']

        # Prepare inputs
        X_train_full = np.hstack([train['X_raw'], train['X_physics']]).astype(np.float32)
        X_val_full = np.hstack([val['X_raw'], val['X_physics']]).astype(np.float32)

        self.curve_feature_mean = X_train_full.mean(axis=0, keepdims=True)
        self.curve_feature_std = X_train_full.std(axis=0, keepdims=True) + 1e-8
        X_train_full = (X_train_full - self.curve_feature_mean) / self.curve_feature_std
        X_val_full = (X_val_full - self.curve_feature_mean) / self.curve_feature_std

        anchors_train = self._predict_curve_anchors(train)
        anchors_val = self._predict_curve_anchors(val)

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

        config = SplitSplineNetConfig(input_dim=X_train_full.shape[1])
        model = SplitSplineCtrlNet(config).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

        from config import V_GRID
        v_grid = torch.from_numpy(V_GRID).float().to(self.device)

        best_val = float('inf')
        best_state = None
        patience = 15
        patience_counter = 0

        for epoch in range(100):
            model.train()
            epoch_losses = []
            for batch_x, batch_anchors, batch_curves in train_loader:
                batch_x = batch_x.to(self.device)
                batch_anchors = batch_anchors.to(self.device)
                batch_curves = batch_curves.to(self.device)

                optimizer.zero_grad()
                ctrl1, ctrl2 = model(batch_x)
                pred_curve = reconstruct_curve(batch_anchors, ctrl1, ctrl2, v_grid)

                loss = F.mse_loss(pred_curve, batch_curves)
                loss = loss + 0.1 * continuity_loss(batch_anchors, ctrl1, ctrl2, v_grid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_anchors, batch_curves in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_anchors = batch_anchors.to(self.device)
                    batch_curves = batch_curves.to(self.device)
                    ctrl1, ctrl2 = model(batch_x)
                    pred_curve = reconstruct_curve(batch_anchors, ctrl1, ctrl2, v_grid)
                    val_loss = F.mse_loss(pred_curve, batch_curves)
                    val_losses.append(val_loss.item())

            avg_val = float(np.mean(val_losses))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={np.mean(epoch_losses):.6f}, val_loss={avg_val:.6f}")

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

    def _predict_curve_anchors(self, split: dict) -> np.ndarray:
        """Predict anchors using LGBM heads (and Voc NN) for curve training."""
        X_raw = split['X_raw']
        X_physics = split['X_physics']

        # Voc NN (normalized)
        X_full = np.hstack([X_raw, X_physics])
        X_norm = (X_full - self.voc_feature_mean) / self.voc_feature_std
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)
        with torch.no_grad():
            voc_pred_norm = self.models['voc_nn'](X_tensor).cpu().numpy()
        voc_pred = voc_pred_norm * self.voc_target_std + self.voc_target_mean

        # LGBM anchors
        jsc_pred = self.models['jsc_lgbm'].predict(X_raw, X_physics, split['jsc_ceiling'])
        vmpp_pred = self.models['vmpp_lgbm'].predict(X_raw, X_physics, voc_pred)
        jmpp_pred = self.models['jmpp_lgbm'].predict(X_raw, X_physics, jsc_pred, vmpp_pred)

        anchors = np.stack([jsc_pred, voc_pred, vmpp_pred, jmpp_pred], axis=1).astype(np.float32)
        return anchors

    def _train_voc_model(self, trainer, train_loader, val_loader, config):
        """Custom training loop for Voc model."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(config.epochs):
            # Train
            trainer.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                trainer.optimizer.zero_grad()
                pred, jac_norm = trainer.model.forward_with_jacobian(batch_x)
                loss = torch.nn.functional.mse_loss(pred, batch_y)
                loss = loss + config.jacobian_weight * jac_norm
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
                trainer.optimizer.step()
                train_losses.append(loss.item())

            # Validate
            trainer.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    pred = trainer.model(batch_x)
                    val_loss = torch.nn.functional.mse_loss(pred, batch_y)
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={np.mean(train_losses):.6f}, val_loss={avg_val_loss:.6f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            trainer.scheduler.step()

        # Restore best model
        if best_state:
            trainer.model.load_state_dict(best_state)

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

        # Save configs
        configs = {
            'voc_nn': self.models['voc_nn'].config.__dict__ if hasattr(self.models['voc_nn'], 'config') else {},
            'jsc_lgbm': self.models['jsc_lgbm'].config.__dict__,
            'vmpp_lgbm': self.models['vmpp_lgbm'].config.__dict__,
            'ff_lgbm': self.models['ff_lgbm'].config.__dict__,
        }

        if 'curve_model' in self.models:
            configs['curve_model'] = {
                'type': 'ctrl_net',
                'input_dim': getattr(self.models['curve_model'], 'config', SplitSplineNetConfig()).input_dim,
                'ctrl_points': getattr(self.models['curve_model'], 'config', SplitSplineNetConfig()).ctrl_points
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
            with open(models_dir / 'normalization.json', 'w') as f:
                json.dump(normalization_params, f)
            print("Saved normalization parameters for Voc NN (features + targets)")

        # Save curve control-point model if trained
        if 'curve_model' in self.models:
            torch.save(self.models['curve_model'].state_dict(), models_dir / 'curve_model.pt')
            print("Saved curve control-point model")

        print(f"Models saved to {models_dir}")

    def run(self):
        """Run full training pipeline."""
        start_time = datetime.now()

        self.load_data()
        self.extract_targets()
        self.compute_features()
        self.split_data()

        if self.run_hpo:
            self.run_hyperparameter_optimization()

        self.train_final_models()
        self.evaluate()
        self.save_models()

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print(f"Pipeline Complete! Duration: {duration}")
        print("=" * 60)

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
    parser.add_argument('--train-curves', action='store_true',
                        help='Train unified split-spline curve model')
    parser.add_argument('--hpo-trials-nn', type=int, default=100,
                        help='Number of HPO trials for NN')
    parser.add_argument('--hpo-trials-lgbm', type=int, default=200,
                        help='Number of HPO trials for LGBM')
    parser.add_argument('--hpo-timeout', type=int, default=7200,
                        help='HPO timeout per model (seconds)')

    args = parser.parse_args()

    hpo_config = HPOConfig(
        n_trials_nn=args.hpo_trials_nn,
        n_trials_lgbm=args.hpo_trials_lgbm,
        timeout_per_model=args.hpo_timeout
    )

    pipeline = ScalarPredictorPipeline(
        params_file=args.params,
        iv_file=args.iv,
        output_dir=args.output,
        device=args.device,
        run_hpo=not args.no_hpo,
        run_curve_model=args.train_curves,
        hpo_config=hpo_config
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
