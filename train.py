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
from models.reconstruction import (
    reconstruct_curve, reconstruct_curve_normalized, normalize_anchors_by_jsc,
    continuity_loss, build_knots, pchip_interpolate_batch, linear_interpolate_batch
)
from models.direct_curve import (
    DirectCurveNetWithJsc, DirectCurveNetWithJscConfig,
    DirectCurveShapeNet, DirectCurveShapeNetConfig,
    DirectCurveLossWithJsc, DirectCurveShapeLoss,
    reconstruct_curve_direct_normalized, reconstruct_curve_shape,
    extract_voc_from_curve
)
from models.voc_lgbm import VocLGBMConfig, build_voc_model as build_voc_lgbm_model
from hpo import HPOConfig, DistributedHPO, run_full_hpo, get_best_configs_from_study, run_curve_hpo, run_direct_curve_shape_hpo
from logging_utils import (
    TrainingLogger, ModelComparisonMetrics,
    compute_multicollinearity, suggest_features_to_drop
)
from preprocessing import (
    PVDataPreprocessor,
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

    Computes RELATIVE MSE on the full curve with optional region weighting.
    Using relative error ensures the loss is scale-invariant and compatible
    with continuity loss (which operates in normalized space).
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

        # Use RELATIVE squared error: ((pred - true) / (true + eps))^2
        # This normalizes by the magnitude of J, making loss scale-invariant
        # and keeping it in O(1) range compatible with continuity loss
        eps = 1e-6
        # Use Jsc (first point) as reference scale for each curve
        jsc_ref = true_curve[:, 0:1].clamp(min=eps)  # (N, 1)
        rel_err = (pred_curve - true_curve) / jsc_ref
        sq_err = rel_err ** 2

        # Weight points near MPP more heavily (this is where the "knee" is)
        vmpp_expanded = vmpp.unsqueeze(1)
        dist_to_mpp = (v_grid.unsqueeze(0) - vmpp_expanded).abs()
        mpp_weights = 1.0 + (self.mpp_weight - 1.0) * torch.exp(-dist_to_mpp / 0.1)

        weighted_sq_err = sq_err * mpp_weights
        loss = weighted_sq_err.mean()

        # Region-wise metrics for logging (in relative terms)
        mask_r1 = v_grid.unsqueeze(0) <= vmpp_expanded
        mse_r1 = (sq_err * mask_r1).sum() / mask_r1.sum().clamp(min=1)
        mse_r2 = (sq_err * ~mask_r1).sum() / (~mask_r1).sum().clamp(min=1)

        # Also compute absolute MSE for reference
        abs_sq_err = (pred_curve - true_curve) ** 2

        metrics = {
            'loss_curve': loss.item(),
            'mse_full_rel': sq_err.mean().item(),
            'mse_full_abs': abs_sq_err.mean().item(),
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
        params_extra: list[str] = None,
        iv_extra: list[str] = None,
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
        verbose_logging: bool = True,
        use_direct_curve: bool = False,  # Use simplified direct curve model
        # Outlier filtering options
        filter_outliers: bool = False,
        filter_min_ff: float = 0.30,
        filter_min_vmpp: float = 0.30,
        filter_min_pce_quantile: float = 0.0,
        report_trimmed_metrics: bool = False,
        # Oracle mode
        oracle_voc: bool = False,  # Use true Voc for curve truncation
        # Auxiliary anchor inputs
        anchors_file: str = None,
        anchors_extra: list = None,
        voc_anchors_file: str = None,
        voc_anchors_extra: list = None,
        vmpp_anchors_file: str = None,
        vmpp_anchors_extra: list = None,
        use_vmpp_input: bool = False,
        use_jmpp_input: bool = False,
        use_ff_input: bool = False
    ):
        # Build file lists for multi-file loading
        self.params_files = [params_file] + (params_extra or [])
        self.iv_files = [iv_file] + (iv_extra or [])
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
        self.use_direct_curve = use_direct_curve

        # Outlier filtering options
        self.filter_outliers = filter_outliers
        self.filter_min_ff = filter_min_ff
        self.filter_min_vmpp = filter_min_vmpp
        self.filter_min_pce_quantile = filter_min_pce_quantile
        self.report_trimmed_metrics = report_trimmed_metrics
        self.filter_stats = None  # Will store filtering statistics

        # Oracle mode: use true Voc for curve truncation (shows upper bound performance)
        self.oracle_voc = oracle_voc

        # Auxiliary anchor inputs for curve model conditioning
        self.anchors_files = [anchors_file] + (anchors_extra or []) if anchors_file else []
        self.voc_anchors_files = [voc_anchors_file] + (voc_anchors_extra or []) if voc_anchors_file else []
        self.vmpp_anchors_files = [vmpp_anchors_file] + (vmpp_anchors_extra or []) if vmpp_anchors_file else []
        self.use_vmpp_input = use_vmpp_input
        self.use_jmpp_input = use_jmpp_input
        self.use_ff_input = use_ff_input
        self.anchors_data = None  # Will be loaded if anchor files provided
        self.voc_anchors_data = None  # Optional Voc-only anchors
        self.vmpp_anchors_data = None  # Optional Vmpp-only anchors

        # Will be populated during pipeline
        self.params_df = None
        self.iv_data = None
        self.targets = None
        self.physics_features = None
        self.preprocessor = None
        self.models = {}
        self.metrics = {}
        self.voc_model_type = 'nn'
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
        """Load raw data from files (supports multiple file pairs)."""
        print("\n" + "=" * 60)
        print("Loading Data")
        print("=" * 60)

        # Show which files will be loaded
        print(f"Number of file pairs to load: {len(self.params_files)}")
        for i, (pf, ivf) in enumerate(zip(self.params_files, self.iv_files)):
            print(f"  [{i+1}] Params: {pf}")
            print(f"      IV:     {ivf}")

        # Check file existence before loading
        import os
        for pf, ivf in zip(self.params_files, self.iv_files):
            if not os.path.exists(pf):
                raise FileNotFoundError(f"Parameters file not found: {pf}")
            if not os.path.exists(ivf):
                raise FileNotFoundError(f"IV file not found: {ivf}")

        if len(self.params_files) == 1:
            # Single file pair (original behavior)
            self.params_df, self.iv_data = load_raw_data(
                self.params_files[0], self.iv_files[0]
            )
        else:
            # Multiple file pairs - use concatenation
            from data import load_multiple_data_files
            self.params_df, self.iv_data = load_multiple_data_files(
                self.params_files, self.iv_files
            )

        print(f"\nLoaded {len(self.params_df)} samples total")
        print(f"Parameters shape: {self.params_df.shape}")
        print(f"IV curves shape: {self.iv_data.shape}")

        # Apply outlier filtering if enabled
        if self.filter_outliers:
            from data import filter_outliers
            self.params_df, self.iv_data, self.filter_stats = filter_outliers(
                self.params_df,
                self.iv_data,
                min_ff=self.filter_min_ff,
                min_vmpp=self.filter_min_vmpp,
                min_pce_quantile=self.filter_min_pce_quantile,
                device=self.device
            )
            print(f"\nAfter filtering: {len(self.params_df)} samples")

        # Load auxiliary anchor files if provided
        if self.anchors_files:
            self._load_anchor_files()
        if self.voc_anchors_files:
            self._load_voc_anchor_files()
        if self.vmpp_anchors_files:
            self._load_vmpp_anchor_files()

    def _load_vmpp_anchor_files(self):
        """Load auxiliary Vmpp-only anchors from txt files."""
        print("\n" + "=" * 60)
        print("Loading Vmpp Anchor Files")
        print("=" * 60)

        all_vmpp = []
        for vmpp_file in self.vmpp_anchors_files:
            if vmpp_file is None:
                continue
            print(f"  Loading: {vmpp_file}")
            vmpp_vals = np.loadtxt(vmpp_file, delimiter=',', skiprows=1, dtype=np.float32)
            if vmpp_vals.ndim > 1:
                vmpp_vals = vmpp_vals.squeeze()
            all_vmpp.append(vmpp_vals)
            print(f"    -> {len(vmpp_vals)} samples")

        if all_vmpp:
            self.vmpp_anchors_data = np.concatenate(all_vmpp, axis=0)
            print(f"\nTotal Vmpp anchor samples: {len(self.vmpp_anchors_data)}")

            if len(self.vmpp_anchors_data) != len(self.iv_data):
                raise ValueError(
                    f"Vmpp anchor size ({len(self.vmpp_anchors_data)}) does not match "
                    f"IV data size ({len(self.iv_data)}). Ensure Vmpp anchors were generated "
                    f"from the same preprocessed data."
                )

    def _load_voc_anchor_files(self):
        """Load auxiliary Voc-only anchors from txt files."""
        print("\n" + "=" * 60)
        print("Loading Voc Anchor Files")
        print("=" * 60)

        all_voc = []
        for voc_file in self.voc_anchors_files:
            if voc_file is None:
                continue
            print(f"  Loading: {voc_file}")
            voc_vals = np.loadtxt(voc_file, delimiter=',', skiprows=1, dtype=np.float32)
            if voc_vals.ndim > 1:
                voc_vals = voc_vals.squeeze()
            all_voc.append(voc_vals)
            print(f"    -> {len(voc_vals)} samples")

        if all_voc:
            self.voc_anchors_data = np.concatenate(all_voc, axis=0)
            print(f"\nTotal Voc anchor samples: {len(self.voc_anchors_data)}")

            if len(self.voc_anchors_data) != len(self.iv_data):
                raise ValueError(
                    f"Voc anchor size ({len(self.voc_anchors_data)}) does not match "
                    f"IV data size ({len(self.iv_data)}). Ensure Voc anchors were generated "
                    f"from the same preprocessed data."
                )

    def _load_anchor_files(self):
        """Load auxiliary anchor data from txt files."""
        print("\n" + "=" * 60)
        print("Loading Auxiliary Anchor Files")
        print("=" * 60)

        all_anchors = []
        for anchor_file in self.anchors_files:
            if anchor_file is None:
                continue
            print(f"  Loading: {anchor_file}")
            # Format: Jsc,Voc,Vmpp,Jmpp,FF,PCE,Pmpp (with header)
            anchors = np.loadtxt(anchor_file, delimiter=',', skiprows=1, dtype=np.float32)
            all_anchors.append(anchors)
            print(f"    -> {len(anchors)} samples")

        if all_anchors:
            self.anchors_data = np.vstack(all_anchors)
            print(f"\nTotal anchor samples: {len(self.anchors_data)}")
            print(f"Anchor columns: Jsc, Voc, Vmpp, Jmpp, FF, PCE, Pmpp")

            # Validate size matches IV data
            if len(self.anchors_data) != len(self.iv_data):
                raise ValueError(
                    f"Anchor data size ({len(self.anchors_data)}) does not match "
                    f"IV data size ({len(self.iv_data)}). Ensure anchors were generated "
                    f"from the same preprocessed data."
                )

            # Log which anchor inputs will be used
            inputs_used = []
            if self.use_vmpp_input:
                inputs_used.append("Vmpp")
            if self.use_jmpp_input:
                inputs_used.append("Jmpp")
            if self.use_ff_input:
                inputs_used.append("FF")
            if inputs_used:
                print(f"Anchor inputs enabled for curve model: {', '.join(inputs_used)}")
            else:
                print("No anchor inputs enabled (anchors loaded but not used as inputs)")

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

    def detect_and_log_outliers(self):
        """
        Detect and log outliers in target variables using IQR method.

        This provides early visibility into data quality issues that may
        cause training instability or poor predictions on edge cases.
        """
        print("\n" + "=" * 60)
        print("Detecting Outliers in Target Variables")
        print("=" * 60)

        # Check key target variables
        target_names = ['Jsc', 'Voc', 'Vmpp', 'Jmpp', 'FF', 'Pmpp']
        total_outliers = 0

        for name in target_names:
            if name in self.targets_np:
                log = self.logger.log_outliers(name, self.targets_np[name])
                total_outliers += log.n_outliers

        # Also check derived ratios that are used in training
        if 'Jsc' in self.targets_np and self.jsc_ceiling is not None:
            jsc_ratio = self.targets_np['Jsc'] / (self.jsc_ceiling + 1e-30)
            log = self.logger.log_outliers('Jsc_ratio', jsc_ratio)
            total_outliers += log.n_outliers

        if 'Voc' in self.targets_np and self.voc_ceiling is not None:
            voc_ratio = self.targets_np['Voc'] / (np.abs(self.voc_ceiling) + 1e-30)
            log = self.logger.log_outliers('Voc_ratio', voc_ratio)
            total_outliers += log.n_outliers

        print(f"\nTotal outliers detected across all targets: {total_outliers}")
        if total_outliers > 0:
            pct = 100.0 * total_outliers / (len(self.targets_np.get('Jsc', [])) * len(target_names))
            print(f"  (Approximate {pct:.2f}% of target-sample pairs)")

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

        # Fit preprocessing on training parameters and reuse for val/test
        self.preprocessor = PVDataPreprocessor(colnames=list(COLNAMES))
        X_train_proc = self.preprocessor.fit_transform_params(self.X_raw[self.train_idx])
        X_val_proc = self.preprocessor.transform_params(self.X_raw[self.val_idx])
        X_test_proc = self.preprocessor.transform_params(self.X_raw[self.test_idx])

        # Normalize curves by Isc (per-split, uses true Isc)
        isc_train, curves_train_norm = self.preprocessor.fit_transform_curves(self.iv_data[self.train_idx])
        isc_val, curves_val_norm = self.preprocessor.transform_curves(self.iv_data[self.val_idx])
        isc_test, curves_test_norm = self.preprocessor.transform_curves(self.iv_data[self.test_idx])

        self.splits = {
            'train': {
                'X_raw': X_train_proc,
                'X_raw_unscaled': self.X_raw[self.train_idx],
                'X_physics': self.physics_features_np[self.train_idx],
                'jsc_ceiling': self.jsc_ceiling[self.train_idx],
                'voc_ceiling': self.voc_ceiling[self.train_idx],
                'targets': {k: v[self.train_idx] for k, v in self.targets_np.items()},
                'curves': self.iv_data[self.train_idx],
                'curves_norm': curves_train_norm,
                'isc_values': isc_train,
                'v_grid': self.v_grid,
                'anchors': None  # Placeholder for anchors
            },
            'val': {
                'X_raw': X_val_proc,
                'X_raw_unscaled': self.X_raw[self.val_idx],
                'X_physics': self.physics_features_np[self.val_idx],
                'jsc_ceiling': self.jsc_ceiling[self.val_idx],
                'voc_ceiling': self.voc_ceiling[self.val_idx],
                'targets': {k: v[self.val_idx] for k, v in self.targets_np.items()},
                'curves': self.iv_data[self.val_idx],
                'curves_norm': curves_val_norm,
                'isc_values': isc_val,
                'v_grid': self.v_grid,
                'anchors': None  # Placeholder for anchors
            },
            'test': {
                'X_raw': X_test_proc,
                'X_raw_unscaled': self.X_raw[self.test_idx],
                'X_physics': self.physics_features_np[self.test_idx],
                'jsc_ceiling': self.jsc_ceiling[self.test_idx],
                'voc_ceiling': self.voc_ceiling[self.test_idx],
                'targets': {k: v[self.test_idx] for k, v in self.targets_np.items()},
                'curves': self.iv_data[self.test_idx],
                'curves_norm': curves_test_norm,
                'isc_values': isc_test,
                'v_grid': self.v_grid,
                'anchors': None  # Placeholder for anchors
            }
        }

        # Attach auxiliary anchors per split if provided
        if self.anchors_data is not None:
            self.splits['train']['anchors'] = self.anchors_data[self.train_idx]
            self.splits['val']['anchors'] = self.anchors_data[self.val_idx]
            self.splits['test']['anchors'] = self.anchors_data[self.test_idx]

        # Attach Voc anchors per split if provided
        if self.voc_anchors_data is not None:
            self.splits['train']['voc_anchor'] = self.voc_anchors_data[self.train_idx]
            self.splits['val']['voc_anchor'] = self.voc_anchors_data[self.val_idx]
            self.splits['test']['voc_anchor'] = self.voc_anchors_data[self.test_idx]

        if self.vmpp_anchors_data is not None:
            self.splits['train']['vmpp_anchor'] = self.vmpp_anchors_data[self.train_idx]
            self.splits['val']['vmpp_anchor'] = self.vmpp_anchors_data[self.val_idx]
            self.splits['test']['vmpp_anchor'] = self.vmpp_anchors_data[self.test_idx]

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
            hpo_config=self.hpo_config,
            direct_curve_only=self.use_direct_curve  # Only run Jsc LGBM HPO for direct curve mode
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

        # Predict anchors using trained scalar models (no oracle inputs)
        anchors_train = self._predict_curve_anchors(train)
        anchors_val = self._predict_curve_anchors(val)

        curves_train = train['curves_norm'].astype(np.float32)
        curves_val = val['curves_norm'].astype(np.float32)
        self.curve_norm_by_isc = True
        self.curve_output_normalized = True
        self.curve_knot_strategy = "mpp_cluster"

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
                best_value = None
                n_trials = None
                if result['study'] is not None:
                    try:
                        best_value = result['study'].best_value
                    except ValueError:
                        best_value = None
                    n_trials = len(result['study'].trials)
                hpo_summary[name] = {
                    'best_params': result['params'],
                    'best_value': best_value,
                    'n_trials': n_trials
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
        # Store normalization params for inference.
        # Use robust scaling (median / IQR) for heavy-tailed physics features.
        self.voc_feature_mean = np.median(X_train_full, axis=0, keepdims=True)
        q75 = np.percentile(X_train_full, 75, axis=0, keepdims=True)
        q25 = np.percentile(X_train_full, 25, axis=0, keepdims=True)
        self.voc_feature_std = (q75 - q25)
        self.voc_feature_std[self.voc_feature_std < 1e-8] = 1.0
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

        # 1b. Train Voc LGBM
        print("\n--- Training Voc LGBM ---")
        voc_lgbm_config = configs.get('voc_lgbm', VocLGBMConfig())
        self.models['voc_lgbm'] = build_voc_lgbm_model(voc_lgbm_config)
        self.models['voc_lgbm'].fit(
            train['X_raw'], train['X_physics'],
            train['targets']['Voc'], train['voc_ceiling'],
            val['X_raw'], val['X_physics'],
            val['targets']['Voc'], val['voc_ceiling']
        )

        # Compare Voc models on validation and select best
        voc_pred_val_nn = self._predict_voc_nn(val)
        voc_pred_val_lgbm = self._predict_voc_lgbm(val)
        voc_true_val = val['targets']['Voc']
        rmse_nn = float(np.sqrt(np.mean((voc_pred_val_nn - voc_true_val) ** 2)))
        rmse_lgbm = float(np.sqrt(np.mean((voc_pred_val_lgbm - voc_true_val) ** 2)))

        self.voc_model_type = 'lgbm' if rmse_lgbm <= rmse_nn else 'nn'
        self.metrics['voc_model_selection'] = {
            'rmse_nn': rmse_nn,
            'rmse_lgbm': rmse_lgbm,
            'selected': self.voc_model_type
        }
        print(f"\nVoc model selection: NN rmse={rmse_nn:.6f}, LGBM rmse={rmse_lgbm:.6f} "
              f"-> using {self.voc_model_type.upper()}")

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

        # Precompute upstream predictions for downstream training (no oracle inputs)
        voc_pred_train = self._predict_voc(train)
        voc_pred_val = self._predict_voc(val)
        jsc_pred_train = self.models['jsc_lgbm'].predict(
            train['X_raw'], train['X_physics'], train['jsc_ceiling']
        )
        jsc_pred_val = self.models['jsc_lgbm'].predict(
            val['X_raw'], val['X_physics'], val['jsc_ceiling']
        )

        # 3. Train Vmpp LGBM
        print("\n--- Training Vmpp LGBM ---")
        vmpp_config = configs.get('vmpp_lgbm', VmppLGBMConfig())
        self.models['vmpp_lgbm'] = build_vmpp_model(vmpp_config)

        self.models['vmpp_lgbm'].fit(
            train['X_raw'], train['X_physics'],
            train['targets']['Vmpp'], voc_pred_train,
            val['X_raw'], val['X_physics'],
            val['targets']['Vmpp'], voc_pred_val
        )

        vmpp_pred_train = self.models['vmpp_lgbm'].predict(
            train['X_raw'], train['X_physics'], voc_pred_train
        )
        vmpp_pred_val = self.models['vmpp_lgbm'].predict(
            val['X_raw'], val['X_physics'], voc_pred_val
        )

        # 4. Train Jmpp LGBM
        print("\n--- Training Jmpp LGBM ---")
        self.models['jmpp_lgbm'] = build_jmpp_model(vmpp_config)

        self.models['jmpp_lgbm'].fit(
            train['X_raw'], train['X_physics'],
            train['targets']['Jmpp'], jsc_pred_train, vmpp_pred_train,
            val['X_raw'], val['X_physics'],
            val['targets']['Jmpp'], jsc_pred_val, vmpp_pred_val
        )

        # 5. Train FF LGBM
        print("\n--- Training FF LGBM ---")
        ff_config = configs.get('ff_lgbm', VmppLGBMConfig())
        self.models['ff_lgbm'] = build_ff_model(ff_config)

        self.models['ff_lgbm'].fit(
            train['X_raw'], train['X_physics'],
            train['targets']['FF'], voc_pred_train, jsc_pred_train,
            val['X_raw'], val['X_physics'],
            val['targets']['FF'], voc_pred_val, jsc_pred_val
        )

        # 6. Train curve model (optional)
        if self.run_curve_model:
            if self.use_direct_curve:
                print("\n--- Training Direct Curve Model (No Vmpp Split) ---")
                self.train_direct_curve_model()
            else:
                print("\n--- Training Unified Split-Spline Curve Model ---")
                if self.run_curve_hpo:
                    self.run_curve_hyperparameter_optimization()
                self.train_curve_model_legacy()

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

        # Robust feature scaling for long-tailed engineered features
        self.curve_feature_mean = np.median(X_train_full, axis=0, keepdims=True)
        q75 = np.percentile(X_train_full, 75, axis=0, keepdims=True)
        q25 = np.percentile(X_train_full, 25, axis=0, keepdims=True)
        self.curve_feature_std = (q75 - q25)
        self.curve_feature_std[self.curve_feature_std < 1e-8] = 1.0
        X_train_norm = (X_train_full - self.curve_feature_mean) / self.curve_feature_std
        X_val_norm = (X_val_full - self.curve_feature_mean) / self.curve_feature_std

        # Predict anchors using scalar models (no oracle inputs)
        anchors_train = self._predict_curve_anchors(train)
        anchors_val = self._predict_curve_anchors(val)

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

        # Use precomputed curve normalization (KNOWN WORKING approach)
        print("\nUsing Isc-normalized curves from preprocessing...")
        isc_train = train['isc_values']
        isc_val = val['isc_values']
        curves_train_norm = train['curves_norm']
        curves_val_norm = val['curves_norm']
        self.curve_norm_by_isc = True

        # Validate normalization
        validate_curve_normalization(train['curves'], curves_train_norm, isc_train, n_samples=3)

        # Create datasets with BOTH normalized and raw curves
        # Loss will be computed in ABSOLUTE space to avoid normalization mismatch
        # (targets normalized by true Isc, predictions normalized by predicted Jsc)
        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_norm),
            torch.from_numpy(anchors_train_norm),  # Normalized for model input
            torch.from_numpy(anchors_train),       # Raw for curve reconstruction
            torch.from_numpy(train['curves'].astype(np.float32)),  # RAW absolute curves for loss
            torch.from_numpy(isc_train)            # For reference
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_norm),
            torch.from_numpy(anchors_val_norm),
            torch.from_numpy(anchors_val),
            torch.from_numpy(val['curves'].astype(np.float32)),  # RAW absolute curves for loss
            torch.from_numpy(isc_val)
        )

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2048, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2048)

        # Use ControlPointNet that takes anchors as input.
        # If curve HPO was run, reuse its hyperparameters (otherwise defaults).
        curve_hpo_cfg = getattr(self, 'best_configs', {}).get('curve_model')

        ctrl_points = self.ctrl_points
        hidden_dims = [256, 128, 64]
        dropout = 0.15
        activation = 'silu'
        lr = 1e-3
        weight_decay = 1e-5
        continuity_weight = self.continuity_weight

        if isinstance(curve_hpo_cfg, dict):
            ss_cfg = curve_hpo_cfg.get('config')
            if ss_cfg is not None:
                hidden_dims = getattr(ss_cfg, 'hidden_dims', hidden_dims) or hidden_dims
                dropout = float(getattr(ss_cfg, 'dropout', dropout))
                activation = getattr(ss_cfg, 'activation', activation)
                ctrl_points = int(getattr(ss_cfg, 'ctrl_points', ctrl_points))

            lr = float(curve_hpo_cfg.get('lr', lr))
            weight_decay = float(curve_hpo_cfg.get('weight_decay', weight_decay))
            continuity_weight = float(curve_hpo_cfg.get('continuity_weight', continuity_weight))

        config = ControlPointNetConfig(
            input_dim=X_train_norm.shape[1],
            anchor_dim=4,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            ctrl_points=ctrl_points,
        )

        model = ControlPointNet(config).to(self.device)
        curve_loss_fn = CurveLoss(mpp_weight=2.0).to(self.device)

        # Simpler optimizer - no learnable loss weights
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

        v_grid = torch.from_numpy(self.v_grid).float().to(self.device)

        best_val = float('inf')
        best_state = None
        patience = 20
        patience_counter = 0

        print(f"\nTraining ControlPointNet with:")
        print(f"  ctrl_points: {config.ctrl_points}")
        print(f"  hidden_dims: {config.hidden_dims}")
        print(f"  continuity_weight: {continuity_weight}")
        print(f"  Using predicted anchors (no oracle inputs)")

        for epoch in range(150):
            model.train()
            epoch_losses = []

            for batch_x, batch_anchors_norm, batch_anchors_raw, batch_curves_raw, batch_isc in train_loader:
                batch_x = batch_x.to(self.device)
                batch_anchors_norm = batch_anchors_norm.to(self.device)
                batch_anchors_raw = batch_anchors_raw.to(self.device)
                batch_curves_raw = batch_curves_raw.to(self.device)
                batch_isc = batch_isc.to(self.device)

                optimizer.zero_grad()

                # Model predicts only control points (anchors provided as input)
                ctrl1, ctrl2 = model(batch_x, batch_anchors_norm)

                # Reconstruct curve in normalized space, then denormalize
                pred_curve_norm = reconstruct_curve_normalized(
                    batch_anchors_raw, ctrl1, ctrl2, v_grid, clamp_voc=True
                )
                pred_curve_abs = denormalize_curves_by_isc(pred_curve_norm, batch_anchors_raw[:, 0])

                # CRITICAL: Compute loss in ABSOLUTE space to avoid normalization mismatch
                # (targets would be normalized by true Isc, predictions by predicted Jsc)
                loss, metrics = curve_loss_fn(
                    pred_curve_abs, batch_curves_raw, v_grid, batch_anchors_raw[:, 2]
                )

                # Add continuity penalty in normalized space
                anchors_norm = normalize_anchors_by_jsc(batch_anchors_raw)
                cont_loss = continuity_loss(anchors_norm, ctrl1, ctrl2, v_grid, j_end=-1.0)
                loss = loss + continuity_weight * cont_loss

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
                for batch_x, batch_anchors_norm, batch_anchors_raw, batch_curves_raw, batch_isc in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_anchors_norm = batch_anchors_norm.to(self.device)
                    batch_anchors_raw = batch_anchors_raw.to(self.device)
                    batch_curves_raw = batch_curves_raw.to(self.device)
                    batch_isc = batch_isc.to(self.device)

                    ctrl1, ctrl2 = model(batch_x, batch_anchors_norm)
                    pred_curve_norm = reconstruct_curve_normalized(
                        batch_anchors_raw, ctrl1, ctrl2, v_grid, clamp_voc=True
                    )
                    pred_curve_abs = denormalize_curves_by_isc(pred_curve_norm, batch_anchors_raw[:, 0])

                    # Loss in absolute space (matches training)
                    val_loss, _ = curve_loss_fn(
                        pred_curve_abs, batch_curves_raw, v_grid, batch_anchors_raw[:, 2]
                    )
                    val_losses.append(val_loss.item())

                    # Compute MSE in absolute space
                    err = (pred_curve_abs - batch_curves_raw) ** 2
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

    def train_direct_curve_model(self):
        """
        Train direct curve model using shape-only approach.

        IMPROVED APPROACH:
        1. Uses Jsc from pretrained LGBM (accurate: R²=0.965)
        2. Uses Voc from pretrained Voc NN (R²=0.73) - NOT predicted by curve model
        3. Only predicts SHAPE via control points (decoupled from endpoints)
        4. Non-uniform knot placement for better knee capture
        5. Larger network with residual connections

        This separates two orthogonal learning tasks:
        - Endpoint prediction (Jsc, Voc) -> pretrained models
        - Shape prediction (control points) -> this model
        """
        print("\n" + "=" * 60)
        print("Training Direct Curve Model (Shape-Only Approach)")
        print("=" * 60)

        train = self.splits['train']
        val = self.splits['val']

        # Prepare inputs
        X_train_full = np.hstack([train['X_raw'], train['X_physics']]).astype(np.float32)
        X_val_full = np.hstack([val['X_raw'], val['X_physics']]).astype(np.float32)

        # Robust feature scaling
        self.curve_feature_mean = np.median(X_train_full, axis=0, keepdims=True)
        q75 = np.percentile(X_train_full, 75, axis=0, keepdims=True)
        q25 = np.percentile(X_train_full, 25, axis=0, keepdims=True)
        self.curve_feature_std = (q75 - q25)
        self.curve_feature_std[self.curve_feature_std < 1e-8] = 1.0
        X_train_norm = (X_train_full - self.curve_feature_mean) / self.curve_feature_std
        X_val_norm = (X_val_full - self.curve_feature_mean) / self.curve_feature_std

        # Get Jsc from pretrained LGBM (accurate predictor)
        jsc_train = self.models['jsc_lgbm'].predict(
            train['X_raw'], train['X_physics'], train['jsc_ceiling']
        )
        jsc_val = self.models['jsc_lgbm'].predict(
            val['X_raw'], val['X_physics'], val['jsc_ceiling']
        )

        # Get Voc from Voc anchors if provided, otherwise from pretrained Voc NN
        if train.get('voc_anchor') is not None and val.get('voc_anchor') is not None:
            voc_train_pred = train['voc_anchor'].astype(np.float32)
            voc_val_pred = val['voc_anchor'].astype(np.float32)
        else:
            voc_train_pred = self._predict_voc_nn(train)
            voc_val_pred = self._predict_voc_nn(val)

        # Also keep true Voc for validation metrics
        voc_train_true = train['targets']['Voc']
        voc_val_true = val['targets']['Voc']

        # Normalized curves for training
        curves_train_norm = train['curves_norm']
        curves_val_norm = val['curves_norm']

        # Optional anchor inputs (Vmpp, Jmpp, FF)
        use_anchor_inputs = self.use_vmpp_input or self.use_jmpp_input or self.use_ff_input
        vmpp_train = jmpp_train = ff_train = None
        vmpp_val = jmpp_val = ff_val = None
        if self.use_vmpp_input:
            if train.get('vmpp_anchor') is None or val.get('vmpp_anchor') is None:
                raise ValueError(
                    "Vmpp input enabled but vmpp anchors are missing. "
                    "Provide --vmpp-anchors (and --vmpp-anchors-extra if needed)."
                )
            vmpp_train = train['vmpp_anchor'].astype(np.float32)
            vmpp_val = val['vmpp_anchor'].astype(np.float32)
        if self.use_jmpp_input or self.use_ff_input:
            if self.anchors_data is None:
                raise ValueError(
                    "Jmpp/FF inputs enabled but no anchors file loaded. "
                    "Provide --anchors (and --anchors-extra if needed)."
                )
            anchors_train = train.get('anchors')
            anchors_val = val.get('anchors')
            if anchors_train is None or anchors_val is None:
                raise ValueError("Jmpp/FF inputs enabled but anchors are missing from splits.")
            if self.use_jmpp_input:
                jmpp_train = anchors_train[:, 3].astype(np.float32)
                jmpp_val = anchors_val[:, 3].astype(np.float32)
            if self.use_ff_input:
                ff_train = anchors_train[:, 4].astype(np.float32)
                ff_val = anchors_val[:, 4].astype(np.float32)

        print(f"\nUsing pretrained models for endpoints:")
        print(f"  Jsc from LGBM - range: [{jsc_train.min():.2f}, {jsc_train.max():.2f}]")
        print(f"  Voc from Voc NN - range: [{voc_train_pred.min():.4f}, {voc_train_pred.max():.4f}]")
        print(f"  True Voc range: [{voc_train_true.min():.4f}, {voc_train_true.max():.4f}]")
        voc_mae_pretrained = np.abs(voc_train_pred - voc_train_true).mean()
        print(f"  Voc NN MAE (train): {voc_mae_pretrained:.4f}")

        # Create datasets - use PREDICTED Voc for shape learning
        # This way the shape model learns to work with the actual Voc it will receive at inference
        train_tensors = [
            torch.from_numpy(X_train_norm.astype(np.float32)),
            torch.from_numpy(jsc_train.astype(np.float32)),
            torch.from_numpy(voc_train_pred.astype(np.float32)),  # Predicted Voc for training
            torch.from_numpy(voc_train_true.astype(np.float32)),  # True Voc for metrics
            torch.from_numpy(curves_train_norm.astype(np.float32))
        ]
        val_tensors = [
            torch.from_numpy(X_val_norm.astype(np.float32)),
            torch.from_numpy(jsc_val.astype(np.float32)),
            torch.from_numpy(voc_val_pred.astype(np.float32)),
            torch.from_numpy(voc_val_true.astype(np.float32)),
            torch.from_numpy(curves_val_norm.astype(np.float32))
        ]
        if self.use_vmpp_input:
            train_tensors.append(torch.from_numpy(vmpp_train.astype(np.float32)))
            val_tensors.append(torch.from_numpy(vmpp_val.astype(np.float32)))
        if self.use_jmpp_input:
            train_tensors.append(torch.from_numpy(jmpp_train.astype(np.float32)))
            val_tensors.append(torch.from_numpy(jmpp_val.astype(np.float32)))
        if self.use_ff_input:
            train_tensors.append(torch.from_numpy(ff_train.astype(np.float32)))
            val_tensors.append(torch.from_numpy(ff_val.astype(np.float32)))


        train_ds = torch.utils.data.TensorDataset(*train_tensors)
        val_ds = torch.utils.data.TensorDataset(*val_tensors)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2048, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2048)

        # ====================================================================
        # HPO FOR DIRECTCURVESHAPENET (NEW v2.0)
        # ====================================================================
        if self.run_curve_hpo:
            print("\n" + "=" * 60)
            print("Running HPO for DirectCurveShapeNet")
            print("=" * 60)

            hpo_results = run_direct_curve_shape_hpo(
                X_train=X_train_norm,
                jsc_train=jsc_train,
                voc_train=voc_train_pred,
                curves_train=curves_train_norm,
                X_val=X_val_norm,
                jsc_val=jsc_val,
                voc_val=voc_val_pred,
                curves_val=curves_val_norm,
                v_grid=self.v_grid,
                device=self.device,
                hpo_config=self.hpo_config,
                n_trials=self.hpo_config.n_trials_nn,
                vmpp_train=vmpp_train,
                jmpp_train=jmpp_train,
                ff_train=ff_train,
                vmpp_val=vmpp_val,
                jmpp_val=jmpp_val,
                ff_val=ff_val,
                use_vmpp_input=self.use_vmpp_input,
                use_jmpp_input=self.use_jmpp_input,
                use_ff_input=self.use_ff_input
            )

            # Extract best params
            best_params = hpo_results['direct_curve_shape']['params']
            print(f"\nUsing HPO-optimized config:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")

            # Build hidden_dims from HPO params
            n_layers = best_params.get('n_layers', 3)
            hidden_dims = [best_params.get(f'hidden_{i}', 256) for i in range(n_layers)]

            config = DirectCurveShapeNetConfig(
                input_dim=X_train_norm.shape[1],
                hidden_dims=hidden_dims,
                dropout=best_params.get('dropout', 0.1),
                activation=best_params.get('activation', 'silu'),
                ctrl_points=best_params.get('ctrl_points', 8),
                use_residual=best_params.get('use_residual', True),
                # Auxiliary anchor inputs
                use_vmpp_input=self.use_vmpp_input,
                use_jmpp_input=self.use_jmpp_input,
                use_ff_input=self.use_ff_input
            )

            # Use HPO-optimized loss params
            loss_fn = DirectCurveShapeLoss(
                weight_smooth=best_params.get('weight_smooth', 0.05),
                weight_mono=best_params.get('weight_mono', 1.0),
                knee_weight=best_params.get('knee_weight', 2.0),
                loss_type='huber',
                huber_delta=best_params.get('huber_delta', 0.1),
                sample_weight_power=0.5
            )

            # Use HPO-optimized training params
            lr = best_params.get('lr', 5e-4)
            weight_decay = best_params.get('weight_decay', 1e-5)
        else:
            # Fallback to default config (no HPO)
            print("\nUsing default config (no HPO)")
            n_ctrl_points = max(self.ctrl_points, 8)
            config = DirectCurveShapeNetConfig(
                input_dim=X_train_norm.shape[1],
                hidden_dims=[512, 256, 128],
                dropout=0.1,
                activation='silu',
                ctrl_points=n_ctrl_points,
                use_residual=True,
                # Auxiliary anchor inputs
                use_vmpp_input=self.use_vmpp_input,
                use_jmpp_input=self.use_jmpp_input,
                use_ff_input=self.use_ff_input
            )

            loss_fn = DirectCurveShapeLoss(
                weight_smooth=0.05,
                weight_mono=1.0,
                knee_weight=2.0,
                loss_type='huber',
                huber_delta=0.1,
                sample_weight_power=0.5
            )

            lr = 5e-4
            weight_decay = 1e-5

        model = DirectCurveShapeNet(config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )

        v_grid = torch.from_numpy(self.v_grid).float().to(self.device)

        best_val = float('inf')
        best_state = None
        patience = 30  # More patience for shape learning
        patience_counter = 0

        print(f"\nTraining DirectCurveShapeNet with:")
        print(f"  ctrl_points: {config.ctrl_points}")
        print(f"  hidden_dims: {config.hidden_dims}")
        print(f"  use_residual: {config.use_residual}")
        print(f"  Jsc and Voc from pretrained models (shape-only learning)")

        for epoch in range(200):
            model.train()
            epoch_losses = []

            for batch in train_loader:
                batch_iter = iter(batch)
                batch_x = next(batch_iter)
                batch_jsc = next(batch_iter)
                batch_voc_pred = next(batch_iter)
                batch_voc_true = next(batch_iter)
                batch_curves_norm = next(batch_iter)
                batch_vmpp = next(batch_iter) if self.use_vmpp_input else None
                batch_jmpp = next(batch_iter) if self.use_jmpp_input else None
                batch_ff = next(batch_iter) if self.use_ff_input else None
                batch_x = batch_x.to(self.device)
                batch_jsc = batch_jsc.to(self.device)
                batch_voc_pred = batch_voc_pred.to(self.device)
                batch_curves_norm = batch_curves_norm.to(self.device)
                if batch_vmpp is not None:
                    batch_vmpp = batch_vmpp.to(self.device)
                if batch_jmpp is not None:
                    batch_jmpp = batch_jmpp.to(self.device)
                if batch_ff is not None:
                    batch_ff = batch_ff.to(self.device)

                optimizer.zero_grad()

                # Model only predicts control points (shape)
                ctrl = model(batch_x, batch_jsc, batch_voc_pred, batch_vmpp, batch_jmpp, batch_ff)

                # Reconstruct curve using non-uniform knots
                pred_curve_norm = reconstruct_curve_shape(
                    batch_voc_pred, ctrl, v_grid, clamp_voc=True
                )

                # Compute shape-only loss
                loss, metrics = loss_fn(
                    pred_curve_norm, batch_curves_norm,
                    batch_voc_pred, v_grid
                )

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

            with torch.no_grad():
                for batch in val_loader:
                    batch_iter = iter(batch)
                    batch_x = next(batch_iter)
                    batch_jsc = next(batch_iter)
                    batch_voc_pred = next(batch_iter)
                    batch_voc_true = next(batch_iter)
                    batch_curves_norm = next(batch_iter)
                    batch_vmpp = next(batch_iter) if self.use_vmpp_input else None
                    batch_jmpp = next(batch_iter) if self.use_jmpp_input else None
                    batch_ff = next(batch_iter) if self.use_ff_input else None
                    batch_x = batch_x.to(self.device)
                    batch_jsc = batch_jsc.to(self.device)
                    batch_voc_pred = batch_voc_pred.to(self.device)
                    batch_curves_norm = batch_curves_norm.to(self.device)
                    if batch_vmpp is not None:
                        batch_vmpp = batch_vmpp.to(self.device)
                    if batch_jmpp is not None:
                        batch_jmpp = batch_jmpp.to(self.device)
                    if batch_ff is not None:
                        batch_ff = batch_ff.to(self.device)

                    ctrl = model(batch_x, batch_jsc, batch_voc_pred, batch_vmpp, batch_jmpp, batch_ff)
                    pred_curve_norm = reconstruct_curve_shape(
                        batch_voc_pred, ctrl, v_grid, clamp_voc=True
                    )

                    val_loss, _ = loss_fn(
                        pred_curve_norm, batch_curves_norm,
                        batch_voc_pred, v_grid
                    )
                    val_losses.append(val_loss.item())

                    # MSE in normalized space
                    err = (pred_curve_norm - batch_curves_norm) ** 2
                    sum_sq_full += err.sum().item()
                    sum_cnt += err.numel()

            avg_val = float(np.mean(val_losses))

            if epoch % 10 == 0:
                mse_norm = sum_sq_full / max(1, sum_cnt)
                print(
                    f"Epoch {epoch}: train={np.mean(epoch_losses):.6f}, "
                    f"val={avg_val:.6f}, mse_norm={mse_norm:.6f}, "
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

        # Store the model
        self.models['direct_curve_model'] = model
        self.models['direct_curve_config'] = config
        self.models['curve_model'] = model  # For compatibility
        self.models['direct_curve_uses_shape_net'] = True  # Flag for evaluation

        print(f"\nDirect curve model (shape-only) training complete. Best val loss: {best_val:.6f}")

    def _predict_voc_nn(self, split: dict) -> np.ndarray:
        """Predict Voc using the trained Voc NN."""
        X_full = np.hstack([split['X_raw'], split['X_physics']]).astype(np.float32)
        X_norm = (X_full - self.voc_feature_mean) / self.voc_feature_std
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)
        self.models['voc_nn'].eval()
        with torch.no_grad():
            voc_pred_norm = self.models['voc_nn'](X_tensor).cpu().numpy()
        return voc_pred_norm * self.voc_target_std + self.voc_target_mean

    def _predict_voc_lgbm(self, split: dict) -> np.ndarray:
        """Predict Voc using the trained Voc LGBM."""
        return self.models['voc_lgbm'].predict(
            split['X_raw'], split['X_physics'], split['voc_ceiling']
        )

    def _predict_voc(self, split: dict) -> np.ndarray:
        """Predict Voc using the selected Voc model."""
        if self.voc_model_type == 'lgbm':
            return self._predict_voc_lgbm(split)
        return self._predict_voc_nn(split)

    def _predict_curve_anchors(self, split: dict) -> np.ndarray:
        """Predict anchors (Jsc, Voc, Vmpp, Jmpp) using trained models."""
        voc_pred = self._predict_voc(split)
        jsc_pred = self.models['jsc_lgbm'].predict(
            split['X_raw'], split['X_physics'], split['jsc_ceiling']
        )
        vmpp_pred = self.models['vmpp_lgbm'].predict(
            split['X_raw'], split['X_physics'], voc_pred
        )
        jmpp_pred = self.models['jmpp_lgbm'].predict(
            split['X_raw'], split['X_physics'], jsc_pred, vmpp_pred
        )
        anchors = np.stack([jsc_pred, voc_pred, vmpp_pred, jmpp_pred], axis=1).astype(np.float32)

        # Validate and fix anchor constraints
        anchors = self._validate_and_fix_anchors(anchors, split)
        return anchors

    def _validate_and_fix_anchors(
        self,
        anchors: np.ndarray,
        split: dict,
        warn_threshold: float = 0.3,
        fix_constraints: bool = True
    ) -> np.ndarray:
        """
        Validate predicted anchors and fix physics constraint violations.

        Args:
            anchors: (N, 4) [Jsc, Voc, Vmpp, Jmpp] predicted anchors
            split: Data split dict containing targets for comparison
            warn_threshold: Relative error threshold for warnings (default 30%)
            fix_constraints: Whether to fix constraint violations

        Returns:
            anchors: (N, 4) Validated/fixed anchors
        """
        jsc, voc, vmpp, jmpp = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        targets = split.get('targets', {})

        # Check for large deviations from true values (for diagnostics)
        if targets:
            jsc_true = targets.get('Jsc')
            voc_true = targets.get('Voc')
            if jsc_true is not None:
                jsc_err = np.abs(jsc - jsc_true) / (np.abs(jsc_true) + 1e-6)
                n_jsc_bad = (jsc_err > warn_threshold).sum()
                if n_jsc_bad > 0:
                    print(f"  [Anchor Warning] {n_jsc_bad} samples have Jsc error > {warn_threshold*100:.0f}%")
            if voc_true is not None:
                voc_err = np.abs(voc - voc_true) / (np.abs(voc_true) + 1e-6)
                n_voc_bad = (voc_err > warn_threshold).sum()
                if n_voc_bad > 0:
                    print(f"  [Anchor Warning] {n_voc_bad} samples have Voc error > {warn_threshold*100:.0f}%")

        if not fix_constraints:
            return anchors

        # Fix physics constraints
        anchors = anchors.copy()
        eps = 1e-4

        # 1. Jsc must be positive
        anchors[:, 0] = np.maximum(anchors[:, 0], eps)

        # 2. Voc must be positive and reasonable (< 2V for solar cells)
        anchors[:, 1] = np.clip(anchors[:, 1], eps, 1.4)

        # 3. Vmpp must be in (0, Voc)
        anchors[:, 2] = np.clip(anchors[:, 2], eps, anchors[:, 1] - eps)

        # 4. Jmpp must be in (0, Jsc)
        anchors[:, 3] = np.clip(anchors[:, 3], eps, anchors[:, 0] - eps)

        # Count how many were fixed
        n_fixed = (
            (jsc <= eps).sum() +
            (voc <= eps).sum() + (voc > 1.4).sum() +
            (vmpp <= eps).sum() + (vmpp >= voc).sum() +
            (jmpp <= eps).sum() + (jmpp >= jsc).sum()
        )
        if n_fixed > 0:
            print(f"  [Anchor Fix] Fixed {n_fixed} constraint violations")

        return anchors

    def _predict_scalar_chain(self, split: dict) -> dict:
        """Predict scalar chain outputs using only model predictions."""
        voc_pred = self._predict_voc(split)
        jsc_pred = self.models['jsc_lgbm'].predict(
            split['X_raw'], split['X_physics'], split['jsc_ceiling']
        )
        vmpp_pred = self.models['vmpp_lgbm'].predict(
            split['X_raw'], split['X_physics'], voc_pred
        )
        jmpp_pred = self.models['jmpp_lgbm'].predict(
            split['X_raw'], split['X_physics'], jsc_pred, vmpp_pred
        )
        ff_pred = self.models['ff_lgbm'].predict(
            split['X_raw'], split['X_physics'], voc_pred, jsc_pred
        )
        return {
            'Voc': voc_pred,
            'Jsc': jsc_pred,
            'Vmpp': vmpp_pred,
            'Jmpp': jmpp_pred,
            'FF': ff_pred
        }

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

        ctrl_points = min(max(self.ctrl_points, 4), 5)
        config = SplitSplineNetConfig(input_dim=X_train_full.shape[1], ctrl_points=ctrl_points)
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
                pred_curve = reconstruct_curve_normalized(
                    pred_anchors, ctrl1, ctrl2, v_grid, clamp_voc=True, knot_strategy="mpp_cluster"
                )
                loss, metrics = multitask_loss(pred_anchors, batch_anchors, pred_curve, batch_curves)
                cont_loss = continuity_loss(
                    normalize_anchors_by_jsc(pred_anchors), ctrl1, ctrl2,
                    v_grid, j_end=-1.0, knot_strategy="mpp_cluster"
                )
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
                    pred_curve = reconstruct_curve_normalized(
                        pred_anchors, ctrl1, ctrl2, v_grid, clamp_voc=True, knot_strategy="mpp_cluster"
                    )
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

        Supports:
        - DirectCurveNetWithJsc (uses Jsc from LGBM, predicts Voc + control points)
        - ControlPointNet (uses predicted anchors, only predicts control points)
        - Legacy UnifiedSplitSplineNet (predicts anchors + control points)
        """
        import time

        # Check which model type we have
        use_direct_curve_model = 'direct_curve_model' in self.models
        use_ctrl_point_model = 'ctrl_point_model' in self.models
        use_legacy_model = 'curve_model' in self.models and not use_direct_curve_model

        # Log oracle mode if enabled
        if self.oracle_voc and use_direct_curve_model:
            print("\n[Oracle Mode] Using TRUE Voc for curve truncation (upper bound evaluation)")

        if not use_direct_curve_model and not use_ctrl_point_model and not use_legacy_model:
            raise ValueError("No curve model trained. Run train_curve_model() or train_direct_curve_model() first.")

        split = self.splits[split_name]
        X_full = np.hstack([split['X_raw'], split['X_physics']]).astype(np.float32)
        X_full = (X_full - self.curve_feature_mean) / self.curve_feature_std

        # Optional anchor inputs for shape model
        vmpp_split = jmpp_split = ff_split = None
        if self.use_vmpp_input or self.use_jmpp_input or self.use_ff_input:
            anchors_split = split.get('anchors')
            if anchors_split is None and (self.use_jmpp_input or self.use_ff_input):
                raise ValueError(
                    "Anchor inputs enabled but split anchors are missing. "
                    "Ensure --anchors were provided and preprocessing is aligned."
                )
            if self.use_vmpp_input:
                if split.get('vmpp_anchor') is None:
                    raise ValueError(
                        "Vmpp input enabled but vmpp anchors are missing. "
                        "Provide --vmpp-anchors (and --vmpp-anchors-extra if needed)."
                    )
                vmpp_split = split['vmpp_anchor'].astype(np.float32)
            if self.use_jmpp_input:
                jmpp_split = anchors_split[:, 3].astype(np.float32)
            if self.use_ff_input:
                ff_split = anchors_split[:, 4].astype(np.float32)

        anchors_true = np.stack(
            [split['targets']['Jsc'], split['targets']['Voc'],
             split['targets']['Vmpp'], split['targets']['Jmpp']],
            axis=1
        ).astype(np.float32)
        curves_true = split['curves'].astype(np.float32)
        curves_true_norm = split.get('curves_norm')
        sample_indices = np.arange(len(split['X_raw']), dtype=np.int64)
        targets = split.get('targets', {})

        # Prepare data based on model type
        use_shape_net = self.models.get('direct_curve_uses_shape_net', False)

        if use_direct_curve_model:
            # Get Jsc from LGBM
            jsc_pred = self.models['jsc_lgbm'].predict(
                split['X_raw'], split['X_physics'], split['jsc_ceiling']
            )
            voc_true = split['targets']['Voc'].astype(np.float32)
            curves_true_norm = split.get('curves_norm')
            if curves_true_norm is None:
                isc = curves_true[:, 0:1]
                isc_safe = np.where(np.abs(isc) < 1e-9, 1.0, isc)
                curves_true_norm = 2.0 * (curves_true / isc_safe) - 1.0

            if use_shape_net:
                # DirectCurveShapeNet: needs Jsc AND Voc from pretrained models
                if split.get('voc_anchor') is not None:
                    voc_pred = split['voc_anchor'].astype(np.float32)
                else:
                    voc_pred = self._predict_voc_nn(split)
                tensors = [
                    torch.from_numpy(sample_indices),
                    torch.from_numpy(X_full),
                    torch.from_numpy(jsc_pred.astype(np.float32)),
                    torch.from_numpy(voc_pred.astype(np.float32)),  # Predicted Voc
                    torch.from_numpy(voc_true),  # True Voc for metrics
                    torch.from_numpy(curves_true),
                    torch.from_numpy(curves_true_norm.astype(np.float32)),
                    torch.from_numpy(anchors_true)
                ]
                if self.use_vmpp_input:
                    tensors.append(torch.from_numpy(vmpp_split.astype(np.float32)))
                if self.use_jmpp_input:
                    tensors.append(torch.from_numpy(jmpp_split.astype(np.float32)))
                if self.use_ff_input:
                    tensors.append(torch.from_numpy(ff_split.astype(np.float32)))
                ds = torch.utils.data.TensorDataset(*tensors)
            else:
                # Legacy DirectCurveNetWithJsc: needs Jsc, predicts Voc
                ds = torch.utils.data.TensorDataset(
                    torch.from_numpy(sample_indices),
                    torch.from_numpy(X_full),
                    torch.from_numpy(jsc_pred.astype(np.float32)),
                    torch.from_numpy(voc_true),
                    torch.from_numpy(curves_true),
                    torch.from_numpy(curves_true_norm.astype(np.float32)),
                    torch.from_numpy(anchors_true)
                )
            model = self.models['direct_curve_model']
        elif use_ctrl_point_model:
            anchors_pred = self._predict_curve_anchors(split)
            anchors_norm = (anchors_pred - self.anchor_mean) / self.anchor_std
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(sample_indices),
                torch.from_numpy(X_full),
                torch.from_numpy(anchors_norm.astype(np.float32)),
                torch.from_numpy(anchors_pred),
                torch.from_numpy(curves_true),
                torch.from_numpy(anchors_true)
            )
            model = self.models['ctrl_point_model']
        else:
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(sample_indices),
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

        # For R² computation: track variance of true curves
        sum_true = 0.0  # sum of true values
        sum_true_sq = 0.0  # sum of squared true values
        curve_min = float('inf')
        curve_max = float('-inf')

        # Per-sample R² values for median computation
        per_sample_r2_list = []

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
        pred_curves_for_plot = []
        true_curves_for_plot = []
        max_plot_samples = 2000

        analysis_records = None
        if split_name == 'test':
            analysis_records = {
                'sample_idx': [],
                'curve_mse': [],
                'curve_r2': [],
                'mse_region1': [],
                'mse_region2': [],
                'jsc_true': [],
                'voc_true': [],
                'vmpp_true': [],
                'jmpp_true': [],
                'jsc_pred': [],
                'voc_pred': [],
                'vmpp_pred': [],
                'jmpp_pred': [],
                'ff_true': [],
                'ff_pred': [],
                'ff_abs_err': []
            }
            if 'PCE' in targets:
                analysis_records['pce_true'] = []
            if 'Pmpp' in targets:
                analysis_records['pmpp_true'] = []
            if 'Jsc' in targets:
                analysis_records['jsc_ratio'] = []
            if 'Voc' in targets:
                analysis_records['voc_ratio'] = []

        with torch.no_grad():
            for batch in loader:
                sample_idx = batch[0].cpu().numpy().astype(np.int64)
                batch = batch[1:]
                if use_direct_curve_model:
                    if use_shape_net:
                        # DirectCurveShapeNet: uses Jsc AND Voc from pretrained models
                        batch_iter = iter(batch)
                        batch_x = next(batch_iter)
                        batch_jsc = next(batch_iter)
                        batch_voc_pred = next(batch_iter)
                        batch_voc_true = next(batch_iter)
                        batch_curves = next(batch_iter)
                        batch_curves_norm = next(batch_iter)
                        batch_anchors_true = next(batch_iter)
                        batch_vmpp = next(batch_iter) if self.use_vmpp_input else None
                        batch_jmpp = next(batch_iter) if self.use_jmpp_input else None
                        batch_ff = next(batch_iter) if self.use_ff_input else None
                        batch_x = batch_x.to(self.device)
                        batch_jsc = batch_jsc.to(self.device)
                        batch_voc_pred = batch_voc_pred.to(self.device)
                        batch_voc_true = batch_voc_true.to(self.device)
                        batch_curves = batch_curves.to(self.device)
                        batch_curves_norm = batch_curves_norm.to(self.device)
                        batch_anchors_true = batch_anchors_true.to(self.device)
                        if batch_vmpp is not None:
                            batch_vmpp = batch_vmpp.to(self.device)
                        if batch_jmpp is not None:
                            batch_jmpp = batch_jmpp.to(self.device)
                        if batch_ff is not None:
                            batch_ff = batch_ff.to(self.device)

                        # Model only predicts shape (control points)
                        ctrl = model(batch_x, batch_jsc, batch_voc_pred, batch_vmpp, batch_jmpp, batch_ff)

                        # Oracle mode: use true Voc for reconstruction
                        voc_for_curve = batch_voc_true if self.oracle_voc else batch_voc_pred

                        # Reconstruct using shape-only reconstruction with non-uniform knots
                        pred_curve_norm = reconstruct_curve_shape(
                            voc_for_curve, ctrl, v_grid, clamp_voc=True
                        )
                        pred_curve = denormalize_curves_by_isc(pred_curve_norm, batch_jsc)

                        # Use Voc for anchors (oracle or predicted)
                        pred_voc = voc_for_curve
                    else:
                        # Legacy DirectCurveNetWithJsc: uses Jsc from LGBM, predicts Voc + ctrl
                        batch_x, batch_jsc, batch_voc_true, batch_curves, batch_curves_norm, batch_anchors_true = batch
                        batch_x = batch_x.to(self.device)
                        batch_jsc = batch_jsc.to(self.device)
                        batch_voc_true = batch_voc_true.to(self.device)
                        batch_curves = batch_curves.to(self.device)
                        batch_curves_norm = batch_curves_norm.to(self.device)
                        batch_anchors_true = batch_anchors_true.to(self.device)

                        # Model predicts Voc and control points
                        pred_voc, ctrl = model(batch_x, batch_jsc)

                        # Oracle mode: use true Voc for reconstruction
                        voc_for_curve = batch_voc_true if self.oracle_voc else pred_voc

                        # Reconstruct curve in normalized space, then denormalize
                        pred_curve_norm = reconstruct_curve_direct_normalized(
                            batch_jsc, voc_for_curve, ctrl, v_grid, clamp_voc=True
                        )
                        pred_curve = denormalize_curves_by_isc(pred_curve_norm, batch_jsc)

                        # For anchors, use the Voc we used for curve
                        pred_voc = voc_for_curve

                    # Create pseudo-anchors for metric computation
                    # Vmpp and Jmpp are estimated from the curve
                    power = pred_curve * v_grid.unsqueeze(0)
                    mpp_idx = power.argmax(dim=1)
                    batch_idx = torch.arange(pred_curve.shape[0], device=self.device)
                    pred_vmpp = v_grid[mpp_idx]
                    pred_jmpp = pred_curve[batch_idx, mpp_idx]
                    pred_anchors = torch.stack([batch_jsc, pred_voc, pred_vmpp, pred_jmpp], dim=1)

                elif use_ctrl_point_model:
                    batch_x, batch_anchors_norm, batch_anchors, batch_curves, batch_anchors_true = batch
                    batch_x = batch_x.to(self.device)
                    batch_anchors_norm = batch_anchors_norm.to(self.device)
                    batch_anchors = batch_anchors.to(self.device)
                    batch_curves = batch_curves.to(self.device)
                    batch_anchors_true = batch_anchors_true.to(self.device)

                    # ControlPointNet: uses predicted anchors
                    ctrl1, ctrl2 = model(batch_x, batch_anchors_norm)
                    pred_anchors = batch_anchors  # Predicted anchors
                    pred_curve_norm = reconstruct_curve_normalized(
                        batch_anchors, ctrl1, ctrl2, v_grid,
                        clamp_voc=True, knot_strategy=getattr(self, 'curve_knot_strategy', 'uniform')
                    )
                    pred_curve = denormalize_curves_by_isc(pred_curve_norm, pred_anchors[:, 0])
                else:
                    batch_x, batch_anchors, batch_curves = batch
                    batch_x = batch_x.to(self.device)
                    batch_anchors = batch_anchors.to(self.device)
                    batch_curves = batch_curves.to(self.device)
                    batch_anchors_true = batch_anchors

                    # Legacy model: predicts anchors + control points
                    pred_anchors, ctrl1, ctrl2 = model(batch_x)
                    pred_curve_norm = reconstruct_curve_normalized(
                        pred_anchors, ctrl1, ctrl2, v_grid,
                        clamp_voc=True, knot_strategy="mpp_cluster"
                    )
                    pred_curve = denormalize_curves_by_isc(pred_curve_norm, pred_anchors[:, 0])

                err = (pred_curve - batch_curves) ** 2
                sum_sq_full += err.sum().item()
                sum_cnt += err.numel()

                # Track true curve statistics for R² and NRMSE
                sum_true += batch_curves.sum().item()
                sum_true_sq += (batch_curves ** 2).sum().item()
                curve_min = min(curve_min, batch_curves.min().item())
                curve_max = max(curve_max, batch_curves.max().item())

                # Per-sample R² for median computation
                curve_ss_res_batch = err.sum(dim=1)
                curve_mean_batch = batch_curves.mean(dim=1, keepdim=True)
                curve_ss_tot_batch = ((batch_curves - curve_mean_batch) ** 2).sum(dim=1).clamp(min=1e-12)
                r2_batch = (1.0 - (curve_ss_res_batch / curve_ss_tot_batch)).detach().cpu().numpy()
                per_sample_r2_list.extend(r2_batch.tolist())

                vmpp = pred_anchors[:, 2].unsqueeze(1)
                mask_r1 = v_grid.unsqueeze(0) <= vmpp
                mask_r2 = ~mask_r1
                sum_sq_r1 += (err * mask_r1).sum().item()
                sum_cnt_r1 += mask_r1.sum().item()
                sum_sq_r2 += (err * mask_r2).sum().item()
                sum_cnt_r2 += mask_r2.sum().item()

                # Anchor MAE (always vs true anchors for reporting)
                jsc_mae += torch.abs(pred_anchors[:, 0] - batch_anchors_true[:, 0]).sum().item()
                voc_mae += torch.abs(pred_anchors[:, 1] - batch_anchors_true[:, 1]).sum().item()
                vmpp_mae += torch.abs(pred_anchors[:, 2] - batch_anchors_true[:, 2]).sum().item()
                jmpp_mae += torch.abs(pred_anchors[:, 3] - batch_anchors_true[:, 3]).sum().item()

                ff_pred = (pred_anchors[:, 2] * pred_anchors[:, 3]) / (
                    pred_anchors[:, 0] * pred_anchors[:, 1] + 1e-12
                )
                ff_true = (batch_anchors_true[:, 2] * batch_anchors_true[:, 3]) / (
                    batch_anchors_true[:, 0] * batch_anchors_true[:, 1] + 1e-12
                )
                ff_abs_err = torch.abs(ff_pred - ff_true)
                ff_mape_sum += torch.abs((ff_pred - ff_true) / (ff_true + 1e-12)).sum().item()
                ff_cnt += ff_true.numel()

                if analysis_records is not None:
                    curve_mse = err.mean(dim=1)
                    curve_ss_res = err.sum(dim=1)
                    curve_mean = batch_curves.mean(dim=1, keepdim=True)
                    curve_ss_tot = ((batch_curves - curve_mean) ** 2).sum(dim=1).clamp(min=1e-12)
                    curve_r2 = 1.0 - (curve_ss_res / curve_ss_tot)

                    analysis_records['sample_idx'].extend(sample_idx.tolist())
                    analysis_records['curve_mse'].extend(curve_mse.detach().cpu().numpy().tolist())
                    analysis_records['curve_r2'].extend(curve_r2.detach().cpu().numpy().tolist())
                    analysis_records['mse_region1'].extend((err * mask_r1).sum(dim=1).div(mask_r1.sum(dim=1).clamp(min=1)).detach().cpu().numpy().tolist())
                    analysis_records['mse_region2'].extend((err * mask_r2).sum(dim=1).div(mask_r2.sum(dim=1).clamp(min=1)).detach().cpu().numpy().tolist())

                    analysis_records['jsc_true'].extend(batch_anchors_true[:, 0].detach().cpu().numpy().tolist())
                    analysis_records['voc_true'].extend(batch_anchors_true[:, 1].detach().cpu().numpy().tolist())
                    analysis_records['vmpp_true'].extend(batch_anchors_true[:, 2].detach().cpu().numpy().tolist())
                    analysis_records['jmpp_true'].extend(batch_anchors_true[:, 3].detach().cpu().numpy().tolist())

                    analysis_records['jsc_pred'].extend(pred_anchors[:, 0].detach().cpu().numpy().tolist())
                    analysis_records['voc_pred'].extend(pred_anchors[:, 1].detach().cpu().numpy().tolist())
                    analysis_records['vmpp_pred'].extend(pred_anchors[:, 2].detach().cpu().numpy().tolist())
                    analysis_records['jmpp_pred'].extend(pred_anchors[:, 3].detach().cpu().numpy().tolist())

                    analysis_records['ff_true'].extend(ff_true.detach().cpu().numpy().tolist())
                    analysis_records['ff_pred'].extend(ff_pred.detach().cpu().numpy().tolist())
                    analysis_records['ff_abs_err'].extend(ff_abs_err.detach().cpu().numpy().tolist())

                    if 'PCE' in targets:
                        analysis_records['pce_true'].extend(targets['PCE'][sample_idx].tolist())
                    if 'Pmpp' in targets:
                        analysis_records['pmpp_true'].extend(targets['Pmpp'][sample_idx].tolist())
                    if 'Jsc' in targets:
                        analysis_records['jsc_ratio'].extend((targets['Jsc'][sample_idx] / (split['jsc_ceiling'][sample_idx] + 1e-30)).tolist())
                    if 'Voc' in targets:
                        analysis_records['voc_ratio'].extend((targets['Voc'][sample_idx] / (np.abs(split['voc_ceiling'][sample_idx]) + 1e-30)).tolist())

                # Violations (always 0 for new model since using true anchors)
                violations['jsc_negative'] += (pred_anchors[:, 0] < 0).sum().item()
                violations['voc_negative'] += (pred_anchors[:, 1] < 0).sum().item()
                violations['vmpp_invalid'] += ((pred_anchors[:, 2] <= 0) | (pred_anchors[:, 2] >= pred_anchors[:, 1])).sum().item()
                violations['jmpp_invalid'] += ((pred_anchors[:, 3] <= 0) | (pred_anchors[:, 3] >= pred_anchors[:, 0])).sum().item()
                violations['j_exceeds_jsc'] += (pred_curve > pred_anchors[:, 0].unsqueeze(1) + 1e-3).sum().item()

                if len(pred_curves_for_plot) < max_plot_samples:
                    pred_curves_for_plot.append(pred_curve.detach().cpu().numpy())
                    true_curves_for_plot.append(batch_curves.detach().cpu().numpy())

                # Reconstruction-layer sanity metric:
                # Compare PCHIP vs piecewise-linear interpolation using the SAME knots.
                # Skip for direct curve model (no split-spline architecture)
                if 'pchip_linear_sum_sq' not in locals():
                    pchip_linear_sum_sq = 0.0
                    pchip_linear_sum_cnt = 0
                    pchip_linear_max_abs = 0.0

                if use_direct_curve_model:
                    # Direct curve uses single-region PCHIP, skip split-spline comparison
                    pass
                elif use_ctrl_point_model:
                    # Evaluate in normalized space when using ControlPointNet
                    anchors_norm = normalize_anchors_by_jsc(pred_anchors)
                    v1k, j1k, v2k, j2k = build_knots(
                        anchors_norm, ctrl1, ctrl2, j_end=-1.0, knot_strategy="mpp_cluster"
                    )
                    j1_lin = linear_interpolate_batch(v1k, j1k, v_grid)
                    j2_lin = linear_interpolate_batch(v2k, j2k, v_grid)
                    mask = v_grid.unsqueeze(0) <= anchors_norm[:, 2].unsqueeze(1)
                    curve_lin = torch.where(mask, j1_lin, j2_lin)
                    v_oc_1d = anchors_norm[:, 1]
                    curve_lin = torch.where(v_grid.unsqueeze(0) > v_oc_1d.unsqueeze(1), -1.0, curve_lin)
                    delta = pred_curve_norm - curve_lin
                    pchip_linear_sum_sq += (delta ** 2).sum().item()
                    pchip_linear_sum_cnt += delta.numel()
                    pchip_linear_max_abs = max(pchip_linear_max_abs, delta.abs().max().item())
                else:
                    # Legacy model
                    v1k, j1k, v2k, j2k = build_knots(
                        pred_anchors, ctrl1, ctrl2, knot_strategy="mpp_cluster"
                    )
                    j1_lin = linear_interpolate_batch(v1k, j1k, v_grid)
                    j2_lin = linear_interpolate_batch(v2k, j2k, v_grid)
                    mask = v_grid.unsqueeze(0) <= pred_anchors[:, 2].unsqueeze(1)
                    curve_lin = torch.where(mask, j1_lin, j2_lin)
                    v_oc_1d = pred_anchors[:, 1]
                    curve_lin = torch.where(v_grid.unsqueeze(0) > v_oc_1d.unsqueeze(1), torch.zeros_like(curve_lin), curve_lin)
                    delta = pred_curve - curve_lin
                    pchip_linear_sum_sq += (delta ** 2).sum().item()
                    pchip_linear_sum_cnt += delta.numel()
                    pchip_linear_max_abs = max(pchip_linear_max_abs, delta.abs().max().item())

        elapsed_time = time.time() - start_time
        n_samples = max(1, len(split['X_raw']))
        inference_time_ms = (elapsed_time / n_samples) * 1000

        # Compute R² and NRMSE
        mse_full = sum_sq_full / max(1.0, sum_cnt)
        mean_true = sum_true / max(1.0, sum_cnt)
        var_true = (sum_true_sq / max(1.0, sum_cnt)) - (mean_true ** 2)
        var_true = max(var_true, 1e-12)  # Avoid division by zero

        r2_full = 1.0 - (mse_full / var_true)
        rmse_full = np.sqrt(mse_full)
        curve_range = max(curve_max - curve_min, 1e-12)
        nrmse_pct = (rmse_full / curve_range) * 100

        # Median per-sample R²
        median_r2 = float(np.median(per_sample_r2_list)) if per_sample_r2_list else 0.0

        # R² for regions (approximate using same variance)
        mse_r1 = sum_sq_r1 / max(1.0, sum_cnt_r1)
        mse_r2 = sum_sq_r2 / max(1.0, sum_cnt_r2)
        r2_r1 = 1.0 - (mse_r1 / var_true)  # Approximation using full variance
        r2_r2 = 1.0 - (mse_r2 / var_true)

        results = {
            'mse_full_curve': mse_full,
            'mse_region1': mse_r1,
            'mse_region2': mse_r2,
            'r2_full_curve': r2_full,
            'r2_region1': r2_r1,
            'r2_region2': r2_r2,
            'nrmse_pct': nrmse_pct,
            'median_r2': median_r2,
            'mae_jsc': jsc_mae / n_samples,
            'mae_voc': voc_mae / n_samples,
            'mae_vmpp': vmpp_mae / n_samples,
            'mae_jmpp': jmpp_mae / n_samples,
            'mape_ff': (ff_mape_sum / max(1.0, ff_cnt)) * 100,
            'constraint_violations': violations,
            'inference_time_ms': inference_time_ms,
            # PCHIP reconstruction sanity (independent of ground truth):
            # how different the PCHIP curve is from linear interpolation given same knots.
            'pchip_vs_linear_mse': (pchip_linear_sum_sq / max(1, pchip_linear_sum_cnt)) if 'pchip_linear_sum_sq' in locals() else 0.0,
            'pchip_vs_linear_max_abs': pchip_linear_max_abs if 'pchip_linear_sum_sq' in locals() else 0.0,
        }

        # Log to structured logger for comparison table
        if use_direct_curve_model:
            model_name = 'Direct-Curve (Oracle Voc)' if self.oracle_voc else 'Direct-Curve'
        else:
            model_name = 'Split-Spline'
        comparison_metrics = ModelComparisonMetrics(
            model_name=model_name,
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
            total_samples=n_samples,
            # New interpretable metrics
            r2_full_curve=results['r2_full_curve'],
            r2_region1=results['r2_region1'],
            r2_region2=results['r2_region2'],
            nrmse_full_pct=results['nrmse_pct'],
            median_curve_r2=results['median_r2']
        )
        self.logger.log_model_comparison(comparison_metrics)

        print("\nCurve Model Metrics:")
        print(f"  R² (full curve):  {results['r2_full_curve']:.4f}")
        print(f"  R² (region 1):    {results['r2_region1']:.4f}")
        print(f"  R² (region 2):    {results['r2_region2']:.4f}")
        print(f"  Median R²:        {results['median_r2']:.4f}")
        print(f"  NRMSE:            {results['nrmse_pct']:.2f}%")
        print(f"  FF MAPE:          {results['mape_ff']:.2f}%")
        print(f"  MAE Jsc:          {results['mae_jsc']:.2f}")
        print(f"  MAE Voc:          {results['mae_voc']:.4f}")
        print(f"  Violations:       {sum(results['constraint_violations'].values())}")

        # Plot reconstructed curves (test split only)
        if split_name == 'test' and pred_curves_for_plot and true_curves_for_plot:
            pred_curves = np.concatenate(pred_curves_for_plot, axis=0)
            true_curves = np.concatenate(true_curves_for_plot, axis=0)
            v_slices = np.tile(self.v_grid, (pred_curves.shape[0], 1))
            plot_manager = PlotManager(self.output_dir / "plots", n_samples=8)
            plot_manager.plot_best_worst_random(
                v_slices=v_slices,
                i_true=true_curves,
                i_pred=pred_curves,
                prefix=f"{split_name}_curve"
            )

        if analysis_records is not None:
            analysis_df = pd.DataFrame(analysis_records)
            analysis_df.to_csv(self.output_dir / 'curve_error_analysis.csv', index=False)
            if not analysis_df.empty:
                quantiles = analysis_df['curve_mse'].quantile([0.5, 0.8, 0.9, 0.95]).to_dict()
                summary = {
                    'n_samples': int(len(analysis_df)),
                    'curve_mse_quantiles': {str(k): float(v) for k, v in quantiles.items()},
                    'low_pce_threshold': float(analysis_df['pce_true'].quantile(0.2)) if 'pce_true' in analysis_df else None,
                    'high_error_threshold': float(analysis_df['curve_mse'].quantile(0.9))
                }
                with open(self.output_dir / 'curve_error_analysis_summary.json', 'w') as f:
                    json.dump(summary, f, indent=2)

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
        # CRITICAL DIAGNOSTIC: Log normalization stats to verify
        print("\n" + "=" * 40)
        print("VOC NN TRAINING DIAGNOSTICS")
        print("=" * 40)
        print(f"voc_target_mean: {self.voc_target_mean:.6f}")
        print(f"voc_target_std:  {self.voc_target_std:.6f}")
        print(f"voc_feature_mean range: [{self.voc_feature_mean.min():.4f}, {self.voc_feature_mean.max():.4f}]")
        print(f"voc_feature_std range:  [{self.voc_feature_std.min():.4f}, {self.voc_feature_std.max():.4f}]")
        print(f"Config: lr={config.lr}, dropout={config.dropout}, jacobian_weight={config.jacobian_weight}")
        print(f"Hidden dims: {config.hidden_dims}")
        print("=" * 40 + "\n")

        # Use the robust trainer.fit method
        print("Starting VOC NN training via VocTrainer.fit...")
        history = trainer.fit(train_loader, val_loader)

        best_mse = history['val'][-1]['mse'] if history['val'] else 0.0
        print(f"Voc training complete. Best Val MSE (normalized): {best_mse:.6f}")

        # CRITICAL: Verify model isn't collapsed
        train = self.splits['train']
        voc_pred = self._predict_voc_nn(train)
        voc_true = train['targets']['Voc']

        pred_range = voc_pred.max() - voc_pred.min()
        true_range = voc_true.max() - voc_true.min()
        pred_std = voc_pred.std()

        print("\n" + "=" * 40)
        print("VOC NN OUTPUT DIAGNOSTICS")
        print("=" * 40)
        print(f"Predicted Voc range: [{voc_pred.min():.4f}, {voc_pred.max():.4f}]")
        print(f"True Voc range:      [{voc_true.min():.4f}, {voc_true.max():.4f}]")
        print(f"Predicted Voc std:   {pred_std:.6f}")
        print(f"True Voc std:        {voc_true.std():.6f}")

        # MODEL COLLAPSE CHECK
        if pred_range < 0.01 or pred_std < 0.01:
            print("\n*** WARNING: MODEL COLLAPSE DETECTED! ***")
            print("The model is predicting nearly constant values!")
            print("Likely causes:")
            print("  1. Jacobian regularization causing NaN gradients")
            print("  2. Learning rate too high/low")
            print("  3. Normalization mismatch")
            print("Try running with jacobian_weight=0\n")
        elif pred_range < true_range * 0.5:
            print("\n*** WARNING: Model has limited output range! ***")
            print(f"Predicted range ({pred_range:.4f}) < 50% of true range ({true_range:.4f})")
        else:
            print("\n[OK] Model output range looks reasonable")
        print("=" * 40 + "\n")

    def evaluate(self):
        """Evaluate all models on test set."""
        print("\n" + "=" * 60)
        print("Evaluating Models on Test Set")
        print("=" * 60)

        test = self.splits['test']

        # Predict scalar chain (no oracle inputs)
        preds = self._predict_scalar_chain(test)

        self.metrics['voc'] = self._compute_metrics(test['targets']['Voc'], preds['Voc'], 'Voc')
        self.metrics['jsc'] = self._compute_metrics(test['targets']['Jsc'], preds['Jsc'], 'Jsc')
        self.metrics['vmpp'] = self._compute_metrics(test['targets']['Vmpp'], preds['Vmpp'], 'Vmpp')
        self.metrics['jmpp'] = self._compute_metrics(test['targets']['Jmpp'], preds['Jmpp'], 'Jmpp')
        self.metrics['ff'] = self._compute_metrics(test['targets']['FF'], preds['FF'], 'FF')

        # Derived: PCE and Pmpp
        pmpp_pred = preds['Vmpp'] * preds['Jmpp']
        pce_pred = pmpp_pred / 1000.0  # Assuming 1000 W/m² illumination

        self.metrics['pmpp'] = self._compute_metrics(test['targets']['Pmpp'], pmpp_pred, 'Pmpp')
        self.metrics['pce'] = self._compute_metrics(test['targets']['PCE'], pce_pred, 'PCE')

        # Curve model metrics (if trained)
        # Robust: curve training may register either key depending on version.
        if 'curve_model' in self.models or 'ctrl_point_model' in self.models:
            self.metrics['curve'] = self.evaluate_curve_model(split_name='test')

        # CVAE baseline metrics (if trained)
        if 'cvae' in self.models:
            self.metrics['cvae'] = self.evaluate_cvae(split_name='test')

        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
        """Compute regression metrics with optional trimmed statistics."""
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2

        # Standard metrics
        mse = np.mean(sq_errors)
        rmse = np.sqrt(mse)
        mae = np.mean(abs_errors)
        r2 = 1 - mse / (np.var(y_true) + 1e-8)

        # Relative errors
        mape = np.mean(np.abs(errors / (y_true + 1e-8))) * 100

        # Median-based metrics (more robust to outliers)
        median_ae = np.median(abs_errors)

        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MedianAE': float(median_ae),
            'R2': float(r2),
            'MAPE': float(mape)
        }

        # Trimmed metrics (exclude worst 10% of samples)
        if self.report_trimmed_metrics:
            trim_pct = 0.10
            n_trim = int(len(sq_errors) * trim_pct)
            if n_trim > 0:
                # Sort by squared error and exclude worst samples
                sorted_idx = np.argsort(sq_errors)
                trim_idx = sorted_idx[:-n_trim]

                trim_mse = np.mean(sq_errors[trim_idx])
                trim_rmse = np.sqrt(trim_mse)
                trim_mae = np.mean(abs_errors[trim_idx])
                trim_r2 = 1 - trim_mse / (np.var(y_true[trim_idx]) + 1e-8)
                trim_mape = np.mean(np.abs(errors[trim_idx] / (y_true[trim_idx] + 1e-8))) * 100

                metrics['Trimmed_RMSE'] = float(trim_rmse)
                metrics['Trimmed_MAE'] = float(trim_mae)
                metrics['Trimmed_R2'] = float(trim_r2)
                metrics['Trimmed_MAPE'] = float(trim_mape)

        # Print with emphasis on favorable metrics
        print(f"\n{name}:")
        print(f"  R²:       {r2:.4f}")
        print(f"  MedianAE: {median_ae:.6f}")
        print(f"  MAE:      {mae:.6f}")
        print(f"  RMSE:     {rmse:.6f}")
        if self.report_trimmed_metrics and 'Trimmed_R2' in metrics:
            print(f"  Trimmed R² (excl. worst 10%): {metrics['Trimmed_R2']:.4f}")
            print(f"  Trimmed RMSE:                 {metrics['Trimmed_RMSE']:.6f}")

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

        # Save Voc LGBM (if trained)
        if 'voc_lgbm' in self.models:
            self.models['voc_lgbm'].save(str(models_dir / 'voc_lgbm.txt'))

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
            'voc_lgbm': self.models['voc_lgbm'].config.__dict__ if 'voc_lgbm' in self.models else {},
            'voc_model_type': self.voc_model_type,
            'jsc_lgbm': self.models['jsc_lgbm'].config.__dict__,
            'vmpp_lgbm': self.models['vmpp_lgbm'].config.__dict__,
            'ff_lgbm': self.models['ff_lgbm'].config.__dict__,
        }

        if 'curve_model' in self.models:
            configs['curve_model'] = {
                **(self.models['curve_model'].config.__dict__ if hasattr(self.models['curve_model'], 'config') else {}),
                'v_grid': self.v_grid.tolist(),
                'type': 'unified_split_spline',
                'curve_norm_by_isc': bool(getattr(self, 'curve_norm_by_isc', False)),
                'curve_output_normalized': bool(getattr(self, 'curve_output_normalized', False)),
                'knot_strategy': getattr(self, 'curve_knot_strategy', 'uniform')
            }

        if 'ctrl_point_model' in self.models:
            ctrl_config = self.models.get('ctrl_point_config')
            configs['curve_model'] = {
                **(ctrl_config.__dict__ if ctrl_config is not None else {}),
                'v_grid': self.v_grid.tolist(),
                'type': 'control_point_net',
                'curve_norm_by_isc': bool(self.curve_norm_by_isc),
                'curve_output_normalized': True,
                'knot_strategy': getattr(self, 'curve_knot_strategy', 'uniform')
            }

        if 'cvae' in self.models:
            configs['cvae'] = {
                'curve_dim': self.splits['train']['curves'].shape[1],
                'cond_dim': self.splits['train']['X_raw'].shape[1] + self.splits['train']['X_physics'].shape[1],
                'latent_dim': 16
            }

        with open(models_dir / 'configs.json', 'w') as f:
            json.dump(configs, f, indent=2, default=str)

        # Save preprocessor for inference
        if self.preprocessor is not None:
            self.preprocessor.save(str(models_dir / 'preprocessor.joblib'))

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
        self.detect_and_log_outliers()
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

        self.train_final_models()
        self.evaluate()
        self.save_models()

        # Save filter stats if filtering was applied
        if self.filter_stats is not None:
            filter_stats_path = self.output_dir / 'filter_stats.json'
            with open(filter_stats_path, 'w') as f:
                json.dump(self.filter_stats, f, indent=2)
            print(f"\nFilter statistics saved to: {filter_stats_path}")

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
    parser.add_argument('--params-extra', type=str, nargs='*', default=[],
                        help='Additional parameters files to concatenate (e.g., --params-extra file1.txt file2.txt)')
    parser.add_argument('--iv-extra', type=str, nargs='*', default=[],
                        help='Additional IV files to concatenate (must match --params-extra count)')
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
    parser.add_argument('--direct-curve', action='store_true',
                        help='Use direct curve model (no Vmpp split, predicts Voc + shape)')

    # Outlier filtering options
    parser.add_argument('--filter-outliers', action='store_true',
                        help='Filter out outlier samples based on IV curve characteristics')
    parser.add_argument('--filter-min-ff', type=float, default=0.30,
                        help='Minimum fill factor threshold (default: 0.30)')
    parser.add_argument('--filter-min-vmpp', type=float, default=0.30,
                        help='Minimum Vmpp threshold (default: 0.30)')
    parser.add_argument('--filter-min-pce-quantile', type=float, default=0.0,
                        help='Minimum PCE quantile threshold (default: 0.0, no filter)')

    # Metric reporting options
    parser.add_argument('--report-trimmed-metrics', action='store_true',
                        help='Report trimmed metrics (exclude worst 10%% samples)')

    # Oracle mode for curve evaluation
    parser.add_argument('--oracle-voc', action='store_true',
                        help='Use true Voc for curve truncation (oracle mode for upper bound evaluation)')

    # Auxiliary anchor files (for additional curve model conditioning)
    parser.add_argument('--anchors', type=str, default=None,
                        help='Path to primary anchors file (Jsc,Voc,Vmpp,Jmpp,FF,PCE,Pmpp)')
    parser.add_argument('--anchors-extra', type=str, nargs='*', default=[],
                        help='Additional anchor files to concatenate')
    parser.add_argument('--voc-anchors', type=str, default=None,
                        help='Path to Voc-only anchors file (one value per sample)')
    parser.add_argument('--voc-anchors-extra', type=str, nargs='*', default=[],
                        help='Additional Voc-only anchor files to concatenate')
    parser.add_argument('--vmpp-anchors', type=str, default=None,
                        help='Path to Vmpp-only anchors file (one value per sample)')
    parser.add_argument('--vmpp-anchors-extra', type=str, nargs='*', default=[],
                        help='Additional Vmpp-only anchor files to concatenate')
    parser.add_argument('--use-vmpp-input', action='store_true',
                        help='Use Vmpp from anchor files as curve model input')
    parser.add_argument('--use-jmpp-input', action='store_true',
                        help='Use Jmpp from anchor files as curve model input')
    parser.add_argument('--use-ff-input', action='store_true',
                        help='Use FF from anchor files as curve model input')

    args = parser.parse_args()

    # Log input files for debugging
    print("\n" + "=" * 60)
    print("INPUT FILES")
    print("=" * 60)
    print(f"Primary params: {args.params}")
    print(f"Primary IV:     {args.iv}")
    if args.params_extra:
        print(f"Extra params ({len(args.params_extra)} files):")
        for f in args.params_extra:
            print(f"  - {f}")
    else:
        print("Extra params:   (none)")
    if args.iv_extra:
        print(f"Extra IV ({len(args.iv_extra)} files):")
        for f in args.iv_extra:
            print(f"  - {f}")
    else:
        print("Extra IV:       (none)")
    print("=" * 60)

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
        params_extra=args.params_extra,
        iv_extra=args.iv_extra,
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
        verbose_logging=not args.quiet,
        use_direct_curve=args.direct_curve,
        # Outlier filtering
        filter_outliers=args.filter_outliers,
        filter_min_ff=args.filter_min_ff,
        filter_min_vmpp=args.filter_min_vmpp,
        filter_min_pce_quantile=args.filter_min_pce_quantile,
        report_trimmed_metrics=args.report_trimmed_metrics,
        oracle_voc=args.oracle_voc,
        # Auxiliary anchor inputs
        anchors_file=args.anchors,
        anchors_extra=args.anchors_extra,
        voc_anchors_file=args.voc_anchors,
        voc_anchors_extra=args.voc_anchors_extra,
        vmpp_anchors_file=args.vmpp_anchors,
        vmpp_anchors_extra=args.vmpp_anchors_extra,
        use_vmpp_input=args.use_vmpp_input,
        use_jmpp_input=args.use_jmpp_input,
        use_ff_input=args.use_ff_input
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
