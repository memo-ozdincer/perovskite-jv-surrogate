"""
Fast inference module for trained scalar predictors.
Supports batch prediction with GPU acceleration.
"""
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import lightgbm as lgb

from config import COLNAMES, V_GRID
from features import compute_all_physics_features, compute_jsc_ceiling, compute_voc_ceiling
from models.voc_nn import VocNN, VocNNConfig, SplitSplineNetConfig, UnifiedSplitSplineNet
from models.jsc_lgbm import JscLGBMConfig
from models.vmpp_lgbm import VmppLGBMConfig
from models.reconstruction import reconstruct_curve


@dataclass
class PredictionResult:
    """Container for all predicted PV parameters."""
    Voc: np.ndarray
    Jsc: np.ndarray
    Vmpp: np.ndarray
    Jmpp: np.ndarray
    FF: np.ndarray
    Pmpp: np.ndarray
    PCE: np.ndarray

    def to_dict(self) -> dict:
        return {
            'Voc': self.Voc,
            'Jsc': self.Jsc,
            'Vmpp': self.Vmpp,
            'Jmpp': self.Jmpp,
            'FF': self.FF,
            'Pmpp': self.Pmpp,
            'PCE': self.PCE
        }

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.to_dict())


class ScalarPredictor:
    """
    Fast inference for all scalar PV predictors.

    Loads trained models and provides batch prediction interface.
    """

    def __init__(self, models_dir: str, device: str = 'cuda'):
        self.models_dir = Path(models_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.configs = {}
        self.physics_feature_mask = None
        self.curve_feature_mean = None
        self.curve_feature_std = None
        self.v_grid = V_GRID.astype(np.float32)

        self._load_models()

    def _load_models(self):
        """Load all trained models."""
        print(f"Loading models from {self.models_dir}")

        # Load configs
        with open(self.models_dir / 'configs.json', 'r') as f:
            self.configs = json.load(f)

        # Load Voc NN
        voc_config_dict = self.configs.get('voc_nn', {})
        # Reconstruct config
        voc_config = VocNNConfig(
            input_dim=voc_config_dict.get('input_dim', 102),
            hidden_dims=voc_config_dict.get('hidden_dims', [512, 512, 256, 256, 128, 64]),
            dropout=voc_config_dict.get('dropout', 0.1),
            use_layer_norm=voc_config_dict.get('use_layer_norm', True),
            use_residual=voc_config_dict.get('use_residual', True),
            activation=voc_config_dict.get('activation', 'gelu'),
        )

        self.models['voc_nn'] = VocNN(voc_config).to(self.device)
        self.models['voc_nn'].load_state_dict(
            torch.load(self.models_dir / 'voc_nn.pt', map_location=self.device)
        )
        self.models['voc_nn'].eval()

        # Load LGBM models
        self.models['jsc_lgbm'] = lgb.Booster(model_file=str(self.models_dir / 'jsc_lgbm.txt'))
        self.models['vmpp_lgbm'] = lgb.Booster(model_file=str(self.models_dir / 'vmpp_lgbm.txt'))
        self.models['jmpp_lgbm'] = lgb.Booster(model_file=str(self.models_dir / 'jmpp_lgbm.txt'))
        self.models['ff_lgbm'] = lgb.Booster(model_file=str(self.models_dir / 'ff_lgbm.txt'))

        # Load normalization parameters for Voc NN
        normalization_file = self.models_dir / 'normalization.json'
        if normalization_file.exists():
            with open(normalization_file, 'r') as f:
                norm_params = json.load(f)
                self.voc_feature_mean = np.array(norm_params['voc_feature_mean'])
                self.voc_feature_std = np.array(norm_params['voc_feature_std'])
                self.voc_target_mean = float(norm_params.get('voc_target_mean', 0.0))
                self.voc_target_std = float(norm_params.get('voc_target_std', 1.0))
                self.curve_feature_mean = np.array(norm_params.get('curve_feature_mean')) if norm_params.get('curve_feature_mean') is not None else None
                self.curve_feature_std = np.array(norm_params.get('curve_feature_std')) if norm_params.get('curve_feature_std') is not None else None
                if 'physics_feature_mask' in norm_params:
                    mask_indices = norm_params['physics_feature_mask']
                    self.physics_feature_mask = np.array(mask_indices, dtype=int)
            print("Loaded normalization parameters for Voc NN")
        else:
            print("WARNING: No normalization parameters found. Predictions may be incorrect.")
            self.voc_feature_mean = None
            self.voc_feature_std = None
            self.voc_target_mean = 0.0
            self.voc_target_std = 1.0

        # Load curve model (optional)
        curve_model_path = self.models_dir / 'curve_model.pt'
        if curve_model_path.exists():
            curve_config_dict = self.configs.get('curve_model', {})
            self.v_grid = np.array(curve_config_dict.get('v_grid', V_GRID), dtype=np.float32)

            curve_config = SplitSplineNetConfig(
                input_dim=curve_config_dict.get('input_dim', 102),
                hidden_dims=curve_config_dict.get('hidden_dims', [512, 256, 128]),
                dropout=curve_config_dict.get('dropout', 0.15),
                activation=curve_config_dict.get('activation', 'silu'),
                ctrl_points=curve_config_dict.get('ctrl_points', 6),
            )
            self.models['curve_model'] = UnifiedSplitSplineNet(curve_config).to(self.device)
            self.models['curve_model'].load_state_dict(
                torch.load(curve_model_path, map_location=self.device)
            )
            self.models['curve_model'].eval()

        print("All models loaded successfully")

    def predict(self, params: np.ndarray) -> PredictionResult:
        """
        Predict all scalar PV parameters from input parameters.

        Args:
            params: (N, 31) array of input parameters

        Returns:
            PredictionResult with all predictions
        """
        params = np.atleast_2d(params).astype(np.float32)
        N = params.shape[0]

        # Convert to tensor for feature computation
        params_tensor = torch.from_numpy(params).to(self.device)

        # Compute physics features
        physics_features = compute_all_physics_features(params_tensor)
        physics_features_np = physics_features.cpu().numpy()

        if self.physics_feature_mask is not None:
            physics_features_np = physics_features_np[:, self.physics_feature_mask]

        # Compute ceilings
        jsc_ceiling = compute_jsc_ceiling(params_tensor).cpu().numpy()
        voc_ceiling = compute_voc_ceiling(params_tensor).cpu().numpy()

        # 1. Predict Voc
        X_full = np.hstack([params, physics_features_np])

        # Apply normalization (critical for NN predictions)
        if self.voc_feature_mean is not None and self.voc_feature_std is not None:
            X_full = (X_full - self.voc_feature_mean) / self.voc_feature_std

        X_tensor = torch.from_numpy(X_full).float().to(self.device)
        voc_ceiling_tensor = torch.from_numpy(voc_ceiling).float().to(self.device)

        with torch.no_grad():
            Voc = self.models['voc_nn'](X_tensor, voc_ceiling_tensor).cpu().numpy()

        # Denormalize Voc predictions if needed
        Voc = Voc * self.voc_target_std + self.voc_target_mean

        # 2. Predict Jsc
        X_jsc = np.hstack([
            params, physics_features_np,
            np.log10(jsc_ceiling + 1e-30).reshape(-1, 1)
        ])
        jsc_efficiency = self.models['jsc_lgbm'].predict(X_jsc)
        Jsc = jsc_efficiency * jsc_ceiling

        # 3. Predict Vmpp
        X_vmpp = np.hstack([
            params, physics_features_np,
            Voc.reshape(-1, 1),
            np.log10(Voc + 1e-30).reshape(-1, 1)
        ])
        vmpp_ratio = self.models['vmpp_lgbm'].predict(X_vmpp)
        Vmpp = vmpp_ratio * Voc

        # 4. Predict Jmpp
        X_jmpp = np.hstack([
            params, physics_features_np,
            Jsc.reshape(-1, 1),
            np.log10(np.abs(Jsc) + 1e-30).reshape(-1, 1),
            Vmpp.reshape(-1, 1),
            np.log10(Vmpp + 1e-30).reshape(-1, 1)
        ])
        jmpp_ratio = self.models['jmpp_lgbm'].predict(X_jmpp)
        Jmpp = jmpp_ratio * Jsc

        # 5. Predict FF
        X_ff = np.hstack([
            params, physics_features_np,
            Voc.reshape(-1, 1),
            np.log10(Voc + 1e-30).reshape(-1, 1),
            Jsc.reshape(-1, 1),
            np.log10(np.abs(Jsc) + 1e-30).reshape(-1, 1),
            (Voc * Jsc).reshape(-1, 1)
        ])
        FF = np.clip(self.models['ff_lgbm'].predict(X_ff), 0, 1)

        # 6. Derived quantities
        Pmpp = Vmpp * Jmpp
        PCE = Pmpp / 1000.0  # Assuming 1000 W/m² illumination

        return PredictionResult(
            Voc=Voc,
            Jsc=Jsc,
            Vmpp=Vmpp,
            Jmpp=Jmpp,
            FF=FF,
            Pmpp=Pmpp,
            PCE=PCE
        )

    def _prepare_curve_inputs(self, params: np.ndarray) -> torch.Tensor:
        """Prepare normalized inputs for curve model."""
        params = np.atleast_2d(params).astype(np.float32)
        params_tensor = torch.from_numpy(params).to(self.device)

        physics_features = compute_all_physics_features(params_tensor)
        physics_features_np = physics_features.cpu().numpy()

        if self.physics_feature_mask is not None:
            physics_features_np = physics_features_np[:, self.physics_feature_mask]

        X_full = np.hstack([params, physics_features_np]).astype(np.float32)

        if self.curve_feature_mean is not None and self.curve_feature_std is not None:
            X_full = (X_full - self.curve_feature_mean) / self.curve_feature_std

        return torch.from_numpy(X_full).float().to(self.device)

    def predict_full_curve(
        self,
        params: np.ndarray,
        return_uncertainty: bool = False,
        n_samples: int = 50
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Predict full J-V curve from material parameters.

        Returns:
            v_grid: (45,) voltage points
            j_curve: (N, 45) current predictions
            j_std: (N, 45) uncertainty (optional)
        """
        if 'curve_model' not in self.models:
            raise ValueError("Curve model not loaded. Train with --train-curves first.")

        x = self._prepare_curve_inputs(params)
        v_grid_tensor = torch.from_numpy(self.v_grid).float().to(self.device)

        model = self.models['curve_model']

        if return_uncertainty:
            was_training = model.training
            model.train()
            curves = []
            for _ in range(n_samples):
                anchors, ctrl1, ctrl2 = model(x)
                curve = reconstruct_curve(anchors, ctrl1, ctrl2, v_grid_tensor, clamp_voc=True)
                curves.append(curve)
            stacked = torch.stack(curves)
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0)
            if not was_training:
                model.eval()
            return self.v_grid, mean.cpu().numpy(), std.cpu().numpy()

        with torch.no_grad():
            anchors, ctrl1, ctrl2 = model(x)
            curve = reconstruct_curve(anchors, ctrl1, ctrl2, v_grid_tensor, clamp_voc=True)
        return self.v_grid, curve.cpu().numpy(), None

    def predict_from_dataframe(self, df) -> PredictionResult:
        """Predict from pandas DataFrame with named columns."""
        # Ensure correct column order
        params = df[COLNAMES].values.astype(np.float32)
        return self.predict(params)

    def predict_single(self, **kwargs) -> dict:
        """
        Predict for a single sample with named parameters.

        Example:
            predictor.predict_single(lH=100, lP=500, lE=100, ...)
        """
        params = np.array([[kwargs[col] for col in COLNAMES]], dtype=np.float32)
        result = self.predict(params)
        return {k: float(v[0]) for k, v in result.to_dict().items()}


class BatchPredictor:
    """
    Optimized batch predictor for very large datasets.

    Uses chunked processing to manage memory.
    """

    def __init__(self, models_dir: str, device: str = 'cuda', chunk_size: int = 50000):
        self.predictor = ScalarPredictor(models_dir, device)
        self.chunk_size = chunk_size

    def predict_large(self, params: np.ndarray) -> PredictionResult:
        """
        Predict on large datasets with chunked processing.

        Args:
            params: (N, 31) array, can be very large

        Returns:
            PredictionResult with concatenated results
        """
        N = params.shape[0]
        n_chunks = (N + self.chunk_size - 1) // self.chunk_size

        results = {
            'Voc': [], 'Jsc': [], 'Vmpp': [], 'Jmpp': [],
            'FF': [], 'Pmpp': [], 'PCE': []
        }

        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, N)

            chunk_result = self.predictor.predict(params[start:end])

            for key in results:
                results[key].append(getattr(chunk_result, key))

        return PredictionResult(
            Voc=np.concatenate(results['Voc']),
            Jsc=np.concatenate(results['Jsc']),
            Vmpp=np.concatenate(results['Vmpp']),
            Jmpp=np.concatenate(results['Jmpp']),
            FF=np.concatenate(results['FF']),
            Pmpp=np.concatenate(results['Pmpp']),
            PCE=np.concatenate(results['PCE'])
        )


def load_predictor(models_dir: str, device: str = 'cuda') -> ScalarPredictor:
    """Convenience function to load predictor."""
    return ScalarPredictor(models_dir, device)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Run inference with trained models')
    parser.add_argument('--models', type=str, required=True,
                        help='Path to models directory')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input parameters CSV')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output predictions CSV')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--predict-curve', action='store_true',
                        help='Predict full J-V curves instead of scalar outputs')
    parser.add_argument('--curve-uncertainty', action='store_true',
                        help='Estimate uncertainty via MC dropout for curves')
    parser.add_argument('--mc-samples', type=int, default=50,
                        help='Number of MC samples for curve uncertainty')

    args = parser.parse_args()

    # Load predictor
    predictor = ScalarPredictor(args.models, args.device)

    # Load input data
    input_df = pd.read_csv(args.input)
    if 'Unnamed: 0' in input_df.columns:
        input_df = input_df.drop('Unnamed: 0', axis=1)

    # Check if columns are named or need to be assigned
    if set(COLNAMES).issubset(set(input_df.columns)):
        params = input_df[COLNAMES].values
    else:
        params = input_df.values

    # Predict
    print(f"Running inference on {len(params)} samples...")

    if args.predict_curve:
        v_grid, j_curve, j_std = predictor.predict_full_curve(
            params.astype(np.float32),
            return_uncertainty=args.curve_uncertainty,
            n_samples=args.mc_samples
        )

        import pandas as pd
        curve_cols = [f"J_{i}" for i in range(j_curve.shape[1])]
        output_df = pd.DataFrame(j_curve, columns=curve_cols)

        if j_std is not None:
            std_cols = [f"Jstd_{i}" for i in range(j_std.shape[1])]
            output_df = pd.concat([output_df, pd.DataFrame(j_std, columns=std_cols)], axis=1)

        output_df.to_csv(args.output, index=False)

        vgrid_path = Path(args.output).with_name(Path(args.output).stem + '_vgrid.csv')
        vgrid_df = pd.DataFrame([v_grid], columns=[f"V_{i}" for i in range(len(v_grid))])
        vgrid_df.to_csv(vgrid_path, index=False)

        print(f"Curve predictions saved to {args.output}")
        print(f"Voltage grid saved to {vgrid_path}")
    else:
        result = predictor.predict(params.astype(np.float32))
        output_df = result.to_dataframe()
        output_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    main()
