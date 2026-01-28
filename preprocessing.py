"""
Data preprocessing utilities matching the KNOWN WORKING pipeline.

Key features:
1. Normalize curves by Isc to [-1, 1] range
2. RobustScaler + MinMaxScaler for parameters with log1p for material properties
3. Proper denormalization for inference
"""
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import COLNAMES


# ============================================================================
# CURVE NORMALIZATION (BY ISC)
# ============================================================================

def normalize_curves_by_isc(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize curves by their Isc value (first point) to [-1, 1] range.

    Formula: normalized = 2.0 * (curve / isc) - 1.0

    Args:
        curves: (N, 45) array of current density curves

    Returns:
        isc_values: (N,) array of Isc values for denormalization
        normalized_curves: (N, 45) array normalized to [-1, 1]
    """
    isc_values = curves[:, 0].copy()

    # Handle zero or near-zero Isc (faulty curves)
    isc_safe = np.where(np.abs(isc_values) < 1e-9, 1.0, isc_values)

    # Normalize: 2.0 * (curve / isc) - 1.0 -> maps [0, isc] to [-1, 1]
    normalized = 2.0 * (curves / isc_safe[:, np.newaxis]) - 1.0

    # Replace faulty curves (isc ~= 0) with zeros
    faulty_mask = np.abs(isc_values) < 1e-9
    normalized[faulty_mask] = 0.0

    return isc_values.astype(np.float32), normalized.astype(np.float32)


def denormalize_curves_by_isc(normalized_curves: np.ndarray | torch.Tensor,
                               isc_values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Denormalize curves back to original scale.

    Formula: curve = (normalized + 1.0) / 2.0 * isc

    Args:
        normalized_curves: (N, 45) normalized curves in [-1, 1]
        isc_values: (N,) or (N, 1) Isc values

    Returns:
        curves: (N, 45) denormalized curves
    """
    is_tensor = isinstance(normalized_curves, torch.Tensor)

    if is_tensor:
        if isc_values.dim() == 1:
            isc_values = isc_values.unsqueeze(1)
    else:
        if isc_values.ndim == 1:
            isc_values = isc_values[:, np.newaxis]

    return (normalized_curves + 1.0) / 2.0 * isc_values


# ============================================================================
# PARAMETER TRANSFORMATIONS (ROBUST + MINMAX + LOG1P)
# ============================================================================

def get_param_transformer(colnames: list[str] = None) -> ColumnTransformer:
    """
    Create parameter transformer matching the KNOWN WORKING pipeline.

    Uses:
    - RobustScaler (removes outliers)
    - MinMaxScaler to [-1, 1]
    - Log1p transformation for material properties

    Args:
        colnames: List of column names (default: from config.COLNAMES)

    Returns:
        ColumnTransformer ready for fitting
    """
    if colnames is None:
        colnames = list(COLNAMES)

    # Define parameter groups (matching reference implementation)
    param_defs = {
        'layer_thickness': ['lH', 'lP', 'lE'],
        'material_properties': [
            'muHh', 'muPh', 'muPe', 'muEe',  # Mobilities
            'NvH', 'NcH', 'NvE', 'NcE', 'NvP', 'NcP',  # DOS
            'chiHh', 'chiHe', 'chiPh', 'chiPe', 'chiEh', 'chiEe',  # Energy levels
            'epsH', 'epsP', 'epsE'  # Permittivities
        ],
        'contacts': ['Wlm', 'Whm'],
        'recombination_gen': ['Gavg', 'Aug', 'Brad', 'Taue', 'Tauh', 'vII', 'vIII']
    }

    transformers = []

    for group, cols in param_defs.items():
        # Get columns that actually exist in the data
        actual_cols = [c for c in cols if c in colnames]
        if not actual_cols:
            continue

        # Build pipeline for this group
        steps = [
            ('robust', RobustScaler()),
            ('minmax', MinMaxScaler(feature_range=(-1, 1)))
        ]

        # Add log1p for material properties (they span many orders of magnitude)
        if group == 'material_properties':
            steps.insert(0, ('log1p', FunctionTransformer(func=np.log1p, inverse_func=np.expm1)))

        transformers.append((group, Pipeline(steps), actual_cols))

    return ColumnTransformer(transformers, remainder='passthrough')


# ============================================================================
# COMBINED PREPROCESSING CLASS
# ============================================================================

class PVDataPreprocessor:
    """
    Complete preprocessing pipeline matching the KNOWN WORKING implementation.

    Features:
    1. Parameter transformation (RobustScaler + MinMaxScaler + log1p)
    2. Curve normalization by Isc
    3. Stores normalization parameters for inference
    """

    def __init__(self, colnames: list[str] = None):
        self.colnames = colnames or list(COLNAMES)
        self.param_transformer = get_param_transformer(self.colnames)

        # Stored for denormalization
        self.isc_train = None
        self.fitted = False

    def fit_transform_params(self, params: np.ndarray) -> np.ndarray:
        """
        Fit parameter transformer on training data and transform.

        Args:
            params: (N, 31) raw parameters

        Returns:
            transformed: (N, 31) transformed parameters in [-1, 1]
        """
        self.fitted = True
        return self.param_transformer.fit_transform(params)

    def transform_params(self, params: np.ndarray) -> np.ndarray:
        """
        Transform parameters using fitted transformer.

        Args:
            params: (N, 31) raw parameters

        Returns:
            transformed: (N, 31) transformed parameters
        """
        if not self.fitted:
            raise ValueError("Must call fit_transform_params first")
        return self.param_transformer.transform(params)

    def fit_transform_curves(self, curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize curves by Isc for training.

        Args:
            curves: (N, 45) raw curves

        Returns:
            isc_values: (N,) Isc values
            normalized_curves: (N, 45) normalized to [-1, 1]
        """
        isc_values, normalized = normalize_curves_by_isc(curves)
        self.isc_train = isc_values  # Store for reference
        return isc_values, normalized

    def transform_curves(self, curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize curves by Isc (same as fit_transform_curves).

        Args:
            curves: (N, 45) raw curves

        Returns:
            isc_values: (N,) Isc values
            normalized_curves: (N, 45) normalized to [-1, 1]
        """
        return normalize_curves_by_isc(curves)

    def denormalize_curves(self, normalized_curves: np.ndarray | torch.Tensor,
                          isc_values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Denormalize curves back to original scale.

        Args:
            normalized_curves: (N, 45) normalized curves
            isc_values: (N,) Isc values

        Returns:
            curves: (N, 45) denormalized curves
        """
        return denormalize_curves_by_isc(normalized_curves, isc_values)

    def save(self, filepath: str):
        """Save preprocessor state for inference."""
        import joblib
        state = {
            'param_transformer': self.param_transformer,
            'isc_train_stats': {
                'mean': float(np.mean(self.isc_train)) if self.isc_train is not None else None,
                'std': float(np.std(self.isc_train)) if self.isc_train is not None else None,
                'min': float(np.min(self.isc_train)) if self.isc_train is not None else None,
                'max': float(np.max(self.isc_train)) if self.isc_train is not None else None,
            },
            'colnames': self.colnames,
            'fitted': self.fitted
        }
        joblib.dump(state, filepath)
        print(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'PVDataPreprocessor':
        """Load preprocessor state from file."""
        import joblib
        state = joblib.load(filepath)

        preprocessor = cls(colnames=state['colnames'])
        preprocessor.param_transformer = state['param_transformer']
        preprocessor.fitted = state['fitted']

        print(f"Preprocessor loaded from {filepath}")
        print(f"  Isc training stats: {state['isc_train_stats']}")

        return preprocessor


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_curve_normalization(curves_raw: np.ndarray, curves_norm: np.ndarray,
                                 isc_values: np.ndarray, n_samples: int = 5):
    """
    Validate that curve normalization/denormalization is correct.

    Args:
        curves_raw: (N, 45) raw curves
        curves_norm: (N, 45) normalized curves
        isc_values: (N,) Isc values
        n_samples: Number of samples to check
    """
    print("\n=== CURVE NORMALIZATION VALIDATION ===")

    # Check normalized range
    print(f"Normalized curve range: [{curves_norm.min():.4f}, {curves_norm.max():.4f}] (should be ~[-1, 1])")

    # Check denormalization round-trip
    curves_denorm = denormalize_curves_by_isc(curves_norm, isc_values)
    max_error = np.abs(curves_raw - curves_denorm).max()
    mean_error = np.abs(curves_raw - curves_denorm).mean()

    print(f"Round-trip error: max={max_error:.2e}, mean={mean_error:.2e} (should be ~0)")

    # Sample-wise validation
    print(f"\nSample validation (first {n_samples} curves):")
    for i in range(min(n_samples, len(curves_raw))):
        isc_raw = curves_raw[i, 0]
        isc_stored = isc_values[i]
        first_norm = curves_norm[i, 0]

        # First point should normalize to 1.0: 2*(isc/isc) - 1 = 2 - 1 = 1
        expected_norm = 2.0 * (isc_raw / isc_stored) - 1.0

        print(f"  Sample {i}: Isc={isc_raw:.2f}, norm[0]={first_norm:.4f}, expected={expected_norm:.4f}")

    if max_error < 1e-4 and -1.1 < curves_norm.min() and curves_norm.max() < 1.1:
        print("\n✓ Curve normalization validation PASSED")
    else:
        print("\n✗ Curve normalization validation FAILED")
        print("  Check normalization formula and denormalization inverse")
