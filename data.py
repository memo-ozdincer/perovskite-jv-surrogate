"""
Ultra-efficient data loading and target extraction using PyTorch CUDA.
Vectorized operations throughout - no loops.
"""
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from config import COLNAMES, V_GRID, RANDOM_SEED

# ============================================================================
# TARGET EXTRACTION - Fully vectorized on GPU
# ============================================================================

def extract_targets_gpu(iv_curves: torch.Tensor, v_grid: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Extract all PV parameters from J-V curves in a single vectorized pass.

    Args:
        iv_curves: (N, 45) tensor of current density values on GPU
        v_grid: (45,) tensor of voltage values on GPU

    Returns:
        Dictionary with Voc, Jsc, Vmpp, Jmpp, FF, Pmpp, PCE tensors
    """
    N = iv_curves.shape[0]

    # J_sc: Current at V=0 (first point)
    Jsc = iv_curves[:, 0]

    # Power curve: P = V * J
    power = v_grid.unsqueeze(0) * iv_curves  # (N, 45)

    # V_mpp, J_mpp, P_mpp: Maximum power point
    Pmpp, mpp_idx = power.max(dim=1)
    Vmpp = v_grid[mpp_idx]
    Jmpp = iv_curves[torch.arange(N, device=iv_curves.device), mpp_idx]

    # V_oc: Voltage where J crosses zero (linear interpolation)
    # Find the crossing point: J changes sign from positive to negative
    Voc = _find_voc_vectorized(iv_curves, v_grid)

    # Fill Factor: FF = Pmpp / (Voc * Jsc)
    FF = Pmpp / (Voc * Jsc + 1e-12)
    FF = FF.clamp(0.0, 1.0)

    # PCE: Assuming 1000 W/m² illumination
    PCE = Pmpp / 1000.0

    return {
        'Voc': Voc,
        'Jsc': Jsc,
        'Vmpp': Vmpp,
        'Jmpp': Jmpp,
        'FF': FF,
        'Pmpp': Pmpp,
        'PCE': PCE
    }


def _find_voc_vectorized(iv_curves: torch.Tensor, v_grid: torch.Tensor) -> torch.Tensor:
    """
    Find V_oc using vectorized linear interpolation.
    V_oc is where current crosses zero (J goes from positive to negative).
    """
    N = iv_curves.shape[0]
    n_pts = iv_curves.shape[1]
    device = iv_curves.device

    # Sign change detection: positive to negative or zero crossing
    signs = torch.sign(iv_curves)
    sign_changes = (signs[:, :-1] > 0) & (signs[:, 1:] <= 0)

    # Find first crossing index for each curve
    # If no crossing found, use last point
    has_crossing = sign_changes.any(dim=1)
    first_crossing_idx = torch.where(
        has_crossing,
        sign_changes.int().argmax(dim=1),
        torch.full((N,), n_pts - 2, device=device, dtype=torch.long)
    )

    # Gather values for linear interpolation
    idx = first_crossing_idx
    idx_next = (idx + 1).clamp(max=n_pts - 1)

    batch_idx = torch.arange(N, device=device)
    J0 = iv_curves[batch_idx, idx]
    J1 = iv_curves[batch_idx, idx_next]
    V0 = v_grid[idx]
    V1 = v_grid[idx_next]

    # Linear interpolation: V_oc = V0 + (0 - J0) * (V1 - V0) / (J1 - J0)
    dJ = J1 - J0
    dJ = torch.where(dJ.abs() < 1e-12, torch.ones_like(dJ) * 1e-12, dJ)
    Voc = V0 - J0 * (V1 - V0) / dJ

    # Clamp to valid range
    Voc = Voc.clamp(v_grid[0], v_grid[-1])

    return Voc


def clamp_iv_curves_at_voc(iv_curves: torch.Tensor, v_grid: torch.Tensor) -> torch.Tensor:
    """
    Clamp IV curves to ensure V_oc is an endpoint and remove negative tail.

    For each curve, find V_oc, then zero out values at and beyond the first
    grid point >= V_oc. This ensures the curve only spans [0, V_oc] with
    J(V_oc) = 0.
    """
    voc = _find_voc_vectorized(iv_curves, v_grid)
    idx_cut = torch.searchsorted(v_grid, voc, right=False)
    idx_cut = idx_cut.clamp(max=v_grid.numel() - 1)
    v_cut = v_grid[idx_cut]

    mask = v_grid.unsqueeze(0) >= v_cut.unsqueeze(1)
    iv_curves = iv_curves.clone()
    iv_curves = torch.where(mask, torch.zeros_like(iv_curves), iv_curves)

    batch_idx = torch.arange(iv_curves.shape[0], device=iv_curves.device)
    iv_curves[batch_idx, idx_cut] = 0.0

    return iv_curves


# ============================================================================
# DATA LOADING
# ============================================================================

def load_raw_data(params_file: str | list[str], iv_file: str | list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load raw parameter and IV data from disk.
    
    Args:
        params_file: Path to parameters file or list of paths
        iv_file: Path to IV curves file or list of paths
        
    Returns:
        params_df: Concatenated parameter DataFrame
        iv_data: Concatenated IV curves array
    """
    # Convert single files to lists for uniform handling
    if isinstance(params_file, str):
        params_file = [params_file]
    if isinstance(iv_file, str):
        iv_file = [iv_file]
    
    if len(params_file) != len(iv_file):
        raise ValueError(
            f"Number of params files ({len(params_file)}) must match IV files ({len(iv_file)})"
        )
    
    # Load all datasets
    all_params = []
    all_iv = []
    
    for pfile, ifile in zip(params_file, iv_file):
        print(f"Loading: {pfile} and {ifile}")
        params_df = pd.read_csv(pfile, header=None, names=COLNAMES)
        iv_data = np.loadtxt(ifile, delimiter=',', dtype=np.float32)
        
        # Basic shape validation
        if iv_data.ndim == 1:
            iv_data = iv_data.reshape(1, -1)
        
        expected_cols = len(V_GRID)
        if iv_data.shape[1] != expected_cols:
            raise ValueError(
                f"IV data in {ifile} has {iv_data.shape[1]} columns, expected {expected_cols} points matching V_GRID"
            )
        if len(params_df) != iv_data.shape[0]:
            raise ValueError(
                f"Params rows ({len(params_df)}) in {pfile} do not match IV rows ({iv_data.shape[0]}) in {ifile}"
            )
        
        all_params.append(params_df)
        all_iv.append(iv_data)
        print(f"  Loaded {len(params_df)} samples")
    
    # Concatenate all datasets
    params_df = pd.concat(all_params, ignore_index=True)
    iv_data = np.vstack(all_iv)
    
    print(f"Total samples after concatenation: {len(params_df)}")

    # Clamp IV curves at Voc
    v_grid = torch.from_numpy(V_GRID.astype(np.float32))
    iv_tensor = torch.from_numpy(iv_data.astype(np.float32))
    iv_tensor = clamp_iv_curves_at_voc(iv_tensor, v_grid)
    iv_data = iv_tensor.cpu().numpy()

    return params_df, iv_data


def prepare_tensors(
    params_df: pd.DataFrame,
    iv_data: np.ndarray,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert data to GPU tensors.

    Returns:
        params: (N, 31) float32 tensor
        iv_curves: (N, 45) float32 tensor
        v_grid: (45,) float32 tensor
    """
    params = torch.from_numpy(params_df.values.astype(np.float32)).to(device)
    iv_curves = torch.from_numpy(iv_data.astype(np.float32)).to(device)
    v_grid = torch.from_numpy(V_GRID).to(device)
    return params, iv_curves, v_grid


# ============================================================================
# DATASET CLASS FOR TRAINING
# ============================================================================

class PVDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that keeps data on GPU."""

    def __init__(
        self,
        params: torch.Tensor,
        targets: dict[str, torch.Tensor],
        features: torch.Tensor | None = None
    ):
        self.params = params
        self.targets = targets
        self.features = features
        self.n_samples = params.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple:
        x = self.params[idx]
        if self.features is not None:
            x = torch.cat([x, self.features[idx]])
        return x, {k: v[idx] for k, v in self.targets.items()}


def create_dataloaders(
    params: torch.Tensor,
    targets: dict[str, torch.Tensor],
    features: torch.Tensor | None = None,
    batch_size: int = 4096,
    val_split: float = 0.1,
    test_split: float = 0.1
) -> tuple:
    """
    Create train/val/test dataloaders with GPU-resident data.
    Uses pin_memory=False since data is already on GPU.
    """
    n_samples = params.shape[0]
    indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(RANDOM_SEED))

    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def subset_targets(t_dict, idx):
        return {k: v[idx] for k, v in t_dict.items()}

    train_ds = PVDataset(
        params[train_idx],
        subset_targets(targets, train_idx),
        features[train_idx] if features is not None else None
    )
    val_ds = PVDataset(
        params[val_idx],
        subset_targets(targets, val_idx),
        features[val_idx] if features is not None else None
    )
    test_ds = PVDataset(
        params[test_idx],
        subset_targets(targets, test_idx),
        features[test_idx] if features is not None else None
    )

    # DataLoader with num_workers=0 since data is on GPU
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


def split_indices(n_samples: int, val_split: float = 0.1, test_split: float = 0.1) -> tuple:
    """Get train/val/test indices for consistent splitting across models."""
    indices = np.random.default_rng(RANDOM_SEED).permutation(n_samples)
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_test - n_val

    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:]
    )
