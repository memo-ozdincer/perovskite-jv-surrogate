#!/usr/bin/env python3
"""
Canonical curve metrics module for ICML paper.
All evaluation must use these functions for consistency.

This module provides:
1. ΔV-weighted metrics for non-uniform voltage grid evaluation
2. Per-curve R², MAPE, MAE calculations
3. Anchor metrics (Jsc, Voc, Vmpp, Jmpp, FF)
4. Physics constraint violation tracking
5. Comprehensive metric aggregation for paper tables
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Import V_GRID from config if available, otherwise define it
try:
    from config import V_GRID
except ImportError:
    V_GRID = np.concatenate([
        np.arange(0, 0.4 + 1e-8, 0.1),
        np.arange(0.425, 1.4 + 1e-8, 0.025)
    ]).astype(np.float32)


@dataclass
class CurveMetrics:
    """Complete metrics for curve prediction evaluation."""
    # Primary metrics
    r2_mean: float
    r2_median: float
    r2_std: float
    r2_p5: float
    r2_p25: float
    r2_p75: float
    r2_p95: float

    # MAPE metrics
    mape_mean: float
    mape_median: float

    # MAE metrics
    mae_mean: float
    mae_max: float

    # Anchor metrics (optional)
    jsc_mae: Optional[float] = None
    voc_mae: Optional[float] = None
    vmpp_mae: Optional[float] = None
    jmpp_mae: Optional[float] = None
    ff_mape: Optional[float] = None
    pce_mape: Optional[float] = None

    # Constraint violations
    violations_per_1000: float = 0.0
    violations_jsc_negative: int = 0
    violations_voc_negative: int = 0
    violations_vmpp_exceeds_voc: int = 0
    violations_jmpp_exceeds_jsc: int = 0
    violations_non_monotonic: int = 0

    # Timing
    inference_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_delta_v(V_grid: np.ndarray = None) -> np.ndarray:
    """
    Compute ΔV weights for non-uniform grid.

    For our grid: 0.1V for indices 0-4, 0.025V for indices 5-44.

    Returns:
        Array of length 45 with ΔV values.
    """
    if V_grid is None:
        V_grid = V_GRID

    delta_v = np.diff(V_grid)
    # Pad last element by copying second-to-last
    delta_v = np.append(delta_v, delta_v[-1])
    return delta_v.astype(np.float32)


def compute_delta_v_torch(V_grid: torch.Tensor = None, device: str = 'cpu') -> torch.Tensor:
    """Torch version of compute_delta_v."""
    if V_grid is None:
        V_grid = torch.tensor(V_GRID, dtype=torch.float32, device=device)

    delta_v = torch.diff(V_grid)
    delta_v = torch.cat([delta_v, delta_v[-1:]])
    return delta_v


def curve_sse_weighted(
    J_pred: torch.Tensor,  # (N, 45)
    J_true: torch.Tensor,  # (N, 45)
    delta_v: torch.Tensor = None,  # (45,)
    mpp_weight: torch.Tensor = None  # (45,) optional MPP weighting
) -> torch.Tensor:
    """
    Compute ΔV-weighted sum of squared errors per curve.

    Returns:
        Tensor of shape (N,) with SSE for each curve.
    """
    if delta_v is None:
        delta_v = compute_delta_v_torch(device=J_pred.device)

    weights = delta_v
    if mpp_weight is not None:
        weights = weights * mpp_weight

    sq_errors = (J_pred - J_true) ** 2  # (N, 45)
    weighted_sse = (sq_errors * weights).sum(dim=1)  # (N,)

    return weighted_sse


def curve_r2_weighted(
    J_pred: torch.Tensor,  # (N, 45)
    J_true: torch.Tensor,  # (N, 45)
    delta_v: torch.Tensor = None,
    return_per_curve: bool = True
) -> Dict[str, Union[torch.Tensor, float]]:
    """
    Compute ΔV-weighted R² per curve.

    Returns:
        dict with keys: 'r2_per_curve', 'mean', 'median', 'p5', 'p25', 'p75', 'p95', 'std'
    """
    if delta_v is None:
        delta_v = compute_delta_v_torch(device=J_pred.device)

    # ΔV-weighted mean of true values
    J_mean = (J_true * delta_v).sum(dim=1, keepdim=True) / delta_v.sum()  # (N, 1)

    # SSE (residual)
    sse = ((J_pred - J_true) ** 2 * delta_v).sum(dim=1)  # (N,)

    # SST (total)
    sst = ((J_true - J_mean) ** 2 * delta_v).sum(dim=1)  # (N,)

    # R² per curve (avoid division by zero)
    r2 = 1 - sse / (sst + 1e-10)  # (N,)

    result = {
        'mean': r2.mean().item(),
        'median': r2.median().item(),
        'std': r2.std().item(),
        'p5': torch.quantile(r2, 0.05).item(),
        'p25': torch.quantile(r2, 0.25).item(),
        'p75': torch.quantile(r2, 0.75).item(),
        'p95': torch.quantile(r2, 0.95).item(),
    }

    if return_per_curve:
        result['r2_per_curve'] = r2

    return result


def curve_mape_safe(
    J_pred: torch.Tensor,
    J_true: torch.Tensor,
    eps: float = 0.1  # mA/cm² floor to avoid explosion near Voc
) -> Dict[str, Union[torch.Tensor, float]]:
    """
    Safe MAPE with clipped denominator.

    Returns:
        dict with 'mape_mean', 'mape_median', 'mape_per_curve'
    """
    denominator = torch.clamp(torch.abs(J_true), min=eps)
    ape = torch.abs(J_pred - J_true) / denominator * 100  # (N, 45)

    mape_per_curve = ape.mean(dim=1)  # (N,)

    return {
        'mape_per_curve': mape_per_curve,
        'mape_mean': mape_per_curve.mean().item(),
        'mape_median': mape_per_curve.median().item(),
    }


def curve_mae(
    J_pred: torch.Tensor,
    J_true: torch.Tensor,
) -> Dict[str, Union[torch.Tensor, float]]:
    """
    Compute MAE metrics.

    Returns:
        dict with 'mae_mean', 'mae_max', 'mae_per_curve'
    """
    ae = torch.abs(J_pred - J_true)  # (N, 45)
    mae_per_curve = ae.mean(dim=1)  # (N,)

    return {
        'mae_per_curve': mae_per_curve,
        'mae_mean': mae_per_curve.mean().item(),
        'mae_max': mae_per_curve.max().item(),
    }


def compute_anchor_metrics(
    anchors_pred: Dict[str, torch.Tensor],
    anchors_true: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute metrics for anchor predictions.

    Args:
        anchors_pred: dict with keys 'jsc', 'voc', 'vmpp', 'jmpp'
        anchors_true: dict with same keys

    Returns:
        dict with MAE and MAPE for each anchor + fill factor
    """
    metrics = {}

    for key in ['jsc', 'voc', 'vmpp', 'jmpp']:
        if key in anchors_pred and key in anchors_true:
            mae = torch.abs(anchors_pred[key] - anchors_true[key]).mean().item()
            metrics[f'{key}_mae'] = mae

            # MAPE with safe division
            denom = torch.clamp(torch.abs(anchors_true[key]), min=1e-6)
            mape = (torch.abs(anchors_pred[key] - anchors_true[key]) / denom * 100).mean().item()
            metrics[f'{key}_mape'] = mape

    # Fill factor
    if all(k in anchors_pred for k in ['vmpp', 'jmpp', 'voc', 'jsc']):
        ff_pred = (anchors_pred['vmpp'] * anchors_pred['jmpp']) / \
                  (anchors_pred['voc'] * anchors_pred['jsc'] + 1e-10)
        ff_true = (anchors_true['vmpp'] * anchors_true['jmpp']) / \
                  (anchors_true['voc'] * anchors_true['jsc'] + 1e-10)

        ff_mape = (torch.abs(ff_pred - ff_true) / (ff_true + 1e-10) * 100).mean().item()
        metrics['ff_mape'] = ff_mape
        metrics['ff_mae'] = torch.abs(ff_pred - ff_true).mean().item()

    return metrics


def count_physics_violations(
    anchors: Dict[str, torch.Tensor],
    curves: torch.Tensor = None,
    eps: float = 1e-6
) -> Dict[str, int]:
    """
    Count physics constraint violations.

    Args:
        anchors: dict with 'jsc', 'voc', 'vmpp', 'jmpp'
        curves: optional (N, 45) tensor for monotonicity check
        eps: tolerance for violation detection

    Returns:
        dict with violation counts
    """
    violations = {
        'jsc_negative': 0,
        'voc_negative': 0,
        'vmpp_exceeds_voc': 0,
        'jmpp_exceeds_jsc': 0,
        'non_monotonic': 0,
        'total': 0,
    }

    n_samples = len(anchors['jsc'])

    violations['jsc_negative'] = int((anchors['jsc'] < eps).sum().item())
    violations['voc_negative'] = int((anchors['voc'] < eps).sum().item())
    violations['vmpp_exceeds_voc'] = int((anchors['vmpp'] >= anchors['voc']).sum().item())
    violations['jmpp_exceeds_jsc'] = int((anchors['jmpp'] >= anchors['jsc']).sum().item())

    # Monotonicity check on curves
    if curves is not None:
        diffs = torch.diff(curves, dim=1)
        non_mono = (diffs > 1e-4).any(dim=1).sum().item()  # J should decrease with V
        violations['non_monotonic'] = int(non_mono)

    violations['total'] = sum(v for k, v in violations.items() if k != 'total')
    violations['per_1000'] = violations['total'] / n_samples * 1000

    return violations


def compute_all_metrics(
    J_pred: torch.Tensor,
    J_true: torch.Tensor,
    anchors_pred: Dict[str, torch.Tensor] = None,
    anchors_true: Dict[str, torch.Tensor] = None,
    inference_time_ms: float = None,
) -> CurveMetrics:
    """
    Compute complete metric suite for a batch.

    Returns:
        CurveMetrics dataclass with all metrics needed for paper tables.
    """
    # Curve R²
    r2_stats = curve_r2_weighted(J_pred, J_true, return_per_curve=False)

    # Curve MAPE
    mape_stats = curve_mape_safe(J_pred, J_true)

    # Curve MAE
    mae_stats = curve_mae(J_pred, J_true)

    # Initialize metrics
    metrics_kwargs = {
        'r2_mean': r2_stats['mean'],
        'r2_median': r2_stats['median'],
        'r2_std': r2_stats['std'],
        'r2_p5': r2_stats['p5'],
        'r2_p25': r2_stats['p25'],
        'r2_p75': r2_stats['p75'],
        'r2_p95': r2_stats['p95'],
        'mape_mean': mape_stats['mape_mean'],
        'mape_median': mape_stats['mape_median'],
        'mae_mean': mae_stats['mae_mean'],
        'mae_max': mae_stats['mae_max'],
        'inference_ms': inference_time_ms,
    }

    # Anchor metrics (if provided)
    if anchors_pred is not None and anchors_true is not None:
        anchor_metrics = compute_anchor_metrics(anchors_pred, anchors_true)
        metrics_kwargs['jsc_mae'] = anchor_metrics.get('jsc_mae')
        metrics_kwargs['voc_mae'] = anchor_metrics.get('voc_mae')
        metrics_kwargs['vmpp_mae'] = anchor_metrics.get('vmpp_mae')
        metrics_kwargs['jmpp_mae'] = anchor_metrics.get('jmpp_mae')
        metrics_kwargs['ff_mape'] = anchor_metrics.get('ff_mape')

        # Violations
        violations = count_physics_violations(anchors_pred, J_pred)
        metrics_kwargs['violations_per_1000'] = violations['per_1000']
        metrics_kwargs['violations_jsc_negative'] = violations['jsc_negative']
        metrics_kwargs['violations_voc_negative'] = violations['voc_negative']
        metrics_kwargs['violations_vmpp_exceeds_voc'] = violations['vmpp_exceeds_voc']
        metrics_kwargs['violations_jmpp_exceeds_jsc'] = violations['jmpp_exceeds_jsc']
        metrics_kwargs['violations_non_monotonic'] = violations['non_monotonic']

    return CurveMetrics(**metrics_kwargs)


def compute_region_metrics(
    J_pred: torch.Tensor,
    J_true: torch.Tensor,
    V_grid: torch.Tensor,
    vmpp: torch.Tensor,  # (N,)
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for Region 1 (0 to Vmpp) and Region 2 (Vmpp to Voc).

    Returns:
        dict with 'region1' and 'region2' sub-dicts containing R² and MAPE
    """
    if V_grid is None:
        V_grid = torch.tensor(V_GRID, device=J_pred.device)

    # Create masks for each region
    V_expanded = V_grid.unsqueeze(0).expand(len(vmpp), -1)  # (N, 45)
    vmpp_expanded = vmpp.unsqueeze(1)  # (N, 1)

    region1_mask = V_expanded <= vmpp_expanded  # (N, 45)
    region2_mask = V_expanded > vmpp_expanded

    results = {}

    for region_name, mask in [('region1', region1_mask), ('region2', region2_mask)]:
        # Masked metrics
        J_pred_masked = J_pred.clone()
        J_true_masked = J_true.clone()

        # Set non-region values to same to avoid contribution
        J_pred_masked[~mask] = 0
        J_true_masked[~mask] = 0

        # Compute region-specific R²
        mask_sum = mask.float().sum(dim=1, keepdim=True)
        J_mean = (J_true_masked * mask.float()).sum(dim=1, keepdim=True) / (mask_sum + 1e-10)

        sse = ((J_pred_masked - J_true_masked) ** 2 * mask.float()).sum(dim=1)
        sst = ((J_true_masked - J_mean) ** 2 * mask.float()).sum(dim=1)
        r2 = 1 - sse / (sst + 1e-10)

        results[region_name] = {
            'r2_mean': r2.mean().item(),
            'r2_median': r2.median().item(),
        }

    return results


def format_metrics_table(
    metrics_list: list,  # List of (name, CurveMetrics) tuples
    latex: bool = False
) -> str:
    """
    Format metrics as a comparison table.

    Args:
        metrics_list: list of (experiment_name, CurveMetrics) tuples
        latex: if True, output LaTeX table format

    Returns:
        Formatted table string
    """
    headers = ['Model', 'R² mean', 'R² p5', 'MAPE (%)', 'FF MAPE (%)', 'Violations/1k', 'Time (ms)']

    rows = []
    for name, m in metrics_list:
        row = [
            name,
            f"{m.r2_mean:.4f}",
            f"{m.r2_p5:.4f}",
            f"{m.mape_mean:.2f}",
            f"{m.ff_mape:.2f}" if m.ff_mape else "-",
            f"{m.violations_per_1000:.1f}",
            f"{m.inference_ms:.2f}" if m.inference_ms else "-",
        ]
        rows.append(row)

    if latex:
        lines = [
            "\\begin{tabular}{l" + "c" * (len(headers) - 1) + "}",
            "\\toprule",
            " & ".join(headers) + " \\\\",
            "\\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(row) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        return "\n".join(lines)
    else:
        # Markdown table
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


def save_metrics_summary(
    metrics_dict: Dict[str, CurveMetrics],
    output_path: Union[str, Path],
    format: str = 'all'  # 'json', 'csv', 'latex', 'all'
) -> None:
    """
    Save metrics summary to file(s).

    Args:
        metrics_dict: dict mapping experiment names to CurveMetrics
        output_path: base path (without extension)
        format: output format(s)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    data = {name: m.to_dict() for name, m in metrics_dict.items()}

    if format in ['json', 'all']:
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(data, f, indent=2)

    if format in ['csv', 'all']:
        import pandas as pd
        df = pd.DataFrame(data).T
        df.to_csv(output_path.with_suffix('.csv'))

    if format in ['latex', 'all']:
        table = format_metrics_table(list(metrics_dict.items()), latex=True)
        with open(output_path.with_suffix('.tex'), 'w') as f:
            f.write(table)

    if format in ['md', 'all']:
        table = format_metrics_table(list(metrics_dict.items()), latex=False)
        with open(output_path.with_suffix('.md'), 'w') as f:
            f.write(table)


# Numpy versions for non-PyTorch evaluation
def curve_r2_weighted_numpy(
    J_pred: np.ndarray,
    J_true: np.ndarray,
    delta_v: np.ndarray = None,
) -> Dict[str, float]:
    """Numpy version of curve_r2_weighted."""
    if delta_v is None:
        delta_v = compute_delta_v()

    J_mean = (J_true * delta_v).sum(axis=1, keepdims=True) / delta_v.sum()
    sse = ((J_pred - J_true) ** 2 * delta_v).sum(axis=1)
    sst = ((J_true - J_mean) ** 2 * delta_v).sum(axis=1)
    r2 = 1 - sse / (sst + 1e-10)

    return {
        'mean': float(np.mean(r2)),
        'median': float(np.median(r2)),
        'std': float(np.std(r2)),
        'p5': float(np.percentile(r2, 5)),
        'p25': float(np.percentile(r2, 25)),
        'p75': float(np.percentile(r2, 75)),
        'p95': float(np.percentile(r2, 95)),
    }


if __name__ == '__main__':
    # Test the metrics
    print("Testing metrics_curve.py...")

    # Create dummy data
    N = 100
    J_true = torch.randn(N, 45).abs() * 20  # Positive currents
    J_true = torch.cumsum(-torch.abs(torch.randn(N, 45)), dim=1) + 25  # Decreasing
    J_pred = J_true + torch.randn(N, 45) * 0.5  # Add noise

    # Compute metrics
    r2_stats = curve_r2_weighted(J_pred, J_true)
    print(f"R² mean: {r2_stats['mean']:.4f}")
    print(f"R² median: {r2_stats['median']:.4f}")
    print(f"R² p5: {r2_stats['p5']:.4f}")

    mape_stats = curve_mape_safe(J_pred, J_true)
    print(f"MAPE mean: {mape_stats['mape_mean']:.2f}%")

    # Test delta_v
    delta_v = compute_delta_v()
    print(f"Delta V shape: {delta_v.shape}")
    print(f"Delta V range: [{delta_v.min():.3f}, {delta_v.max():.3f}]")

    print("\nmetrics_curve.py tests passed!")
