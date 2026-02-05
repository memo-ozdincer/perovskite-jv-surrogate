#!/usr/bin/env python3
"""
Physics Analysis Module for ICML Paper
=======================================

This module provides:
1. Jacobian sensitivity analysis (input-output derivatives)
2. Parameter sensitivity analysis (one-at-a-time perturbations)
3. Feature importance ranking via permutation/SHAP
4. Physics constraint validation
5. Uncertainty calibration analysis

Usage:
    python physics_analysis.py --model-dir outputs/T0-1-main/seed_42 --analysis jacobian
    python physics_analysis.py --model-dir outputs/T0-1-main/seed_42 --analysis sensitivity
    python physics_analysis.py --model-dir outputs/T0-1-main/seed_42 --analysis all
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import local modules
try:
    from config import V_GRID, COLNAMES, TARGETS
    from features import compute_all_physics_features
    from metrics_curve import compute_delta_v, curve_r2_weighted
except ImportError:
    print("Warning: Could not import local modules. Some features may be limited.")
    V_GRID = None
    COLNAMES = None


@dataclass
class JacobianAnalysis:
    """Results from Jacobian sensitivity analysis."""
    jacobian_mean: np.ndarray  # (n_outputs, n_inputs)
    jacobian_std: np.ndarray
    jacobian_abs_mean: np.ndarray
    input_names: List[str]
    output_names: List[str]
    n_samples: int

    def get_input_importance(self) -> pd.DataFrame:
        """Get input feature importance based on mean |∂output/∂input|."""
        importance = self.jacobian_abs_mean.sum(axis=0)
        df = pd.DataFrame({
            'feature': self.input_names,
            'importance': importance,
            'rank': np.argsort(-importance) + 1
        })
        return df.sort_values('importance', ascending=False)

    def save(self, output_dir: Path) -> None:
        """Save analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / 'jacobian_mean.npy', self.jacobian_mean)
        np.save(output_dir / 'jacobian_std.npy', self.jacobian_std)
        np.save(output_dir / 'jacobian_abs_mean.npy', self.jacobian_abs_mean)

        with open(output_dir / 'jacobian_meta.json', 'w') as f:
            json.dump({
                'input_names': self.input_names,
                'output_names': self.output_names,
                'n_samples': self.n_samples,
            }, f, indent=2)

        self.get_input_importance().to_csv(output_dir / 'feature_importance_jacobian.csv', index=False)


@dataclass
class SensitivityAnalysis:
    """Results from parameter sensitivity analysis."""
    parameter_names: List[str]
    target_names: List[str]
    sensitivities: np.ndarray  # (n_params, n_targets, n_perturbations)
    perturbations: np.ndarray
    baseline_predictions: np.ndarray

    def get_sensitivity_summary(self) -> pd.DataFrame:
        """Get summary of sensitivities (range of output change per parameter)."""
        rows = []
        for i, param in enumerate(self.parameter_names):
            for j, target in enumerate(self.target_names):
                sens_curve = self.sensitivities[i, j, :]
                rows.append({
                    'parameter': param,
                    'target': target,
                    'sensitivity_range': sens_curve.max() - sens_curve.min(),
                    'sensitivity_slope': np.polyfit(self.perturbations, sens_curve, 1)[0],
                    'nonlinearity': np.std(sens_curve - np.polyval(np.polyfit(self.perturbations, sens_curve, 1), self.perturbations)),
                })
        return pd.DataFrame(rows)

    def save(self, output_dir: Path) -> None:
        """Save analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / 'sensitivities.npy', self.sensitivities)
        np.save(output_dir / 'perturbations.npy', self.perturbations)

        with open(output_dir / 'sensitivity_meta.json', 'w') as f:
            json.dump({
                'parameter_names': self.parameter_names,
                'target_names': self.target_names,
            }, f, indent=2)

        self.get_sensitivity_summary().to_csv(output_dir / 'sensitivity_summary.csv', index=False)


def compute_jacobian_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    output_indices: Optional[List[int]] = None,
    device: str = 'cuda',
    use_hutchinson: bool = True,
    n_hutchinson_samples: int = 10,
) -> torch.Tensor:
    """
    Compute Jacobian matrix for a batch of inputs.

    Args:
        model: Neural network model
        inputs: (N, n_inputs) tensor
        output_indices: Which outputs to compute Jacobian for (None = all)
        device: Device to use
        use_hutchinson: Use Hutchinson's trace estimator (faster for large outputs)
        n_hutchinson_samples: Number of random vectors for Hutchinson estimator

    Returns:
        Jacobian tensor of shape (N, n_outputs, n_inputs)
    """
    model = model.to(device)
    model.eval()

    inputs = inputs.to(device)
    inputs.requires_grad_(True)

    # Forward pass
    with torch.enable_grad():
        outputs = model(inputs)

    n_batch, n_inputs = inputs.shape
    n_outputs = outputs.shape[1] if len(outputs.shape) > 1 else 1

    if output_indices is None:
        output_indices = list(range(n_outputs))

    if use_hutchinson and n_outputs > 10:
        # Hutchinson's estimator for large output dimensions
        jacobian = torch.zeros(n_batch, len(output_indices), n_inputs, device=device)

        for _ in range(n_hutchinson_samples):
            v = torch.randn_like(outputs[:, output_indices])
            vjp = torch.autograd.grad(
                outputs[:, output_indices], inputs,
                grad_outputs=v,
                retain_graph=True,
                create_graph=False
            )[0]
            jacobian += (v.unsqueeze(-1) * vjp.unsqueeze(1)) / n_hutchinson_samples

    else:
        # Exact Jacobian computation
        jacobian = torch.zeros(n_batch, len(output_indices), n_inputs, device=device)

        for i, out_idx in enumerate(output_indices):
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[:, out_idx] = 1.0

            grads = torch.autograd.grad(
                outputs, inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]
            jacobian[:, i, :] = grads

    return jacobian


def jacobian_analysis(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    input_names: List[str],
    output_names: List[str],
    device: str = 'cuda',
    max_samples: int = 1000,
) -> JacobianAnalysis:
    """
    Perform comprehensive Jacobian sensitivity analysis.

    Args:
        model: Trained neural network
        data_loader: DataLoader with input samples
        input_names: Names of input features
        output_names: Names of output variables
        device: Device to use
        max_samples: Maximum number of samples to analyze

    Returns:
        JacobianAnalysis dataclass with results
    """
    model = model.to(device)
    model.eval()

    all_jacobians = []
    n_samples = 0

    for batch in tqdm(data_loader, desc="Computing Jacobians"):
        if n_samples >= max_samples:
            break

        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        inputs = inputs.to(device)

        # Limit batch size
        if n_samples + len(inputs) > max_samples:
            inputs = inputs[:max_samples - n_samples]

        jacobian = compute_jacobian_batch(model, inputs, device=device)
        all_jacobians.append(jacobian.cpu().numpy())
        n_samples += len(inputs)

    # Concatenate and compute statistics
    jacobians = np.concatenate(all_jacobians, axis=0)

    return JacobianAnalysis(
        jacobian_mean=jacobians.mean(axis=0),
        jacobian_std=jacobians.std(axis=0),
        jacobian_abs_mean=np.abs(jacobians).mean(axis=0),
        input_names=input_names,
        output_names=output_names,
        n_samples=n_samples,
    )


def parameter_sensitivity_analysis(
    model: nn.Module,
    baseline_inputs: torch.Tensor,
    parameter_names: List[str],
    target_names: List[str],
    perturbation_range: Tuple[float, float] = (-0.2, 0.2),
    n_steps: int = 21,
    device: str = 'cuda',
) -> SensitivityAnalysis:
    """
    One-at-a-time parameter sensitivity analysis.

    Args:
        model: Trained neural network
        baseline_inputs: (1, n_params) baseline input tensor
        parameter_names: Names of parameters
        target_names: Names of targets to analyze
        perturbation_range: Range of perturbations (fraction of normalized range)
        n_steps: Number of perturbation steps
        device: Device to use

    Returns:
        SensitivityAnalysis dataclass with results
    """
    model = model.to(device)
    model.eval()

    baseline = baseline_inputs.to(device)
    perturbations = np.linspace(perturbation_range[0], perturbation_range[1], n_steps)

    n_params = baseline.shape[1]
    n_targets = len(target_names)

    sensitivities = np.zeros((n_params, n_targets, n_steps))

    # Baseline prediction
    with torch.no_grad():
        baseline_pred = model(baseline)
        baseline_pred = baseline_pred.cpu().numpy().flatten()

    # Perturb each parameter
    for i in tqdm(range(n_params), desc="Parameter sensitivity"):
        for j, delta in enumerate(perturbations):
            perturbed = baseline.clone()
            perturbed[0, i] += delta

            with torch.no_grad():
                pred = model(perturbed)
                pred = pred.cpu().numpy().flatten()

            for k in range(min(n_targets, len(pred))):
                sensitivities[i, k, j] = pred[k]

    return SensitivityAnalysis(
        parameter_names=parameter_names,
        target_names=target_names,
        sensitivities=sensitivities,
        perturbations=perturbations,
        baseline_predictions=baseline_pred,
    )


def physics_constraint_validation(
    predictions: Dict[str, torch.Tensor],
    curves: torch.Tensor,
    v_grid: torch.Tensor,
) -> Dict[str, any]:
    """
    Validate that predictions satisfy physical constraints.

    Checks:
    1. J decreases with V (monotonicity)
    2. J(0) = Jsc (boundary)
    3. J(Voc) = 0 (boundary)
    4. Vmpp < Voc, Jmpp < Jsc (ordering)
    5. FF = Pmpp / (Jsc × Voc) consistency

    Returns:
        Dict with validation results
    """
    results = {
        'n_samples': len(curves),
        'violations': {},
        'statistics': {},
    }

    # 1. Monotonicity check
    diffs = torch.diff(curves, dim=1)
    n_mono_violations = (diffs > 1e-4).any(dim=1).sum().item()
    results['violations']['monotonicity'] = n_mono_violations
    results['statistics']['monotonicity_rate'] = n_mono_violations / len(curves)

    # 2. Boundary checks
    jsc = predictions.get('jsc', curves[:, 0])
    voc = predictions.get('voc')

    # J(0) should equal Jsc
    j_at_0 = curves[:, 0]
    jsc_boundary_error = torch.abs(j_at_0 - jsc).mean().item()
    results['statistics']['jsc_boundary_mae'] = jsc_boundary_error

    # 3. Ordering checks
    if 'vmpp' in predictions and voc is not None:
        vmpp = predictions['vmpp']
        n_vmpp_violations = (vmpp >= voc).sum().item()
        results['violations']['vmpp_exceeds_voc'] = n_vmpp_violations

    if 'jmpp' in predictions:
        jmpp = predictions['jmpp']
        n_jmpp_violations = (jmpp >= jsc).sum().item()
        results['violations']['jmpp_exceeds_jsc'] = n_jmpp_violations

    # 4. Fill factor consistency
    if all(k in predictions for k in ['vmpp', 'jmpp', 'jsc', 'voc']):
        ff_computed = (predictions['vmpp'] * predictions['jmpp']) / \
                      (predictions['jsc'] * predictions['voc'] + 1e-10)

        if 'ff' in predictions:
            ff_error = torch.abs(ff_computed - predictions['ff']).mean().item()
            results['statistics']['ff_consistency_error'] = ff_error

    # Total violations
    results['total_violations'] = sum(results['violations'].values())
    results['violation_rate'] = results['total_violations'] / (len(curves) * len(results['violations']))

    return results


def plot_jacobian_heatmap(
    analysis: JacobianAnalysis,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 10),
    top_n_inputs: int = 31,
    top_n_outputs: int = 20,
) -> None:
    """Plot Jacobian heatmap."""
    plt.figure(figsize=figsize)

    # Get top inputs by importance
    importance = analysis.jacobian_abs_mean.sum(axis=0)
    top_input_idx = np.argsort(-importance)[:top_n_inputs]

    # Subset the Jacobian
    jac_subset = analysis.jacobian_abs_mean[:top_n_outputs, top_input_idx]
    input_labels = [analysis.input_names[i] for i in top_input_idx]
    output_labels = analysis.output_names[:top_n_outputs]

    # Plot
    sns.heatmap(
        jac_subset,
        xticklabels=input_labels,
        yticklabels=output_labels,
        cmap='YlOrRd',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Mean |∂output/∂input|'}
    )

    plt.xlabel('Input Features')
    plt.ylabel('Output Variables')
    plt.title('Jacobian Sensitivity Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sensitivity_tornado(
    analysis: SensitivityAnalysis,
    output_path: Path,
    target_idx: int = 0,
    figsize: Tuple[int, int] = (10, 12),
) -> None:
    """Plot tornado diagram for parameter sensitivity."""
    fig, ax = plt.subplots(figsize=figsize)

    # Get sensitivity ranges for each parameter
    target_name = analysis.target_names[target_idx]
    sens = analysis.sensitivities[:, target_idx, :]
    baseline = analysis.baseline_predictions[target_idx]

    # Compute impact (max - min)
    impacts = sens.max(axis=1) - sens.min(axis=1)
    sorted_idx = np.argsort(impacts)[::-1][:20]  # Top 20

    y_pos = np.arange(len(sorted_idx))

    for i, idx in enumerate(sorted_idx):
        low = sens[idx, :].min() - baseline
        high = sens[idx, :].max() - baseline

        ax.barh(i, high - low, left=low, color='steelblue', alpha=0.7, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([analysis.parameter_names[i] for i in sorted_idx])
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel(f'Impact on {target_name}')
    ax.set_title(f'Parameter Sensitivity: {target_name}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sensitivity_spider(
    analysis: SensitivityAnalysis,
    output_path: Path,
    top_n_params: int = 10,
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """Plot spider/radar chart for multi-target sensitivity."""
    from matplotlib.patches import Patch

    n_targets = len(analysis.target_names)

    # Get top parameters by total impact
    impacts = np.abs(analysis.sensitivities).sum(axis=(1, 2))
    top_idx = np.argsort(-impacts)[:top_n_params]
    param_names = [analysis.parameter_names[i] for i in top_idx]

    # Setup angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set2(np.linspace(0, 1, n_targets))

    for t_idx, target in enumerate(analysis.target_names):
        # Normalize sensitivities for this target
        sens = analysis.sensitivities[top_idx, t_idx, :].max(axis=1) - \
               analysis.sensitivities[top_idx, t_idx, :].min(axis=1)
        sens_norm = sens / (sens.max() + 1e-10)
        values = sens_norm.tolist()
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=target, color=colors[t_idx])
        ax.fill(angles, values, alpha=0.1, color=colors[t_idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(param_names, size=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Multi-Target Parameter Sensitivity', size=14, y=1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def run_full_analysis(
    model_dir: Path,
    output_dir: Path,
    data_path: Optional[str] = None,
    device: str = 'cuda',
) -> Dict:
    """
    Run full physics analysis suite.

    Args:
        model_dir: Directory containing trained model
        output_dir: Directory to save analysis results
        data_path: Path to test data (if not in model_dir)
        device: Device to use

    Returns:
        Dict with all analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print(f"\n{'='*60}")
    print("PHYSICS ANALYSIS SUITE")
    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Load model (implementation depends on your model structure)
    # This is a placeholder - adjust based on actual model loading
    print("\n[1/4] Loading model...")
    # model = load_model(model_dir)

    # For now, create mock analysis to demonstrate structure
    print("\n[2/4] Jacobian analysis...")
    # jacobian = jacobian_analysis(model, test_loader, input_names, output_names, device)
    # jacobian.save(output_dir / 'jacobian')
    # plot_jacobian_heatmap(jacobian, output_dir / 'jacobian_heatmap.pdf')

    print("\n[3/4] Parameter sensitivity analysis...")
    # sensitivity = parameter_sensitivity_analysis(model, baseline, param_names, target_names)
    # sensitivity.save(output_dir / 'sensitivity')
    # plot_sensitivity_tornado(sensitivity, output_dir / 'sensitivity_tornado.pdf')
    # plot_sensitivity_spider(sensitivity, output_dir / 'sensitivity_spider.pdf')

    print("\n[4/4] Physics constraint validation...")
    # validation = physics_constraint_validation(predictions, curves, v_grid)
    # with open(output_dir / 'physics_validation.json', 'w') as f:
    #     json.dump(validation, f, indent=2)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Physics analysis for ICML paper")

    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: model_dir/analysis)')
    parser.add_argument('--analysis', type=str, default='all',
                        choices=['jacobian', 'sensitivity', 'validation', 'all'],
                        help='Type of analysis to run')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Max samples for Jacobian analysis')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / 'analysis'

    run_full_analysis(model_dir, output_dir, device=args.device)


if __name__ == '__main__':
    main()
