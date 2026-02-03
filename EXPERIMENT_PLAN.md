# PINN-Coupled-PDE-Solver: Experiment Production Plan

**Version**: 1.0
**Target**: ICML 2025 Submission
**Status**: Ready for Implementation

---

## Executive Summary

This document provides the complete experiment production plan to generate all results, figures, and tables for the ICML paper. The plan is organized into:

1. **Code Changes** (must implement before running experiments)
2. **Run Manifest** (exact experiments to run)
3. **Figure/Table Specifications** (what to produce from results)
4. **Quality Checks** (assertions to verify correctness)

**Total Compute Budget**: ~66 runs minimum (Tier 0 + Tier 1)

---

## 1. Code Changes Required

### 1.1 Create `metrics_curve.py` (NEW FILE)

```python
"""
Canonical curve metrics module for ICML paper.
All evaluation must use these functions for consistency.
"""
import torch
import numpy as np
from config import V_GRID

def compute_delta_v(V_grid: np.ndarray = None) -> np.ndarray:
    """
    Compute ΔV weights for non-uniform grid.

    Returns:
        Array of length 45 with ΔV values.
        For our grid: 0.1V for indices 0-4, 0.025V for indices 5-44.
    """
    if V_grid is None:
        V_grid = V_GRID

    delta_v = np.diff(V_grid)
    # Pad last element by copying second-to-last
    delta_v = np.append(delta_v, delta_v[-1])
    return delta_v.astype(np.float32)

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
        delta_v = torch.tensor(compute_delta_v(), device=J_pred.device)

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
    return_stats: bool = True
) -> dict:
    """
    Compute ΔV-weighted R² per curve.

    Returns:
        dict with keys: 'r2_per_curve', 'mean', 'median', 'p5', 'p25', 'p75', 'p95'
    """
    if delta_v is None:
        delta_v = torch.tensor(compute_delta_v(), device=J_pred.device)

    # ΔV-weighted mean of true values
    J_mean = (J_true * delta_v).sum(dim=1, keepdim=True) / delta_v.sum()  # (N, 1)

    # SSE (residual)
    sse = ((J_pred - J_true) ** 2 * delta_v).sum(dim=1)  # (N,)

    # SST (total)
    sst = ((J_true - J_mean) ** 2 * delta_v).sum(dim=1)  # (N,)

    # R² per curve (avoid division by zero)
    r2 = 1 - sse / (sst + 1e-10)  # (N,)

    if not return_stats:
        return {'r2_per_curve': r2}

    return {
        'r2_per_curve': r2,
        'mean': r2.mean().item(),
        'median': r2.median().item(),
        'std': r2.std().item(),
        'p5': torch.quantile(r2, 0.05).item(),
        'p25': torch.quantile(r2, 0.25).item(),
        'p75': torch.quantile(r2, 0.75).item(),
        'p95': torch.quantile(r2, 0.95).item(),
    }

def curve_mape_safe(
    J_pred: torch.Tensor,
    J_true: torch.Tensor,
    eps: float = 0.1  # mA/cm² floor to avoid explosion near Voc
) -> dict:
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

def compute_all_metrics(
    J_pred: torch.Tensor,
    J_true: torch.Tensor,
    anchors_pred: dict = None,  # {'jsc': ..., 'voc': ..., 'vmpp': ..., 'jmpp': ...}
    anchors_true: dict = None,
) -> dict:
    """
    Compute complete metric suite for a batch.

    Returns:
        dict with all metrics needed for paper tables.
    """
    metrics = {}

    # Curve R²
    r2_stats = curve_r2_weighted(J_pred, J_true)
    metrics['r2_mean'] = r2_stats['mean']
    metrics['r2_median'] = r2_stats['median']
    metrics['r2_std'] = r2_stats['std']
    metrics['r2_p5'] = r2_stats['p5']
    metrics['r2_p95'] = r2_stats['p95']

    # Curve MAPE
    mape_stats = curve_mape_safe(J_pred, J_true)
    metrics['mape_mean'] = mape_stats['mape_mean']
    metrics['mape_median'] = mape_stats['mape_median']

    # Anchor metrics (if provided)
    if anchors_pred is not None and anchors_true is not None:
        for key in ['jsc', 'voc', 'vmpp', 'jmpp']:
            if key in anchors_pred and key in anchors_true:
                mae = torch.abs(anchors_pred[key] - anchors_true[key]).mean().item()
                metrics[f'{key}_mae'] = mae

        # Fill factor
        ff_pred = (anchors_pred['vmpp'] * anchors_pred['jmpp']) / \
                  (anchors_pred['voc'] * anchors_pred['jsc'] + 1e-10)
        ff_true = (anchors_true['vmpp'] * anchors_true['jmpp']) / \
                  (anchors_true['voc'] * anchors_true['jsc'] + 1e-10)

        ff_mape = torch.abs(ff_pred - ff_true) / (ff_true + 1e-10) * 100
        metrics['ff_mape'] = ff_mape.mean().item()

    return metrics
```

### 1.2 Create `run_manifest.yaml`

```yaml
# Experiment Run Manifest for ICML Paper
# Run with: python run_experiments.py --manifest run_manifest.yaml

meta:
  project: "PINN-Coupled-PDE-Solver"
  target: "ICML 2025"
  primary_metric: "r2_mean"
  seeds: [42, 123, 456]
  output_base: "outputs/icml_experiments"

# Tier 0: Must-have ablations (main paper)
tier0:
  - id: "T0-1-main"
    name: "Main Model (Full)"
    config:
      model: "UnifiedSplitSplineNet"
      use_split: true
      use_anchors: true
      use_projection: true
      use_physics_features: true
      ctrl_points: 4
      continuity_weight: 0.1
    description: "Baseline: full model with all components"

  - id: "T0-2-no-split"
    name: "No Split (Single Spline)"
    config:
      model: "UnifiedSplitSplineNet"
      use_split: false  # Single spline over [0, Voc]
      use_anchors: true
      use_projection: true
      use_physics_features: true
    description: "Ablation: remove split at MPP"

  - id: "T0-3-no-anchors"
    name: "No Anchors (Direct 45-out)"
    config:
      model: "DirectCurveNet"  # Direct 45-point output
      use_anchors: false
      use_projection: false  # Can't project without anchors
      use_physics_features: true
    description: "Ablation: remove anchor-based reconstruction"

  - id: "T0-4-no-projection"
    name: "No Physics Projection"
    config:
      model: "UnifiedSplitSplineNet"
      use_split: true
      use_anchors: true
      use_projection: false  # Disable Π_C
      use_physics_features: true
    description: "Ablation: remove hard constraint projection"

  - id: "T0-5-no-physics-features"
    name: "Raw 31 Features Only"
    config:
      model: "UnifiedSplitSplineNet"
      use_split: true
      use_anchors: true
      use_projection: true
      use_physics_features: false  # Only raw 31 params
    description: "Ablation: remove engineered physics features"

  - id: "T0-6-cvae-baseline"
    name: "CVAE Baseline"
    config:
      model: "ConditionalVAE"
      latent_dim: 16
      beta: 0.001
    description: "Baseline: generative approach"

  - id: "T0-7-mlp-baseline"
    name: "Direct MLP Baseline"
    config:
      model: "DirectMLP"
      hidden_dims: [256, 128]
      output_dim: 45
    description: "Baseline: simple feedforward"

# Tier 1: Sweeps (appendix)
tier1:
  control_points_sweep:
    base_config:
      model: "UnifiedSplitSplineNet"
      use_split: true
      use_anchors: true
      use_projection: true
    sweep_param: "ctrl_points"
    values: [2, 4, 6, 8]

  continuity_sweep:
    base_config:
      model: "UnifiedSplitSplineNet"
      use_split: true
      ctrl_points: 4
    sweep_param: "continuity_weight"
    values: [0.0, 0.01, 0.1, 0.5, 1.0]

  feature_count_sweep:
    base_config:
      model: "UnifiedSplitSplineNet"
      use_split: true
    sweep_param: "n_physics_features"
    values: [0, 2, 4, 8, 16, 71]  # top-m after selection
```

### 1.3 Create `run_experiments.py`

```python
#!/usr/bin/env python3
"""
Experiment runner for ICML paper.
Reads manifest, launches jobs, collects results.
"""
import argparse
import yaml
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
from typing import List, Dict

def load_manifest(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_run_commands(manifest: dict) -> List[Dict]:
    """Generate all experiment configurations."""
    runs = []
    seeds = manifest['meta']['seeds']
    output_base = manifest['meta']['output_base']

    # Tier 0 experiments
    for exp in manifest.get('tier0', []):
        for seed in seeds:
            run = {
                'exp_id': exp['id'],
                'seed': seed,
                'name': exp['name'],
                'config': exp['config'],
                'output_dir': f"{output_base}/{exp['id']}/seed_{seed}",
                'description': exp.get('description', ''),
            }
            runs.append(run)

    # Tier 1 sweeps
    for sweep_name, sweep_config in manifest.get('tier1', {}).items():
        base = sweep_config['base_config']
        param = sweep_config['sweep_param']
        for value in sweep_config['values']:
            for seed in seeds:
                config = base.copy()
                config[param] = value
                exp_id = f"T1-{sweep_name}-{param}_{value}"
                run = {
                    'exp_id': exp_id,
                    'seed': seed,
                    'name': f"{sweep_name}: {param}={value}",
                    'config': config,
                    'output_dir': f"{output_base}/{exp_id}/seed_{seed}",
                }
                runs.append(run)

    return runs

def run_experiment(run: Dict, dry_run: bool = False) -> Dict:
    """Execute a single experiment run."""
    output_dir = Path(run['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(run, f, indent=2)

    # Build command
    cmd = [
        'python', 'train.py',
        '--output', str(output_dir),
        '--seed', str(run['seed']),
    ]

    # Add config flags
    config = run['config']
    if config.get('model') == 'UnifiedSplitSplineNet':
        cmd.append('--train-curves')
        if not config.get('use_split', True):
            cmd.append('--no-split')
        if not config.get('use_projection', True):
            cmd.append('--no-physics-projection')
        if not config.get('use_physics_features', True):
            cmd.append('--no-physics-features')
        if 'ctrl_points' in config:
            cmd.extend(['--ctrl-points', str(config['ctrl_points'])])
        if 'continuity_weight' in config:
            cmd.extend(['--continuity-weight', str(config['continuity_weight'])])

    elif config.get('model') == 'ConditionalVAE':
        cmd.append('--train-cvae')
        if 'beta' in config:
            cmd.extend(['--cvae-beta', str(config['beta'])])

    elif config.get('model') == 'DirectMLP':
        cmd.append('--direct-mlp')

    elif config.get('model') == 'DirectCurveNet':
        cmd.append('--direct-curve')

    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return {'status': 'dry_run', 'cmd': cmd}

    # Execute
    print(f"[RUNNING] {run['exp_id']} seed={run['seed']}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        'status': 'success' if result.returncode == 0 else 'failed',
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }

def collect_results(output_base: str) -> pd.DataFrame:
    """Collect all results into a single DataFrame."""
    results = []

    for exp_dir in Path(output_base).glob('*/seed_*'):
        metrics_path = exp_dir / 'metrics.json'
        config_path = exp_dir / 'config.json'

        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(config_path) as f:
            config = json.load(f)

        row = {
            'exp_id': config['exp_id'],
            'seed': config['seed'],
            'name': config['name'],
            **metrics,
        }
        results.append(row)

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='run_manifest.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tier', choices=['0', '1', 'all'], default='all')
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    output_base = manifest['meta']['output_base']

    if args.collect_only:
        df = collect_results(output_base)
        df.to_csv(f'{output_base}/results.csv', index=False)
        print(f"Collected {len(df)} results to {output_base}/results.csv")
        return

    runs = generate_run_commands(manifest)

    # Filter by tier
    if args.tier == '0':
        runs = [r for r in runs if r['exp_id'].startswith('T0')]
    elif args.tier == '1':
        runs = [r for r in runs if r['exp_id'].startswith('T1')]

    print(f"Total runs: {len(runs)}")

    for run in runs:
        result = run_experiment(run, dry_run=args.dry_run)
        if result['status'] == 'failed':
            print(f"[FAILED] {run['exp_id']}: {result.get('stderr', '')[:200]}")

    if not args.dry_run:
        df = collect_results(output_base)
        df.to_csv(f'{output_base}/results.csv', index=False)
        print(f"\nResults saved to {output_base}/results.csv")

if __name__ == '__main__':
    main()
```

### 1.4 Create `plot_paper_figs.py`

```python
#!/usr/bin/env python3
"""
Generate all figures for ICML paper from experiment results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE_SINGLE = (4, 3)
FIGSIZE_WIDE = (8, 3)
FIGSIZE_SQUARE = (4, 4)

def load_results(results_path: str) -> pd.DataFrame:
    return pd.read_csv(results_path)

def fig1_r2_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Figure 1: R² distribution (violin plot)
    Main model vs key baselines.
    """
    # Filter to relevant experiments
    experiments = ['T0-1-main', 'T0-2-no-split', 'T0-6-cvae-baseline', 'T0-7-mlp-baseline']
    df_plot = df[df['exp_id'].isin(experiments)].copy()

    # Map to display names
    name_map = {
        'T0-1-main': 'Split-Spline\n(Ours)',
        'T0-2-no-split': 'Single Spline',
        'T0-6-cvae-baseline': 'CVAE',
        'T0-7-mlp-baseline': 'Direct MLP',
    }
    df_plot['Model'] = df_plot['exp_id'].map(name_map)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Violin plot of R² distributions
    # NOTE: This requires per-curve R² values stored in results
    # For now, use mean ± std approximation
    sns.barplot(data=df_plot, x='Model', y='r2_mean', ax=ax, capsize=0.1,
                errorbar=('ci', 95), palette='Set2')

    ax.set_ylabel('Curve $R^2$')
    ax.set_xlabel('')
    ax.set_ylim(0.9, 1.0)
    ax.axhline(0.99, color='gray', linestyle='--', alpha=0.5, label='Target')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_r2_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("Generated fig1_r2_distribution.pdf")

def fig2_jv_overlays(curves_pred, curves_true, V_grid, output_dir: Path, n_samples=9):
    """
    Figure 2: J-V curve overlays (3x3 grid)
    True vs predicted with grid point markers.
    """
    fig, axes = plt.subplots(3, 3, figsize=(8, 7))
    axes = axes.flatten()

    indices = np.random.choice(len(curves_pred), n_samples, replace=False)

    for ax, idx in zip(axes, indices):
        ax.plot(V_grid, curves_true[idx], 'b-', label='True', linewidth=1.5)
        ax.plot(V_grid, curves_pred[idx], 'r--', label='Predicted', linewidth=1.5)
        ax.scatter(V_grid[::5], curves_true[idx][::5], c='blue', s=10, zorder=5)
        ax.scatter(V_grid[::5], curves_pred[idx][::5], c='red', s=10, marker='x', zorder=5)

        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('J (mA/cm²)')
        ax.set_xlim(0, 1.4)

    axes[0].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_jv_overlays.pdf', bbox_inches='tight')
    plt.close()
    print("Generated fig2_jv_overlays.pdf")

def fig3_ablation_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Figure 3: Ablation heatmap
    Rows = ablation configs, Columns = metrics
    """
    # Filter to Tier 0 experiments
    df_t0 = df[df['exp_id'].str.startswith('T0')].copy()

    # Aggregate across seeds
    agg = df_t0.groupby('exp_id').agg({
        'r2_mean': 'mean',
        'mape_mean': 'mean',
        'ff_mape': 'mean',
        'violations_per_1000': 'mean',
    }).reset_index()

    # Reorder
    order = ['T0-1-main', 'T0-2-no-split', 'T0-3-no-anchors', 'T0-4-no-projection',
             'T0-5-no-physics-features', 'T0-6-cvae-baseline', 'T0-7-mlp-baseline']
    agg = agg.set_index('exp_id').loc[order].reset_index()

    # Create heatmap data
    metrics = ['r2_mean', 'mape_mean', 'ff_mape', 'violations_per_1000']
    metric_labels = ['$R^2$', 'Curve MAPE', 'FF MAPE', 'Violations/1k']

    # Normalize for visualization (higher is better for R², lower for others)
    heatmap_data = agg[metrics].values.T

    fig, ax = plt.subplots(figsize=(8, 4))

    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn')

    # Labels
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['Main', 'No Split', 'No Anchors', 'No Proj.',
                        'No Phys.', 'CVAE', 'MLP'], rotation=45, ha='right')
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels)

    # Add values
    for i in range(len(metric_labels)):
        for j in range(len(order)):
            val = heatmap_data[i, j]
            fmt = '.3f' if i == 0 else '.1f'
            ax.text(j, i, f'{val:{fmt}}', ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_ablation_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("Generated fig3_ablation_heatmap.pdf")

def fig4_violation_learning_curve(violation_logs: dict, output_dir: Path):
    """
    Figure 4: Constraint violation learning curve
    Pre-projection violation rate vs epoch.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for exp_name, log in violation_logs.items():
        epochs = log['epochs']
        violations = log['violation_rate']
        ax.semilogy(epochs, violations, label=exp_name)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pre-projection Violation Rate')
    ax.set_ylim(1e-4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_violation_curve.pdf', bbox_inches='tight')
    plt.close()
    print("Generated fig4_violation_curve.pdf")

def fig5_method_schematic(output_dir: Path):
    """
    Figure 5: Method schematic (placeholder - hand-draw this)
    """
    print("NOTE: fig5_method_schematic.pdf should be hand-drawn")
    print("  Contents: Input → Feature Selection → Backbone → Heads → Π_C → Split PCHIP → Curve")

def table1_baselines(df: pd.DataFrame, output_dir: Path):
    """
    Table 1: Baseline comparison (mean ± std)
    """
    experiments = ['T0-1-main', 'T0-6-cvae-baseline', 'T0-7-mlp-baseline']
    df_filtered = df[df['exp_id'].isin(experiments)]

    # Aggregate
    agg = df_filtered.groupby('exp_id').agg({
        'r2_mean': ['mean', 'std'],
        'mape_mean': ['mean', 'std'],
        'ff_mape': ['mean', 'std'],
        'violations_per_1000': ['mean', 'std'],
        'inference_ms': ['mean', 'std'],
    })

    # Format as LaTeX
    latex = agg.to_latex(float_format='%.3f')

    with open(output_dir / 'table1_baselines.tex', 'w') as f:
        f.write(latex)
    print("Generated table1_baselines.tex")

def table2_ablations(df: pd.DataFrame, output_dir: Path):
    """
    Table 2: Ablation results
    """
    experiments = ['T0-1-main', 'T0-2-no-split', 'T0-3-no-anchors',
                   'T0-4-no-projection', 'T0-5-no-physics-features']
    df_filtered = df[df['exp_id'].isin(experiments)]

    agg = df_filtered.groupby('exp_id').agg({
        'r2_mean': ['mean', 'std'],
        'ff_mape': ['mean', 'std'],
        'violations_per_1000': ['mean', 'std'],
    })

    latex = agg.to_latex(float_format='%.3f')

    with open(output_dir / 'table2_ablations.tex', 'w') as f:
        f.write(latex)
    print("Generated table2_ablations.tex")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='outputs/icml_experiments/results.csv')
    parser.add_argument('--output', default='figures')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    df = load_results(args.results)

    print(f"Loaded {len(df)} results")
    print(f"Experiments: {df['exp_id'].unique()}")

    # Generate figures
    fig1_r2_distribution(df, output_dir)
    fig3_ablation_heatmap(df, output_dir)
    table1_baselines(df, output_dir)
    table2_ablations(df, output_dir)
    fig5_method_schematic(output_dir)

    # Note: fig2 and fig4 require additional data
    print("\nNOTE: fig2_jv_overlays and fig4_violation_curve require additional data files")

if __name__ == '__main__':
    main()
```

### 1.5 Modify `train.py` - Add Missing Flags

Add these command-line arguments:

```python
# In argument parser section:
parser.add_argument('--no-split', action='store_true',
                    help='Use single spline instead of split at MPP')
parser.add_argument('--no-physics-projection', action='store_true',
                    help='Disable hard physics constraint projection')
parser.add_argument('--no-physics-features', action='store_true',
                    help='Use only raw 31 parameters, no engineered features')
parser.add_argument('--direct-mlp', action='store_true',
                    help='Train simple direct MLP baseline')
parser.add_argument('--ctrl-points', type=int, default=4,
                    help='Number of control points per region')
parser.add_argument('--continuity-weight', type=float, default=0.1,
                    help='Weight for C1 continuity loss')
parser.add_argument('--n-physics-features', type=int, default=None,
                    help='Number of top physics features to keep (None=auto)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
```

### 1.6 Create Baseline Adapters

#### `baselines/direct_mlp.py`

```python
"""Direct MLP baseline - no physics structure."""
import torch
import torch.nn as nn

class DirectMLPBaseline(nn.Module):
    """Simple MLP: 31 inputs → 45 outputs."""

    def __init__(self, input_dim=31, hidden_dims=[256, 128], output_dim=45):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

---

## 2. Run Manifest Summary

### Tier 0 (Main Paper) - 21 runs total

| ID | Configuration | Seeds | Purpose |
|----|--------------|-------|---------|
| T0-1 | Main model (full) | 3 | Baseline |
| T0-2 | No split | 3 | Justify MPP split |
| T0-3 | No anchors | 3 | Justify anchor prediction |
| T0-4 | No projection | 3 | Justify hard constraints |
| T0-5 | No physics features | 3 | Justify feature engineering |
| T0-6 | CVAE baseline | 3 | Compare vs generative |
| T0-7 | Direct MLP | 3 | Compare vs black-box |

### Tier 1 (Appendix) - 45 runs total

| Sweep | Values | Runs |
|-------|--------|------|
| Control points K | {2, 4, 6, 8} | 12 |
| Continuity λ | {0, 0.01, 0.1, 0.5, 1.0} | 15 |
| Feature count m | {0, 2, 4, 8, 16, 71} | 18 |

### Total: 66 runs

---

## 3. Figure Specifications

### Main Paper (5 figures)

| Fig | Name | Type | Data Source |
|-----|------|------|-------------|
| 1 | Method Schematic | Hand-drawn | N/A |
| 2 | R² Distribution | Violin | T0-1,2,6,7 r2_per_curve |
| 3 | J-V Overlays | Line plots | T0-1 predictions |
| 4 | Ablation Heatmap | Heatmap | T0-* aggregated |
| 5 | Violation Curve | Line (log y) | T0-1 violation logs |

### Main Paper (2 tables)

| Table | Name | Columns |
|-------|------|---------|
| 1 | Baseline Comparison | R², MAPE, FF MAPE, Violations, Inference ms |
| 2 | Ablation Results | R², FF MAPE, Violations |

### Appendix (8 figures)

1. Voltage grid visualization
2. Feature selection stability
3. K sweep results
4. λ_cont sweep results
5. Metric comparison (weighted vs unweighted)
6. Error vs voltage
7. CVAE β sweep
8. Feature importance (SHAP)

---

## 4. Quality Checks

### Pre-Run Assertions

```python
# In train.py or a test file:

def assert_no_leakage():
    """Verify feature selection only uses train indices."""
    # Load feature mask
    with open('outputs/exp/feature_mask.json') as f:
        mask_info = json.load(f)

    # Verify train hash matches
    assert mask_info['train_indices_hash'] == compute_hash(train_indices)

def assert_monotonicity():
    """Verify reconstructed curves are monotone."""
    for curve in predicted_curves:
        diffs = np.diff(curve)
        assert np.all(diffs <= 1e-6), "Curve not monotone decreasing"

def assert_knot_ordering():
    """Verify knot voltages increasing, currents decreasing."""
    for v_knots, j_knots in zip(all_v_knots, all_j_knots):
        assert np.all(np.diff(v_knots) > 0), "V knots not increasing"
        assert np.all(np.diff(j_knots) <= 1e-6), "J knots not decreasing"

def assert_weighted_metrics():
    """Verify evaluation uses ΔV-weighted metrics."""
    # Check that reported R² matches weighted computation
    r2_reported = metrics['r2_mean']
    r2_computed = curve_r2_weighted(J_pred, J_true)['mean']
    assert abs(r2_reported - r2_computed) < 1e-6
```

### Post-Run Validation

```bash
# Run after all experiments complete:
python validate_results.py --results outputs/icml_experiments/results.csv

# Checks:
# - All 66 runs completed
# - No NaN/inf in metrics
# - Main model R² > 0.99
# - No monotonicity violations in main model
# - Feature selection stable (>80% overlap across seeds)
```

---

## 5. Execution Order

1. **Implement code changes** (§1.1-1.6)
2. **Run Tier 0** (21 runs, ~24 GPU-hours)
   ```bash
   python run_experiments.py --manifest run_manifest.yaml --tier 0
   ```
3. **Generate main figures** (verify story holds)
   ```bash
   python plot_paper_figs.py --results outputs/icml_experiments/results.csv
   ```
4. **If Tier 0 results good, run Tier 1** (45 runs, ~48 GPU-hours)
   ```bash
   python run_experiments.py --manifest run_manifest.yaml --tier 1
   ```
5. **Generate appendix figures**
6. **Run quality checks**
7. **Write paper using generated figures/tables**

---

## 6. Expected Results (Sanity Check)

Based on existing runs:

| Model | Expected R² | Expected FF MAPE |
|-------|-------------|------------------|
| Main (T0-1) | >0.99 | <2% |
| No split (T0-2) | ~0.97 | ~5% |
| No projection (T0-4) | ~0.98 | ~3% (+ violations) |
| CVAE (T0-6) | ~0.95 | ~8% |
| Direct MLP (T0-7) | ~0.90 | ~15% |

If results deviate significantly, investigate before proceeding.

---

## Appendix: SLURM Job Template

```bash
#!/bin/bash
#SBATCH --job-name=pinn_icml
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-20  # For Tier 0

# Load environment
source ~/.bashrc
conda activate pinn

# Run experiment based on array index
python run_experiments.py \
    --manifest run_manifest.yaml \
    --tier 0 \
    --run-index $SLURM_ARRAY_TASK_ID
```
