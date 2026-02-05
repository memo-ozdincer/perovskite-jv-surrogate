#!/usr/bin/env python3
"""
Master Experiment Runner for ICML Paper
========================================

This script orchestrates all experiments defined in ablation_configs.yaml.
It can run experiments sequentially or generate SLURM array job commands.

Usage:
    # Run all Tier 0 experiments
    python run_all_experiments.py --tier 0

    # Run specific experiment
    python run_all_experiments.py --exp-id T0-1-main --seed 42

    # Dry run (show commands without executing)
    python run_all_experiments.py --tier 0 --dry-run

    # Generate SLURM array script
    python run_all_experiments.py --tier 0 --slurm-array

    # Collect results after runs complete
    python run_all_experiments.py --collect-only

    # Run all experiments and generate figures
    python run_all_experiments.py --all --generate-figures
"""

import argparse
import yaml
import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


@dataclass
class ExperimentRun:
    """Single experiment run configuration."""
    exp_id: str
    seed: int
    name: str
    config: dict
    output_dir: str
    description: str = ""
    tier: str = "0"


def load_config(config_path: str = "ablation_configs.yaml") -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_tier0_runs(manifest: dict) -> List[ExperimentRun]:
    """Generate all Tier 0 experiment runs."""
    runs = []
    seeds = manifest['meta']['seeds']
    output_base = manifest['meta']['output_base']

    for exp in manifest.get('tier0', []):
        for seed in seeds:
            run = ExperimentRun(
                exp_id=exp['id'],
                seed=seed,
                name=exp['name'],
                config=exp['config'],
                output_dir=f"{output_base}/{exp['id']}/seed_{seed}",
                description=exp.get('description', ''),
                tier="0"
            )
            runs.append(run)

    return runs


def generate_tier1_runs(manifest: dict) -> List[ExperimentRun]:
    """Generate all Tier 1 sweep runs."""
    runs = []
    seeds = manifest['meta']['seeds']
    output_base = manifest['meta']['output_base']

    for sweep_name, sweep_config in manifest.get('tier1', {}).items():
        base = sweep_config['base_config'].copy()
        param = sweep_config['sweep_param']

        for value in sweep_config['values']:
            config = base.copy()
            config[param] = value
            exp_id = f"T1-{sweep_name}-{param}_{value}"

            for seed in seeds:
                run = ExperimentRun(
                    exp_id=exp_id,
                    seed=seed,
                    name=f"{sweep_config['name']}: {param}={value}",
                    config=config,
                    output_dir=f"{output_base}/{exp_id}/seed_{seed}",
                    description=sweep_config.get('description', ''),
                    tier="1"
                )
                runs.append(run)

    return runs


def generate_tier2_runs(manifest: dict) -> List[ExperimentRun]:
    """Generate Tier 2 analysis runs."""
    runs = []
    seeds = manifest['meta']['seeds']
    output_base = manifest['meta']['output_base']

    tier2 = manifest.get('tier2', {})

    # Jacobian analysis
    if 'jacobian_analysis' in tier2:
        config = tier2['jacobian_analysis']['config']
        runs.append(ExperimentRun(
            exp_id="T2-jacobian",
            seed=seeds[0],  # Only one seed needed
            name=tier2['jacobian_analysis']['name'],
            config=config,
            output_dir=f"{output_base}/T2-jacobian",
            description=tier2['jacobian_analysis']['description'],
            tier="2"
        ))

    # Parameter sensitivity
    if 'parameter_sensitivity' in tier2:
        config = tier2['parameter_sensitivity']['config']
        runs.append(ExperimentRun(
            exp_id="T2-sensitivity",
            seed=seeds[0],
            name=tier2['parameter_sensitivity']['name'],
            config=config,
            output_dir=f"{output_base}/T2-sensitivity",
            description=tier2['parameter_sensitivity']['description'],
            tier="2"
        ))

    return runs


def config_to_cli_args(config: dict) -> List[str]:
    """Convert configuration dict to CLI arguments."""
    args = []

    # Boolean flags
    bool_flags = {
        'train_curves': '--train-curves',
        'train_cvae': '--train-cvae',
        'use_split': None,  # Default is True, so we use --no-split for False
        'use_anchors': None,
        'use_projection': None,
        'use_physics_features': None,
        'drop_weak_features': '--drop-weak-features',
        'drop_multicollinear': '--drop-multicollinear',
        'oracle_voc': '--oracle-voc',
        'no_hpo': '--no-hpo',
        'direct_curve': '--direct-curve',
        'direct_mlp': '--direct-mlp',
        'compute_jacobian': '--compute-jacobian',
        'sensitivity_analysis': '--sensitivity-analysis',
    }

    # Negative flags (when False, add the --no-* flag)
    negative_flags = {
        'use_split': '--no-split',
        'use_projection': '--no-physics-projection',
        'use_physics_features': '--no-physics-features',
    }

    for key, flag in bool_flags.items():
        if key in config:
            if config[key] and flag:
                args.append(flag)
            elif not config[key] and key in negative_flags:
                args.append(negative_flags[key])

    # Value flags
    value_flags = {
        'ctrl_points': '--ctrl-points',
        'continuity_weight': '--continuity-weight',
        'cvae_latent_dim': '--cvae-latent-dim',
        'cvae_beta': '--cvae-beta',
        'n_physics_features': '--n-physics-features',
    }

    for key, flag in value_flags.items():
        if key in config and config[key] is not None:
            args.extend([flag, str(config[key])])

    # Hidden dims (special handling for lists)
    if 'hidden_dims' in config:
        args.extend(['--hidden-dims'] + [str(d) for d in config['hidden_dims']])

    return args


def build_train_command(
    run: ExperimentRun,
    params_file: str = "preprocessed/LHS_parameters_m_clean.txt",
    iv_file: str = "preprocessed/IV_m_clean.txt",
    params_extra: str = None,
    iv_extra: str = None,
    device: str = "cuda"
) -> List[str]:
    """Build the full training command for an experiment."""
    cmd = [
        "python", "train.py",
        "--params", params_file,
        "--iv", iv_file,
        "--output", run.output_dir,
        "--device", device,
        "--seed", str(run.seed),
    ]

    # Add extra data files if provided
    if params_extra:
        cmd.extend(["--params-extra", params_extra])
    if iv_extra:
        cmd.extend(["--iv-extra", iv_extra])

    # Add config-specific arguments
    cmd.extend(config_to_cli_args(run.config))

    return cmd


def run_experiment(
    run: ExperimentRun,
    dry_run: bool = False,
    verbose: bool = True,
    **kwargs
) -> Dict:
    """Execute a single experiment run."""
    cmd = build_train_command(run, **kwargs)

    # Create output directory
    output_dir = Path(run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config_path = output_dir / "run_config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(run), f, indent=2)

    if dry_run:
        print(f"[DRY RUN] {run.exp_id} seed={run.seed}")
        print(f"  Command: {' '.join(cmd)}")
        return {'status': 'dry_run', 'cmd': cmd}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {run.exp_id} (seed={run.seed})")
        print(f"Output: {run.output_dir}")
        print(f"{'='*60}")

    # Execute
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    elapsed = time.time() - start_time

    # Save stdout/stderr
    (output_dir / "stdout.txt").write_text(result.stdout)
    (output_dir / "stderr.txt").write_text(result.stderr)

    status = 'success' if result.returncode == 0 else 'failed'

    # Save run summary
    summary = {
        'exp_id': run.exp_id,
        'seed': run.seed,
        'status': status,
        'returncode': result.returncode,
        'elapsed_seconds': elapsed,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / "run_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"  Status: {status}")
        print(f"  Time: {elapsed:.1f}s")

    if status == 'failed' and verbose:
        print(f"  Error: {result.stderr[:500]}")

    return summary


def collect_results(output_base: str) -> pd.DataFrame:
    """Collect all experiment results into a single DataFrame."""
    results = []

    for exp_dir in Path(output_base).glob('*/seed_*'):
        metrics_path = exp_dir / 'metrics.json'
        config_path = exp_dir / 'run_config.json'
        summary_path = exp_dir / 'run_summary.json'

        if not metrics_path.exists():
            print(f"Warning: No metrics.json in {exp_dir}")
            continue

        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            with open(config_path) as f:
                config = json.load(f)
            with open(summary_path) as f:
                summary = json.load(f)

            # Flatten metrics (handle nested dicts)
            flat_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        flat_metrics[f"{k}_{k2}"] = v2
                else:
                    flat_metrics[k] = v

            row = {
                'exp_id': config['exp_id'],
                'seed': config['seed'],
                'name': config['name'],
                'tier': config.get('tier', '0'),
                'elapsed_seconds': summary.get('elapsed_seconds'),
                **flat_metrics,
            }
            results.append(row)

        except Exception as e:
            print(f"Error loading {exp_dir}: {e}")

    return pd.DataFrame(results)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across seeds (mean ± std)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['seed']]

    agg_funcs = {col: ['mean', 'std'] for col in numeric_cols}

    grouped = df.groupby('exp_id').agg(agg_funcs)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Add name and tier
    names = df.groupby('exp_id').first()[['name', 'tier']]
    result = pd.concat([names, grouped], axis=1)

    return result.reset_index()


def generate_slurm_array_script(
    runs: List[ExperimentRun],
    output_path: str = "slurm_experiment_array.sh",
    **kwargs
) -> str:
    """Generate a SLURM array job script."""
    # Write run configs to a file
    configs_path = Path(output_path).parent / "experiment_configs.json"
    configs = [asdict(r) for r in runs]
    with open(configs_path, 'w') as f:
        json.dump(configs, f, indent=2)

    script = f'''#!/bin/bash
#SBATCH --job-name=pinn_icml_experiments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err
#SBATCH --account=aip-aspuru
#SBATCH --array=0-{len(runs)-1}

# ============================================================================
# ICML Experiment Array Job
# Generated: {datetime.now().isoformat()}
# Total runs: {len(runs)}
# ============================================================================

echo "==========================================="
echo "SLURM Array Job"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "==========================================="

# Load modules
module purge
module load gcc/12.3 cuda/12.2 python/3.11

# Setup environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors"
cd $WORK_DIR
source ../venv/bin/activate

mkdir -p logs

# Run the experiment for this array task
python run_all_experiments.py \\
    --run-index $SLURM_ARRAY_TASK_ID \\
    --configs-file {configs_path}

echo ""
echo "==========================================="
echo "Task $SLURM_ARRAY_TASK_ID Complete"
echo "End time: $(date)"
echo "==========================================="
'''

    with open(output_path, 'w') as f:
        f.write(script)

    print(f"Generated SLURM array script: {output_path}")
    print(f"  Total runs: {len(runs)}")
    print(f"  Submit with: sbatch {output_path}")

    return output_path


def run_from_index(index: int, configs_file: str) -> None:
    """Run a single experiment from the configs file by index."""
    with open(configs_file) as f:
        configs = json.load(f)

    if index >= len(configs):
        print(f"Error: Index {index} out of range (max {len(configs)-1})")
        sys.exit(1)

    config = configs[index]
    run = ExperimentRun(**config)

    result = run_experiment(run, dry_run=False, verbose=True)

    if result['status'] == 'failed':
        sys.exit(1)


def print_summary(manifest: dict, tier0_runs: List, tier1_runs: List, tier2_runs: List):
    """Print experiment summary."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    print(f"\nTier 0 (Main Paper): {len(tier0_runs)} runs")
    for exp in manifest.get('tier0', []):
        print(f"  - {exp['id']}: {exp['name']}")

    print(f"\nTier 1 (Sweeps): {len(tier1_runs)} runs")
    for sweep_name in manifest.get('tier1', {}):
        n_values = len(manifest['tier1'][sweep_name]['values'])
        n_seeds = len(manifest['meta']['seeds'])
        print(f"  - {sweep_name}: {n_values} values × {n_seeds} seeds = {n_values * n_seeds} runs")

    print(f"\nTier 2 (Analysis): {len(tier2_runs)} runs")

    total = len(tier0_runs) + len(tier1_runs) + len(tier2_runs)
    print(f"\nTotal: {total} runs")
    print(f"Estimated GPU-hours: {total * 1.5:.0f}")  # ~1.5h per run


def main():
    parser = argparse.ArgumentParser(
        description="Master experiment runner for ICML paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Tier selection
    parser.add_argument('--tier', choices=['0', '1', '2', 'all'], default=None,
                        help='Run experiments from specified tier')
    parser.add_argument('--exp-id', type=str, default=None,
                        help='Run specific experiment by ID')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run specific seed (use with --exp-id)')

    # Config
    parser.add_argument('--config', default='ablation_configs.yaml',
                        help='Path to experiment configuration YAML')

    # Execution modes
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--slurm-array', action='store_true',
                        help='Generate SLURM array job script')
    parser.add_argument('--collect-only', action='store_true',
                        help='Only collect results, no new runs')
    parser.add_argument('--generate-figures', action='store_true',
                        help='Generate figures after running experiments')

    # For SLURM array execution
    parser.add_argument('--run-index', type=int, default=None,
                        help='Run single experiment by index (for SLURM array)')
    parser.add_argument('--configs-file', type=str, default=None,
                        help='Path to experiment configs JSON (for SLURM array)')

    # Data paths
    parser.add_argument('--params', default='preprocessed/LHS_parameters_m_clean.txt')
    parser.add_argument('--iv', default='preprocessed/IV_m_clean.txt')
    parser.add_argument('--params-extra', default=None)
    parser.add_argument('--iv-extra', default=None)
    parser.add_argument('--device', default='cuda')

    # Output
    parser.add_argument('--output-base', default=None,
                        help='Override output base directory')

    args = parser.parse_args()

    # Handle SLURM array execution
    if args.run_index is not None and args.configs_file:
        run_from_index(args.run_index, args.configs_file)
        return

    # Load configuration
    manifest = load_config(args.config)

    if args.output_base:
        manifest['meta']['output_base'] = args.output_base

    output_base = manifest['meta']['output_base']

    # Generate runs
    tier0_runs = generate_tier0_runs(manifest)
    tier1_runs = generate_tier1_runs(manifest)
    tier2_runs = generate_tier2_runs(manifest)

    # Print summary
    print_summary(manifest, tier0_runs, tier1_runs, tier2_runs)

    # Collect only mode
    if args.collect_only:
        print(f"\nCollecting results from {output_base}...")
        df = collect_results(output_base)
        print(f"Found {len(df)} completed runs")

        # Save raw results
        df.to_csv(f"{output_base}/results_raw.csv", index=False)

        # Aggregate across seeds
        if len(df) > 0:
            agg_df = aggregate_results(df)
            agg_df.to_csv(f"{output_base}/results_aggregated.csv", index=False)
            print(f"\nResults saved to:")
            print(f"  - {output_base}/results_raw.csv")
            print(f"  - {output_base}/results_aggregated.csv")

            # Print summary table
            print("\n" + "="*60)
            print("RESULTS SUMMARY (mean ± std)")
            print("="*60)
            for _, row in agg_df.iterrows():
                print(f"\n{row['exp_id']}: {row['name']}")
                if 'r2_mean_mean' in row:
                    print(f"  R² mean: {row['r2_mean_mean']:.4f} ± {row.get('r2_mean_std', 0):.4f}")
                if 'mape_mean_mean' in row:
                    print(f"  MAPE: {row['mape_mean_mean']:.2f}% ± {row.get('mape_mean_std', 0):.2f}%")

        return

    # Select runs based on tier
    runs = []
    if args.tier == '0' or args.tier == 'all':
        runs.extend(tier0_runs)
    if args.tier == '1' or args.tier == 'all':
        runs.extend(tier1_runs)
    if args.tier == '2' or args.tier == 'all':
        runs.extend(tier2_runs)

    # Filter by specific experiment ID
    if args.exp_id:
        runs = [r for r in runs if r.exp_id == args.exp_id]
        if args.seed:
            runs = [r for r in runs if r.seed == args.seed]

    if not runs:
        print("No runs selected. Use --tier or --exp-id to select experiments.")
        parser.print_help()
        return

    print(f"\nSelected {len(runs)} runs to execute")

    # SLURM array mode
    if args.slurm_array:
        generate_slurm_array_script(runs)
        return

    # Execute runs
    results = []
    for i, run in enumerate(runs):
        print(f"\n[{i+1}/{len(runs)}]", end="")
        result = run_experiment(
            run,
            dry_run=args.dry_run,
            params_file=args.params,
            iv_file=args.iv,
            params_extra=args.params_extra,
            iv_extra=args.iv_extra,
            device=args.device,
        )
        results.append(result)

    # Summary
    if not args.dry_run:
        n_success = sum(1 for r in results if r['status'] == 'success')
        n_failed = sum(1 for r in results if r['status'] == 'failed')
        print(f"\n{'='*60}")
        print(f"EXECUTION COMPLETE")
        print(f"  Success: {n_success}")
        print(f"  Failed: {n_failed}")
        print(f"{'='*60}")

        # Collect and save results
        df = collect_results(output_base)
        df.to_csv(f"{output_base}/results_raw.csv", index=False)

        if len(df) > 0:
            agg_df = aggregate_results(df)
            agg_df.to_csv(f"{output_base}/results_aggregated.csv", index=False)

    # Generate figures if requested
    if args.generate_figures and not args.dry_run:
        print("\nGenerating figures...")
        subprocess.run([
            "python", "generate_paper_figures.py",
            "--results", f"{output_base}/results_raw.csv",
            "--output", f"{output_base}/figures"
        ])


if __name__ == '__main__':
    main()
