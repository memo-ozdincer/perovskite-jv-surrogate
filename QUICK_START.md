# PINN-Coupled-PDE-Solver: Quick Start Guide for ICML Experiments

This guide provides the fastest path to running all experiments and generating publication-ready figures.

## TL;DR - One Command to Run Everything

```bash
# Submit the master pipeline job (runs all experiments + generates figures)
sbatch slurm_master_pipeline.sh
```

This will:
1. Preprocess data (quality filtering)
2. Run all Tier 0 ablations (21 experiments)
3. Run all Tier 1 sweeps (54 experiments)
4. Run physics analysis
5. Generate all figures and tables

## Quick Overview of Files

| File | Purpose |
|------|---------|
| `slurm_master_pipeline.sh` | **Main entry point** - runs everything |
| `slurm_figures_only.sh` | Regenerate figures from existing results |
| `run_all_experiments.py` | Python experiment orchestrator |
| `generate_paper_figures.py` | Figure generation from results |
| `ablation_configs.yaml` | Experiment configurations |
| `metrics_curve.py` | ΔV-weighted metrics for evaluation |
| `physics_analysis.py` | Jacobian and sensitivity analysis |

## Step-by-Step Instructions

### Option A: Run All Experiments (Recommended)

```bash
# 1. Submit the master job
sbatch slurm_master_pipeline.sh

# 2. Monitor progress
tail -f /scratch/memoozd/ts-tools-scratch/dbe/logs/master_pipeline_*.out

# 3. Results will be in:
#    outputs/icml_experiments_YYYYMMDD/
#    ├── figures/main_paper/    # 5 main figures
#    ├── figures/appendix/      # Appendix figures
#    ├── figures/tables/        # LaTeX tables
#    ├── all_results.csv        # Aggregated metrics
#    └── T0-*/seed_*/           # Individual experiment outputs
```

### Option B: Run Only Tier 0 (Main Paper Ablations)

```bash
# Faster: only 21 runs instead of 75
sbatch slurm_master_pipeline.sh --tier0-only
```

### Option C: Run Specific Experiments

```bash
# Run a single experiment
python run_all_experiments.py --exp-id T0-1-main --seed 42

# Dry run (show commands without executing)
python run_all_experiments.py --tier 0 --dry-run
```

### Option D: Only Generate Figures (from existing results)

```bash
# After experiments complete, regenerate figures
sbatch slurm_figures_only.sh /path/to/outputs/icml_experiments_YYYYMMDD
```

## Experiment Configuration

All experiments are defined in `ablation_configs.yaml`:

### Tier 0 (Main Paper) - 7 configs × 3 seeds = 21 runs

| ID | Name | What it Tests |
|----|------|--------------|
| T0-1-main | Full Model | Baseline performance |
| T0-2-no-split | No Split | Importance of MPP split |
| T0-3-no-anchors | Direct 45-out | Importance of anchor prediction |
| T0-4-no-projection | No Projection | Importance of hard constraints |
| T0-5-no-physics | Raw 31 Only | Value of physics features |
| T0-6-cvae | CVAE Baseline | vs generative approach |
| T0-7-mlp | Direct MLP | vs black-box baseline |

### Tier 1 (Appendix) - Hyperparameter sweeps

- Control points: K ∈ {2, 4, 6, 8, 10, 12}
- Continuity weight: λ ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0}

## Generated Outputs

### Figures (PDF format)

**Main Paper:**
- `fig1_method_schematic.pdf` - Architecture diagram
- `fig2_r2_distribution.pdf` - R² comparison violin plot
- `fig3_jv_overlays.pdf` - True vs predicted J-V curves (3×3 grid)
- `fig4_ablation_heatmap.pdf` - Ablation results heatmap
- `fig5_violation_curve.pdf` - Constraint violation learning curve

**Appendix:**
- `fig_a1_voltage_grid.pdf` - Non-uniform grid visualization
- `fig_a2_ctrl_points_sweep.pdf` - K sweep results
- `fig_a3_continuity_sweep.pdf` - λ sweep results
- `fig_a4_feature_importance.pdf` - Top physics features
- `fig_a5_error_vs_voltage.pdf` - Error distribution

### Tables (LaTeX format)

- `table1_baselines.tex` - Baseline comparison
- `table2_ablations.tex` - Ablation results

### Metrics

- `all_results.csv` - All experiment metrics
- `summary_statistics.csv` - Aggregated mean ± std

## New CLI Flags for Ablations

The following flags were added to `train.py`:

```bash
# Ablation flags
--seed INT              # Random seed (default: 42)
--no-split              # Single spline (no MPP split)
--no-physics-projection # Disable hard constraints
--no-physics-features   # Use only raw 31 params
--direct-mlp            # Simple MLP baseline
--hidden-dims INT...    # MLP architecture (default: 256 128)
--cvae-latent-dim INT   # CVAE latent dimension
--cvae-beta FLOAT       # CVAE KL weight
--n-physics-features INT # Limit physics features
--compute-jacobian      # Compute Jacobian sensitivity
--sensitivity-analysis  # Run parameter sensitivity
```

## Expected Results

Based on prior runs, you should see:

| Model | R² mean | FF MAPE | Violations/1k |
|-------|---------|---------|---------------|
| T0-1-main | >0.99 | <2% | <1 |
| T0-2-no-split | ~0.97 | ~5% | <5 |
| T0-3-no-anchors | ~0.95 | ~8% | >10 |
| T0-6-cvae | ~0.95 | ~8% | >20 |
| T0-7-mlp | ~0.90 | ~15% | >50 |

## Troubleshooting

### Job fails with CUDA error
```bash
# Check GPU availability
nvidia-smi
# Or use CPU
python train.py --device cpu ...
```

### Out of memory
```bash
# Reduce batch size (in train.py, default is 256)
# Or reduce HPO trials
python train.py --hpo-trials-nn 50 --hpo-trials-lgbm 100
```

### Missing results
```bash
# Rerun figure generation
python generate_paper_figures.py --results path/to/all_results.csv --output figures/
```

### Check experiment status
```bash
# Count completed experiments
find outputs/icml_experiments_* -name "metrics.json" | wc -l
```

## Runtime Estimates

| Tier | Runs | GPU-Hours | Wall Time (1 GPU) |
|------|------|-----------|-------------------|
| Tier 0 | 21 | ~24 | ~24h |
| Tier 1 | 54 | ~48 | ~48h |
| Full Pipeline | 75 | ~72 | ~72h |

With 4 GPUs in parallel: divide wall time by 4.

## Integrating Results into Paper

1. Copy `figures/main_paper/*.pdf` to your paper's figures directory
2. Copy `figures/tables/*.tex` and include with `\input{table1_baselines.tex}`
3. Use `all_results.csv` for any additional analysis

## File Structure After Running

```
outputs/icml_experiments_YYYYMMDD/
├── all_results.csv                    # All metrics
├── summary_statistics.csv             # Aggregated stats
├── timing.log                         # Runtime info
├── figures/
│   ├── main_paper/
│   │   ├── fig1_method_schematic.pdf
│   │   ├── fig2_r2_distribution.pdf
│   │   ├── fig3_jv_overlays.pdf
│   │   ├── fig4_ablation_heatmap.pdf
│   │   └── fig5_violation_curve.pdf
│   ├── appendix/
│   │   └── fig_a*.pdf
│   └── tables/
│       ├── table1_baselines.tex
│       └── table2_ablations.tex
├── analysis/
│   ├── jacobian_heatmap.pdf
│   └── sensitivity_*.pdf
├── T0-1-main/
│   ├── seed_42/
│   │   ├── metrics.json
│   │   ├── training_summary.json
│   │   └── models/*.pt
│   ├── seed_123/
│   └── seed_456/
├── T0-2-no-split/
│   └── ...
└── T1-ctrl_points-*/
    └── ...
```

## Questions?

See the full documentation in:
- `EXPERIMENT_PLAN.md` - Detailed experiment specifications
- `icml_documentation.tex` - Technical documentation
- `docs.txt` - Implementation details
