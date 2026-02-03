# Implementation Checklist for ICML Paper

**Quick Reference**: What to implement, in what order.

---

## Phase 1: Core Code Changes (Before Running Experiments)

### ☐ 1.1 Create `metrics_curve.py`
- [ ] `compute_delta_v()` - returns 45-length ΔV weights
- [ ] `curve_r2_weighted()` - per-curve R² with ΔV weighting
- [ ] `curve_mape_safe()` - clipped-denominator MAPE (eps=0.1)
- [ ] `compute_all_metrics()` - wrapper for all metrics

### ☐ 1.2 Add CLI Flags to `train.py`
- [ ] `--no-split` - single spline ablation
- [ ] `--no-physics-projection` - disable Π_C
- [ ] `--no-physics-features` - raw 31 only
- [ ] `--direct-mlp` - baseline flag
- [ ] `--ctrl-points N` - control point count
- [ ] `--continuity-weight W` - λ_cont
- [ ] `--n-physics-features M` - feature count
- [ ] `--seed S` - random seed

### ☐ 1.3 Create Baseline Adapters
- [ ] `baselines/direct_mlp.py` - 31→256→128→45 MLP
- [ ] Verify `models/cvae.py` outputs 45 points on same grid

### ☐ 1.4 Update Loss/Evaluation
- [ ] Training loss uses ΔV-weighted MSE
- [ ] Evaluation uses `curve_r2_weighted()` from metrics_curve.py
- [ ] Log per-curve R² distribution (not just mean)

### ☐ 1.5 Create Experiment Infrastructure
- [ ] `run_manifest.yaml` - experiment configs
- [ ] `run_experiments.py` - launcher script
- [ ] `plot_paper_figs.py` - figure generator

---

## Phase 2: Run Experiments

### ☐ 2.1 Tier 0 (Must-Have) - 21 runs
```bash
python run_experiments.py --manifest run_manifest.yaml --tier 0
```

| ID | Config | Seeds |
|----|--------|-------|
| T0-1 | Main (full) | 42, 123, 456 |
| T0-2 | No split | 42, 123, 456 |
| T0-3 | No anchors | 42, 123, 456 |
| T0-4 | No projection | 42, 123, 456 |
| T0-5 | No physics features | 42, 123, 456 |
| T0-6 | CVAE baseline | 42, 123, 456 |
| T0-7 | Direct MLP | 42, 123, 456 |

### ☐ 2.2 Verify Results Before Tier 1
- [ ] Main model R² > 0.99
- [ ] No monotonicity violations
- [ ] FF MAPE < 2%
- [ ] Clear separation from baselines

### ☐ 2.3 Tier 1 (Sweeps) - 45 runs
```bash
python run_experiments.py --manifest run_manifest.yaml --tier 1
```

| Sweep | Values | Runs |
|-------|--------|------|
| K (ctrl points) | 2, 4, 6, 8 | 12 |
| λ_cont | 0, 0.01, 0.1, 0.5, 1.0 | 15 |
| Feature count | 0, 2, 4, 8, 16, 71 | 18 |

---

## Phase 3: Generate Figures & Tables

### ☐ 3.1 Main Paper Figures (5)
```bash
python plot_paper_figs.py --results outputs/icml_experiments/results.csv
```

1. [ ] `fig1_method_schematic.pdf` - hand-draw
2. [ ] `fig2_r2_distribution.pdf` - violin plot
3. [ ] `fig3_jv_overlays.pdf` - 3×3 curve grid
4. [ ] `fig4_ablation_heatmap.pdf` - metrics heatmap
5. [ ] `fig5_violation_curve.pdf` - learning curve

### ☐ 3.2 Main Paper Tables (2)
1. [ ] `table1_baselines.tex` - vs CVAE, MLP
2. [ ] `table2_ablations.tex` - structural ablations

### ☐ 3.3 Appendix Figures (8)
1. [ ] Voltage grid visualization
2. [ ] Feature selection stability (10 seeds)
3. [ ] K sweep plot
4. [ ] λ_cont sweep plot
5. [ ] Weighted vs unweighted metrics
6. [ ] Error vs voltage position
7. [ ] CVAE β sweep
8. [ ] SHAP / feature importance

---

## Phase 4: Quality Checks

### ☐ 4.1 Pre-Run Assertions
- [ ] Feature selection uses train split only
- [ ] Logged train indices hash in feature_mask.json
- [ ] No ground truth leakage in projection

### ☐ 4.2 Post-Run Validation
- [ ] All 66 runs completed successfully
- [ ] No NaN/inf in any metrics
- [ ] Monotonicity violation rate = 0 for main model
- [ ] Results match expected ranges (see EXPERIMENT_PLAN.md)

### ☐ 4.3 Reproducibility
- [ ] All random seeds logged
- [ ] HPO configs saved to JSON
- [ ] Feature masks saved with split hash
- [ ] Full results in `results.csv`

---

## Quick Commands

```bash
# Run Tier 0 experiments
python run_experiments.py --manifest run_manifest.yaml --tier 0

# Run Tier 1 experiments
python run_experiments.py --manifest run_manifest.yaml --tier 1

# Collect results
python run_experiments.py --manifest run_manifest.yaml --collect-only

# Generate figures
python plot_paper_figs.py --results outputs/icml_experiments/results.csv --output figures/

# Feature selection stability (10 seeds)
for seed in 1 2 3 4 5 6 7 8 9 10; do
    python train.py --only-feature-selection --seed $seed
done

# SHAP analysis
python analysis/shap_analysis.py --model outputs/T0-1-main/seed_42/model.pt
```

---

## Expected Timeline

| Phase | Duration | GPU-hours |
|-------|----------|-----------|
| Code changes | 1 day | 0 |
| Tier 0 runs | 1 day | ~24 |
| Verify + iterate | 0.5 day | ~6 |
| Tier 1 runs | 2 days | ~48 |
| Figure generation | 0.5 day | 0 |
| Quality checks | 0.5 day | 0 |
| **Total** | **~5-6 days** | **~78** |

---

## Files to Create/Modify

### New Files
- `metrics_curve.py`
- `run_manifest.yaml`
- `run_experiments.py`
- `plot_paper_figs.py`
- `baselines/direct_mlp.py`
- `validate_results.py`

### Modified Files
- `train.py` (add CLI flags)
- `models/reconstruction.py` (use ΔV-weighted loss)

### Output Files
- `outputs/icml_experiments/results.csv`
- `figures/*.pdf`
- `tables/*.tex`
