# Release Manifest (GitHub + Hugging Face)

This manifest defines the publish-facing structure for the dilated-convolution Stage-2 pipeline.

## Scope

- Primary architecture: bidirectional dilated convolution (`conv + dilated`, no attention).
- `TCN` naming is retained only for backward compatibility aliases.
- Stage-1 scalar predictions (`Voc`, `Vmpp`) are external (MATLAB model), not generated from true IV curves in this repo.

## New publish-facing entrypoints

- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/slurm_dilated_conv_master_pipeline.sh`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/slurm_dilated_conv_single.sh`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/train_dilated_conv.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/dilated_collect_results.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/dilated_generate_figures.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/dilated_analysis.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/inference_tcn_dilated.py`

Compatibility files are intentionally kept and not moved/deleted.

## Required GitHub code files

- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/config.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/data.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/features.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/train_attention_tcn.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/train_dilated_conv.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/inference_tcn_dilated.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/scripts/preprocess_data.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/scripts/generate_scalar_txt.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/slurm_dilated_conv_single.sh`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/slurm_dilated_conv_master_pipeline.sh`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/dilated_collect_results.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/dilated_generate_figures.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/dilated_analysis.py`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/requirements.txt`
- `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/README.md`

## Stage-1 scalar policy

- Scalars must come from external predictions (MATLAB Stage-1).
- `scripts/generate_scalar_txt.py` now only repackages externally predicted scalar files into the expected naming convention.
- Pipelines no longer compute scalar txt files from cleaned true IV curves.

## Physics feature selection

Implemented in Stage-2 trainer and enabled by pipeline flags:

- `--physics-feature-selection`
- `--physics-weak-threshold` (default `0.30`)
- `--physics-corr-threshold` (default `0.85`)
- `--physics-max-features` (optional cap)

Current master/single pipelines pass `--physics-feature-selection`.

## Analysis policy

- Partial Jacobian placeholder logic was removed from `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/tcn_analysis.py`.
- Analysis now focuses on complete aggregate experiment statistics and ranking outputs.

## Recommended GitHub root navigation

Use this in README:

- `Quickstart (single run)`: `slurm_dilated_conv_single.sh`
- `Full ablation pipeline`: `slurm_dilated_conv_master_pipeline.sh`
- `Trainer`: `train_dilated_conv.py`
- `Inference (31 params + Voc/Vmpp)`: `inference_tcn_dilated.py`
- `Preprocessing`: `scripts/preprocess_data.py`
- `External scalar packaging`: `scripts/generate_scalar_txt.py`
- `Result aggregation`: `dilated_collect_results.py`
- `Figures/tables`: `dilated_generate_figures.py`

## Hugging Face split

### HF Dataset repo

- `raw/`:
  - `LHS_parameters_m.txt`, `LHS_parameters_m_300k.txt`
  - `IV_m.txt`, `IV_m_300k.txt`
- `processed/`:
  - cleaned params/iv files
  - externally predicted scalar files (`voc_clean_100k.txt`, `vmpp_clean_100k.txt`, etc.)
  - preprocessing stats JSON
- `README.md` dataset card:
  - provenance, filtering criteria, units, split policy, license

### HF Model repo

- `checkpoints/`:
  - champion and seed checkpoints (`best-model.ckpt`)
- `cache/`:
  - matching `atcn_param_transformer.joblib`
  - matching `atcn_scalar_transformer.joblib`
  - matching `atcn_physics_transformer.joblib` (if physics enabled)
- `metrics/`:
  - `test_stats.json`, aggregated CSV summaries
- `README.md` model card:
  - exact input contract (31 params + Voc/Vmpp)
  - output contract (8-point normalized + optional 45-point reconstruction)
  - known failure modes and intended use

## Known paper/code consistency note (no paper edits applied)

- Paper claims feature selection compresses 71 features to `m=5`.
- Code now supports train-only feature filtering with optional top-k cap.
- Current master/single release pipelines set `--physics-max-features 5` to enforce the paper-aligned `m=5` setting.
