# PINN-Coupled-PDE-Solver (ICML Release)

Physics-constrained Stage-2 surrogate for perovskite J-V curve reconstruction.

This release is centered on the **bidirectional dilated convolution** model (no attention), with a modular two-stage setup:

- Stage-1: external scalar predictor (MATLAB) provides `Voc`, `Vmpp`
- Stage-2: this repo reconstructs 8-point curves (and optional 45-point PCHIP curves)

## Repository Entry Points

- Single-run pipeline: `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/slurm_dilated_conv_single.sh`
- Full ablation pipeline: `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/slurm_dilated_conv_master_pipeline.sh`
- Trainer CLI: `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/train_dilated_conv.py`
- Inference CLI (31 params + `Voc`/`Vmpp`): `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/inference_tcn_dilated.py`
- Preprocessing: `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/scripts/preprocess_data.py`
- External scalar packaging: `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/scripts/generate_scalar_txt.py`
- Result aggregation: `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/dilated_collect_results.py`
- Figure/table generation: `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/dilated_generate_figures.py`
- Aggregate analysis: `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/dilated_analysis.py`

## Quickstart

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytorch-lightning scipy seaborn pillow tqdm
```

### 2) Prepare inputs

- Parameters: 31-column file (`.csv` or `.txt`)
- External scalar files from Stage-1 (MATLAB):
  - `voc_clean_100k.txt`, `vmpp_clean_100k.txt`
  - `voc_clean_300k.txt`, `vmpp_clean_300k.txt`

Important: this repo does **not** compute publishing scalars from true IV curves.

### 3) Train Stage-2 model

```bash
python3 train_dilated_conv.py \
  --params /abs/path/LHS_parameters_m_clean.txt \
  --iv /abs/path/IV_m_clean.txt \
  --params-extra /abs/path/LHS_parameters_m_300k_clean.txt \
  --iv-extra /abs/path/IV_m_300k_clean.txt \
  --scalar-files /abs/path/voc_clean_100k.txt /abs/path/vmpp_clean_100k.txt \
  --scalar-files-extra /abs/path/voc_clean_300k.txt /abs/path/vmpp_clean_300k.txt \
  --architecture conv --no-attention --use-dilated \
  --use-physics-features --physics-feature-selection \
  --output-dir /abs/path/out \
  --data-dir /abs/path/cache
```

### 4) Run inference

```bash
python3 inference_tcn_dilated.py \
  --params /abs/path/params.csv \
  --voc /abs/path/voc.txt \
  --vmpp /abs/path/vmpp.txt \
  --checkpoint /abs/path/best-model.ckpt \
  --cache-dir /abs/path/cache \
  --output-dir /abs/path/inference_out
```

## Feature Selection

The trainer supports train-only physics feature filtering:

- `--physics-feature-selection`
- `--physics-weak-threshold` (default `0.30`)
- `--physics-corr-threshold` (default `0.85`)
- `--physics-max-features` (optional)

If exact `m=5` is required, use `--physics-max-features 5`.

## Publishing Layout

See `/Users/memoozdincer/Desktop/NUS2026/PINN-Coupled-PDE-Solver/RELEASE_MANIFEST.md` for:

- GitHub keep list
- Hugging Face dataset/model split
- release notes and compatibility aliases
