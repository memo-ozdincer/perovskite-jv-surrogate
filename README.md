# Physics-Informed Convolutional Surrogates for Coupled Drift-Diffusion Equations

Reconstructing full J-V curves of perovskite solar cells from 31 coupled drift-diffusion parameters.

**Paper (PDF):** [src/ArXiV_paper.pdf](src/ArXiV_paper.pdf) — *Physics-Informed Convolutional Surrogates for Coupled Drift-Diffusion Equations: Reconstructing Full J-V Curves of Perovskite Solar Cells* (Ozdincer, 2026)

**Pre-trained models:** [huggingface.co/memo-ozdincer/perovskite-jv-surrogate](https://huggingface.co/memo-ozdincer/perovskite-jv-surrogate)

**Dataset (~150k devices):** [huggingface.co/datasets/memo-ozdincer/perovskite-jv-comsol-150k](https://huggingface.co/datasets/memo-ozdincer/perovskite-jv-comsol-150k)

## Key Results

| Metric | Mean | Median |
|---|---|---|
| R² (per-curve, ΔV-weighted) | 0.9858 ± 0.0009 | **0.9975 ± 0.0004** |
| MAE (A/m²) | 4.75 ± 0.26 | 2.96 ± 0.34 |
| \|V_oc error\| (V) | 0.0056 ± 0.0058 | **0.00039 ± 0.00016** |

- Median V_oc error of 0.39 mV is an order of magnitude below experimental measurement uncertainty
- \>10⁴× speedup over COMSOL FEM (10⁶-device screening in <1 hour on a single GPU)
- Training completes in ~10 min on a single NVIDIA L40S GPU

## Method Overview

A two-stage surrogate pipeline:

1. **Stage 1** (external, Zhao et al.): Three-layer ANNs predict scalar metrics (V_oc, V_mpp) from 31 raw COMSOL parameters
2. **Stage 2** (this repository): A physics-constrained dilated convolutional network reconstructs 8-point J-V curves from the 31 parameters + Stage-1 scalars + 5 physics-derived features

### Architecture

```
31 params + 2 scalars + 5 physics features
    → ParamMLP (3-layer, 256→128→128)
    → Broadcast + Gaussian RBF voltage encoding (8 positions × 256 channels)
    → 3× Dilated 1D Conv blocks (bidirectional, kernel=5, dilation up to 2)
    → Pointwise linear → 8-point normalized J-V curve
    → PCHIP interpolation → 45-point full curve
```

### Physics-Constrained Loss

```
L = 0.98·L_MSE + 0.005·L_monotonicity + 0.005·L_convexity + 0.01·L_curvature + λ·L_jacobian
```

### Physics Feature Engineering

71 analytically derived drift-diffusion features are distilled to 5 based on independent physical mechanisms:

1. **J_max^log** — Beer-Lambert photocurrent ceiling (generation)
2. **E_g** — Perovskite band gap (energetics)
3. **E_g^offset** — Band gap vs built-in potential mismatch (field alignment)
4. **χ_P^e** — Perovskite electron affinity (interface band alignment)
5. **V_oc^loss** — Composite recombination metric (recombination losses)

## Repository Structure

```
├── src/                        # Core Python modules
│   ├── train.py                # Main training script (PyTorch Lightning)
│   ├── inference.py            # Inference CLI
│   ├── config.py               # 31-parameter definitions, voltage grid, constants
│   ├── features.py             # 71 physics-derived feature computation
│   ├── data.py                 # Dataset loaders and transforms
│   ├── preprocessing.py        # Parameter scaling pipeline (log1p → Robust → MinMax)
│   ├── metrics_curve.py        # ΔV-weighted evaluation metrics
│   ├── physics_analysis.py     # Jacobian sensitivity analysis
│   ├── logging_utils.py        # Training logging utilities
│   └── plotting_utils.py       # Visualization utilities
├── scripts/                    # Shell entry points
│   ├── run_paper_training.sh   # Training entrypoint with paper defaults
│   ├── run_paper_inference.sh  # Inference entrypoint
│   └── generate_scalar_txt.py  # Utility to package scalar predictions
├── paper/                      # LaTeX source
│   └── icml_paper.tex
├── NEWRESULTS/                 # Pre-trained model and data (HuggingFace)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU for training (CPU inference is supported).

## Data

The model expects:
- **31-parameter file**: Tab/comma-separated, one row per device, 31 COMSOL drift-diffusion parameters (see `src/config.py` for column names)
- **I-V curve file**: Tab/comma-separated, one row per device, 45 current-density values on the non-uniform voltage grid
- **Scalar files**: One-column text files for V_oc and V_mpp (from Stage-1 predictions or COMSOL ground truth)

The 31 parameters span >20 orders of magnitude and cover: layer thicknesses, carrier mobilities, density of states, band edges, work functions, permittivities, generation rate, recombination coefficients, SRH lifetimes, and surface recombination velocities.

## Training

```bash
bash scripts/run_paper_training.sh \
    --params /path/to/LHS_parameters.txt \
    --iv /path/to/IV_curves.txt \
    --voc /path/to/voc_predictions.txt \
    --vmpp /path/to/vmpp_predictions.txt \
    --output-dir ./outputs/paper_train \
    --batch-size 512 \
    --seed 42
```

Key training flags:
- `--batch-size 512`: Paper default (largest single-factor improvement)
- `--no-physics-features`: Ablation without physics feature engineering
- `--physics-max-features N`: Number of selected physics features (default: 5)
- `--force-preprocess`: Regenerate cached preprocessing artifacts

Outputs: model checkpoint (`best-model.ckpt`), preprocessing transformers (`cnn_*.joblib`), and diagnostic plots.

## Inference

```bash
bash scripts/run_paper_inference.sh \
    --params /path/to/params.txt \
    --voc /path/to/voc.txt \
    --vmpp /path/to/vmpp.txt \
    --checkpoint /path/to/best-model.ckpt \
    --cache-dir /path/to/training_cache \
    --output-dir ./outputs/inference
```

Outputs:
- `predictions_8pt_normalized.csv` — 8-point J_sc-normalized curve predictions
- `predictions_45pt_normalized.csv` — PCHIP-interpolated 45-point curves
- `predictions_*_absolute.csv` — Absolute current (requires `--jsc` flag)

## Citation

```bibtex
@article{ozdincer2026physics,
  title={Physics-Informed Convolutional Surrogates for Coupled Drift-Diffusion Equations: Reconstructing Full {J-V} Curves of Perovskite Solar Cells},
  author={Ozdincer, Mehmet},
  year={2026}
}
```

## Acknowledgments

Stage-1 scalar predictors are from Zhao et al. (2025), "Accelerating device characterization in perovskite solar cells via neural network approach," *Applied Energy*, 392:125922.

## License

See [LICENSE](LICENSE) for details.
