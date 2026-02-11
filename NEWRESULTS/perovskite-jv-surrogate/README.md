---
license: apache-2.0
tags:
  - physics-informed
  - solar-cells
  - perovskite
  - surrogate-model
  - drift-diffusion
  - pytorch
  - pytorch-lightning
datasets:
  - memo-ozdincer/perovskite-jv-comsol-150k
language:
  - en
pipeline_tag: other
---

# Perovskite J-V Curve Surrogate Model

**Paper:** *Physics-Informed Convolutional Surrogates for Coupled Drift-Diffusion Equations: Reconstructing Full J-V Curves of Perovskite Solar Cells* (Ozdincer, 2026)

**Code:** [github.com/memo-ozdincer/perovskite-jv-surrogate](https://github.com/memo-ozdincer/perovskite-jv-surrogate)

**Dataset:** [huggingface.co/datasets/memo-ozdincer/perovskite-jv-comsol-150k](https://huggingface.co/datasets/memo-ozdincer/perovskite-jv-comsol-150k)

## Model Description

A physics-constrained dilated convolutional network that reconstructs full current-voltage (J-V) curves of perovskite solar cells from 31 coupled drift-diffusion parameters spanning >20 orders of magnitude. The model is the **Stage-2** component of a two-stage surrogate pipeline:

1. **Stage 1** (Zhao et al., 2025 — included in `matlab_stage1/`): MATLAB neural networks predict scalar device metrics (V_oc, V_mpp, J_sc) from the 31 raw parameters.
2. **Stage 2** (this model): A dilated 1D convolutional network reconstructs 8-point J-V curves from the 31 scaled parameters + Stage-1 scalar predictions.

The full 45-point J-V curve is recovered via monotone PCHIP interpolation from the 8 predicted points.

### Architecture

```
Input: 31 params + 2 scalars (V_oc, V_mpp) = 33 features
  -> ParamMLP: 33 -> 256 -> 128 -> 128 (GELU, BatchNorm, Dropout=0.036)
  -> Broadcast across 8 voltage query points
  -> Concat with Gaussian RBF voltage encoding (18 bands, sigma=0.077)
  -> 8 x 256 sequence
  -> Conv Block 1: 256 -> 128, kernel=5, dilation=1 (residual)
  -> Conv Block 2: 128 -> 128, kernel=5, dilation=1 (residual)
  -> Conv Block 3: 128 -> 64,  kernel=5, dilation=2 (full receptive field > 8)
  -> Pointwise Linear: 64 -> 1
Output: 8-point J_sc-normalized J-V curve in [-1, 1]
```

Total parameters: ~0.49M

### Physics-Constrained Loss

```
L = 0.98 * L_MSE + 0.005 * L_monotonicity + 0.005 * L_convexity + 0.01 * L_curvature
```

- **Monotonicity:** Penalizes dJ/dV > 0 (current must decrease with voltage)
- **Convexity:** Penalizes concave-up regions in the knee
- **Excess curvature:** Bounds second finite differences to prevent oscillation

## Performance

Evaluated on ~15k held-out test devices, averaged over 3 random seeds:

| Metric | Mean | Median |
|---|---|---|
| R² (per-curve, ΔV-weighted) | 0.9858 +/- 0.0009 | **0.9975 +/- 0.0004** |
| MAE (A/m²) | 4.75 +/- 0.26 | 2.96 +/- 0.34 |
| RMSE (A/m²) | 6.66 +/- 0.26 | 4.27 +/- 0.34 |
| \|V_oc error\| (V) | 0.0056 +/- 0.0058 | **0.00039 +/- 0.00016** |
| \|I_sc error\| (A/m²) | 0.59 +/- 0.53 | 0.36 +/- 0.62 |

- Median V_oc error (0.39 mV) is an order of magnitude below experimental measurement uncertainty
- \>10,000x speedup over COMSOL finite-element simulation
- Training: ~10 min on a single NVIDIA L40S GPU
- Inference: ~10 s for ~15k devices

### Per-Seed Breakdown

| Seed | R² (median) | MAE (median) | V_oc error (median, mV) |
|---|---|---|---|
| 123 | 0.9979 | 2.62 | 0.38 |
| 42 | 0.9976 | 2.96 | 0.23 |
| 456 | 0.9971 | 3.31 | 0.56 |

## Repository Structure

```
perovskite-jv-surrogate/
├── README.md                           # This model card
├── checkpoint/
│   ├── hparams.yaml                    # Full hyperparameter configuration
│   ├── seed_123/
│   │   ├── best-model.ckpt            # PyTorch Lightning checkpoint
│   │   └── test_stats.json            # Test-set evaluation metrics
│   ├── seed_42/
│   │   ├── best-model.ckpt
│   │   └── test_stats.json
│   └── seed_456/
│       ├── best-model.ckpt
│       └── test_stats.json
├── preprocessing/
│   ├── seed_123/
│   │   ├── cnn_param_transformer.joblib    # Parameter normalization (sklearn)
│   │   ├── cnn_scalar_transformer.joblib   # Scalar normalization (sklearn)
│   │   └── cnn_preprocessed.npz            # Cached metadata (split indices, feature names)
│   ├── seed_42/
│   │   └── ...
│   └── seed_456/
│       └── ...
├── matlab_stage1/                      # Stage-1 scalar predictors (Zhao et al., 2025)
│   ├── NN_Perf.mat                     # Trained NN weights: V_oc, FF, PCE
│   ├── NN_RMPP.mat                     # Trained NN weights: MPP-related metrics
│   ├── NN_ROC.mat                      # Trained NN weights: additional metrics
│   ├── NN_predict.m                    # MATLAB inference script
│   └── NN_predict_v2.m                 # Alternative inference script
└── results/
    ├── champion_model.json             # Aggregated champion statistics
    ├── all_results.csv                 # Full 30-run ablation results (10 configs x 3 seeds)
    └── results_leaderboard.csv         # Ranked experiment performance
```

## Usage

### Inference with the training code

```bash
git clone https://github.com/memo-ozdincer/perovskite-jv-surrogate.git
cd perovskite-jv-surrogate
pip install -r requirements.txt

bash scripts/run_paper_inference.sh \
    --params /path/to/params.txt \
    --voc /path/to/voc.txt \
    --vmpp /path/to/vmpp.txt \
    --checkpoint /path/to/best-model.ckpt \
    --cache-dir /path/to/preprocessing/seed_42 \
    --output-dir ./outputs
```

### Loading the checkpoint directly

```python
import torch
import sys
sys.path.insert(0, "src")
from train import PhysicsIVSystem

model = PhysicsIVSystem.load_from_checkpoint("best-model.ckpt", map_location="cpu")
model.eval()

# x: (batch, 33) tensor of scaled parameters + scalars
# v: (batch, 8) tensor of voltage query points
with torch.no_grad():
    j_pred = model(x, v)  # (batch, 8) normalized current predictions
```

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (beta1=0.9, beta2=0.999) |
| Learning rate | 5.5e-3 (OneCycleLR, cosine to lr/150) |
| Weight decay | 5.4e-5 |
| Batch size | 512 |
| Precision | bf16-mixed (fp32 for Jacobian) |
| Max epochs | 100 |
| Early stopping | 20 epochs patience on val loss |
| Gradient clipping | max_norm=1.0 |

## MATLAB Stage-1 Models (Zhao et al.)

The `matlab_stage1/` directory contains pre-trained MATLAB neural networks from **Zhao et al. (2025)**, "Accelerating device characterization in perovskite solar cells via neural network approach," *Applied Energy*, 392:125922.

**These models are NOT authored by the maintainer of this repository.** They are included here for pipeline completeness and are redistributed with attribution. The Stage-1 models predict scalar device metrics (V_oc, FF, PCE) from the 31 raw COMSOL parameters using 3-layer ANNs with Bayesian regularization, achieving MSE < 3.1e-4 for V_oc. See the original publication for full details.

**Usage (MATLAB):**
```matlab
load('NN_Perf.mat');
input = mapminmax('apply', X_P, PS_input);
output = net(input);
Perf = mapminmax('reverse', output, PS_output);
% Perf(1,:) = Voc, Perf(2,:) = FF, Perf(3,:) = PCE
```

## Limitations

- **S-shaped curves:** Devices with large interface band offsets (|ΔE_v^HP| >> k_BT) produce barrier-dominated transport that violates the monotone-sigmoid assumption. The model correctly refuses unphysical fits for these cases.
- **Extreme V_oc:** Devices with V_oc < 0.5 V or V_oc > 1.2 V may have mislocated knee regions due to the fixed query-point grid.
- **High series resistance:** R_s > 10³ Ohm cm² produces nearly linear J-V curves that the convexity penalty actively resists.
- These failure modes affect <1% of devices and are physically interpretable (see paper Section 4.3).

## Citation

```bibtex
@article{ozdincer2026physics,
  title={Physics-Informed Convolutional Surrogates for Coupled Drift-Diffusion Equations: Reconstructing Full {J-V} Curves of Perovskite Solar Cells},
  author={Ozdincer, Mehmet},
  year={2026}
}

@article{zhao2025ae,
  title={Accelerating device characterization in perovskite solar cells via neural network approach},
  author={Zhao, X. and Huang, C. and Birgersson, E. and Suprun, N. and Tan, H.Q. and Zhang, Y. and Jiang, Y. and Shou, C. and Sun, J. and Peng, J. and Xue, H.},
  journal={Applied Energy},
  volume={392},
  pages={125922},
  year={2025}
}
```
