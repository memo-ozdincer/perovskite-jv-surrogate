---
license: apache-2.0
tags:
  - solar-cells
  - perovskite
  - drift-diffusion
  - comsol
  - simulation
  - materials-science
  - physics
language:
  - en
size_categories:
  - 100K<n<1M
task_categories:
  - tabular-regression
---

# Perovskite Solar Cell J-V Curve Dataset (COMSOL, ~150k devices)

**Paper:** *Physics-Informed Convolutional Surrogates for Coupled Drift-Diffusion Equations: Reconstructing Full J-V Curves of Perovskite Solar Cells* (Ozdincer, 2026)

**Model:** [huggingface.co/memo-ozdincer/perovskite-jv-surrogate](https://huggingface.co/memo-ozdincer/perovskite-jv-surrogate)

**Code:** [github.com/memo-ozdincer/perovskite-jv-surrogate](https://github.com/memo-ozdincer/perovskite-jv-surrogate)

## Dataset Description

This dataset contains coupled Poisson-drift-diffusion simulation results for perovskite solar cells, generated using COMSOL Multiphysics finite-element methods. It provides the ground-truth data used to train and evaluate the physics-informed J-V curve surrogate model described in the accompanying paper.

Each sample represents a simulated perovskite solar cell device defined by 31 physical parameters, with its corresponding full J-V (current density vs. voltage) curve computed by solving the coupled drift-diffusion equations numerically.

### Key Statistics

- **Total devices:** ~150,000 (after COMSOL convergence and quality filtering)
- **Campaign 1:** 100,000 Latin Hypercube Samples (LHS)
- **Campaign 2:** 300,000 LHS samples (usable subset after filtering)
- **Combined:** ~150k usable devices, split 80/10/10 (train/val/test)
- **Simulation cost:** ~4,800 seconds per device (~80 minutes each via COMSOL FEM)

### Physical System

The simulated device is a planar p-i-n perovskite solar cell stack:

```
Glass / Front-contact / HTL / Perovskite absorber / ETL / Back-contact
```

Transport is governed by three coupled PDEs (Poisson equation + electron/hole drift-diffusion) solved at each applied voltage to produce the J(V) curve. Recombination includes radiative, Shockley-Read-Hall (SRH), Auger, and interface terms.

Full details of the simulation methodology are described in Zhao (2023), "Mathematical Modeling of Two-terminal Perovskite-Based Thin-film Tandem Solar Cells," Ph.D. thesis, National University of Singapore.

## Dataset Structure

```
perovskite-jv-comsol-150k/
├── README.md                           # This data card
├── campaign_100k/
│   ├── LHS_parameters_m.txt            # 100k devices, 31 parameters each
│   └── iV_m.txt                        # 100k J-V curves, 45 voltage points each
├── campaign_300k/
│   ├── LHS_parameters_m_300k.txt       # 300k devices, 31 parameters each
│   └── IV_m_300k.txt                   # 300k J-V curves, 45 voltage points each
└── scalar_predictions/                 # Stage-1 NN predictions (Zhao et al., 2025)
    ├── NN_predicted_Voc.txt            # Open-circuit voltage
    ├── NN_predicted_isc.txt            # Short-circuit current density
    ├── NN_predicted_FF.txt             # Fill factor
    ├── NN_predicted_PCE.txt            # Power conversion efficiency
    └── NN_predicted_Perf.txt           # Combined performance metrics
```

## Input Parameters (31 dimensions)

Each device is described by 31 COMSOL drift-diffusion parameters spanning >20 orders of magnitude:

| Group | Count | Parameters | Units |
|---|---|---|---|
| Layer thicknesses | 3 | l_H, l_P, l_E | nm |
| Carrier mobilities | 4 | mu_Hh, mu_Ph, mu_Pe, mu_Ee | m²/V/s (log10) |
| Density of states | 6 | N_vH, N_cH, N_vE, N_cE, N_vP, N_cP | m⁻³ (log10) |
| Band edges | 6 | chi_Hh, chi_He, chi_Ph, chi_Pe, chi_Eh, chi_Ee | eV |
| Contact work functions | 2 | W_lm, W_hm | eV |
| Relative permittivities | 3 | eps_H, eps_P, eps_E | dimensionless |
| Generation rate | 1 | G_avg | m⁻³s⁻¹ (log10) |
| Recombination coefficients | 2 | Aug, B_rad | cm⁶/s, cm³/s (log10) |
| SRH lifetimes | 2 | tau_e, tau_h | s (log10) |
| Surface recombination | 2 | v_II, v_III | m⁴/s (log10) |

Parameters are sampled via Latin Hypercube Sampling (LHS) to ensure near-orthogonal coverage of the design space. Most pairwise correlations are < 0.2.

**Column order** (tab-separated, no header):
```
lH, lP, lE, muHh, muPh, muPe, muEe, NvH, NcH, NvE, NcE, NvP, NcP,
chiHh, chiHe, chiPh, chiPe, chiEh, chiEe, Wlm, Whm, epsH, epsP, epsE,
Gavg, Aug, Brad, Taue, Tauh, vII, vIII
```

Note: Mobility, DOS, generation, recombination, lifetime, and surface velocity parameters are stored in **log10 scale**.

## Output: J-V Curves (45 voltage points)

Each row contains 45 current density values (A/m²) evaluated on a **non-uniform voltage grid**:

| Region | Voltage range | Points | Spacing |
|---|---|---|---|
| Flat-band (low V) | 0.0 - 0.4 V | 5 | ΔV = 0.1 V |
| Knee/MPP (high V) | 0.425 - 1.4 V | 40 | ΔV = 0.025 V |
| **Total** | 0.0 - 1.4 V | **45** | non-uniform |

The non-uniform grid concentrates resolution in the knee/MPP region where device-performance sensitivity is highest and curvature is largest, while using coarser spacing in the flat-band region where J ~ J_sc with near-zero curvature.

Convention: **positive photocurrent** (J > 0 for V < V_oc).

## Scalar Predictions (Zhao et al., 2025)

The `scalar_predictions/` directory contains scalar device metrics predicted by the **Stage-1 MATLAB neural networks** from Zhao et al. (2025), "Accelerating device characterization in perovskite solar cells via neural network approach," *Applied Energy*, 392:125922.

**These predictions are NOT generated by the maintainer of this dataset.** They are included for pipeline completeness and are produced by 3-layer ANNs with Bayesian regularization trained on the same COMSOL data. See the original publication for methodology and validation details (MSE < 3.1e-4 for V_oc, validated against 9 fabricated devices).

## Data Quality

- **Filtering:** Only devices where COMSOL failed to converge or produced grossly unphysical curves (negative J_sc, V_oc < 0) are removed. No curvature-based or performance-based filtering is applied, preserving the full LHS distribution including pathological devices.
- **LHS design:** Near-orthogonal parameter coverage confirmed by Pearson correlation analysis (most pairwise |r| < 0.2).
- **Simulation fidelity:** Each J-V curve is the steady-state solution of the coupled Poisson-drift-diffusion system at each applied voltage, computed via COMSOL finite-element methods with appropriate mesh refinement.

## Usage

```python
import numpy as np

# Load 100k campaign
params = np.loadtxt("campaign_100k/LHS_parameters_m.txt")     # (N, 31)
iv_curves = np.loadtxt("campaign_100k/iV_m.txt")              # (N, 45)

# Voltage grid
V_GRID = np.concatenate([
    np.arange(0, 0.4 + 1e-8, 0.1),       # 5 coarse points
    np.arange(0.425, 1.4 + 1e-8, 0.025)   # 40 dense points
]).astype(np.float32)  # 45 points total

# Load scalar predictions
voc = np.loadtxt("scalar_predictions/NN_predicted_Voc.txt")
```

## Citation

```bibtex
@article{ozdincer2026physics,
  title={Physics-Informed Convolutional Surrogates for Coupled Drift-Diffusion Equations: Reconstructing Full {J-V} Curves of Perovskite Solar Cells},
  author={Ozdincer, Mehmet},
  year={2026}
}

@phdthesis{zhao2023phd,
  title={Mathematical Modeling of Two-terminal Perovskite-Based Thin-film Tandem Solar Cells},
  author={Zhao, X.},
  school={National University of Singapore},
  year={2023}
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
