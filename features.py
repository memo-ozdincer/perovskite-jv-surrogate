"""
Comprehensive physics-informed feature engineering.
Uses ALL analytical relationships from drift-diffusion physics.
Fully vectorized on GPU using PyTorch.
"""
import torch
import numpy as np
from config import COLNAMES, KB_T

# Column index mapping for fast tensor access
COL_IDX = {name: i for i, name in enumerate(COLNAMES)}


def compute_all_physics_features(params: torch.Tensor) -> torch.Tensor:
    """
    Compute ALL physics-informed features from the 31 input parameters.

    Args:
        params: (N, 31) tensor of raw parameters

    Returns:
        features: (N, n_features) tensor of physics features
    """
    device = params.device
    N = params.shape[0]

    # Extract raw columns (some are log10 scale, some linear)
    lH = params[:, COL_IDX['lH']]        # nm, linear
    lP = params[:, COL_IDX['lP']]        # nm, linear
    lE = params[:, COL_IDX['lE']]        # nm, linear

    # Mobilities (log10 scale) -> convert to linear
    muHh = 10 ** params[:, COL_IDX['muHh']]  # m²/V/s
    muPh = 10 ** params[:, COL_IDX['muPh']]
    muPe = 10 ** params[:, COL_IDX['muPe']]
    muEe = 10 ** params[:, COL_IDX['muEe']]

    # Density of states (log10 scale) -> convert to linear
    NvH = 10 ** params[:, COL_IDX['NvH']]    # m⁻³
    NcH = 10 ** params[:, COL_IDX['NcH']]
    NvE = 10 ** params[:, COL_IDX['NvE']]
    NcE = 10 ** params[:, COL_IDX['NcE']]
    NvP = 10 ** params[:, COL_IDX['NvP']]
    NcP = 10 ** params[:, COL_IDX['NcP']]

    # Energy levels (eV, linear)
    chiHh = params[:, COL_IDX['chiHh']]      # Hole ionization potential HTL
    chiHe = params[:, COL_IDX['chiHe']]      # Electron affinity HTL
    chiPh = params[:, COL_IDX['chiPh']]      # Hole ionization potential Perovskite
    chiPe = params[:, COL_IDX['chiPe']]      # Electron affinity Perovskite
    chiEh = params[:, COL_IDX['chiEh']]      # Hole ionization potential ETL
    chiEe = params[:, COL_IDX['chiEe']]      # Electron affinity ETL

    # Work functions (eV, linear)
    Wlm = params[:, COL_IDX['Wlm']]          # Cathode (low work function)
    Whm = params[:, COL_IDX['Whm']]          # Anode (high work function)

    # Permittivities (linear)
    epsH = params[:, COL_IDX['epsH']]
    epsP = params[:, COL_IDX['epsP']]
    epsE = params[:, COL_IDX['epsE']]

    # Generation rate (log10 scale) -> convert to linear
    Gavg = 10 ** params[:, COL_IDX['Gavg']]  # m⁻⁴s⁻¹
    Gavg_log = params[:, COL_IDX['Gavg']]    # Keep log version too

    # Recombination coefficients (log10 scale) -> convert to linear
    Aug = 10 ** params[:, COL_IDX['Aug']]    # m⁶/s (Auger)
    Brad = 10 ** params[:, COL_IDX['Brad']]  # m³/s (Radiative)
    Aug_log = params[:, COL_IDX['Aug']]
    Brad_log = params[:, COL_IDX['Brad']]

    # Lifetimes (log10 scale) -> convert to linear
    Taue = 10 ** params[:, COL_IDX['Taue']]  # s
    Tauh = 10 ** params[:, COL_IDX['Tauh']]
    Taue_log = params[:, COL_IDX['Taue']]
    Tauh_log = params[:, COL_IDX['Tauh']]

    # Surface recombination velocities (log10 scale) -> convert to linear
    vII = 10 ** params[:, COL_IDX['vII']]    # m⁴/s
    vIII = 10 ** params[:, COL_IDX['vIII']]

    # Convert thicknesses to meters for physics calculations
    lH_m = lH * 1e-9
    lP_m = lP * 1e-9
    lE_m = lE * 1e-9

    # ========================================================================
    # ENERGETIC FEATURES
    # ========================================================================

    # Bandgap of perovskite (fundamental)
    Eg = chiPh - chiPe

    # Built-in voltage (contact potential difference)
    V_bi = Whm - Wlm

    # Bandgap-Vbi offset (measures Schottky vs Ohmic contacts)
    Eg_offset = Eg - V_bi

    # HTL bandgap and ETL bandgap
    Eg_HTL = chiHh - chiHe
    Eg_ETL = chiEh - chiEe

    # ========================================================================
    # INTERFACE BARRIER FEATURES (Critical for carrier selectivity)
    # ========================================================================

    # Valence band offsets (hole injection/blocking)
    VBO_HP = chiHh - chiPh          # HTL/Perovskite VB offset
    VBO_PE = chiPh - chiEh          # Perovskite/ETL VB offset

    # Conduction band offsets (electron injection/blocking)
    CBO_HP = chiPe - chiHe          # HTL/Perovskite CB offset
    CBO_PE = chiEe - chiPe          # Perovskite/ETL CB offset

    # Hole blocking barrier at ETL (should be large positive)
    Barrier_h_ETL = chiEh - chiPh

    # Electron blocking barrier at HTL (should be large positive)
    Barrier_e_HTL = chiHe - chiPe

    # Contact barriers (injection from electrodes)
    Barrier_anode = Whm - chiPh     # Hole injection from anode to perovskite
    Barrier_cathode = chiPe - Wlm   # Electron injection from cathode to perovskite

    # Total barrier asymmetry
    Barrier_asymmetry = Barrier_h_ETL - Barrier_e_HTL

    # ========================================================================
    # TRANSPORT FEATURES
    # ========================================================================

    # Diffusion lengths in perovskite
    L_diff_e = torch.sqrt(muPe * Taue * KB_T)   # Electron diffusion length
    L_diff_h = torch.sqrt(muPh * Tauh * KB_T)   # Hole diffusion length
    L_diff_min = torch.minimum(L_diff_e, L_diff_h)
    L_diff_max = torch.maximum(L_diff_e, L_diff_h)

    # Collection efficiency ratios (critical for Jsc)
    Collection_e = L_diff_e / lP_m
    Collection_h = L_diff_h / lP_m
    Collection_min = L_diff_min / lP_m
    Collection_ratio = L_diff_min / L_diff_max  # Balance

    # Mu-Tau products (transport quality)
    MuTau_e = muPe * Taue
    MuTau_h = muPh * Tauh
    MuTau_balance = MuTau_e / (MuTau_h + 1e-30)  # Should be ~1 for good FF
    MuTau_log_balance = torch.log10(MuTau_balance + 1e-30)

    # Drift lengths under built-in field (Hecht equation approximation)
    # L_drift = mu * tau * E, where E ~ V_bi / lP
    E_builtin = V_bi / lP_m  # Built-in electric field
    L_drift_e = muPe * Taue * E_builtin
    L_drift_h = muPh * Tauh * E_builtin

    # Extraction figures of merit (Theta)
    Theta_e = (muPe * Taue * V_bi) / (lP_m ** 2)
    Theta_h = (muPh * Tauh * V_bi) / (lP_m ** 2)
    Theta_min = torch.minimum(Theta_e, Theta_h)
    Theta_product = Theta_e * Theta_h

    # Log versions for better scaling
    Theta_e_log = torch.log10(Theta_e + 1e-30)
    Theta_h_log = torch.log10(Theta_h + 1e-30)
    Theta_min_log = torch.log10(Theta_min + 1e-30)

    # ========================================================================
    # SERIES RESISTANCE FEATURES (Critical for FF)
    # ========================================================================

    # Conductivity = q * mu * n, approximate n from DOS
    sigma_HTL = muHh * NvH
    sigma_ETL = muEe * NcE
    sigma_P_e = muPe * NcP
    sigma_P_h = muPh * NvP

    # Layer resistances (R = L / sigma)
    Rs_HTL = lH_m / (sigma_HTL + 1e-30)
    Rs_ETL = lE_m / (sigma_ETL + 1e-30)
    Rs_P = lP_m / (torch.sqrt(sigma_P_e * sigma_P_h) + 1e-30)
    Rs_total = Rs_HTL + Rs_ETL + Rs_P

    # Log versions
    Rs_HTL_log = torch.log10(Rs_HTL + 1e-30)
    Rs_ETL_log = torch.log10(Rs_ETL + 1e-30)
    Rs_total_log = torch.log10(Rs_total + 1e-30)

    # Resistance ratios
    Rs_ratio_HTL_ETL = Rs_HTL / (Rs_ETL + 1e-30)

    # ========================================================================
    # GENERATION FEATURES (Critical for Jsc)
    # ========================================================================

    # Maximum theoretical Jsc (generation limit)
    # J_max = q * G * L_active
    Q_E = 1.602e-19
    J_max = Q_E * Gavg * lP_m
    J_max_log = torch.log10(J_max + 1e-30)

    # Generation per unit thickness
    G_per_nm = Gavg / lP
    G_per_nm_log = Gavg_log - torch.log10(lP)

    # ========================================================================
    # RECOMBINATION FEATURES (Critical for Voc)
    # ========================================================================

    # Recombination lifetime (effective)
    # Protect against division by very small Taue/Tauh values
    # Clamp lifetimes to minimum value to prevent inf
    Taue_safe = Taue.clamp(min=1e-30)
    Tauh_safe = Tauh.clamp(min=1e-30)
    Tau_eff = 1.0 / (1.0/Taue_safe + 1.0/Tauh_safe)
    Tau_eff_log = torch.log10(Tau_eff.clamp(min=1e-30))

    # Recombination rate proxy
    Recomb_proxy = 1.0 / (Taue_safe + Tauh_safe)
    Recomb_proxy_log = torch.log10(Recomb_proxy.clamp(min=1e-30))

    # SRH recombination strength
    SRH_strength = (Taue_log + Tauh_log) / 2.0

    # Auger recombination (density dependent)
    Auger_factor = Aug_log

    # Radiative recombination
    Rad_factor = Brad_log

    # Total recombination proxy (sum of mechanisms in log space)
    # J0 ~ exp(-Eg/kT) * recomb_factors
    Voc_loss_proxy = -Eg / KB_T + Recomb_proxy_log

    # Intrinsic carrier concentration proxy: ni² ~ Nc*Nv*exp(-Eg/kT)
    ni2_log = torch.log10(NcP) + torch.log10(NvP) - Eg / (KB_T * np.log(10))

    # Voc theoretical (Shockley-Queisser inspired)
    # Voc ~ Eg - kT*ln(J0/Jsc)
    Voc_SQ_proxy = Eg - KB_T * (ni2_log - J_max_log) * np.log(10)

    # ========================================================================
    # SURFACE RECOMBINATION FEATURES
    # ========================================================================

    # Surface recombination velocities (already in log form effectively)
    vII_log = params[:, COL_IDX['vII']]
    vIII_log = params[:, COL_IDX['vIII']]

    # Surface vs bulk recombination competition
    # S_eff / (D/L) ratio
    Surface_bulk_ratio_e = vII * lP_m / (muPe * KB_T + 1e-30)
    Surface_bulk_ratio_h = vIII * lP_m / (muPh * KB_T + 1e-30)

    # ========================================================================
    # DIELECTRIC FEATURES
    # ========================================================================

    # Dielectric contrast
    eps_ratio_HP = epsH / epsP
    eps_ratio_PE = epsP / epsE
    eps_total = epsH + epsP + epsE

    # Debye length proxy (screening length)
    # L_D ~ sqrt(eps * kT / (q² * n))
    eps_0 = 8.854e-12
    Debye_P = torch.sqrt(epsP * eps_0 * KB_T / (Q_E * torch.sqrt(NcP * NvP) + 1e-30))
    Debye_ratio = Debye_P / lP_m

    # ========================================================================
    # GEOMETRY FEATURES
    # ========================================================================

    # Thickness ratios
    thickness_total = lH + lP + lE
    thickness_ratio_P = lP / thickness_total
    thickness_ratio_H = lH / thickness_total
    thickness_ratio_E = lE / thickness_total
    thickness_ratio_HE = lH / (lE + 1e-6)

    # ========================================================================
    # COMPOSITE PHYSICS FEATURES (Higher order)
    # ========================================================================

    # Fill factor predictor: combines Theta and Rs effects
    FF_predictor = Theta_min_log - Rs_total_log

    # Voc predictor: combines Eg, recombination, and barriers
    Voc_predictor = Eg + 0.5 * (Barrier_h_ETL + Barrier_e_HTL) - KB_T * Recomb_proxy_log

    # Jsc predictor: combines generation and collection
    Jsc_predictor = J_max_log + torch.log10(Collection_min.clamp(min=1e-6, max=1.0))

    # Vmpp predictor: combination of Voc predictors scaled by FF-like term
    Vmpp_predictor = Voc_predictor * torch.sigmoid(FF_predictor)

    # Quality factor (overall device quality metric)
    Quality_factor = Theta_min_log + SRH_strength - Rs_total_log

    # Energy alignment score
    Energy_alignment = -(torch.abs(CBO_HP) + torch.abs(VBO_PE))

    # ========================================================================
    # STACK ALL FEATURES
    # ========================================================================

    features = torch.stack([
        # Energetics (7)
        Eg, V_bi, Eg_offset, Eg_HTL, Eg_ETL,
        chiPh, chiPe,  # Raw energy levels useful

        # Interface barriers (9)
        VBO_HP, VBO_PE, CBO_HP, CBO_PE,
        Barrier_h_ETL, Barrier_e_HTL,
        Barrier_anode, Barrier_cathode, Barrier_asymmetry,

        # Transport - diffusion (8)
        torch.log10(L_diff_e + 1e-30), torch.log10(L_diff_h + 1e-30),
        torch.log10(L_diff_min + 1e-30), torch.log10(L_diff_max + 1e-30),
        torch.log10(Collection_e + 1e-30), torch.log10(Collection_h + 1e-30),
        torch.log10(Collection_min + 1e-30), Collection_ratio,

        # Transport - MuTau (4)
        torch.log10(MuTau_e + 1e-30), torch.log10(MuTau_h + 1e-30),
        MuTau_log_balance, Tau_eff_log,

        # Drift/Extraction - Theta (6)
        Theta_e_log, Theta_h_log, Theta_min_log,
        torch.log10(L_drift_e + 1e-30), torch.log10(L_drift_h + 1e-30),
        torch.log10(Theta_product + 1e-30),

        # Series resistance (5)
        Rs_HTL_log, Rs_ETL_log, Rs_total_log,
        torch.log10(Rs_ratio_HTL_ETL + 1e-30), torch.log10(Rs_P + 1e-30),

        # Generation (4)
        J_max_log, G_per_nm_log, Gavg_log, torch.log10(lP_m),

        # Recombination (8)
        Recomb_proxy_log, SRH_strength, Auger_factor, Rad_factor,
        Voc_loss_proxy, ni2_log, Voc_SQ_proxy, Taue_log + Tauh_log,

        # Surface recombination (4)
        vII_log, vIII_log,
        torch.log10(Surface_bulk_ratio_e + 1e-30),
        torch.log10(Surface_bulk_ratio_h + 1e-30),

        # Dielectric (4)
        eps_ratio_HP, eps_ratio_PE, eps_total, torch.log10(Debye_ratio + 1e-30),

        # Geometry (5)
        thickness_total, thickness_ratio_P, thickness_ratio_H,
        thickness_ratio_E, thickness_ratio_HE,

        # Composite predictors (7)
        FF_predictor, Voc_predictor, Jsc_predictor, Vmpp_predictor,
        Quality_factor, Energy_alignment,
        torch.log10(E_builtin + 1e-30),

    ], dim=1)

    # Validate: replace any remaining inf/nan with safe values
    features = torch.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

    return features


def validate_features(features: torch.Tensor, verbose: bool = True) -> dict:
    """
    Validate features for inf/nan values.
    Returns dict with validation stats.
    """
    n_inf = torch.isinf(features).sum().item()
    n_nan = torch.isnan(features).sum().item()
    n_total = features.numel()

    stats = {
        'n_inf': n_inf,
        'n_nan': n_nan,
        'n_total': n_total,
        'pct_invalid': (n_inf + n_nan) / n_total * 100,
        'has_issues': n_inf > 0 or n_nan > 0
    }

    if verbose and stats['has_issues']:
        print(f"WARNING: Features contain {n_inf} inf and {n_nan} nan values ({stats['pct_invalid']:.4f}%)")
        # Find which features have issues
        inf_mask = torch.isinf(features).any(dim=0)
        nan_mask = torch.isnan(features).any(dim=0)
        feature_names = get_feature_names()
        for i, name in enumerate(feature_names):
            if inf_mask[i] or nan_mask[i]:
                print(f"  - Feature {i} ({name}): inf={inf_mask[i].item()}, nan={nan_mask[i].item()}")

    return stats


def get_feature_names() -> list[str]:
    """Return names for all physics features in order."""
    return [
        # Energetics
        'Eg', 'V_bi', 'Eg_offset', 'Eg_HTL', 'Eg_ETL', 'chiPh', 'chiPe',
        # Interface barriers
        'VBO_HP', 'VBO_PE', 'CBO_HP', 'CBO_PE',
        'Barrier_h_ETL', 'Barrier_e_ETL',
        'Barrier_anode', 'Barrier_cathode', 'Barrier_asymmetry',
        # Transport - diffusion
        'L_diff_e_log', 'L_diff_h_log', 'L_diff_min_log', 'L_diff_max_log',
        'Collection_e_log', 'Collection_h_log', 'Collection_min_log', 'Collection_ratio',
        # Transport - MuTau
        'MuTau_e_log', 'MuTau_h_log', 'MuTau_log_balance', 'Tau_eff_log',
        # Drift/Extraction
        'Theta_e_log', 'Theta_h_log', 'Theta_min_log',
        'L_drift_e_log', 'L_drift_h_log', 'Theta_product_log',
        # Series resistance
        'Rs_HTL_log', 'Rs_ETL_log', 'Rs_total_log', 'Rs_ratio_HE_log', 'Rs_P_log',
        # Generation
        'J_max_log', 'G_per_nm_log', 'Gavg_log', 'lP_m_log',
        # Recombination
        'Recomb_proxy_log', 'SRH_strength', 'Auger_factor', 'Rad_factor',
        'Voc_loss_proxy', 'ni2_log', 'Voc_SQ_proxy', 'Tau_sum_log',
        # Surface
        'vII_log', 'vIII_log', 'Surf_bulk_e_log', 'Surf_bulk_h_log',
        # Dielectric
        'eps_ratio_HP', 'eps_ratio_PE', 'eps_total', 'Debye_ratio_log',
        # Geometry
        'thickness_total', 'thickness_ratio_P', 'thickness_ratio_H',
        'thickness_ratio_E', 'thickness_ratio_HE',
        # Composite
        'FF_predictor', 'Voc_predictor', 'Jsc_predictor', 'Vmpp_predictor',
        'Quality_factor', 'Energy_alignment', 'E_builtin_log',
    ]


# ============================================================================
# TARGET-SPECIFIC ANALYTICAL BOUNDS
# ============================================================================

def compute_jsc_ceiling(params: torch.Tensor) -> torch.Tensor:
    """
    Compute analytical ceiling for Jsc (before losses).
    J_ceiling = q * G_avg * L_P
    """
    Q_E = 1.602e-19
    lP_m = params[:, COL_IDX['lP']] * 1e-9
    Gavg = 10 ** params[:, COL_IDX['Gavg']]
    return Q_E * Gavg * lP_m


def compute_voc_ceiling(params: torch.Tensor) -> torch.Tensor:
    """
    Compute analytical ceiling for Voc.
    V_oc_max = min(V_bi, Eg)
    """
    Eg = params[:, COL_IDX['chiPh']] - params[:, COL_IDX['chiPe']]
    V_bi = params[:, COL_IDX['Whm']] - params[:, COL_IDX['Wlm']]
    return torch.minimum(Eg, V_bi)


def compute_vmpp_estimate(params: torch.Tensor, voc: torch.Tensor) -> torch.Tensor:
    """
    Compute analytical estimate for Vmpp.
    V_mpp ~ V_oc - kT/q * ln(1 + V_oc*q/(kT))  (ideal diode approx)
    Typically V_mpp ~ 0.8 * V_oc for good cells
    """
    # Simplified estimate based on Shockley diode
    voc_norm = voc / KB_T
    return voc - KB_T * torch.log(1 + voc_norm)


def compute_ff_estimate(params: torch.Tensor, voc: torch.Tensor) -> torch.Tensor:
    """
    Analytical FF estimate (empirical Green formula).
    FF = (v_oc - ln(v_oc + 0.72)) / (v_oc + 1)
    where v_oc = V_oc / (kT/q)
    """
    v_oc = voc / KB_T
    v_oc = v_oc.clamp(min=1.0)  # Avoid log issues
    ff = (v_oc - torch.log(v_oc + 0.72)) / (v_oc + 1)
    return ff.clamp(0.25, 0.9)
