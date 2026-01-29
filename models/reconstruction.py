"""
Split-spline reconstruction utilities for J-V curves.
Differentiable PCHIP interpolation implemented in PyTorch.
"""
from __future__ import annotations

import torch


def build_knots(
    anchors: torch.Tensor,
    ctrl1: torch.Tensor,
    ctrl2: torch.Tensor,
    j_end: float = 0.0,
    validate_monotonicity: bool = False,
    return_violation_counts: bool = False
):
    """
    Build knot points for split spline reconstruction.

    The cumulative scaling ensures EXACT endpoint matching:
    - Region 1: J goes from Jsc down to Jmpp (monotonically decreasing)
    - Region 2: J goes from Jmpp down to 0 (monotonically decreasing)

    Args:
        anchors: (N, 4) [Jsc, Voc, Vmpp, Jmpp]
        ctrl1: (N, K) control points for region 1 (0->Vmpp), sigmoid outputs in [0,1]
        ctrl2: (N, K) control points for region 2 (Vmpp->Voc), sigmoid outputs in [0,1]
        validate_monotonicity: If True, raise error on violations
        return_violation_counts: If True, return violation counts dict

    Returns:
        v1_knots, j1_knots, v2_knots, j2_knots, (violations if return_violation_counts)
    """
    j_sc, v_oc, v_mpp, j_mpp = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    n_ctrl = ctrl1.shape[1]
    device = anchors.device

    # Build voltage knots (uniformly spaced in each region)
    v1_norm = torch.linspace(0, 1, n_ctrl + 2, device=device)
    v1_knots = v1_norm.unsqueeze(0) * v_mpp.unsqueeze(1)

    eps = 1e-4  # Increased epsilon for better numerical stability

    # Region 1: Jsc -> Jmpp (cumulative scaling guarantees monotonic decrease)
    # ctrl1 values are sigmoid outputs in [0,1], cumsum gives increasing sequence
    # Normalizing by final cumsum ensures exact endpoint matching
    j1_cumsum = torch.cumsum(ctrl1, dim=1)
    j1_final = j1_cumsum[:, -1:].clamp(min=eps)  # Ensure non-zero divisor
    j1_scale = j1_cumsum / j1_final  # Normalized to [0, 1]
    # j1_interior goes from near Jsc (scale~0) to near Jmpp (scale~1)
    # Clamp (j_sc - j_mpp) to ensure non-negative range
    j_range1 = (j_sc - j_mpp).clamp(min=0).unsqueeze(1)
    j1_interior = j_sc.unsqueeze(1) - j1_scale * j_range1
    j1_knots = torch.cat([j_sc.unsqueeze(1), j1_interior, j_mpp.unsqueeze(1)], dim=1)

    # Region 2: Jmpp -> J_end (same cumulative scaling approach)
    v2_norm = torch.linspace(0, 1, n_ctrl + 2, device=device)
    # Clamp (v_oc - v_mpp) to ensure non-negative range
    v_range2 = (v_oc - v_mpp).clamp(min=eps).unsqueeze(1)
    v2_knots = v_mpp.unsqueeze(1) + v2_norm.unsqueeze(0) * v_range2

    j2_cumsum = torch.cumsum(ctrl2, dim=1)
    j2_final = j2_cumsum[:, -1:].clamp(min=eps)  # Ensure non-zero divisor
    j2_scale = j2_cumsum / j2_final
    # j2_interior goes from near Jmpp (scale~0) to near J_end (scale~1)
    if isinstance(j_end, torch.Tensor):
        j_end_tensor = j_end.to(device)
    else:
        j_end_tensor = torch.full_like(j_mpp, float(j_end))
    j2_interior = j_mpp.unsqueeze(1) + (j_end_tensor.unsqueeze(1) - j_mpp.unsqueeze(1)) * j2_scale
    j2_knots = torch.cat(
        [j_mpp.unsqueeze(1), j2_interior, j_end_tensor.unsqueeze(1)],
        dim=1
    )

    # Count monotonicity violations (for logging)
    viol1 = j1_knots[:, 1:] > (j1_knots[:, :-1] + 1e-6)
    viol2 = j2_knots[:, 1:] > (j2_knots[:, :-1] + 1e-6)
    n_viol1 = viol1.any(dim=1).sum().item()
    n_viol2 = viol2.any(dim=1).sum().item()

    if validate_monotonicity:
        if viol1.any():
            idx = viol1.any(dim=1).nonzero(as_tuple=False)[0].item()
            raise ValueError(f"Region 1 knots not monotonic for sample {idx}: {j1_knots[idx].detach().cpu()}")
        if viol2.any():
            idx = viol2.any(dim=1).nonzero(as_tuple=False)[0].item()
            raise ValueError(f"Region 2 knots not monotonic for sample {idx}: {j2_knots[idx].detach().cpu()}")

    if return_violation_counts:
        violations = {
            'region1_violations': n_viol1,
            'region2_violations': n_viol2,
            'total_samples': anchors.shape[0]
        }
        return v1_knots, j1_knots, v2_knots, j2_knots, violations

    return v1_knots, j1_knots, v2_knots, j2_knots


def _pchip_slopes(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute PCHIP slopes using Fritsch-Carlson method with robust NaN handling."""
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]

    # Robust division: use sign-aware epsilon to avoid division by tiny values
    dx_safe = torch.where(dx.abs() < eps, torch.sign(dx + 1e-12) * eps, dx)
    m = dy / dx_safe

    d = torch.zeros_like(y)

    # Interior points
    m0 = m[:, :-1]
    m1 = m[:, 1:]
    h0 = dx[:, :-1]
    h1 = dx[:, 1:]

    w1 = 2 * h1 + h0
    w2 = h1 + 2 * h0
    cond = (m0 * m1) > 0

    # Robust division for harmonic mean computation
    m0_safe = torch.where(m0.abs() < eps, torch.sign(m0 + 1e-12) * eps, m0)
    m1_safe = torch.where(m1.abs() < eps, torch.sign(m1 + 1e-12) * eps, m1)
    denom = (w1 / m0_safe) + (w2 / m1_safe)
    denom_safe = torch.where(denom.abs() < eps, torch.sign(denom + 1e-12) * eps, denom)
    d[:, 1:-1] = torch.where(cond, (w1 + w2) / denom_safe, torch.zeros_like(m0))

    # Endpoints - with robust division
    denom_d0 = dx[:, 0] + dx[:, 1]
    denom_d0_safe = torch.where(denom_d0.abs() < eps, eps * torch.ones_like(denom_d0), denom_d0)
    d0 = ((2 * dx[:, 0] + dx[:, 1]) * m[:, 0] - dx[:, 0] * m[:, 1]) / denom_d0_safe

    denom_dn = dx[:, -1] + dx[:, -2]
    denom_dn_safe = torch.where(denom_dn.abs() < eps, eps * torch.ones_like(denom_dn), denom_dn)
    dn = ((2 * dx[:, -1] + dx[:, -2]) * m[:, -1] - dx[:, -1] * m[:, -2]) / denom_dn_safe

    def _adjust(d_end, m_end, m_adj):
        cond1 = torch.sign(d_end) != torch.sign(m_end)
        cond2 = (torch.sign(m_end) != torch.sign(m_adj)) & (d_end.abs() > (3 * m_end).abs())
        return torch.where(cond1 | cond2, torch.zeros_like(d_end), d_end)

    d[:, 0] = _adjust(d0, m[:, 0], m[:, 1])
    d[:, -1] = _adjust(dn, m[:, -1], m[:, -2])

    return d


def pchip_interpolate_batch(
    x_knots: torch.Tensor,
    y_knots: torch.Tensor,
    x_eval: torch.Tensor
) -> torch.Tensor:
    """
    Batch PCHIP interpolation for 1D data.
    Vectorized implementation for speed.

    Args:
        x_knots: (N, K) sorted knot locations
        y_knots: (N, K) knot values
        x_eval: (M,) evaluation points (will be broadcast to (N, M))

    Returns:
        y_eval: (N, M) interpolated values
    """
    batch, k = x_knots.shape
    n_eval = x_eval.numel()
    device = x_knots.device

    slopes = _pchip_slopes(x_knots, y_knots)
    
    # Expand x_eval to correlate with batch: (N, M)
    x_eval_batch = x_eval.unsqueeze(0).expand(batch, n_eval)
    
    # Find indices: (N, M)
    # searchsorted returns indices where elements should be inserted to maintain order
    # right=False: a[i-1] < v <= a[i]
    idx = torch.searchsorted(x_knots, x_eval_batch, right=False) - 1
    
    # Clamp to valid segment indices [0, k-2]
    idx = idx.clamp(0, k - 2)
    
    # Gather values for the corresponding segments
    # All gathered tensors will be (N, M)
    x0 = torch.gather(x_knots, 1, idx)
    x1 = torch.gather(x_knots, 1, idx + 1)
    y0 = torch.gather(y_knots, 1, idx)
    y1 = torch.gather(y_knots, 1, idx + 1)
    d0 = torch.gather(slopes, 1, idx)
    d1 = torch.gather(slopes, 1, idx + 1)

    # Perform interpolation
    h = x1 - x0
    # Add epsilon to handle cases where h=0 (though knots should be distinct)
    t = (x_eval_batch - x0) / (h + 1e-12)

    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t ** 2 * (3 - 2 * t)
    h11 = t ** 2 * (t - 1)

    y_eval = h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1

    return y_eval


def linear_interpolate_batch(
    x_knots: torch.Tensor,
    y_knots: torch.Tensor,
    x_eval: torch.Tensor
) -> torch.Tensor:
    """
    Batch piecewise-linear interpolation matching `pchip_interpolate_batch` API.

    Args:
        x_knots: (N, K) sorted knot locations
        y_knots: (N, K) knot values
        x_eval: (M,) evaluation points

    Returns:
        y_eval: (N, M) linearly interpolated values
    """
    batch, k = x_knots.shape
    n_eval = x_eval.numel()

    x_eval_batch = x_eval.unsqueeze(0).expand(batch, n_eval)
    idx = torch.searchsorted(x_knots, x_eval_batch, right=False) - 1
    idx = idx.clamp(0, k - 2)

    x0 = torch.gather(x_knots, 1, idx)
    x1 = torch.gather(x_knots, 1, idx + 1)
    y0 = torch.gather(y_knots, 1, idx)
    y1 = torch.gather(y_knots, 1, idx + 1)

    t = (x_eval_batch - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)


def reconstruct_curve(
    anchors: torch.Tensor,
    ctrl1: torch.Tensor,
    ctrl2: torch.Tensor,
    v_grid: torch.Tensor,
    j_end: float = 0.0,
    clamp_voc: bool = True,
    validate_monotonicity: bool = False
) -> torch.Tensor:
    """
    Reconstruct full J-V curve using split PCHIP interpolation.

    Args:
        anchors: (N, 4)
        ctrl1: (N, K)
        ctrl2: (N, K)
        v_grid: (M,)

    Returns:
        j_curve: (N, M)
    """
    v1_knots, j1_knots, v2_knots, j2_knots = build_knots(
        anchors, ctrl1, ctrl2, j_end=j_end, validate_monotonicity=validate_monotonicity
    )
    j1 = pchip_interpolate_batch(v1_knots, j1_knots, v_grid)
    j2 = pchip_interpolate_batch(v2_knots, j2_knots, v_grid)

    # IMPORTANT: keep per-sample voltages 1D for indexing.
    # Using (N, 1) here causes broadcasting in advanced indexing below.
    v_mpp_1d = anchors[:, 2]
    mask = v_grid.unsqueeze(0) <= v_mpp_1d.unsqueeze(1)
    j_curve = torch.where(mask, j1, j2)

    if clamp_voc:
        v_oc_1d = anchors[:, 1]
        if isinstance(j_end, torch.Tensor):
            j_end_tensor = j_end.to(j_curve.device).unsqueeze(1)
        else:
            j_end_tensor = torch.full_like(v_oc_1d, float(j_end)).unsqueeze(1)
        j_curve = torch.where(v_grid.unsqueeze(0) > v_oc_1d.unsqueeze(1), j_end_tensor, j_curve)
        j_curve = torch.where(v_grid.unsqueeze(0) < 0, j_end_tensor, j_curve)

    # CRITICAL: Replace any NaN values with interpolated fallback to prevent loss explosion
    if torch.isnan(j_curve).any():
        # Fallback: linear interpolation from Jsc to 0 over voltage range
        j_sc = anchors[:, 0].unsqueeze(1)
        v_oc = anchors[:, 1].unsqueeze(1)
        v_norm = v_grid.unsqueeze(0) / (v_oc + 1e-6)
        if isinstance(j_end, torch.Tensor):
            j_end_tensor = j_end.to(j_curve.device).unsqueeze(1)
        else:
            j_end_tensor = torch.full_like(j_sc, float(j_end))
        fallback = j_sc + (j_end_tensor - j_sc) * torch.clamp(v_norm, 0, 1)
        j_curve = torch.where(torch.isnan(j_curve), fallback, j_curve)

    batch_idx = torch.arange(j_curve.shape[0], device=j_curve.device)
    j_curve[batch_idx, 0] = anchors[:, 0]

    # Enforce exact knot at Vmpp (1D indexing!)
    idx_vmpp = torch.searchsorted(v_grid, v_mpp_1d, right=False).clamp(max=v_grid.numel() - 1)
    j_curve[batch_idx, idx_vmpp] = anchors[:, 3]

    if clamp_voc:
        v_oc_1d = anchors[:, 1]
        idx_cut = torch.searchsorted(v_grid, v_oc_1d, right=False).clamp(max=v_grid.numel() - 1)
        v_cut = v_grid[idx_cut]
        mask_cut = v_grid.unsqueeze(0) >= v_cut.unsqueeze(1)
        if isinstance(j_end, torch.Tensor):
            j_end_tensor = j_end.to(j_curve.device)
        else:
            j_end_tensor = torch.full_like(v_oc_1d, float(j_end))
        j_curve = torch.where(mask_cut, j_end_tensor.unsqueeze(1), j_curve)
        j_curve[batch_idx, idx_cut] = j_end_tensor

    return j_curve


def normalize_anchors_by_jsc(anchors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize anchors for Jsc-based curve normalization.

    Normalized space ([-1, 1] via 2*(J/Jsc)-1):
      - Jsc -> 1
      - Jmpp -> 2*(Jmpp/Jsc) - 1
      - J(Voc) -> -1 (handled via j_end in reconstruction)
    """
    anchors_norm = anchors.clone()
    j_sc = anchors[:, 0].clamp(min=eps)
    anchors_norm[:, 0] = 1.0
    anchors_norm[:, 3] = 2.0 * (anchors[:, 3] / j_sc) - 1.0
    return anchors_norm


def reconstruct_curve_normalized(
    anchors: torch.Tensor,
    ctrl1: torch.Tensor,
    ctrl2: torch.Tensor,
    v_grid: torch.Tensor,
    clamp_voc: bool = True,
    validate_monotonicity: bool = False
) -> torch.Tensor:
    """
    Reconstruct curve in normalized space ([-1, 1]) using Jsc normalization.
    """
    anchors_norm = normalize_anchors_by_jsc(anchors)
    return reconstruct_curve(
        anchors_norm,
        ctrl1,
        ctrl2,
        v_grid,
        j_end=-1.0,
        clamp_voc=clamp_voc,
        validate_monotonicity=validate_monotonicity
    )


def continuity_loss(
    anchors: torch.Tensor,
    ctrl1: torch.Tensor,
    ctrl2: torch.Tensor,
    v_grid: torch.Tensor,
    j_end: float = 0.0,
    validate_monotonicity: bool = False
) -> torch.Tensor:
    """
    Enforce C1 continuity at Vmpp using finite differences.
    """
    v1_knots, j1_knots, v2_knots, j2_knots = build_knots(
        anchors, ctrl1, ctrl2, j_end=j_end, validate_monotonicity=validate_monotonicity
    )
    j1 = pchip_interpolate_batch(v1_knots, j1_knots, v_grid)
    j2 = pchip_interpolate_batch(v2_knots, j2_knots, v_grid)

    v_mpp = anchors[:, 2]
    idx = torch.argmin((v_grid.unsqueeze(0) - v_mpp.unsqueeze(1)).abs(), dim=1)

    batch = anchors.shape[0]
    batch_idx = torch.arange(batch, device=anchors.device)
    j1_at = j1[batch_idx, idx]
    j2_at = j2[batch_idx, idx]
    l_value = ((j1_at - j2_at) ** 2).mean()

    idx_left = (idx - 1).clamp(min=0)
    idx_right = (idx + 1).clamp(max=v_grid.numel() - 1)

    dv = (v_grid[idx_right] - v_grid[idx_left]).clamp(min=1e-6)
    dj1 = (j1[batch_idx, idx_right] - j1[batch_idx, idx_left]) / dv
    dj2 = (j2[batch_idx, idx_right] - j2[batch_idx, idx_left]) / dv
    l_deriv = ((dj1 - dj2) ** 2).mean()

    return l_value + l_deriv
