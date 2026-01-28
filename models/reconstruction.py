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
    validate_monotonicity: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build knot points for split spline reconstruction.

    Args:
        anchors: (N, 4) [Jsc, Voc, Vmpp, Jmpp]
        ctrl1: (N, K) control points for region 1 (0->Vmpp)
        ctrl2: (N, K) control points for region 2 (Vmpp->Voc)

    Returns:
        v1_knots, j1_knots, v2_knots, j2_knots
    """
    j_sc, v_oc, v_mpp, j_mpp = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    n_ctrl = ctrl1.shape[1]
    device = anchors.device

    v1_norm = torch.linspace(0, 1, n_ctrl + 2, device=device)
    v1_knots = v1_norm.unsqueeze(0) * v_mpp.unsqueeze(1)

    eps = 1e-6
    j1_cumsum = torch.cumsum(ctrl1, dim=1)
    j1_scale = j1_cumsum / (j1_cumsum[:, -1:] + eps)
    j1_interior = j_sc.unsqueeze(1) - j1_scale * (j_sc - j_mpp).unsqueeze(1)
    j1_knots = torch.cat([j_sc.unsqueeze(1), j1_interior, j_mpp.unsqueeze(1)], dim=1)

    v2_norm = torch.linspace(0, 1, n_ctrl + 2, device=device)
    v2_knots = v_mpp.unsqueeze(1) + v2_norm.unsqueeze(0) * (v_oc - v_mpp).unsqueeze(1)

    j2_cumsum = torch.cumsum(ctrl2, dim=1)
    j2_scale = j2_cumsum / (j2_cumsum[:, -1:] + eps)
    j2_interior = j_mpp.unsqueeze(1) * (1 - j2_scale)
    j2_knots = torch.cat(
        [j_mpp.unsqueeze(1), j2_interior, torch.zeros_like(j_mpp).unsqueeze(1)],
        dim=1
    )

    if validate_monotonicity:
        viol1 = j1_knots[:, 1:] > (j1_knots[:, :-1] + 1e-6)
        viol2 = j2_knots[:, 1:] > (j2_knots[:, :-1] + 1e-6)
        if viol1.any():
            idx = viol1.any(dim=1).nonzero(as_tuple=False)[0].item()
            raise ValueError(f"Region 1 knots not monotonic for sample {idx}: {j1_knots[idx].detach().cpu()}")
        if viol2.any():
            idx = viol2.any(dim=1).nonzero(as_tuple=False)[0].item()
            raise ValueError(f"Region 2 knots not monotonic for sample {idx}: {j2_knots[idx].detach().cpu()}")

    return v1_knots, j1_knots, v2_knots, j2_knots


def _pchip_slopes(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute PCHIP slopes using Fritsch-Carlson method."""
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    m = dy / (dx + eps)

    d = torch.zeros_like(y)

    # Interior points
    m0 = m[:, :-1]
    m1 = m[:, 1:]
    h0 = dx[:, :-1]
    h1 = dx[:, 1:]

    w1 = 2 * h1 + h0
    w2 = h1 + 2 * h0
    cond = (m0 * m1) > 0
    denom = (w1 / (m0 + eps)) + (w2 / (m1 + eps))
    d[:, 1:-1] = torch.where(cond, (w1 + w2) / (denom + eps), torch.zeros_like(m0))

    # Endpoints
    d0 = ((2 * dx[:, 0] + dx[:, 1]) * m[:, 0] - dx[:, 0] * m[:, 1]) / (dx[:, 0] + dx[:, 1] + eps)
    dn = ((2 * dx[:, -1] + dx[:, -2]) * m[:, -1] - dx[:, -1] * m[:, -2]) / (dx[:, -1] + dx[:, -2] + eps)

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

    Args:
        x_knots: (N, K) sorted knot locations
        y_knots: (N, K) knot values
        x_eval: (M,) evaluation points (shared across batch)

    Returns:
        y_eval: (N, M) interpolated values
    """
    batch, k = x_knots.shape
    n_eval = x_eval.numel()
    device = x_knots.device

    slopes = _pchip_slopes(x_knots, y_knots)
    y_eval = torch.empty(batch, n_eval, device=device, dtype=y_knots.dtype)

    for i in range(batch):
        xi = x_knots[i]
        yi = y_knots[i]
        di = slopes[i]

        idx = torch.bucketize(x_eval, xi) - 1
        idx = idx.clamp(0, k - 2)

        x0 = xi[idx]
        x1 = xi[idx + 1]
        h = x1 - x0
        t = (x_eval - x0) / (h + 1e-12)

        y0 = yi[idx]
        y1 = yi[idx + 1]
        d0 = di[idx]
        d1 = di[idx + 1]

        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t ** 2 * (3 - 2 * t)
        h11 = t ** 2 * (t - 1)

        y_eval[i] = h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1

    return y_eval


def reconstruct_curve(
    anchors: torch.Tensor,
    ctrl1: torch.Tensor,
    ctrl2: torch.Tensor,
    v_grid: torch.Tensor,
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
        anchors, ctrl1, ctrl2, validate_monotonicity=validate_monotonicity
    )
    j1 = pchip_interpolate_batch(v1_knots, j1_knots, v_grid)
    j2 = pchip_interpolate_batch(v2_knots, j2_knots, v_grid)

    v_mpp = anchors[:, 2].unsqueeze(1)
    mask = v_grid.unsqueeze(0) <= v_mpp
    j_curve = torch.where(mask, j1, j2)

    if clamp_voc:
        v_oc = anchors[:, 1].unsqueeze(1)
        j_curve = torch.where(v_grid.unsqueeze(0) > v_oc, torch.zeros_like(j_curve), j_curve)
    return j_curve


def continuity_loss(
    anchors: torch.Tensor,
    ctrl1: torch.Tensor,
    ctrl2: torch.Tensor,
    v_grid: torch.Tensor,
    validate_monotonicity: bool = False
) -> torch.Tensor:
    """
    Enforce C1 continuity at Vmpp using finite differences.
    """
    v1_knots, j1_knots, v2_knots, j2_knots = build_knots(
        anchors, ctrl1, ctrl2, validate_monotonicity=validate_monotonicity
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
