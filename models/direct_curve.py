"""
Direct curve prediction model - simplified architecture.

Two variants:
1. DirectCurveNet: Predicts Jsc, Voc, and control points (fully self-contained)
2. DirectCurveNetWithJsc: Takes Jsc as input, only predicts Voc + control points
   (recommended since Jsc LGBM is already accurate: R²=0.965)

No Vmpp split - uses single-region PCHIP from (0, Jsc) to (Voc, 0).
This avoids cascade errors from inaccurate Voc/Vmpp/Jmpp predictions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

from .reconstruction import pchip_interpolate_batch


@dataclass
class DirectCurveNetConfig:
    """Configuration for direct curve prediction network."""
    input_dim: int = 31 + 71  # raw params + physics features
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.15
    activation: str = 'silu'
    ctrl_points: int = 6  # Interior control points between Jsc and 0


@dataclass
class DirectCurveNetWithJscConfig:
    """Configuration for direct curve network that takes Jsc as input."""
    input_dim: int = 31 + 71  # raw params + physics features
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.15
    activation: str = 'silu'
    ctrl_points: int = 6  # Interior control points


class DirectCurveNetWithJsc(nn.Module):
    """
    Direct curve prediction network that takes Jsc as input.

    RECOMMENDED: Use this variant since Jsc LGBM is already accurate (R²=0.965).

    Inputs:
    - features: (N, input_dim) raw params + physics features
    - jsc: (N,) short-circuit current from pretrained LGBM

    Outputs:
    - voc: (N,) open-circuit voltage
    - ctrl: (N, K) control points for curve shape

    Curve reconstruction:
    - Single-region PCHIP from (V=0, J=Jsc) to (V=Voc, J=0)
    - Monotonically decreasing J guaranteed by cumulative scaling
    - Jsc normalization preserved (2*(J/Jsc) - 1)
    """

    def __init__(self, config: DirectCurveNetWithJscConfig):
        super().__init__()
        self.config = config

        # Input: features + Jsc scalar
        total_input = config.input_dim + 1

        # Shared backbone
        layers = []
        prev_dim = total_input
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Voc head - predict positive voltage
        # Output is a residual around typical Voc (~1.0V)
        self.head_voc = nn.Sequential(
            nn.Linear(prev_dim, 32),
            self._get_activation(config.activation),
            nn.Linear(32, 1)
        )

        # Control points head - predict shape weights
        self.head_ctrl = nn.Sequential(
            nn.Linear(prev_dim, 32),
            self._get_activation(config.activation),
            nn.Linear(32, config.ctrl_points),
            nn.Softplus()  # Positive for cumsum
        )

        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'relu': nn.ReLU(),
        }
        return activations.get(name, nn.SiLU())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Voc bias: center around 1.0V
        with torch.no_grad():
            self.head_voc[-1].bias.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        jsc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (N, input_dim) normalized features
            jsc: (N,) short-circuit current from LGBM

        Returns:
            voc: (N,) open-circuit voltage (positive, clamped)
            ctrl: (N, K) control points (positive, for cumsum)
        """
        # Concatenate features with Jsc
        jsc_input = jsc.unsqueeze(1) / 50.0  # Normalize Jsc to ~O(1)
        combined = torch.cat([x, jsc_input], dim=1)

        features = self.backbone(combined)

        # Voc prediction with positivity guarantee
        voc_raw = self.head_voc(features).squeeze(-1)
        voc = torch.clamp(voc_raw, min=0.1, max=2.0)  # Reasonable range

        # Control points
        ctrl = self.head_ctrl(features)

        return voc, ctrl


class DirectCurveNet(nn.Module):
    """
    Fully self-contained direct curve prediction network.

    Predicts everything: Jsc, Voc, and control points.
    Use this if you don't have a pretrained Jsc model.

    For most cases, use DirectCurveNetWithJsc instead since Jsc LGBM is accurate.
    """

    def __init__(self, config: DirectCurveNetConfig):
        super().__init__()
        self.config = config

        # Shared backbone
        layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Jsc head
        self.head_jsc = nn.Sequential(
            nn.Linear(prev_dim, 64),
            self._get_activation(config.activation),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        # Voc head
        self.head_voc = nn.Sequential(
            nn.Linear(prev_dim, 64),
            self._get_activation(config.activation),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        # Control points head
        self.head_ctrl = nn.Sequential(
            nn.Linear(prev_dim, 64),
            self._get_activation(config.activation),
            nn.Linear(64, config.ctrl_points),
            nn.Softplus()
        )

        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'relu': nn.ReLU(),
        }
        return activations.get(name, nn.SiLU())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.head_jsc[-2].bias.fill_(3.0)
            self.head_voc[-2].bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (N, input_dim) normalized features

        Returns:
            jsc: (N,) short-circuit current
            voc: (N,) open-circuit voltage
            ctrl: (N, K) control points
        """
        features = self.backbone(x)

        jsc = self.head_jsc(features).squeeze(-1) * 10.0
        voc = self.head_voc(features).squeeze(-1) + 0.5
        ctrl = self.head_ctrl(features)

        return jsc, voc, ctrl


def build_knots_single_region(
    jsc: torch.Tensor,
    voc: torch.Tensor,
    ctrl: torch.Tensor,
    eps: float = 1e-4,
    normalized: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build voltage and current knots for single-region PCHIP.

    Creates K+2 knots:
    - (V=0, J=Jsc or 1.0) at start
    - K interior points with monotonically decreasing J
    - (V=Voc, J=0 or -1.0) at end

    Args:
        jsc: (N,) short-circuit current
        voc: (N,) open-circuit voltage
        ctrl: (N, K) control point weights (positive)
        normalized: If True, build knots in normalized space [-1, 1]
                   where J=1 at V=0 and J=-1 at V=Voc

    Returns:
        v_knots: (N, K+2) voltage positions
        j_knots: (N, K+2) current values (monotonically decreasing)
    """
    n_ctrl = ctrl.shape[1]
    device = ctrl.device
    batch = ctrl.shape[0]

    # Voltage knots: uniformly spaced from 0 to Voc
    v_norm = torch.linspace(0, 1, n_ctrl + 2, device=device)
    v_knots = v_norm.unsqueeze(0) * voc.unsqueeze(1)  # (N, K+2)

    # Current knots: monotonically decreasing
    # Use cumulative sum to guarantee monotonicity
    ctrl_cumsum = torch.cumsum(ctrl, dim=1)  # (N, K)
    ctrl_final = ctrl_cumsum[:, -1:].clamp(min=eps)
    ctrl_scale = ctrl_cumsum / ctrl_final  # Normalized to [0, 1]

    if normalized:
        # Normalized space: J goes from 1.0 to -1.0
        # j_interior = 1.0 - 2.0 * scale (scale: 0->1 maps to J: 1->-1)
        j_interior = 1.0 - 2.0 * ctrl_scale  # (N, K)
        j_start = torch.ones(batch, 1, device=device)
        j_end = torch.full((batch, 1), -1.0, device=device)
    else:
        # Absolute space: J goes from Jsc to 0
        j_interior = jsc.unsqueeze(1) * (1.0 - ctrl_scale)  # (N, K)
        j_start = jsc.unsqueeze(1)
        j_end = torch.zeros(batch, 1, device=device)

    j_knots = torch.cat([j_start, j_interior, j_end], dim=1)  # (N, K+2)

    return v_knots, j_knots


def reconstruct_curve_direct(
    jsc: torch.Tensor,
    voc: torch.Tensor,
    ctrl: torch.Tensor,
    v_grid: torch.Tensor,
    clamp_voc: bool = True
) -> torch.Tensor:
    """
    Reconstruct J-V curve using single-region PCHIP interpolation.

    Args:
        jsc: (N,) short-circuit current
        voc: (N,) open-circuit voltage
        ctrl: (N, K) control point weights
        v_grid: (M,) voltage evaluation points
        clamp_voc: If True, set J=0 for V > Voc

    Returns:
        j_curve: (N, M) reconstructed current values in ABSOLUTE space
    """
    v_knots, j_knots = build_knots_single_region(jsc, voc, ctrl, normalized=False)

    # PCHIP interpolation
    j_curve = pchip_interpolate_batch(v_knots, j_knots, v_grid)

    # Enforce exact endpoints
    batch_idx = torch.arange(j_curve.shape[0], device=j_curve.device)
    j_curve[batch_idx, 0] = jsc

    # Clamp to 0 for V > Voc
    if clamp_voc:
        voc_expanded = voc.unsqueeze(1)
        j_curve = torch.where(v_grid.unsqueeze(0) > voc_expanded,
                              torch.zeros_like(j_curve), j_curve)

    # Ensure non-negative (should already be, but safety check)
    j_curve = torch.clamp(j_curve, min=0.0)

    return j_curve


def reconstruct_curve_direct_normalized(
    jsc: torch.Tensor,
    voc: torch.Tensor,
    ctrl: torch.Tensor,
    v_grid: torch.Tensor,
    clamp_voc: bool = True
) -> torch.Tensor:
    """
    Reconstruct curve in NORMALIZED space [-1, 1].

    The curve is normalized by Jsc: J_norm = 2 * (J / Jsc) - 1
    So J=Jsc -> 1, J=0 -> -1

    This is used for training with Isc-normalized curves.

    Args:
        jsc: (N,) short-circuit current (used only for Voc knot placement)
        voc: (N,) open-circuit voltage
        ctrl: (N, K) control point weights
        v_grid: (M,) voltage evaluation points
        clamp_voc: If True, set J=-1 for V > Voc

    Returns:
        j_curve_norm: (N, M) normalized current values in [-1, 1]
    """
    # Build knots directly in normalized space
    v_knots, j_knots = build_knots_single_region(jsc, voc, ctrl, normalized=True)

    # PCHIP interpolation in normalized space
    j_curve_norm = pchip_interpolate_batch(v_knots, j_knots, v_grid)

    # Enforce exact endpoints
    batch_idx = torch.arange(j_curve_norm.shape[0], device=j_curve_norm.device)
    j_curve_norm[batch_idx, 0] = 1.0  # J/Jsc = 1 at V=0

    # Clamp to -1 for V > Voc
    if clamp_voc:
        voc_expanded = voc.unsqueeze(1)
        j_curve_norm = torch.where(v_grid.unsqueeze(0) > voc_expanded,
                                   torch.full_like(j_curve_norm, -1.0), j_curve_norm)

    # Clamp to valid range
    j_curve_norm = torch.clamp(j_curve_norm, min=-1.0, max=1.0)

    return j_curve_norm


class DirectCurveLoss(nn.Module):
    """
    Loss function for fully self-contained DirectCurveNet.

    Components:
    1. Curve MSE: Match predicted curve to target
    2. Jsc MSE: Ensure first point matches (high weight)
    3. Voc MSE: Ensure zero-crossing matches (moderate weight)
    4. Smoothness: Penalize large second derivatives
    """

    def __init__(
        self,
        weight_jsc: float = 10.0,
        weight_voc: float = 5.0,
        weight_smooth: float = 0.1
    ):
        super().__init__()
        self.weight_jsc = weight_jsc
        self.weight_voc = weight_voc
        self.weight_smooth = weight_smooth

    def forward(
        self,
        pred_curve: torch.Tensor,
        true_curve: torch.Tensor,
        pred_jsc: torch.Tensor,
        true_jsc: torch.Tensor,
        pred_voc: torch.Tensor,
        true_voc: torch.Tensor,
        v_grid: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss.

        All inputs should be in absolute space (not normalized).
        """
        # Curve MSE - relative error normalized by Jsc
        jsc_ref = true_jsc.unsqueeze(1).clamp(min=1e-6)
        rel_err = (pred_curve - true_curve) / jsc_ref
        loss_curve = (rel_err ** 2).mean()

        # Jsc MSE - match first point exactly
        loss_jsc = ((pred_jsc - true_jsc) ** 2).mean()

        # Voc MSE - match zero-crossing
        loss_voc = ((pred_voc - true_voc) ** 2).mean()

        # Smoothness - penalize non-smooth curves via second derivative
        # d2J/dV2 approximated by finite differences
        dv = v_grid[1:] - v_grid[:-1]
        dj = pred_curve[:, 1:] - pred_curve[:, :-1]
        slopes = dj / (dv.unsqueeze(0) + 1e-6)
        d2j = slopes[:, 1:] - slopes[:, :-1]
        loss_smooth = (d2j ** 2).mean()

        # Total loss
        loss_total = (
            loss_curve +
            self.weight_jsc * loss_jsc +
            self.weight_voc * loss_voc +
            self.weight_smooth * loss_smooth
        )

        metrics = {
            'loss_total': loss_total.item(),
            'loss_curve': loss_curve.item(),
            'loss_jsc': loss_jsc.item(),
            'loss_voc': loss_voc.item(),
            'loss_smooth': loss_smooth.item(),
        }

        return loss_total, metrics


class DirectCurveLossWithJsc(nn.Module):
    """
    Loss function for DirectCurveNetWithJsc (Jsc provided as input).

    Since Jsc is provided, we only penalize:
    1. Curve MSE: Match predicted curve to target (in normalized space)
    2. Voc MSE: Ensure zero-crossing matches
    3. Smoothness: Penalize large second derivatives
    """

    def __init__(
        self,
        weight_voc: float = 5.0,
        weight_smooth: float = 0.1
    ):
        super().__init__()
        self.weight_voc = weight_voc
        self.weight_smooth = weight_smooth

    def forward(
        self,
        pred_curve_norm: torch.Tensor,
        true_curve_norm: torch.Tensor,
        pred_voc: torch.Tensor,
        true_voc: torch.Tensor,
        v_grid: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss in NORMALIZED space.

        Args:
            pred_curve_norm: (N, M) predicted curve in [-1, 1] space
            true_curve_norm: (N, M) target curve in [-1, 1] space
            pred_voc: (N,) predicted Voc
            true_voc: (N,) true Voc
            v_grid: (M,) voltage grid

        Returns:
            loss: Total loss
            metrics: Dict of individual loss components
        """
        # Curve MSE in normalized space
        loss_curve = ((pred_curve_norm - true_curve_norm) ** 2).mean()

        # Voc MSE (relative)
        voc_rel_err = (pred_voc - true_voc) / (true_voc.clamp(min=0.1))
        loss_voc = (voc_rel_err ** 2).mean()

        # Smoothness in normalized space
        dv = v_grid[1:] - v_grid[:-1]
        dj = pred_curve_norm[:, 1:] - pred_curve_norm[:, :-1]
        slopes = dj / (dv.unsqueeze(0) + 1e-6)
        d2j = slopes[:, 1:] - slopes[:, :-1]
        loss_smooth = (d2j ** 2).mean()

        # Total loss
        loss_total = (
            loss_curve +
            self.weight_voc * loss_voc +
            self.weight_smooth * loss_smooth
        )

        metrics = {
            'loss_total': loss_total.item(),
            'loss_curve': loss_curve.item(),
            'loss_voc': loss_voc.item(),
            'loss_smooth': loss_smooth.item(),
        }

        return loss_total, metrics


def extract_voc_from_curve(curves: torch.Tensor, v_grid: torch.Tensor) -> torch.Tensor:
    """
    Extract Voc from J-V curves by finding zero-crossing.

    Voc is the voltage where J first becomes <= 0.

    Args:
        curves: (N, M) current values
        v_grid: (M,) voltage grid

    Returns:
        voc: (N,) open-circuit voltage
    """
    # Find first index where J <= 0
    zero_mask = curves <= 0

    # For each sample, find first True index
    # If no zero crossing, use last voltage
    first_zero = zero_mask.float().argmax(dim=1)
    no_crossing = ~zero_mask.any(dim=1)
    first_zero[no_crossing] = len(v_grid) - 1

    # Linear interpolation to find exact crossing
    # Use the point before and at the crossing
    batch = curves.shape[0]
    device = curves.device

    idx_after = first_zero.clamp(min=1, max=len(v_grid)-1)
    idx_before = idx_after - 1

    batch_idx = torch.arange(batch, device=device)
    j_before = curves[batch_idx, idx_before]
    j_after = curves[batch_idx, idx_after]
    v_before = v_grid[idx_before]
    v_after = v_grid[idx_after]

    # Linear interpolation: V = V_before + (0 - J_before) * (V_after - V_before) / (J_after - J_before)
    dj = j_after - j_before
    dj_safe = torch.where(dj.abs() < 1e-8, torch.ones_like(dj), dj)
    t = -j_before / dj_safe
    t = t.clamp(0, 1)

    voc = v_before + t * (v_after - v_before)

    # Handle no-crossing case
    voc[no_crossing] = v_grid[-1]

    return voc
