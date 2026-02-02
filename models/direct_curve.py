"""
Direct curve prediction model - simplified architecture.

Three variants:
1. DirectCurveNet: Predicts Jsc, Voc, and control points (fully self-contained)
2. DirectCurveNetWithJsc: Takes Jsc as input, only predicts Voc + control points
3. DirectCurveShapeNet: Takes Jsc AND Voc as inputs, only predicts shape (RECOMMENDED)
   - Uses pretrained Jsc LGBM (R²=0.965) and Voc NN (R²=0.73)
   - Separates shape learning from endpoint prediction
   - Uses non-uniform knot placement for better knee capture

No Vmpp split - uses single-region PCHIP from (0, Jsc) to (Voc, 0).
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


@dataclass
class DirectCurveShapeNetConfig:
    """
    Configuration for shape-only curve network.

    RECOMMENDED: Uses pretrained Jsc LGBM and Voc NN for endpoints,
    only predicts the curve shape via control points.
    """
    input_dim: int = 31 + 71  # raw params + physics features
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1
    activation: str = 'silu'
    ctrl_points: int = 8  # More control points for better shape capture
    use_residual: bool = True  # Residual connections for better training


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


class DirectCurveShapeNet(nn.Module):
    """
    Shape-only curve prediction network (RECOMMENDED).

    Takes BOTH Jsc AND Voc as inputs from pretrained models.
    Only predicts the curve SHAPE via control points.

    This separates two orthogonal learning tasks:
    1. Endpoint prediction (Jsc, Voc) - handled by pretrained models
    2. Shape prediction (control points) - handled by this network

    Uses non-uniform knot placement with more density near the knee region.
    """

    def __init__(self, config: DirectCurveShapeNetConfig):
        super().__init__()
        self.config = config

        # Input: features + Jsc + Voc
        total_input = config.input_dim + 2

        # Shared backbone with optional residual connections
        if config.use_residual and len(config.hidden_dims) >= 2:
            self.backbone = self._build_residual_backbone(total_input, config)
        else:
            self.backbone = self._build_simple_backbone(total_input, config)

        # Final dimension after backbone
        final_dim = config.hidden_dims[-1]

        # Control points head - predict shape coefficients
        # Using sigmoid to get values in [0, 1] directly, avoiding cumsum issues
        self.head_ctrl = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.LayerNorm(64),
            self._get_activation(config.activation),
            nn.Linear(64, config.ctrl_points),
            nn.Sigmoid()  # Output in [0, 1] for monotonic mapping
        )

        self._init_weights()

    def _build_simple_backbone(self, input_dim: int, config) -> nn.Module:
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    def _build_residual_backbone(self, input_dim: int, config) -> nn.Module:
        """Build backbone with residual connections for better gradient flow."""
        class ResidualBlock(nn.Module):
            def __init__(self, dim, dropout, activation):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    activation,
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                )
                self.activation = activation

            def forward(self, x):
                return self.activation(x + self.block(x))

        layers = []
        # Initial projection
        layers.append(nn.Linear(input_dim, config.hidden_dims[0]))
        layers.append(nn.LayerNorm(config.hidden_dims[0]))
        layers.append(self._get_activation(config.activation))

        # Residual blocks at each hidden dimension
        for i, hidden_dim in enumerate(config.hidden_dims):
            if i > 0:
                # Dimension change
                layers.append(nn.Linear(config.hidden_dims[i-1], hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(self._get_activation(config.activation))
            # Add residual block
            layers.append(ResidualBlock(hidden_dim, config.dropout, self._get_activation(config.activation)))

        return nn.Sequential(*layers)

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
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        jsc: torch.Tensor,
        voc: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (N, input_dim) normalized features
            jsc: (N,) short-circuit current from LGBM
            voc: (N,) open-circuit voltage from Voc NN

        Returns:
            ctrl: (N, K) control points in [0, 1] for monotonic J interpolation
        """
        # Normalize inputs to ~O(1)
        jsc_input = jsc.unsqueeze(1) / 50.0
        voc_input = voc.unsqueeze(1)  # Already ~O(1)
        combined = torch.cat([x, jsc_input, voc_input], dim=1)

        features = self.backbone(combined)
        ctrl = self.head_ctrl(features)

        return ctrl


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


def build_knots_nonuniform(
    voc: torch.Tensor,
    ctrl: torch.Tensor,
    normalized: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build voltage and current knots with NON-UNIFORM spacing.

    Places more knots near the knee region (around 0.7-0.9 * Voc) where
    the curve shape changes most rapidly.

    Uses sigmoid control points directly as the J/Jsc fraction at each knot.
    Monotonicity is enforced by sorting.

    Args:
        voc: (N,) open-circuit voltage
        ctrl: (N, K) control points from sigmoid, values in [0, 1]
        normalized: If True, J goes from 1 to -1, else from 1 to 0

    Returns:
        v_knots: (N, K+2) voltage positions (non-uniform in [0, Voc])
        j_knots: (N, K+2) current values (monotonically decreasing)
    """
    n_ctrl = ctrl.shape[1]
    device = ctrl.device
    batch = ctrl.shape[0]

    # Non-uniform voltage positions: denser near the knee (0.7-0.95 of Voc)
    # Use a quadratic mapping that places more points near the end
    # v_norm_raw = [0, 0.1, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 1.0] for 8 ctrl points
    if n_ctrl == 8:
        v_norm_raw = torch.tensor([0.0, 0.1, 0.25, 0.45, 0.65, 0.78, 0.88, 0.94, 0.98, 1.0], device=device)
    elif n_ctrl == 6:
        v_norm_raw = torch.tensor([0.0, 0.15, 0.35, 0.6, 0.8, 0.92, 0.98, 1.0], device=device)
    else:
        # Fallback: use quadratic spacing for any number of control points
        t = torch.linspace(0, 1, n_ctrl + 2, device=device)
        # Quadratic mapping: more points near 1.0
        v_norm_raw = t ** 0.7  # Slightly skewed toward higher V

    v_knots = v_norm_raw.unsqueeze(0) * voc.unsqueeze(1)  # (N, K+2)

    # Control points represent J/Jsc ratio at each interior knot
    # They should be monotonically DECREASING from ~1 to ~0
    # Sigmoid output is [0, 1], we use it as "fraction of J remaining"
    # Sort to ensure monotonicity
    ctrl_sorted, _ = torch.sort(ctrl, dim=1, descending=True)  # Descending: high to low

    if normalized:
        # Normalized: J goes from 1 (at V=0) to -1 (at V=Voc)
        # Interior J = 2 * (J/Jsc) - 1 = 2 * ctrl_sorted - 1
        j_interior = 2.0 * ctrl_sorted - 1.0  # (N, K)
        j_start = torch.ones(batch, 1, device=device)
        j_end = torch.full((batch, 1), -1.0, device=device)
    else:
        # Absolute: J goes from 1 (=Jsc/Jsc) to 0
        j_interior = ctrl_sorted  # (N, K), fraction of Jsc
        j_start = torch.ones(batch, 1, device=device)
        j_end = torch.zeros(batch, 1, device=device)

    j_knots = torch.cat([j_start, j_interior, j_end], dim=1)  # (N, K+2)

    return v_knots, j_knots


def reconstruct_curve_shape(
    voc: torch.Tensor,
    ctrl: torch.Tensor,
    v_grid: torch.Tensor,
    clamp_voc: bool = True
) -> torch.Tensor:
    """
    Reconstruct curve using shape-only control points (for DirectCurveShapeNet).

    Works in NORMALIZED space where J/Jsc goes from 1 to -1.
    Uses non-uniform knot placement for better knee capture.

    Args:
        voc: (N,) open-circuit voltage
        ctrl: (N, K) control points from sigmoid [0, 1]
        v_grid: (M,) voltage evaluation points
        clamp_voc: If True, set J=-1 for V > Voc

    Returns:
        j_curve_norm: (N, M) normalized curve in [-1, 1]
    """
    v_knots, j_knots = build_knots_nonuniform(voc, ctrl, normalized=True)

    # PCHIP interpolation
    j_curve_norm = pchip_interpolate_batch(v_knots, j_knots, v_grid)

    # Enforce exact start point
    batch_idx = torch.arange(j_curve_norm.shape[0], device=j_curve_norm.device)
    j_curve_norm[batch_idx, 0] = 1.0  # J/Jsc = 1 at V=0

    # Clamp to -1 for V > Voc
    if clamp_voc:
        voc_expanded = voc.unsqueeze(1)
        j_curve_norm = torch.where(
            v_grid.unsqueeze(0) > voc_expanded,
            torch.full_like(j_curve_norm, -1.0),
            j_curve_norm
        )

    # Clamp to valid range
    j_curve_norm = torch.clamp(j_curve_norm, min=-1.0, max=1.0)

    return j_curve_norm


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


class DirectCurveShapeLoss(nn.Module):
    """
    Loss function for DirectCurveShapeNet (Jsc and Voc provided as inputs).

    Since both endpoints are provided, we ONLY penalize curve shape:
    1. Curve MSE with regional weighting (knee region weighted higher)
    2. Smoothness (penalize non-physical oscillations)
    3. Monotonicity (soft penalty for J increasing with V)
    """

    def __init__(
        self,
        weight_smooth: float = 0.05,
        weight_mono: float = 1.0,
        knee_weight: float = 2.0
    ):
        super().__init__()
        self.weight_smooth = weight_smooth
        self.weight_mono = weight_mono
        self.knee_weight = knee_weight

    def forward(
        self,
        pred_curve_norm: torch.Tensor,
        true_curve_norm: torch.Tensor,
        voc: torch.Tensor,
        v_grid: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute shape-only loss in NORMALIZED space.

        Args:
            pred_curve_norm: (N, M) predicted curve in [-1, 1] space
            true_curve_norm: (N, M) target curve in [-1, 1] space
            voc: (N,) Voc values (for regional weighting)
            v_grid: (M,) voltage grid

        Returns:
            loss: Total loss
            metrics: Dict of individual loss components
        """
        batch_size = pred_curve_norm.shape[0]

        # Regional weighting: higher weight in knee region (0.6-0.95 of Voc)
        # This is where the curve shape matters most for FF
        voc_expanded = voc.unsqueeze(1)  # (N, 1)
        v_ratio = v_grid.unsqueeze(0) / (voc_expanded + 1e-6)  # (N, M)

        # Weight mask: 1.0 everywhere, knee_weight in knee region
        weights = torch.ones_like(pred_curve_norm)
        knee_mask = (v_ratio >= 0.5) & (v_ratio <= 0.95)
        weights = torch.where(knee_mask, self.knee_weight * weights, weights)

        # Weighted MSE
        sq_err = (pred_curve_norm - true_curve_norm) ** 2
        loss_curve = (weights * sq_err).sum() / weights.sum()

        # Smoothness: penalize second derivative
        dv = v_grid[1:] - v_grid[:-1]
        dj = pred_curve_norm[:, 1:] - pred_curve_norm[:, :-1]
        slopes = dj / (dv.unsqueeze(0) + 1e-6)
        d2j = slopes[:, 1:] - slopes[:, :-1]
        loss_smooth = (d2j ** 2).mean()

        # Monotonicity penalty: J should decrease with V (dJ/dV <= 0)
        # In normalized space, this means dj should be <= 0
        mono_violation = torch.relu(dj)  # Positive where J increases
        loss_mono = (mono_violation ** 2).mean()

        # Total loss
        loss_total = (
            loss_curve +
            self.weight_smooth * loss_smooth +
            self.weight_mono * loss_mono
        )

        metrics = {
            'loss_total': loss_total.item(),
            'loss_curve': loss_curve.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_mono': loss_mono.item(),
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
