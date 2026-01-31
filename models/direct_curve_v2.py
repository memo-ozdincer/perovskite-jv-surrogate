"""
Direct Curve Prediction V2 - High-accuracy architecture inspired by:
- Zbinden et al. (2026): Autoencoder with latent parameter loss
- Toprak (2025): Simple MLP with high R² (>0.9996)

Key improvements over V1:
1. Direct 45-point prediction (no control point bottleneck)
2. Conv1D decoder for spatially coherent curves
3. Parameter reconstruction head (latent loss a la Zbinden)
4. Hard monotonicity via cumulative decrements
5. Physics-informed endpoint constraints
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DirectCurveNetV2Config:
    """Configuration for V2 direct curve network."""
    input_dim: int = 102  # 31 raw params + 71 physics features
    n_curve_points: int = 45  # Number of output curve points
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 256])
    use_conv_decoder: bool = True  # Use Conv1D for smoother curves
    conv_channels: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    activation: str = 'silu'
    # Parameter reconstruction (latent loss like Zbinden)
    use_param_reconstruction: bool = True
    n_params_to_reconstruct: int = 6  # Key physics params to reconstruct
    param_hidden_dim: int = 128


class DirectCurveNetV2(nn.Module):
    """
    Direct curve prediction with optional parameter reconstruction.

    Architecture:
    1. Encoder: MLP that processes input features
    2. Curve Head: Predicts all 45 curve points directly
       - Option A (Conv1D): Uses transposed convolutions for spatial coherence
       - Option B (MLP): Direct linear projection
    3. Parameter Head (optional): Reconstructs key physics parameters
       - Enables latent parameter loss (Zbinden approach)

    Monotonicity is enforced via cumulative decrement scaling:
    - Network predicts positive decrements
    - Curve = Jsc - cumsum(decrements) * Jsc / sum(decrements)
    - Guarantees monotonic decrease from Jsc to 0
    """

    def __init__(self, config: DirectCurveNetV2Config):
        super().__init__()
        self.config = config

        # Activation function
        self.activation = self._get_activation(config.activation)

        # Encoder: MLP backbone
        encoder_layers = []
        prev_dim = config.input_dim + 1  # +1 for Jsc input
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout),
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Curve decoder
        if config.use_conv_decoder:
            self.curve_decoder = self._build_conv_decoder(prev_dim, config)
        else:
            self.curve_decoder = self._build_mlp_decoder(prev_dim, config)

        # Voc prediction head (needed for endpoint)
        self.voc_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.LayerNorm(64),
            self._get_activation(config.activation),
            nn.Linear(64, 1),
        )

        # Parameter reconstruction head (optional, for latent loss)
        if config.use_param_reconstruction:
            self.param_head = nn.Sequential(
                nn.Linear(prev_dim, config.param_hidden_dim),
                nn.LayerNorm(config.param_hidden_dim),
                self._get_activation(config.activation),
                nn.Linear(config.param_hidden_dim, config.n_params_to_reconstruct),
            )
        else:
            self.param_head = None

        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),  # Like Toprak's tansig
        }
        return activations.get(name, nn.SiLU())

    def _build_conv_decoder(self, input_dim: int, config: DirectCurveNetV2Config) -> nn.Module:
        """
        Build Conv1D-based decoder for spatially coherent curve prediction.

        Uses transposed convolutions to upsample from latent to curve space.
        This produces smoother curves than direct MLP prediction.
        """
        # First, project to a small spatial dimension
        initial_len = 8  # Start with 8 points, upsample to 45

        return nn.Sequential(
            # Project to initial spatial dimension
            nn.Linear(input_dim, config.conv_channels[0] * initial_len),
            nn.Unflatten(1, (config.conv_channels[0], initial_len)),

            # Conv1D blocks with upsampling
            nn.ConvTranspose1d(config.conv_channels[0], config.conv_channels[0],
                              kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm1d(config.conv_channels[0]),
            self._get_activation(config.activation),

            nn.ConvTranspose1d(config.conv_channels[0], config.conv_channels[1],
                              kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm1d(config.conv_channels[1]),
            self._get_activation(config.activation),

            # Final conv to get close to 45 points
            nn.ConvTranspose1d(config.conv_channels[1], config.conv_channels[1],
                              kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm1d(config.conv_channels[1]),
            self._get_activation(config.activation),

            # Output projection
            nn.Conv1d(config.conv_channels[1], 1, kernel_size=1),
            nn.Flatten(1),

            # Exact size projection
            nn.Linear(64, config.n_curve_points),
            nn.Softplus(),  # Positive decrements for monotonicity
        )

    def _build_mlp_decoder(self, input_dim: int, config: DirectCurveNetV2Config) -> nn.Module:
        """Build MLP-based decoder (simpler but potentially less smooth)."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.n_curve_points),
            nn.Softplus(),  # Positive decrements
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize Voc head bias to typical value
        with torch.no_grad():
            self.voc_head[-1].bias.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        jsc: torch.Tensor,
        v_grid: torch.Tensor,
        return_params: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (N, input_dim) normalized features
            jsc: (N,) short-circuit current from pretrained LGBM
            v_grid: (M,) voltage grid (M=45)
            return_params: If True, return reconstructed parameters

        Returns:
            curve: (N, M) J-V curve in absolute space
            voc: (N,) predicted open-circuit voltage
            params: (N, n_params) reconstructed parameters (if return_params)
        """
        batch_size = x.shape[0]
        device = x.device

        # Normalize Jsc for input
        jsc_input = jsc.unsqueeze(1) / 50.0  # Scale to ~O(1)
        combined = torch.cat([x, jsc_input], dim=1)

        # Encode
        features = self.encoder(combined)

        # Predict Voc
        voc_raw = self.voc_head(features).squeeze(-1)
        voc = torch.clamp(voc_raw, min=0.1, max=1.5)

        # Predict curve via decrements (guarantees monotonicity)
        decrements = self.curve_decoder(features)  # (N, 45), positive values

        # Build monotonically decreasing curve
        # Method: J(i) = Jsc * (1 - cumsum(decrements) / sum(decrements))
        cumsum = torch.cumsum(decrements, dim=1)
        total = cumsum[:, -1:].clamp(min=1e-6)
        curve_fraction = cumsum / total  # Normalized to [0, 1]

        # Curve goes from Jsc (fraction=0) to 0 (fraction=1)
        curve = jsc.unsqueeze(1) * (1.0 - curve_fraction)

        # Apply Voc cutoff: set J=0 for V > Voc
        voc_expanded = voc.unsqueeze(1)
        curve = torch.where(
            v_grid.unsqueeze(0) > voc_expanded,
            torch.zeros_like(curve),
            curve
        )

        # Enforce exact Jsc at V=0
        curve[:, 0] = jsc

        # Optional parameter reconstruction
        if return_params and self.param_head is not None:
            params = self.param_head(features)
            return curve, voc, params

        return curve, voc, None


class DirectCurveNetV2WithVoc(nn.Module):
    """
    Variant that takes BOTH Jsc AND Voc as inputs.

    Uses pretrained models for both endpoints, only learns the curve shape.
    This is the most modular approach (like DirectCurveShapeNet but with
    direct 45-point output instead of control points).
    """

    def __init__(self, config: DirectCurveNetV2Config):
        super().__init__()
        self.config = config

        # Encoder: +2 for Jsc and Voc inputs
        encoder_layers = []
        prev_dim = config.input_dim + 2
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Curve decoder - direct 45-point output
        self.curve_decoder = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.n_curve_points),
            nn.Softplus(),  # Positive decrements
        )

        # Optional parameter reconstruction
        if config.use_param_reconstruction:
            self.param_head = nn.Sequential(
                nn.Linear(prev_dim, config.param_hidden_dim),
                nn.LayerNorm(config.param_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.param_hidden_dim, config.n_params_to_reconstruct),
            )
        else:
            self.param_head = None

        self._init_weights()

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
        voc: torch.Tensor,
        v_grid: torch.Tensor,
        return_params: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (N, input_dim) normalized features
            jsc: (N,) short-circuit current from pretrained LGBM
            voc: (N,) open-circuit voltage from pretrained model
            v_grid: (M,) voltage grid
            return_params: If True, return reconstructed parameters

        Returns:
            curve: (N, M) J-V curve
            params: (N, n_params) reconstructed parameters (if return_params)
        """
        # Normalize inputs
        jsc_input = jsc.unsqueeze(1) / 50.0
        voc_input = voc.unsqueeze(1)
        combined = torch.cat([x, jsc_input, voc_input], dim=1)

        # Encode
        features = self.encoder(combined)

        # Predict decrements
        decrements = self.curve_decoder(features)

        # Build monotonic curve
        cumsum = torch.cumsum(decrements, dim=1)
        total = cumsum[:, -1:].clamp(min=1e-6)
        curve_fraction = cumsum / total
        curve = jsc.unsqueeze(1) * (1.0 - curve_fraction)

        # Apply Voc cutoff
        curve = torch.where(
            v_grid.unsqueeze(0) > voc.unsqueeze(1),
            torch.zeros_like(curve),
            curve
        )

        # Enforce exact Jsc at V=0
        curve[:, 0] = jsc

        # Optional parameter reconstruction
        if return_params and self.param_head is not None:
            params = self.param_head(features)
            return curve, params

        return curve, None


class DirectCurveLossV2(nn.Module):
    """
    Loss function for DirectCurveNetV2.

    Components (following Zbinden's approach):
    1. Curve MSE: Match predicted curve to target (relative error)
    2. Parameter Loss: Reconstruct key physics parameters (latent loss)
    3. Region Weighting: Higher weight near knee region
    4. Endpoint Constraints: Ensure Jsc and ~0 at endpoints
    5. Smoothness: Penalize non-physical oscillations
    """

    def __init__(
        self,
        lambda_params: float = 0.5,  # Weight for parameter reconstruction loss
        lambda_smooth: float = 0.1,  # Weight for smoothness loss
        lambda_endpoint: float = 1.0,  # Weight for endpoint constraints
        knee_weight: float = 2.0,  # Extra weight for knee region
        knee_position: float = 0.75,  # Typical knee at ~0.75 * Voc
    ):
        super().__init__()
        self.lambda_params = lambda_params
        self.lambda_smooth = lambda_smooth
        self.lambda_endpoint = lambda_endpoint
        self.knee_weight = knee_weight
        self.knee_position = knee_position

    def forward(
        self,
        pred_curve: torch.Tensor,
        true_curve: torch.Tensor,
        pred_voc: torch.Tensor,
        true_voc: torch.Tensor,
        true_jsc: torch.Tensor,
        v_grid: torch.Tensor,
        pred_params: Optional[torch.Tensor] = None,
        true_params: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            pred_curve: (N, M) predicted curve
            true_curve: (N, M) target curve
            pred_voc: (N,) predicted Voc
            true_voc: (N,) true Voc
            true_jsc: (N,) true Jsc (for relative error normalization)
            v_grid: (M,) voltage grid
            pred_params: (N, P) predicted parameters (optional)
            true_params: (N, P) true parameters (optional)

        Returns:
            loss: Total loss
            metrics: Dict with component losses
        """
        batch_size = pred_curve.shape[0]
        device = pred_curve.device

        # 1. Curve MSE with region weighting
        jsc_ref = true_jsc.unsqueeze(1).clamp(min=1e-6)
        rel_err = (pred_curve - true_curve) / jsc_ref
        sq_err = rel_err ** 2

        # Region weighting: higher weight near knee (~0.75 * Voc)
        knee_v = self.knee_position * true_voc.unsqueeze(1)
        dist_to_knee = (v_grid.unsqueeze(0) - knee_v).abs()
        weights = 1.0 + (self.knee_weight - 1.0) * torch.exp(-dist_to_knee / 0.15)

        loss_curve = (weights * sq_err).mean()

        # 2. Voc loss
        loss_voc = F.mse_loss(pred_voc, true_voc)

        # 3. Endpoint constraints
        # Jsc constraint: pred_curve[:, 0] should equal true_jsc
        loss_jsc = ((pred_curve[:, 0] - true_jsc) ** 2).mean()

        # Voc constraint: curve should be ~0 at V=Voc
        # Find index closest to Voc for each sample
        voc_idx = torch.searchsorted(v_grid.unsqueeze(0).expand(batch_size, -1),
                                     true_voc.unsqueeze(1)).clamp(max=v_grid.numel()-1)
        batch_idx = torch.arange(batch_size, device=device)
        loss_voc_endpoint = (pred_curve[batch_idx, voc_idx.squeeze()] ** 2).mean()

        loss_endpoint = loss_jsc + loss_voc_endpoint

        # 4. Smoothness: penalize large second derivatives
        dj = pred_curve[:, 1:] - pred_curve[:, :-1]
        d2j = dj[:, 1:] - dj[:, :-1]
        loss_smooth = (d2j ** 2).mean()

        # 5. Parameter reconstruction loss (Zbinden's latent loss)
        loss_params = torch.tensor(0.0, device=device)
        if pred_params is not None and true_params is not None:
            loss_params = F.mse_loss(pred_params, true_params)

        # Total loss
        loss = (
            loss_curve +
            loss_voc +
            self.lambda_endpoint * loss_endpoint +
            self.lambda_smooth * loss_smooth +
            self.lambda_params * loss_params
        )

        metrics = {
            'loss_total': loss.item(),
            'loss_curve': loss_curve.item(),
            'loss_voc': loss_voc.item(),
            'loss_endpoint': loss_endpoint.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_params': loss_params.item() if pred_params is not None else 0.0,
            'mse_curve_abs': ((pred_curve - true_curve) ** 2).mean().item(),
        }

        return loss, metrics


class DirectCurveLossV2WithVoc(nn.Module):
    """
    Loss function for DirectCurveNetV2WithVoc (when Voc is provided as input).

    Since Voc is given, we don't need Voc loss. Focus on:
    1. Curve MSE with region weighting
    2. Parameter reconstruction (latent loss)
    3. Smoothness
    """

    def __init__(
        self,
        lambda_params: float = 0.5,
        lambda_smooth: float = 0.1,
        knee_weight: float = 2.0,
        knee_position: float = 0.75,
    ):
        super().__init__()
        self.lambda_params = lambda_params
        self.lambda_smooth = lambda_smooth
        self.knee_weight = knee_weight
        self.knee_position = knee_position

    def forward(
        self,
        pred_curve: torch.Tensor,
        true_curve: torch.Tensor,
        true_jsc: torch.Tensor,
        true_voc: torch.Tensor,
        v_grid: torch.Tensor,
        pred_params: Optional[torch.Tensor] = None,
        true_params: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss for shape-only model."""
        device = pred_curve.device

        # Curve MSE with region weighting
        jsc_ref = true_jsc.unsqueeze(1).clamp(min=1e-6)
        rel_err = (pred_curve - true_curve) / jsc_ref
        sq_err = rel_err ** 2

        # Knee region weighting
        knee_v = self.knee_position * true_voc.unsqueeze(1)
        dist_to_knee = (v_grid.unsqueeze(0) - knee_v).abs()
        weights = 1.0 + (self.knee_weight - 1.0) * torch.exp(-dist_to_knee / 0.15)

        loss_curve = (weights * sq_err).mean()

        # Smoothness
        dj = pred_curve[:, 1:] - pred_curve[:, :-1]
        d2j = dj[:, 1:] - dj[:, :-1]
        loss_smooth = (d2j ** 2).mean()

        # Parameter loss
        loss_params = torch.tensor(0.0, device=device)
        if pred_params is not None and true_params is not None:
            loss_params = F.mse_loss(pred_params, true_params)

        loss = loss_curve + self.lambda_smooth * loss_smooth + self.lambda_params * loss_params

        metrics = {
            'loss_total': loss.item(),
            'loss_curve': loss_curve.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_params': loss_params.item() if pred_params is not None else 0.0,
            'mse_curve_abs': ((pred_curve - true_curve) ** 2).mean().item(),
        }

        return loss, metrics


def get_params_for_reconstruction(
    params_raw: torch.Tensor,
    param_indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Extract key physics parameters for latent reconstruction loss.

    Following Zbinden's approach, we select parameters that most
    directly influence the J-V curve shape.

    Default indices correspond to (from config.py COLNAMES):
    - Index 3: mue_P (electron mobility in perovskite) - affects transport
    - Index 4: muh_P (hole mobility) - affects transport
    - Index 15: Gavg (generation rate) - affects Jsc
    - Index 16: Aug (Auger coefficient) - affects recombination
    - Index 17: Brad (radiative recombination) - affects Voc
    - Index 30: SRV_E (surface recombination velocity) - affects FF

    Args:
        params_raw: (N, 31) raw parameters (unscaled)
        param_indices: List of parameter indices to extract

    Returns:
        params_selected: (N, len(param_indices)) selected parameters
    """
    if param_indices is None:
        # Default: key parameters for J-V shape
        param_indices = [3, 4, 15, 16, 17, 30]

    return params_raw[:, param_indices]


def normalize_params_for_loss(params: torch.Tensor, log_scale: bool = True) -> torch.Tensor:
    """
    Normalize parameters for loss computation.

    Following Zbinden's approach: log-normalize then scale to [0, 1].

    Args:
        params: (N, P) raw parameters
        log_scale: If True, apply log10 transform first

    Returns:
        params_norm: (N, P) normalized parameters
    """
    if log_scale:
        # Log-scale (handle zeros with clamp)
        params = torch.log10(params.clamp(min=1e-30))

    # Min-max scale per-parameter to [0, 1]
    p_min = params.min(dim=0, keepdim=True).values
    p_max = params.max(dim=0, keepdim=True).values
    params_norm = (params - p_min) / (p_max - p_min + 1e-8)

    return params_norm
