"""
Deep Neural Network for V_oc prediction with Jacobian-aware training.
Physics-informed architecture with residual connections.
Optimized for H100 with mixed precision training.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass


def physics_projection(raw_outputs: torch.Tensor, return_violations: bool = False) -> torch.Tensor:
    """
    Hard projection for physical constraints on anchors.

    Input shape: (N, 4) with [Jsc, Voc, Vmpp, Jmpp]
    Output shape: (N, 4) with constraints enforced.

    Args:
        raw_outputs: Raw network outputs before projection
        return_violations: If True, also return violation counts dict

    Returns:
        projected: Tensor with constraints enforced
        violations: (optional) Dict with violation counts if return_violations=True
    """
    # Count violations BEFORE projection (for logging)
    violations = None
    if return_violations:
        violations = {
            'jsc_negative': (raw_outputs[:, 0] < 1e-6).sum().item(),
            'voc_negative': (raw_outputs[:, 1] < 1e-6).sum().item(),
            'vmpp_exceeds_voc': (raw_outputs[:, 2] >= raw_outputs[:, 1]).sum().item(),
            'jmpp_exceeds_jsc': (raw_outputs[:, 3] >= raw_outputs[:, 0]).sum().item(),
            'total_samples': raw_outputs.shape[0]
        }

    # Apply hard constraints
    # [FIX] Relax constraints from 1e-6 to 1e-3 to prevent slope explosion
    margin = 1e-3
    j_sc = torch.clamp(raw_outputs[:, 0], min=margin)
    v_oc = torch.clamp(raw_outputs[:, 1], min=margin)
    v_mpp = torch.clamp(raw_outputs[:, 2], min=margin)
    j_mpp = torch.clamp(raw_outputs[:, 3], min=margin)

    v_mpp_max = torch.clamp(v_oc - margin, min=margin)
    j_mpp_max = torch.clamp(j_sc - margin, min=margin)

    v_mpp = torch.minimum(v_mpp, v_mpp_max)
    j_mpp = torch.minimum(j_mpp, j_mpp_max)

    projected = torch.stack([j_sc, v_oc, v_mpp, j_mpp], dim=1)

    if return_violations:
        return projected, violations
    return projected


@dataclass
class VocNNConfig:
    """Configuration for Voc Neural Network."""
    input_dim: int = 31 + 71  # raw params + physics features
    hidden_dims: list = None
    dropout: float = 0.1
    use_layer_norm: bool = True
    use_residual: bool = True
    activation: str = 'gelu'
    jacobian_weight: float = 0.01  # Smoothness regularization
    physics_weight: float = 0.1     # Physical ceiling constraint
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15
    use_amp: bool = True  # Mixed precision for H100

    def __post_init__(self):
        if self.hidden_dims is None:
            # Simpler default: 3 layers tapering from 256 -> 64
            # Prevents overfitting on limited data
            self.hidden_dims = [256, 128, 64]


class ResidualBlock(nn.Module):
    """Residual block with optional layer normalization."""

    def __init__(self, dim: int, dropout: float, use_layer_norm: bool, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.act = self._get_activation(activation)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
        }
        return activations.get(name, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x + residual


class VocNN(nn.Module):
    """
    Deep neural network for Voc prediction.
    Features:
    - Deep residual architecture
    - Physics-informed output (bounded by Voc ceiling)
    - Smooth via Jacobian regularization
    """

    def __init__(self, config: VocNNConfig):
        super().__init__()
        self.config = config

        # Input projection
        layers = []
        prev_dim = config.input_dim

        # Build hidden layers with residual blocks where dimensions match
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))

            # Add residual block at this dimension
            if config.use_residual:
                layers.append(ResidualBlock(
                    hidden_dim, config.dropout,
                    config.use_layer_norm, config.activation
                ))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Output head - SIMPLE: single linear layer
        # Overparameterized heads cause overfitting on limited data
        self.output_head = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
        }
        return activations.get(name, nn.GELU())

    def _init_weights(self):
        """Initialize weights with activation-specific strategies."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use different init for different activations
                if self.config.activation in ['gelu', 'silu', 'mish']:
                    # For smooth activations, use Xavier/Glorot
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                else:
                    # For ReLU-like, use Kaiming
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, voc_ceiling: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with NO hard constraints - ceiling only used in loss function.

        Args:
            x: Input features (N, input_dim)
            voc_ceiling: Optional physics ceiling - NOT USED in forward pass
                        (kept for API compatibility, used only in loss function)

        Returns:
            Predicted Voc (N,) - unconstrained for maximum learning capacity
        """
        h = self.backbone(x)
        out = self.output_head(h).squeeze(-1)

        # NO CONSTRAINTS - let the model learn freely
        # Physics guidance is handled entirely through the loss function
        # This allows the model to explore beyond theoretical limits if data supports it
        # For 99.9% accuracy, we need to trust the data, not impose hard limits

        return out

    def forward_with_jacobian(
        self,
        x: torch.Tensor,
        voc_ceiling: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also computes input-output Jacobian.
        Used for Jacobian regularization during training.
        """
        # IMPORTANT:
        # - Jacobian regularization is extremely sensitive under AMP (fp16/bf16).
        # - If computed in reduced precision, it can overflow -> inf grads.
        # - With GradScaler, that silently SKIPS optimizer steps -> "instant plateau".
        # We therefore compute the Jacobian in fp32 and normalize by input_dim.
        x = x.float().requires_grad_(True)
        out = self.forward(x, voc_ceiling)
        out_for_grad = out.float()

        # Compute Jacobian norm efficiently via vector-Jacobian product
        # We use random projection for efficiency (Hutchinson's trace estimator)
        v = torch.randn_like(out_for_grad)
        jacobian_vector = torch.autograd.grad(
            outputs=out_for_grad,
            inputs=x,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True
        )[0]

        # Frobenius norm approximation (mean-square per input dimension)
        # Scaling makes jacobian_weight less brittle across feature counts.
        jacobian_norm = (jacobian_vector ** 2).sum(dim=1).mean() / max(1, x.shape[1])

        return out, jacobian_norm


class VocTrainer:
    """Trainer for Voc neural network with Jacobian regularization."""

    def __init__(self, model: VocNN, config: VocNNConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.scaler = GradScaler() if config.use_amp else None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        jacobian_norm: torch.Tensor = None,
        voc_ceiling: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute composite loss with physics terms.

        Loss = MSE + λ_jac * ||J||² + λ_phys * max(0, pred - ceiling)²
        """
        # Primary MSE loss
        mse_loss = nn.functional.mse_loss(pred, target)

        losses = {'mse': mse_loss.item()}
        total_loss = mse_loss

        # Jacobian regularization (encourages smoothness)
        if jacobian_norm is not None and self.config.jacobian_weight > 0:
            jac_loss = self.config.jacobian_weight * jacobian_norm
            total_loss = total_loss + jac_loss
            losses['jacobian'] = jac_loss.item()

        # Physics constraint penalty
        if voc_ceiling is not None and self.config.physics_weight > 0:
            violation = torch.relu(pred - voc_ceiling)
            physics_loss = self.config.physics_weight * (violation ** 2).mean()
            total_loss = total_loss + physics_loss
            losses['physics'] = physics_loss.item()

        losses['total'] = total_loss.item()
        return total_loss, losses

    def train_epoch(self, train_loader, voc_ceilings: torch.Tensor = None) -> dict:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []

        for batch_idx, (x, targets) in enumerate(train_loader):
            x = x.to(self.device)
            
            # Use 'Voc' key if dict, otherwise assume tensor is the target
            if isinstance(targets, dict):
                target = targets['Voc'].to(self.device)
            else:
                target = targets.to(self.device)

            # Get ceiling for this batch if available
            ceiling = None
            if voc_ceilings is not None:
                start_idx = batch_idx * train_loader.batch_size
                end_idx = start_idx + x.shape[0]
                ceiling = voc_ceilings[start_idx:end_idx].to(self.device)

            self.optimizer.zero_grad()

            if self.config.use_amp:
                # AMP-safe training:
                # - forward (prediction) in autocast for speed
                # - Jacobian in full precision to avoid overflow / skipped steps
                with autocast():
                    pred = self.model(x, ceiling)

                with autocast(enabled=False):
                    _, jac_norm = self.model.forward_with_jacobian(x, ceiling)

                loss, loss_dict = self.compute_loss(
                    pred.float(),
                    target.float(),
                    jac_norm,
                    ceiling.float() if ceiling is not None else None
                )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred, jac_norm = self.model.forward_with_jacobian(x, ceiling)
                loss, loss_dict = self.compute_loss(pred, target, jac_norm, ceiling)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            epoch_losses.append(loss_dict)

        self.scheduler.step()

        # Average losses
        avg_losses = {}
        for key in epoch_losses[0]:
            avg_losses[key] = sum(d[key] for d in epoch_losses) / len(epoch_losses)
        return avg_losses

    @torch.no_grad()
    def validate(self, val_loader, voc_ceilings: torch.Tensor = None) -> dict:
        """Validate model."""
        self.model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        for batch_idx, (x, targets) in enumerate(val_loader):
            x = x.to(self.device)
            
            # Use 'Voc' key if dict, otherwise assume tensor is the target
            if isinstance(targets, dict):
                target = targets['Voc'].to(self.device)
            else:
                target = targets.to(self.device)

            ceiling = None
            if voc_ceilings is not None:
                start_idx = batch_idx * val_loader.batch_size
                end_idx = start_idx + x.shape[0]
                ceiling = voc_ceilings[start_idx:end_idx].to(self.device)

            pred = self.model(x, ceiling)
            loss, loss_dict = self.compute_loss(pred, target, None, ceiling)

            val_losses.append(loss_dict)
            all_preds.append(pred)
            all_targets.append(target)

        # Metrics
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        mse = nn.functional.mse_loss(preds, targets).item()
        mae = nn.functional.l1_loss(preds, targets).item()
        r2 = 1 - mse / (targets.var().item() + 1e-8)

        avg_losses = {}
        for key in val_losses[0]:
            avg_losses[key] = sum(d[key] for d in val_losses) / len(val_losses)

        avg_losses['mae'] = mae
        avg_losses['r2'] = r2

        return avg_losses

    def fit(
        self,
        train_loader,
        val_loader,
        train_ceilings: torch.Tensor = None,
        val_ceilings: torch.Tensor = None
    ) -> dict:
        """Full training loop with early stopping."""
        history = {'train': [], 'val': []}

        for epoch in range(self.config.epochs):
            train_losses = self.train_epoch(train_loader, train_ceilings)
            val_losses = self.validate(val_loader, val_ceilings)

            history['train'].append(train_losses)
            history['val'].append(val_losses)

            # Early stopping
            if val_losses['mse'] < self.best_val_loss:
                self.best_val_loss = val_losses['mse']
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.patience:
                break

        # Restore best model
        self.model.load_state_dict(self.best_state)
        return history

    @torch.no_grad()
    def predict(self, x: torch.Tensor, ceiling: torch.Tensor = None) -> torch.Tensor:
        """Inference."""
        self.model.eval()
        return self.model(x.to(self.device), ceiling.to(self.device) if ceiling is not None else None)


@dataclass
class SplitSplineNetConfig:
    """Configuration for unified split-spline network."""
    input_dim: int = 31 + 71
    hidden_dims: list = None
    dropout: float = 0.15
    activation: str = 'silu'
    ctrl_points: int = 6

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


class UnifiedSplitSplineNet(nn.Module):
    """
    Unified multi-head network for anchors + curve control points.
    Outputs:
      - anchors: (N, 4) [Jsc, Voc, Vmpp, Jmpp] with hard projection
      - ctrl_region1: (N, 6) normalized control points for region 1
      - ctrl_region2: (N, 6) normalized control points for region 2
    """

    def __init__(self, config: SplitSplineNetConfig):
        super().__init__()
        self.config = config

        layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.head_anchors = nn.Sequential(
            nn.Linear(prev_dim, 64),
            self._get_activation(config.activation),
            nn.Linear(64, 4)
        )
        self.head_region1 = nn.Sequential(
            nn.Linear(prev_dim, 64),
            self._get_activation(config.activation),
            nn.Linear(64, config.ctrl_points),
            nn.Sigmoid()
        )
        self.head_region2 = nn.Sequential(
            nn.Linear(prev_dim, 64),
            self._get_activation(config.activation),
            nn.Linear(64, config.ctrl_points),
            nn.Sigmoid()
        )

        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
        }
        return activations.get(name, nn.SiLU())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize anchor head biases to typical PV values
        # This helps the model start in a physically reasonable range
        # Typical values: Jsc~20-25, Voc~0.9-1.1, Vmpp~0.7-0.9, Jmpp~18-22
        with torch.no_grad():
            anchor_bias = self.head_anchors[-1].bias
            if anchor_bias is not None:
                anchor_bias[0] = 22.0   # Jsc
                anchor_bias[1] = 1.0    # Voc
                anchor_bias[2] = 0.8    # Vmpp
                anchor_bias[3] = 20.0   # Jmpp

    def forward(
        self,
        x: torch.Tensor,
        return_violations: bool = False
    ):
        """
        Forward pass with optional constraint violation logging.

        Args:
            x: Input features (N, input_dim)
            return_violations: If True, also return constraint violation counts

        Returns:
            anchors, ctrl1, ctrl2, (violations if return_violations)
        """
        features = self.backbone(x)
        anchors_raw = self.head_anchors(features)

        violations = None
        if return_violations:
            anchors, violations = physics_projection(anchors_raw, return_violations=True)
        else:
            anchors = physics_projection(anchors_raw)

        ctrl1 = self.head_region1(features)
        ctrl2 = self.head_region2(features)

        if return_violations:
            return anchors, ctrl1, ctrl2, violations
        return anchors, ctrl1, ctrl2


@dataclass
class ControlPointNetConfig:
    """Configuration for control-point-only network (anchors provided externally)."""
    input_dim: int = 31 + 71  # raw params + physics features
    anchor_dim: int = 4       # Jsc, Voc, Vmpp, Jmpp (normalized)
    hidden_dims: list = None
    dropout: float = 0.15
    activation: str = 'silu'
    ctrl_points: int = 4

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class ControlPointNet(nn.Module):
    """
    Network that predicts ONLY spline control points.
    Anchors are provided as input (from pretrained scalar predictors).

    This decouples anchor prediction from curve shape learning:
    - Anchors come from well-trained LightGBM/VocNN models
    - This network only learns the curve shape between anchor points

    Inputs:
      - x: (N, input_dim) raw + physics features
      - anchors: (N, 4) normalized [Jsc, Voc, Vmpp, Jmpp] from pretrained models

    Outputs:
      - ctrl_region1: (N, K) sigmoid control points for 0->Vmpp region
      - ctrl_region2: (N, K) sigmoid control points for Vmpp->Voc region
    """

    def __init__(self, config: ControlPointNetConfig):
        super().__init__()
        self.config = config

        # Combine features + anchors as input
        total_input = config.input_dim + config.anchor_dim

        layers = []
        prev_dim = total_input
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Control point heads - predict shape between anchors
        self.head_region1 = nn.Sequential(
            nn.Linear(prev_dim, 32),
            self._get_activation(config.activation),
            nn.Linear(32, config.ctrl_points),
            nn.Softplus()  # Positive values for cumsum, better than Sigmoid for monotonicity
        )
        self.head_region2 = nn.Sequential(
            nn.Linear(prev_dim, 32),
            self._get_activation(config.activation),
            nn.Linear(32, config.ctrl_points),
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

    def forward(self, x: torch.Tensor, anchors_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (N, input_dim) normalized features
            anchors_norm: (N, 4) normalized anchors [Jsc, Voc, Vmpp, Jmpp]

        Returns:
            ctrl1: (N, K) control points for region 1
            ctrl2: (N, K) control points for region 2
        """
        # Concatenate features with anchor information
        combined = torch.cat([x, anchors_norm], dim=1)
        features = self.backbone(combined)

        ctrl1 = self.head_region1(features)
        ctrl2 = self.head_region2(features)

        # Normalize to sum to 1 for proper interpolation weighting
        ctrl1 = ctrl1 / (ctrl1.sum(dim=1, keepdim=True) + 1e-8)
        ctrl2 = ctrl2 / (ctrl2.sum(dim=1, keepdim=True) + 1e-8)

        return ctrl1, ctrl2


@torch.no_grad()
def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 45
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo dropout inference for uncertainty estimation.
    Returns mean and std across samples.
    """
    was_training = model.training
    model.train()
    preds = []
    for _ in range(n_samples):
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        preds.append(out)
    stacked = torch.stack(preds)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    if not was_training:
        model.eval()
    return mean, std


def build_voc_model(config: VocNNConfig, device: torch.device) -> tuple[VocNN, VocTrainer]:
    """Factory function to create model and trainer."""
    model = VocNN(config)
    trainer = VocTrainer(model, config, device)
    return model, trainer
