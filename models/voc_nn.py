"""
Deep Neural Network for V_oc prediction with Jacobian-aware training.
Physics-informed architecture with residual connections.
Optimized for H100 with mixed precision training.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass


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
        x.requires_grad_(True)
        out = self.forward(x, voc_ceiling)

        # Compute Jacobian norm efficiently via vector-Jacobian product
        # We use random projection for efficiency (Hutchinson's trace estimator)
        v = torch.randn_like(out)
        jacobian_vector = torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True
        )[0]

        # Frobenius norm approximation
        jacobian_norm = (jacobian_vector ** 2).sum(dim=1).mean()

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
            target = targets['Voc'].to(self.device)

            # Get ceiling for this batch if available
            ceiling = None
            if voc_ceilings is not None:
                start_idx = batch_idx * train_loader.batch_size
                end_idx = start_idx + x.shape[0]
                ceiling = voc_ceilings[start_idx:end_idx].to(self.device)

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    pred, jac_norm = self.model.forward_with_jacobian(x, ceiling)
                    loss, loss_dict = self.compute_loss(pred, target, jac_norm, ceiling)

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
            target = targets['Voc'].to(self.device)

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


def build_voc_model(config: VocNNConfig, device: torch.device) -> tuple[VocNN, VocTrainer]:
    """Factory function to create model and trainer."""
    model = VocNN(config)
    trainer = VocTrainer(model, config, device)
    return model, trainer
