#!/usr/bin/env python3
"""
Direct MLP Baseline for ICML Paper
===================================

Simple feedforward MLP that directly predicts 45-point J-V curves from
31 input parameters. No physics structure, constraints, or spline interpolation.

This serves as a baseline to demonstrate the value of:
1. Physics-informed feature engineering
2. Split-spline architecture
3. Hard constraint projection
4. Anchor-based reconstruction

Architecture: 31 → 256 → 128 → 45 (configurable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class DirectMLPBaseline(nn.Module):
    """
    Simple MLP baseline for curve prediction.

    This model directly maps input parameters to full J-V curves without
    any physics structure or constraints.

    Args:
        input_dim: Number of input features (default: 31 for raw params)
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output points (default: 45 for J-V curve)
        dropout: Dropout probability
        activation: Activation function ('silu', 'gelu', 'relu')
        use_batchnorm: Whether to use batch normalization
    """

    def __init__(
        self,
        input_dim: int = 31,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 45,
        dropout: float = 0.2,
        activation: str = 'silu',
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Select activation function
        if activation == 'silu':
            act_fn = nn.SiLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer (no activation - raw curve values)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predicted J-V curve of shape (batch_size, output_dim)
        """
        return self.net(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC-Dropout uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            Tuple of (mean predictions, std predictions)
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        self.eval()
        return mean, std


class DirectMLPWithPhysicsFeatures(DirectMLPBaseline):
    """
    Direct MLP that uses physics-informed features as input.

    This allows comparison between:
    - Raw 31 params → MLP (DirectMLPBaseline)
    - 31 params + physics features → MLP (this class)
    - Full split-spline model (main model)
    """

    def __init__(
        self,
        raw_param_dim: int = 31,
        physics_feature_dim: int = 71,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 45,
        dropout: float = 0.2,
        activation: str = 'silu',
        use_batchnorm: bool = True,
    ):
        input_dim = raw_param_dim + physics_feature_dim
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

        self.raw_param_dim = raw_param_dim
        self.physics_feature_dim = physics_feature_dim


class DirectMLPWithAnchors(nn.Module):
    """
    MLP that predicts anchors first, then uses them for curve prediction.

    This is an intermediate between full direct MLP and split-spline:
    - Predicts [Jsc, Voc, Vmpp, Jmpp] as intermediate outputs
    - Uses these to condition the curve prediction
    - But no hard constraints or spline structure

    Architecture:
        Input → Backbone → [Anchor Head, Curve Head]
                              ↓            ↓
                          [4 anchors]  [45 points]
    """

    def __init__(
        self,
        input_dim: int = 102,  # 31 raw + 71 physics
        hidden_dims: List[int] = [384, 256, 128],
        dropout: float = 0.2,
        activation: str = 'gelu',
    ):
        super().__init__()

        if activation == 'gelu':
            act_fn = nn.GELU
        else:
            act_fn = nn.SiLU

        # Shared backbone
        backbone_layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            backbone_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*backbone_layers)

        # Anchor prediction head (4 outputs: Jsc, Voc, Vmpp, Jmpp)
        self.anchor_head = nn.Linear(hidden_dims[-1], 4)

        # Curve prediction head (45 outputs)
        # Conditioned on both backbone features and predicted anchors
        self.curve_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] + 4, 128),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(128, 45),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_anchors: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, input_dim)
            return_anchors: If True, also return predicted anchors

        Returns:
            Predicted curve (batch, 45), optionally with anchors
        """
        # Backbone features
        features = self.backbone(x)

        # Predict anchors
        anchors = self.anchor_head(features)

        # Concatenate features with anchors for curve prediction
        combined = torch.cat([features, anchors], dim=1)
        curve = self.curve_head(combined)

        if return_anchors:
            return curve, anchors
        return curve


def train_direct_mlp(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 30,
) -> dict:
    """
    Train a direct MLP model.

    Args:
        model: The MLP model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            X, y = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                X, y = batch[0].to(device), batch[1].to(device)
                pred = model(X)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Restore best model
    model.load_state_dict(best_state)

    history['best_val_loss'] = best_val_loss
    return history


if __name__ == '__main__':
    # Test the models
    print("Testing DirectMLP models...")

    # Test basic MLP
    model = DirectMLPBaseline(input_dim=31, hidden_dims=[256, 128], output_dim=45)
    x = torch.randn(32, 31)
    y = model(x)
    print(f"DirectMLPBaseline: input {x.shape} → output {y.shape}")

    # Test with physics features
    model2 = DirectMLPWithPhysicsFeatures(
        raw_param_dim=31,
        physics_feature_dim=71,
        hidden_dims=[256, 128]
    )
    x2 = torch.randn(32, 102)
    y2 = model2(x2)
    print(f"DirectMLPWithPhysicsFeatures: input {x2.shape} → output {y2.shape}")

    # Test with anchors
    model3 = DirectMLPWithAnchors(input_dim=102, hidden_dims=[384, 256, 128])
    y3, anchors = model3(x2, return_anchors=True)
    print(f"DirectMLPWithAnchors: input {x2.shape} → curve {y3.shape}, anchors {anchors.shape}")

    # Test MC-Dropout uncertainty
    mean, std = model.predict_with_uncertainty(x, n_samples=10)
    print(f"MC-Dropout uncertainty: mean {mean.shape}, std {std.shape}")

    print("\nAll DirectMLP tests passed!")
