"""
Conditional Variational Autoencoder baseline for J-V curve reconstruction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalVAE(nn.Module):
    """
    Baseline CVAE: (curve + params) -> latent -> reconstructed curve.
    """

    def __init__(self, curve_dim: int = 45, cond_dim: int = 102, latent_dim: int = 16):
        super().__init__()
        self.curve_dim = curve_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        enc_in = curve_dim + cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        dec_in = latent_dim + cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, curve_dim)
        )

    def encode(self, curve: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([curve, cond], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, cond], dim=1))

    def forward(self, curve: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(curve, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar


def cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl
