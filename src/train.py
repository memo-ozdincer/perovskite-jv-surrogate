#!/usr/bin/env python3
"""
Physics-Informed Neural Network for I-V Curve Reconstruction.

Adapted from Google Colab notebook for SLURM/CLI execution.
All scalar features are loaded exclusively from external txt files
(no internal scalar computation) so that true scalars can be swapped
for predicted ones without code changes.

Architectures (--architecture):
  cnn       : Causal dilated 1D convolutions
  conv      : Standard (non-causal) 1D convolutions (paper default)
  pointwise : Position-independent 1x1 convolutions

Usage:
  python train.py \\
      --params LHS_parameters_m.txt --iv iV_m.txt \\
      --params-extra LHS_parameters_m_300k.txt --iv-extra iV_m_300k.txt \\
      --scalar-files voc_clean_100k.txt vmpp_clean_100k.txt \\
      --scalar-files-extra voc_clean_300k.txt vmpp_clean_300k.txt \\
      --architecture conv --no-attention \\
      --output-dir ./lightning_output --run-name Conv-NoAttn --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import multiprocessing as mp
import os
import time
import typing
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Use Tensor Cores on Ampere+ GPUs
torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ──────────────────────────────────────────────────────────────────────────────
#   CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

COLNAMES = [
    "lH", "lP", "lE",
    "muHh", "muPh", "muPe", "muEe",
    "NvH", "NcH", "NvE", "NcE", "NvP", "NcP",
    "chiHh", "chiHe", "chiPh", "chiPe", "chiEh", "chiEe",
    "Wlm", "Whm",
    "epsH", "epsP", "epsE",
    "Gavg", "Aug", "Brad", "Taue", "Tauh", "vII", "vIII",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#   CONFIG BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_paths(paths: list[str] | None) -> list[str]:
    """Resolve a list of paths, filtering None."""
    if not paths:
        return []
    return [str(Path(p).resolve()) for p in paths]


def build_config(args: argparse.Namespace) -> dict:
    data_dir = str(Path(args.data_dir).resolve())
    dataloader_cfg = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        dataloader_cfg["persistent_workers"] = True
        dataloader_cfg["prefetch_factor"] = args.prefetch_factor

    return {
        "train": {
            "seed": args.seed,
            "run_name": args.run_name,
        },
        "model": {
            "param_dim": 31,  # updated during preprocessing
            "dense_units": [256, 128, 128],
            "filters": [128, 64],
            "kernel": 5,
            "heads": 4,
            "dropout": 0.03631232181608377,
            "embedding_type": "gaussian",
            "gaussian_bands": 18,
            "gaussian_sigma": 0.07749512610240868,
            "architecture": args.architecture,
            "use_attention": args.use_attention,
            "use_dilated_conv": args.use_dilated,
            "loss_weights": {
                "mse": 0.98,
                "mono": 0.005,
                "convex": 0.005,
                "excurv": 0.01,
                "excess_threshold": 0.8,
                "jacobian": args.jacobian_weight,
            },
        },
        "optimizer": {
            "lr": 0.005545402750717978,
            "weight_decay": 5.403751961152276e-05,
            "final_lr_ratio": 0.00666,
            "warmup_epochs": 7,
        },
        "dataset": {
            "paths": {
                "params_csv": str(Path(args.params).resolve()),
                "iv_raw_txt": str(Path(args.iv).resolve()),
                "params_csv_extra": str(Path(args.params_extra).resolve()) if args.params_extra else None,
                "iv_raw_txt_extra": str(Path(args.iv_extra).resolve()) if args.iv_extra else None,
                "scalar_files": _resolve_paths(args.scalar_files),
                "scalar_files_extra": _resolve_paths(args.scalar_files_extra),
                "output_dir": data_dir,
                "preprocessed_npz": os.path.join(data_dir, "cnn_preprocessed.npz"),
                "param_transformer": os.path.join(data_dir, "cnn_param_transformer.joblib"),
                "scalar_transformer": os.path.join(data_dir, "cnn_scalar_transformer.joblib"),
                "physics_transformer": os.path.join(data_dir, "cnn_physics_transformer.joblib"),
                "v_fine_memmap": os.path.join(data_dir, "cnn_v_fine_curves.mmap"),
                "i_fine_memmap": os.path.join(data_dir, "cnn_i_fine_curves.mmap"),
            },
            "pchip": {
                "v_max": 1.4,
                "n_fine": 2000,
                "n_pre_mpp": 3,
                "n_post_mpp": 4,
                "seq_len": 8,
            },
            "dataloader": {
                **dataloader_cfg,
            },
            "use_physics_features": args.use_physics_features,
            "physics_feature_selection": {
                "enabled": bool(args.physics_feature_selection),
                "weak_threshold": float(args.physics_weak_threshold),
                "corr_threshold": float(args.physics_corr_threshold),
                "max_features": args.physics_max_features,
            },
            "curvature_weighting": {
                "alpha": 4.0,
                "power": 1.5,
            },
        },
        "trainer": {
            "max_epochs": args.max_epochs,
            "accelerator": "auto",
            "devices": "auto",
            "precision": "bf16-mixed",
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 25,
            "benchmark": True,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
#   UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)


def process_iv_with_pchip(
    iv_raw, full_v_grid, n_pre, n_post, v_max, n_fine
) -> typing.Optional[tuple]:
    seq_len = n_pre + 1 + n_post
    try:
        if np.count_nonzero(~np.isnan(iv_raw)) < 4:
            return None
        pi = PchipInterpolator(full_v_grid, iv_raw, extrapolate=False)
        v_fine = np.linspace(0, v_max, n_fine)
        i_fine = pi(v_fine)
        valid_mask = ~np.isnan(i_fine) & ~np.isinf(i_fine)
        v_fine, i_fine = v_fine[valid_mask], i_fine[valid_mask]
        if v_fine.size < 2:
            return None
        zero_cross_idx = np.where(i_fine <= 0)[0]
        voc_v = v_fine[zero_cross_idx[0]] if len(zero_cross_idx) > 0 else v_fine[-1]
        v_search_mask = v_fine <= voc_v
        v_search, i_search = v_fine[v_search_mask], i_fine[v_search_mask]
        if v_search.size == 0:
            return None
        power = v_search * i_search
        mpp_idx = np.argmax(power)
        v_mpp = v_search[mpp_idx]
        v_pre_mpp = np.linspace(v_search[0], v_mpp, n_pre + 2, endpoint=True)[:-1]
        v_post_mpp = np.linspace(v_mpp, v_search[-1], n_post + 2, endpoint=True)[1:]
        v_mpp_grid = np.unique(np.concatenate([v_pre_mpp, v_post_mpp]))
        v_slice = np.interp(
            np.linspace(0, 1, seq_len),
            np.linspace(0, 1, len(v_mpp_grid)),
            v_mpp_grid,
        )
        i_slice = pi(v_slice)
        if np.any(np.isnan(i_slice)) or i_slice.shape[0] != seq_len:
            return None
        f16_info = np.finfo(np.float16)
        v_fine_clipped = np.clip(v_fine, f16_info.min, f16_info.max)
        i_fine_clipped = np.clip(i_fine, f16_info.min, f16_info.max)
        return (
            v_slice.astype(np.float32),
            i_slice.astype(np.float32),
            (v_fine_clipped.astype(np.float16), i_fine_clipped.astype(np.float16)),
        )
    except (ValueError, IndexError):
        return None


def normalize_and_scale_by_isc(curve: np.ndarray) -> tuple[float, np.ndarray]:
    isc_val = float(curve[0])
    if abs(isc_val) < 1e-9:
        return 0.0, np.zeros_like(curve, dtype=np.float32)
    return isc_val, (2.0 * (curve / isc_val) - 1.0).astype(np.float32)


def compute_curvature_weights(
    y_curves: np.ndarray, alpha: float, power: float
) -> np.ndarray:
    padded = np.pad(y_curves, ((0, 0), (1, 1)), mode="edge")
    kappa = np.abs(padded[:, 2:] - 2 * padded[:, 1:-1] + padded[:, :-2])
    max_kappa = np.max(kappa, axis=1, keepdims=True)
    max_kappa[max_kappa < 1e-9] = 1.0
    return (1.0 + alpha * np.power(kappa / max_kappa, power)).astype(np.float32)


def _compute_curve_targets_for_feature_selection(
    i_slices: np.ndarray, v_slices: np.ndarray
) -> np.ndarray:
    """
    Build scalar targets from 8-point curves for train-only feature filtering.
    Columns: Jsc, Voc, Vmpp, Jmpp, FF.
    """
    jsc = i_slices[:, 0].astype(np.float32, copy=False)
    power = v_slices * i_slices
    mpp_idx = np.argmax(power, axis=1)
    vmpp = v_slices[np.arange(len(v_slices)), mpp_idx].astype(np.float32, copy=False)
    jmpp = i_slices[np.arange(len(i_slices)), mpp_idx].astype(np.float32, copy=False)

    voc = np.full(len(i_slices), v_slices[:, -1], dtype=np.float32)
    for i in range(len(i_slices)):
        y = i_slices[i]
        x = v_slices[i]
        crossing = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
        if len(crossing) > 0:
            j = crossing[0]
            y0, y1 = float(y[j]), float(y[j + 1])
            x0, x1 = float(x[j]), float(x[j + 1])
            dy = y1 - y0
            if abs(dy) < 1e-12:
                voc[i] = x1
            else:
                voc[i] = np.float32(x0 - y0 * (x1 - x0) / dy)
        else:
            voc[i] = np.float32(x[-1])

    ff = power[np.arange(len(power)), mpp_idx] / np.maximum(voc * jsc, 1e-12)
    ff = np.clip(ff, 0.0, 1.0).astype(np.float32, copy=False)
    return np.column_stack([jsc, voc, vmpp, jmpp, ff]).astype(np.float32)


def _max_abs_target_corr(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-feature max absolute Pearson correlation across targets."""
    out = np.zeros(features.shape[1], dtype=np.float32)
    for i in range(features.shape[1]):
        fx = features[:, i]
        if np.std(fx) < 1e-12:
            continue
        best = 0.0
        for j in range(targets.shape[1]):
            ty = targets[:, j]
            if np.std(ty) < 1e-12:
                continue
            r = np.corrcoef(fx, ty)[0, 1]
            if np.isnan(r):
                r = 0.0
            best = max(best, abs(float(r)))
        out[i] = np.float32(best)
    return out


def _select_physics_features(
    physics_train: np.ndarray,
    target_train: np.ndarray,
    feature_names: list[str],
    corr_threshold: float = 0.85,
    weak_threshold: float = 0.30,
    max_features: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Train-only filter:
      1) Drop multicollinear features using pairwise |r| > corr_threshold
      2) Keep features with max |corr(target)| >= weak_threshold
      3) Optional cap to top-k by target correlation
    """
    n_feat = physics_train.shape[1]
    if n_feat == 0:
        return np.array([], dtype=int), []

    # --- Step 1: multicollinearity ---
    std = physics_train.std(axis=0)
    std[std < 1e-10] = 1.0
    z = (physics_train - physics_train.mean(axis=0)) / std
    corr = np.corrcoef(z.T)
    corr = np.nan_to_num(corr, nan=0.0)
    target_corr = _max_abs_target_corr(physics_train, target_train)

    keep = np.ones(n_feat, dtype=bool)
    for i in range(n_feat):
        if not keep[i]:
            continue
        for j in range(i + 1, n_feat):
            if not keep[j]:
                continue
            if abs(corr[i, j]) > corr_threshold:
                if target_corr[i] >= target_corr[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    # --- Step 2: relevance ---
    cand_idx = np.where(keep)[0]
    if len(cand_idx) == 0:
        return np.array([], dtype=int), []
    strong = cand_idx[target_corr[cand_idx] >= weak_threshold]
    selected = strong if len(strong) > 0 else cand_idx

    # --- Step 3: optional top-k cap ---
    if max_features is not None and max_features > 0 and len(selected) > max_features:
        order = np.argsort(target_corr[selected])[::-1]
        selected = selected[order[:max_features]]

    selected = np.array(sorted(selected.tolist()), dtype=int)
    names = [feature_names[i] for i in selected]
    return selected, names


def get_param_transformer(colnames: list[str]) -> ColumnTransformer:
    param_defs = {
        "layer_thickness": ["lH", "lP", "lE"],
        "material_properties": [
            "muHh", "muPh", "muPe", "muEe",
            "NvH", "NcH", "NvE", "NcE", "NvP", "NcP",
            "chiHh", "chiHe", "chiPh", "chiPe", "chiEh", "chiEe",
            "epsH", "epsP", "epsE",
        ],
        "contacts": ["Wlm", "Whm"],
        "recombination_gen": ["Gavg", "Aug", "Brad", "Taue", "Tauh", "vII", "vIII"],
    }
    transformers = []
    for group, cols in param_defs.items():
        actual_cols = [c for c in cols if c in colnames]
        if not actual_cols:
            continue
        steps = [
            ("robust", RobustScaler()),
            ("minmax", MinMaxScaler(feature_range=(-1, 1))),
        ]
        if group == "material_properties":
            steps.insert(0, ("log1p", FunctionTransformer(func=np.log1p)))
        transformers.append((group, Pipeline(steps), actual_cols))
    return ColumnTransformer(transformers, remainder="passthrough")


def load_scalar_txt(path: str) -> tuple[str, np.ndarray]:
    """Load a scalar txt file.  Returns (column_name, values_array)."""
    with open(path) as f:
        header = f.readline().strip().strip(",")
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    return header, values.ravel()


def denormalize(scaled_current, isc):
    if isinstance(scaled_current, torch.Tensor):
        isc = isc.unsqueeze(1)
    else:
        isc = isc[:, np.newaxis]
    return (scaled_current + 1.0) / 2.0 * isc


# ──────────────────────────────────────────────────────────────────────────────
#   MODEL COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────

class FourierFeatures(nn.Module):
    def __init__(self, num_bands: int, v_max: float = 1.4):
        super().__init__()
        self.v_max = v_max
        B = torch.logspace(0, 3, num_bands)
        self.register_buffer("B", B, persistent=False)
        self.out_dim = num_bands * 2

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        two_pi = torch.tensor(2 * math.pi, device=v.device, dtype=v.dtype)
        v_norm = v / self.v_max
        v_proj = v_norm.unsqueeze(-1) * self.B
        return torch.cat([(two_pi * v_proj).sin(), (two_pi * v_proj).cos()], dim=-1)


class ClippedFourierFeatures(nn.Module):
    def __init__(self, num_bands: int, v_max: float = 1.4):
        super().__init__()
        self.v_max = v_max
        B = torch.logspace(0, 3, num_bands)
        B_mask = (B >= 1.0).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer("B", B, persistent=False)
        self.register_buffer("B_mask", B_mask, persistent=False)
        self.out_dim = num_bands * 2

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        two_pi = torch.tensor(2 * math.pi, device=v.device, dtype=v.dtype)
        v_norm = v / self.v_max
        v_proj = v_norm.unsqueeze(-1) * self.B
        sines = (two_pi * v_proj).sin() * self.B_mask
        coses = (two_pi * v_proj).cos() * self.B_mask
        return torch.cat([sines, coses], dim=-1)


class GaussianRBFFeatures(nn.Module):
    def __init__(self, num_bands: int, sigma: float = 0.1, v_max: float = 1.4):
        super().__init__()
        self.v_max = v_max
        self.sigma = sigma
        mu = torch.linspace(0, 1, num_bands)
        self.register_buffer("mu", mu, persistent=False)
        self.out_dim = num_bands

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_norm = v / self.v_max
        diff = v_norm.unsqueeze(-1) - self.mu
        return torch.exp(-0.5 * (diff / self.sigma) ** 2)


def make_positional_embedding(cfg: dict) -> nn.Module:
    etype = cfg["model"]["embedding_type"]
    if etype == "fourier":
        return FourierFeatures(
            cfg["model"]["fourier_bands"], cfg["dataset"]["pchip"]["v_max"]
        )
    elif etype == "fourier_clipped":
        return ClippedFourierFeatures(
            cfg["model"]["fourier_bands"], cfg["dataset"]["pchip"]["v_max"]
        )
    elif etype == "gaussian":
        return GaussianRBFFeatures(
            cfg["model"]["gaussian_bands"],
            cfg["model"]["gaussian_sigma"],
            cfg["dataset"]["pchip"]["v_max"],
        )
    raise ValueError(f"Unknown embedding type: {etype}")


def physics_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    sample_w: torch.Tensor,
    loss_w: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mse_loss = (((y_true - y_pred) ** 2) * sample_w).mean()
    mono_violations = torch.relu(y_pred[:, 1:] - y_pred[:, :-1])
    mono_loss = mono_violations.pow(2).mean()
    convex_violations = torch.relu(
        2 * y_pred[:, 1:-1] - y_pred[:, :-2] - y_pred[:, 2:]
    )
    convex_loss = convex_violations.pow(2).mean()
    curvature = torch.abs(
        y_pred[:, :-2] - 2 * y_pred[:, 1:-1] + y_pred[:, 2:]
    )
    excurv_violations = torch.relu(curvature - loss_w["excess_threshold"])
    excurv_loss = excurv_violations.pow(2).mean()
    total_loss = (
        loss_w["mse"] * mse_loss
        + loss_w["mono"] * mono_loss
        + loss_w["convex"] * convex_loss
        + loss_w["excurv"] * excurv_loss
    )
    return total_loss, {
        "mse": mse_loss,
        "mono": mono_loss,
        "convex": convex_loss,
        "excurv": excurv_loss,
    }


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


# --- Architecture: Causal CNN (causal dilated convolutions) ---

class TemporalBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dropout: float, dilation: int
    ):
        super().__init__()
        self.padding = ((kernel_size - 1) * dilation, 0)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.act2 = nn.GELU()
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = F.pad(x, self.padding)
        out = self.drop1(self.norm1(self.act1(self.conv1(out))))
        out = F.pad(out, self.padding)
        out = self.drop2(self.norm2(self.act2(self.conv2(out))))
        return out + res


# --- Architecture: Conv (standard non-causal 1D convolutions) ---

class ConvResBlock(nn.Module):
    """Non-causal 1D convolutional residual block with optional dilation.

    Uses symmetric (same) padding so every position can attend to
    neighbours in both directions — appropriate for I-V curves where
    the underlying drift-diffusion physics is a boundary-value problem
    with information flow in both voltage directions.

    Dilation expands the receptive field without adding parameters,
    letting the network capture both local knee curvature and
    full-curve shape from a single block.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float,
                 dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.act2 = nn.GELU()
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.drop1(self.norm1(self.act1(self.conv1(x))))
        out = self.drop2(self.norm2(self.act2(self.conv2(out))))
        return out + res


# --- Architecture: Pointwise (1x1 convolutions, position-independent) ---

class PointwiseResBlock(nn.Module):
    """Position-independent residual block using 1x1 convolutions.

    Each voltage query point is processed independently through the
    feature dimension — no spatial mixing at all.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Conv1d(in_ch, out_ch, 1)
        self.act1 = nn.GELU()
        self.norm1 = ChannelLayerNorm(out_ch)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv1d(out_ch, out_ch, 1)
        self.act2 = nn.GELU()
        self.norm2 = ChannelLayerNorm(out_ch)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.drop1(self.norm1(self.act1(self.fc1(x))))
        out = self.drop2(self.norm2(self.act2(self.fc2(out))))
        return out + res


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_transposed = x.transpose(1, 2)
        seq_len = x.size(2)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.attn(
            x_transposed, x_transposed, x_transposed, attn_mask=causal_mask
        )
        return self.norm(x_transposed + self.drop(attn_out)).transpose(1, 2)


# ──────────────────────────────────────────────────────────────────────────────
#   DATA MODULES
# ──────────────────────────────────────────────────────────────────────────────

class IVDataset(Dataset):
    def __init__(self, cfg: dict, split: str, param_tf, scalar_tf):
        paths = cfg["dataset"]["paths"]
        data = np.load(paths["preprocessed_npz"], allow_pickle=True)
        split_labels = data["split_labels"]
        indices = np.where(split_labels == split)[0]

        self.v_slices = torch.from_numpy(data["v_slices"][indices])
        self.i_slices = torch.from_numpy(data["i_slices"][indices])
        self.i_slices_scaled = torch.from_numpy(data["i_slices_scaled"][indices])
        self.sample_weights = torch.from_numpy(data["sample_weights"][indices])
        self.isc_vals = torch.from_numpy(data["isc_vals"][indices])

        # Device parameters
        params_df = pd.read_csv(
            paths["params_csv"], header=None, names=COLNAMES
        )
        if paths.get("params_csv_extra") and Path(paths["params_csv_extra"]).exists():
            params_df_extra = pd.read_csv(
                paths["params_csv_extra"], header=None, names=COLNAMES
            )
            params_df = pd.concat([params_df, params_df_extra], ignore_index=True)
        params_df_valid = params_df.iloc[data["valid_indices"]].reset_index(drop=True)
        X_params = param_tf.transform(params_df_valid).astype(np.float32)

        # Scalar features from txt files (NO internal computation)
        scalar_names = list(data["scalar_names"]) if "scalar_names" in data else []
        if scalar_names:
            scalar_data = data["scalar_data"].astype(np.float32)
            X_scalar = scalar_tf.transform(
                pd.DataFrame(scalar_data, columns=scalar_names)
            ).astype(np.float32)
            X_combined = np.concatenate([X_params, X_scalar], axis=1)
        else:
            X_combined = X_params

        # Physics features (if computed during preprocessing)
        physics_names = list(data["physics_feature_names"]) if "physics_feature_names" in data else []
        if physics_names:
            physics_tf = joblib.load(paths["physics_transformer"])
            physics_data_raw = data["physics_data"].astype(np.float32)
            X_physics = physics_tf.transform(
                pd.DataFrame(physics_data_raw, columns=physics_names)
            ).astype(np.float32)
            X_combined = np.concatenate([X_combined, X_physics], axis=1)

        self.X = torch.from_numpy(X_combined[indices])

    def __len__(self):
        return len(self.v_slices)

    def __getitem__(self, idx):
        return {
            "X_combined": self.X[idx],
            "voltage": self.v_slices[idx],
            "current_scaled": self.i_slices_scaled[idx],
            "sample_w": self.sample_weights[idx],
            "isc": self.isc_vals[idx],
            "v_true_slice": self.v_slices[idx],
            "i_true_slice": self.i_slices[idx],
        }


class IVDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.param_tf = None
        self.scalar_tf = None

    def prepare_data(self):
        if not Path(self.cfg["dataset"]["paths"]["preprocessed_npz"]).exists():
            log.info("Preprocessed data not found. Running preprocessing...")
            self._preprocess_and_save()
        else:
            log.info("Found preprocessed data. Skipping preprocessing.")

    def setup(self, stage: str | None = None):
        if self.param_tf is None:
            self.param_tf = joblib.load(
                self.cfg["dataset"]["paths"]["param_transformer"]
            )
            # Infer param_dim from saved transformers + cached scalar count
            data = np.load(
                self.cfg["dataset"]["paths"]["preprocessed_npz"], allow_pickle=True
            )
            params_df = pd.read_csv(
                self.cfg["dataset"]["paths"]["params_csv"], header=None, names=COLNAMES
            )
            if (self.cfg["dataset"]["paths"].get("params_csv_extra")
                    and Path(self.cfg["dataset"]["paths"]["params_csv_extra"]).exists()):
                params_df_extra = pd.read_csv(
                    self.cfg["dataset"]["paths"]["params_csv_extra"],
                    header=None, names=COLNAMES,
                )
                params_df = pd.concat([params_df, params_df_extra], ignore_index=True)
            params_df_valid = params_df.iloc[data["valid_indices"]].reset_index(drop=True)
            param_dim = self.param_tf.transform(params_df_valid[:1]).shape[1]

            scalar_names = list(data["scalar_names"]) if "scalar_names" in data else []
            scalar_dim = len(scalar_names)
            if scalar_dim > 0:
                self.scalar_tf = joblib.load(
                    self.cfg["dataset"]["paths"]["scalar_transformer"]
                )
            else:
                self.scalar_tf = None

            # Physics features dimension
            physics_names = list(data["physics_feature_names"]) if "physics_feature_names" in data else []
            physics_dim = len(physics_names)
            if self.cfg["dataset"].get("use_physics_features", False) and physics_dim == 0:
                raise RuntimeError(
                    "Physics features requested but not in cache. "
                    "Re-run with --force-preprocess."
                )

            self.cfg["model"]["param_dim"] = param_dim + scalar_dim + physics_dim
            log.info(
                f"Inferred param_dim from cache: {param_dim + scalar_dim + physics_dim} "
                f"({param_dim} device params + {scalar_dim} scalars + {physics_dim} physics)"
            )

        if stage == "fit" or stage is None:
            self.train_dataset = IVDataset(
                self.cfg, "train", self.param_tf, self.scalar_tf
            )
            self.val_dataset = IVDataset(
                self.cfg, "val", self.param_tf, self.scalar_tf
            )
        if stage == "test" or stage is None:
            self.test_dataset = IVDataset(
                self.cfg, "test", self.param_tf, self.scalar_tf
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, **self.cfg["dataset"]["dataloader"],
            shuffle=True, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg["dataset"]["dataloader"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg["dataset"]["dataloader"])

    def _preprocess_and_save(self):
        log.info("--- Starting PCHIP Data Preprocessing ---")
        cfg = self.cfg
        paths = cfg["dataset"]["paths"]
        pchip_cfg = cfg["dataset"]["pchip"]

        # ── Load & concatenate primary + extra datasets ──
        params_df = pd.read_csv(paths["params_csv"], header=None, names=COLNAMES)
        iv_data_raw = np.loadtxt(paths["iv_raw_txt"], delimiter=",")
        n_primary = len(iv_data_raw)

        if paths.get("params_csv_extra") and paths.get("iv_raw_txt_extra"):
            log.info("Loading extra (300k) dataset...")
            params_df_extra = pd.read_csv(
                paths["params_csv_extra"], header=None, names=COLNAMES
            )
            iv_extra = np.loadtxt(paths["iv_raw_txt_extra"], delimiter=",")
            params_df = pd.concat([params_df, params_df_extra], ignore_index=True)
            iv_data_raw = np.vstack([iv_data_raw, iv_extra])
            log.info(
                f"Combined dataset: {n_primary} + {len(iv_extra)} = {len(iv_data_raw)}"
            )
            del iv_extra, params_df_extra

        full_v_grid = np.concatenate([
            np.arange(0, 0.4 + 1e-8, 0.1),
            np.arange(0.425, 1.4 + 1e-8, 0.025),
        ]).astype(np.float32)

        N_raw = len(iv_data_raw)
        log.info(f"Total raw dataset size: {N_raw}")
        Path(paths["output_dir"]).mkdir(parents=True, exist_ok=True)

        v_fine_mm = np.memmap(
            paths["v_fine_memmap"], dtype=np.float16, mode="w+",
            shape=(N_raw, pchip_cfg["n_fine"]),
        )
        i_fine_mm = np.memmap(
            paths["i_fine_memmap"], dtype=np.float16, mode="w+",
            shape=(N_raw, pchip_cfg["n_fine"]),
        )
        v_fine_mm[:] = np.nan
        i_fine_mm[:] = np.nan

        valid_indices, v_slices, i_slices = [], [], []
        pchip_args = (
            full_v_grid,
            pchip_cfg["n_pre_mpp"],
            pchip_cfg["n_post_mpp"],
            pchip_cfg["v_max"],
            pchip_cfg["n_fine"],
        )
        # Parallel PCHIP processing for speed
        n_workers = min(mp.cpu_count(), 16)
        _pchip_fn = partial(
            process_iv_with_pchip,
            full_v_grid=full_v_grid,
            n_pre=pchip_cfg["n_pre_mpp"],
            n_post=pchip_cfg["n_post_mpp"],
            v_max=pchip_cfg["v_max"],
            n_fine=pchip_cfg["n_fine"],
        )
        log.info(f"PCHIP processing {N_raw} curves with {n_workers} workers...")
        with mp.Pool(n_workers) as pool:
            results_iter = pool.imap(
                _pchip_fn,
                iv_data_raw,
                chunksize=max(1, N_raw // (n_workers * 4)),
            )
            for i, res in enumerate(tqdm(results_iter, total=N_raw, desc="PCHIP processing")):
                if res is not None and res[1][0] > 1e-9:
                    valid_indices.append(i)
                    v_slices.append(res[0])
                    i_slices.append(res[1])
                    v_fine, i_fine = res[2]
                    v_fine_mm[i, : len(v_fine)] = v_fine
                    i_fine_mm[i, : len(i_fine)] = i_fine

        v_fine_mm.flush()
        i_fine_mm.flush()
        del v_fine_mm, i_fine_mm

        log.info(
            f"Retained {len(valid_indices)} / {N_raw} valid curves after PCHIP."
        )
        if not valid_indices:
            raise RuntimeError("No valid curves found after preprocessing.")

        v_slices = np.array(v_slices)
        i_slices = np.array(i_slices)
        valid_indices = np.array(valid_indices)

        isc_vals, i_slices_scaled = zip(
            *[normalize_and_scale_by_isc(c) for c in i_slices]
        )
        isc_vals = np.array(isc_vals)
        i_slices_scaled = np.array(i_slices_scaled)
        sample_weights = compute_curvature_weights(
            i_slices_scaled, **cfg["dataset"]["curvature_weighting"]
        )

        # ── Train / val / test split ──
        all_idx = np.arange(len(valid_indices))
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=0.2, random_state=cfg["train"]["seed"]
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.15, random_state=cfg["train"]["seed"]
        )
        split_labels = np.array([""] * len(all_idx), dtype=object)
        split_labels[train_idx] = "train"
        split_labels[val_idx] = "val"
        split_labels[test_idx] = "test"
        train_mask = split_labels == "train"

        # ── Fit parameter transformer (train split only) ──
        params_df_valid = params_df.iloc[valid_indices].reset_index(drop=True)
        params_df_train = params_df_valid.iloc[train_idx].reset_index(drop=True)
        param_transformer = get_param_transformer(COLNAMES)
        param_transformer.fit(params_df_train)
        joblib.dump(param_transformer, paths["param_transformer"])
        param_dim = param_transformer.transform(params_df_valid).shape[1]

        # ── Load scalar features exclusively from external txt files ──
        scalar_files = paths.get("scalar_files", [])
        scalar_files_extra = paths.get("scalar_files_extra", [])
        scalar_names: list[str] = []
        scalar_columns: list[np.ndarray] = []

        if scalar_files:
            for sf, sf_extra in zip(scalar_files, scalar_files_extra or [None] * len(scalar_files)):
                name, vals_primary = load_scalar_txt(sf)
                if sf_extra:
                    _, vals_extra = load_scalar_txt(sf_extra)
                    vals_all = np.concatenate([vals_primary, vals_extra])
                else:
                    vals_all = vals_primary
                if len(vals_all) < N_raw:
                    log.warning(
                        f"Scalar file '{name}' has {len(vals_all)} rows but "
                        f"expected {N_raw}. Padding with NaN."
                    )
                    vals_all = np.pad(
                        vals_all, (0, N_raw - len(vals_all)),
                        constant_values=np.nan,
                    )
                scalar_names.append(name)
                scalar_columns.append(vals_all[valid_indices])

        scalar_dim = len(scalar_names)
        if scalar_dim > 0:
            scalar_data = np.column_stack(scalar_columns).astype(np.float32)
            scalar_df = pd.DataFrame(scalar_data, columns=scalar_names)
            scalar_df_train = scalar_df.iloc[train_idx].reset_index(drop=True)
            scalar_transformer = Pipeline([
                ("scaler", MinMaxScaler(feature_range=(-1, 1)))
            ])
            scalar_transformer.fit(scalar_df_train)
            joblib.dump(scalar_transformer, paths["scalar_transformer"])
            log.info(f"Scalar features from txt: {scalar_names}")
        else:
            scalar_data = np.empty((len(valid_indices), 0), dtype=np.float32)
            log.info("No scalar files provided. Using device parameters only.")

        # ── Compute physics features from raw params (if enabled) ──
        physics_feature_names: list[str] = []
        physics_selected_indices = np.array([], dtype=np.int64)
        if cfg["dataset"].get("use_physics_features", False):
            from features import compute_all_physics_features, get_feature_names

            log.info("Computing physics features from raw parameters...")
            raw_params_tensor = torch.from_numpy(
                params_df_valid.values.astype(np.float32)
            )
            with torch.no_grad():
                physics_features = compute_all_physics_features(raw_params_tensor)
            physics_all = physics_features.numpy().astype(np.float32)
            all_feature_names = get_feature_names()

            sel_cfg = cfg["dataset"].get("physics_feature_selection", {})
            if sel_cfg.get("enabled", False):
                target_all = _compute_curve_targets_for_feature_selection(i_slices, v_slices)
                selected_idx, selected_names = _select_physics_features(
                    physics_train=physics_all[train_mask],
                    target_train=target_all[train_mask],
                    feature_names=all_feature_names,
                    corr_threshold=float(sel_cfg.get("corr_threshold", 0.85)),
                    weak_threshold=float(sel_cfg.get("weak_threshold", 0.30)),
                    max_features=sel_cfg.get("max_features"),
                )
                if len(selected_idx) == 0:
                    log.warning(
                        "Physics feature selection produced 0 features; falling back to all."
                    )
                    selected_idx = np.arange(physics_all.shape[1], dtype=int)
                    selected_names = all_feature_names
                physics_selected_indices = selected_idx.astype(np.int64)
                physics_feature_names = selected_names
                physics_data = physics_all[:, physics_selected_indices]
                log.info(
                    "Physics feature selection enabled: "
                    f"{physics_all.shape[1]} -> {len(physics_feature_names)} features"
                )
            else:
                physics_data = physics_all
                physics_feature_names = all_feature_names
                physics_selected_indices = np.arange(len(physics_feature_names), dtype=np.int64)

            physics_df = pd.DataFrame(physics_data, columns=physics_feature_names)
            physics_df_train = physics_df.iloc[train_idx].reset_index(drop=True)
            physics_transformer = Pipeline([
                ("scaler", MinMaxScaler(feature_range=(-1, 1)))
            ])
            physics_transformer.fit(physics_df_train)
            joblib.dump(physics_transformer, paths["physics_transformer"])
            log.info(
                f"Physics features: {len(physics_feature_names)} features computed and scaled"
            )
        else:
            physics_data = np.empty((len(valid_indices), 0), dtype=np.float32)

        physics_dim = len(physics_feature_names)
        self.cfg["model"]["param_dim"] = param_dim + scalar_dim + physics_dim
        log.info(
            f"Total input dimension: {param_dim + scalar_dim + physics_dim} "
            f"({param_dim} device params + {scalar_dim} scalars + {physics_dim} physics)"
        )

        np.savez(
            paths["preprocessed_npz"],
            v_slices=v_slices,
            i_slices=i_slices,
            i_slices_scaled=i_slices_scaled,
            sample_weights=sample_weights,
            isc_vals=isc_vals,
            valid_indices=valid_indices,
            split_labels=split_labels,
            scalar_names=np.array(scalar_names),
            scalar_data=scalar_data,
            physics_feature_names=np.array(physics_feature_names),
            physics_data=physics_data,
            physics_selected_indices=physics_selected_indices,
        )
        log.info(f"Saved preprocessed data to {paths['preprocessed_npz']}")


# ──────────────────────────────────────────────────────────────────────────────
#   LIGHTNING MODEL
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsIVSystem(pl.LightningModule):
    def __init__(self, cfg: dict, warmup_steps: int, total_steps: int):
        super().__init__()
        cfg_copy = copy.deepcopy(cfg)
        self.save_hyperparameters(cfg_copy)
        self.hparams.warmup_steps = warmup_steps
        self.hparams.total_steps = total_steps
        mcfg = self.hparams.model

        # ── Parameter MLP ──
        mlp_layers = []
        in_dim = mcfg["param_dim"]
        for units in mcfg["dense_units"]:
            mlp_layers.extend([
                nn.Linear(in_dim, units),
                nn.BatchNorm1d(units),
                nn.GELU(),
                nn.Dropout(mcfg["dropout"]),
            ])
            in_dim = units
        self.param_mlp = nn.Sequential(*mlp_layers)

        # ── Positional embedding ──
        self.pos_embed = make_positional_embedding(self.hparams)
        seq_input_dim = mcfg["dense_units"][-1] + self.pos_embed.out_dim

        filters = mcfg["filters"]
        kernel = mcfg["kernel"]
        dropout = mcfg["dropout"]
        heads = mcfg["heads"]
        arch = mcfg["architecture"]

        # ── Build backbone blocks based on architecture choice ──
        if arch == "cnn":
            d1 = 2 ** 0 if mcfg["use_dilated_conv"] else 1
            d2 = 2 ** 1 if mcfg["use_dilated_conv"] else 1
            self.block1 = TemporalBlock(seq_input_dim, filters[0], kernel, dropout, dilation=d1)
            if mcfg["use_attention"]:
                self.block2 = SelfAttentionBlock(filters[0], heads, dropout)
            else:
                self.block2 = TemporalBlock(filters[0], filters[0], kernel, dropout, dilation=d1)
            self.block3 = TemporalBlock(filters[0], filters[1], kernel, dropout, dilation=d2)

        elif arch == "conv":
            d1 = 2 ** 0 if mcfg["use_dilated_conv"] else 1
            d2 = 2 ** 1 if mcfg["use_dilated_conv"] else 1
            self.block1 = ConvResBlock(seq_input_dim, filters[0], kernel, dropout, dilation=d1)
            if mcfg["use_attention"]:
                self.block2 = SelfAttentionBlock(filters[0], heads, dropout)
            else:
                self.block2 = ConvResBlock(filters[0], filters[0], kernel, dropout, dilation=d1)
            self.block3 = ConvResBlock(filters[0], filters[1], kernel, dropout, dilation=d2)

        elif arch == "pointwise":
            self.block1 = PointwiseResBlock(seq_input_dim, filters[0], dropout)
            if mcfg["use_attention"]:
                self.block2 = SelfAttentionBlock(filters[0], heads, dropout)
            else:
                self.block2 = PointwiseResBlock(filters[0], filters[0], dropout)
            self.block3 = PointwiseResBlock(filters[0], filters[1], dropout)

        else:
            raise ValueError(f"Unknown architecture: {arch}")

        self.out_head = nn.Linear(filters[1], 1)
        self.apply(self._init_weights)

        # Buffers for test-time aggregation
        self.test_preds: list = []
        self.test_trues: list = []
        self.test_v_slices: list = []
        self.all_test_preds_np = None
        self.all_test_trues_np = None
        self.all_test_v_slices_np = None
        self.all_test_stats = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, X_combined: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        B, L = voltage.shape
        p = self.param_mlp(X_combined)
        v_emb = self.pos_embed(voltage)
        p_rep = p.unsqueeze(1).expand(-1, L, -1)
        x = torch.cat([p_rep, v_emb], dim=-1).transpose(1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x).transpose(1, 2)
        return self.out_head(x).squeeze(-1)

    def compute_jacobian_norm(
        self,
        X_combined: torch.Tensor,
        voltage: torch.Tensor,
        n_hutchinson_samples: int = 4,
    ) -> torch.Tensor:
        """Approximate ||J||_F^2 where J = d(output)/d(X_combined).
        Uses Hutchinson trace estimator in fp32 for numerical stability.
        Returns scalar: mean Jacobian Frobenius norm squared, normalized by input dim.
        """
        x = X_combined.detach().float().requires_grad_(True)
        v = voltage.detach()

        dev_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            y_pred = self.forward(x, v)

        if torch.isnan(y_pred).any():
            return torch.tensor(0.0, device=X_combined.device, requires_grad=True)

        jac_norm_accum = torch.tensor(0.0, device=X_combined.device)
        for _ in range(n_hutchinson_samples):
            v_rand = torch.randn_like(y_pred)
            try:
                vjp = torch.autograd.grad(
                    outputs=y_pred,
                    inputs=x,
                    grad_outputs=v_rand,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                jac_norm_accum = jac_norm_accum + (vjp ** 2).sum(dim=1).mean()
            except RuntimeError:
                return torch.tensor(0.0, device=X_combined.device, requires_grad=True)

        jac_norm = jac_norm_accum / n_hutchinson_samples
        jac_norm = jac_norm / max(1, X_combined.shape[1])

        if torch.isnan(jac_norm) or torch.isinf(jac_norm):
            return torch.tensor(0.0, device=X_combined.device, requires_grad=True)

        return jac_norm.clamp(max=10.0)

    def _step(self, batch, stage: str):
        y_pred = self(batch["X_combined"], batch["voltage"])
        loss, comps = physics_loss(
            y_pred,
            batch["current_scaled"],
            batch["sample_w"],
            self.hparams.model["loss_weights"],
        )

        # Jacobian regularization (training only, when weight > 0)
        jac_weight = self.hparams.model["loss_weights"].get("jacobian", 0.0)
        if stage == "train" and jac_weight > 0:
            jac_norm = self.compute_jacobian_norm(
                batch["X_combined"], batch["voltage"]
            )
            loss = loss + jac_weight * jac_norm
            comps["jacobian"] = jac_norm

        self.log_dict(
            {f"{stage}_{k}": v for k, v in comps.items()},
            on_step=False, on_epoch=True,
            batch_size=len(batch["voltage"]),
        )
        self.log(
            f"{stage}_loss", loss,
            prog_bar=(stage == "val"),
            on_step=False, on_epoch=True,
            batch_size=len(batch["voltage"]),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        pred_scaled = self(batch["X_combined"], batch["voltage"])
        self.test_preds.append(denormalize(pred_scaled.cpu(), batch["isc"].cpu()))
        self.test_trues.append(batch["i_true_slice"].cpu())
        self.test_v_slices.append(batch["v_true_slice"].cpu())
        loss, _ = physics_loss(
            pred_scaled, batch["current_scaled"], batch["sample_w"],
            self.hparams.model["loss_weights"],
        )
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True,
            batch_size=len(batch["voltage"]),
        )

    def on_test_epoch_start(self):
        self.test_preds.clear()
        self.test_trues.clear()
        self.test_v_slices.clear()

    def on_test_epoch_end(self):
        if not self.test_preds:
            return
        self.all_test_preds_np = torch.cat(self.test_preds, dim=0).numpy()
        self.all_test_trues_np = torch.cat(self.test_trues, dim=0).numpy()
        self.all_test_v_slices_np = torch.cat(self.test_v_slices, dim=0).numpy()

        stats = self._compute_detailed_statistics(
            self.all_test_preds_np, self.all_test_trues_np, self.all_test_v_slices_np
        )
        self.log_dict({f"test/stats/{k}": v for k, v in stats.items()}, prog_bar=True)
        self.all_test_stats = stats

    def _compute_detailed_statistics(self, preds, trues, v_slices):
        stats = {}
        valid_mask = np.all(np.isfinite(preds), axis=1) & np.all(
            np.isfinite(trues), axis=1
        )
        if not np.any(valid_mask):
            return {"error": 1.0}
        preds, trues, v_slices = (
            preds[valid_mask], trues[valid_mask], v_slices[valid_mask],
        )
        per_curve_mae = np.mean(np.abs(preds - trues), axis=1)
        per_curve_rmse = np.sqrt(np.mean((preds - trues) ** 2, axis=1))
        var_mask = np.var(trues, axis=1) > 1e-6
        per_curve_r2 = np.array([
            r2_score(trues[i], preds[i]) if var_mask[i] else -1.0
            for i in range(len(trues))
        ])
        for name, arr in [
            ("mae", per_curve_mae),
            ("rmse", per_curve_rmse),
            ("r2", per_curve_r2[var_mask]),
        ]:
            stats[f"{name}_mean"] = np.mean(arr)
            stats[f"{name}_std"] = np.std(arr)
            stats[f"{name}_median"] = np.median(arr)

        isc_true, isc_pred = trues[:, 0], preds[:, 0]
        stats["isc_error_abs_mean"] = np.mean(np.abs(isc_true - isc_pred))
        stats["isc_error_abs_median"] = np.median(np.abs(isc_true - isc_pred))

        voc_true, voc_pred = [], []
        for i in range(len(trues)):
            if trues[i, -1] < 0 and np.any(trues[i] > 0):
                v_t = np.interp(0, trues[i, ::-1], v_slices[i, ::-1])
                if preds[i, -1] < 0 and np.any(preds[i] > 0):
                    v_p = np.interp(0, preds[i, ::-1], v_slices[i, ::-1])
                    voc_true.append(v_t)
                    voc_pred.append(v_p)
        if voc_true:
            voc_err = np.abs(np.array(voc_true) - np.array(voc_pred))
            stats["voc_error_abs_mean"] = np.mean(voc_err)
            stats["voc_error_abs_median"] = np.median(voc_err)
            n_with_voc = np.sum((trues[:, -1] < 0) & np.any(trues > 0, axis=1))
            stats["voc_prediction_rate"] = len(voc_true) / max(n_with_voc, 1)
        else:
            stats["voc_error_abs_mean"] = -1.0
            stats["voc_error_abs_median"] = -1.0
            stats["voc_prediction_rate"] = 0.0
        return stats

    def configure_optimizers(self):
        opt_cfg = self.hparams.optimizer
        trainer_cfg = self.hparams.trainer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=opt_cfg["lr"],
            total_steps=self.hparams.total_steps,
            pct_start=opt_cfg["warmup_epochs"] / trainer_cfg["max_epochs"],
            final_div_factor=1 / opt_cfg["final_lr_ratio"],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# ──────────────────────────────────────────────────────────────────────────────
#   PLOTTING CALLBACK
# ──────────────────────────────────────────────────────────────────────────────

class ExamplePlotsCallback(pl.Callback):
    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        log.info("--- Generating reconstructed plots ---")
        if pl_module.all_test_preds_np is None:
            log.warning("No predictions found. Skipping plotting.")
            return

        preds = pl_module.all_test_preds_np
        trues = pl_module.all_test_trues_np
        valid_mask = [i for i in range(len(trues)) if np.var(trues[i]) > 1e-6]
        if not valid_mask:
            return

        metrics_df = pd.DataFrame(
            {"r2": [r2_score(trues[i], preds[i]) for i in valid_mask]},
            index=valid_mask,
        )
        n_samples = min(self.num_samples, len(metrics_df))
        if n_samples == 0:
            return

        plot_groups = {
            "Random_Samples": np.random.choice(
                metrics_df.index, n_samples, replace=False
            ),
            "Best_R2_Samples": metrics_df.nlargest(n_samples, "r2").index.values,
            "Worst_R2_Samples": metrics_df.nsmallest(n_samples, "r2").index.values,
        }
        for name, indices in plot_groups.items():
            if not trainer.logger:
                continue
            filename = Path(trainer.logger.log_dir) / f"test_plots_{name.lower()}.png"
            self._generate_and_log_plot(
                trainer, pl_module, filename, name, indices, preds, trues, metrics_df
            )

    def _generate_and_log_plot(
        self, trainer, pl_module, filename, title, indices, preds, trues, metrics_df
    ):
        hparams = pl_module.hparams
        paths = hparams.dataset["paths"]
        try:
            preprocessed_data = np.load(
                paths["preprocessed_npz"], allow_pickle=True
            )
        except FileNotFoundError:
            log.error(f"Could not find {paths['preprocessed_npz']}.")
            return

        test_indices_in_valid_set = np.where(
            preprocessed_data["split_labels"] == "test"
        )[0]
        try:
            n_fine = hparams.dataset["pchip"]["n_fine"]
            v_fine_mm = np.memmap(
                paths["v_fine_memmap"], dtype=np.float16, mode="r"
            ).reshape(-1, n_fine)
            i_fine_mm = np.memmap(
                paths["i_fine_memmap"], dtype=np.float16, mode="r"
            ).reshape(-1, n_fine)
        except (FileNotFoundError, ValueError) as e:
            log.error(f"Error loading memmap files: {e}.")
            return

        n_samples = len(indices)
        nrows, ncols = (n_samples + 3) // 4, 4
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(20, 5 * nrows),
            squeeze=False, constrained_layout=True,
        )
        fig.suptitle(title.replace("_", " "), fontsize=20, weight="bold")

        for i, test_set_idx in enumerate(indices):
            ax = axes.flatten()[i]
            valid_set_idx = test_indices_in_valid_set[test_set_idx]
            raw_data_idx = preprocessed_data["valid_indices"][valid_set_idx]
            v_slice = pl_module.all_test_v_slices_np[test_set_idx]
            i_true_slice = trues[test_set_idx]
            i_pred_slice = preds[test_set_idx]
            v_fine = v_fine_mm[raw_data_idx].astype(np.float32)
            i_fine = i_fine_mm[raw_data_idx].astype(np.float32)
            mask = ~np.isnan(v_fine) & ~np.isinf(v_fine)
            v_fine, i_fine = v_fine[mask], i_fine[mask]

            if len(v_fine) > 0:
                ax.plot(v_fine, i_fine, "k-", alpha=0.7, lw=2, label="Actual (Fine)")
                recon = PchipInterpolator(v_slice, i_pred_slice, extrapolate=False)(
                    v_fine
                )
                ax.plot(v_fine, recon, "r--", lw=2, label="Predicted (Recon)")
            ax.plot(v_slice, i_true_slice, "bo", ms=6, label="Actual Pts")
            ax.plot(v_slice, i_pred_slice, "rx", ms=6, mew=2, label="Predicted Pts")
            ax.set_title(
                f"Sample #{test_set_idx} (R²={metrics_df.loc[test_set_idx, 'r2']:.4f})"
            )
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current (mA/cm²)")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            if len(v_fine) > 0 and len(i_fine) > 0:
                ax.set_xlim(left=-0.05, right=max(v_fine.max() * 1.05, 0.1))
                ax.set_ylim(bottom=-max(i_fine.max() * 0.05, 1))

        for j in range(n_samples, len(axes.flatten())):
            fig.delaxes(axes.flatten()[j])
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        try:
            trainer.logger.experiment.add_image(
                title, np.array(Image.open(filename)), 0, dataformats="HWC"
            )
            log.info(f"Saved plot: {filename}")
        except Exception as e:
            log.warning(f"Could not log image to TensorBoard: {e}")
        del v_fine_mm, i_fine_mm, preprocessed_data


# ──────────────────────────────────────────────────────────────────────────────
#   EXPERIMENT RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(cfg: dict) -> dict:
    total_start = time.perf_counter()
    run_name = cfg["train"]["run_name"]
    log.info(f"--- Starting Experiment: {run_name} ---")
    seed_everything(cfg["train"]["seed"])

    output_dir = Path(cfg["_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = IVDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    batches_per_epoch = len(datamodule.train_dataloader())
    if batches_per_epoch == 0:
        log.error("Train dataloader is empty. Aborting.")
        return {"error": "Empty train dataloader"}

    total_steps = cfg["trainer"]["max_epochs"] * batches_per_epoch
    model = PhysicsIVSystem(
        cfg,
        warmup_steps=cfg["optimizer"]["warmup_epochs"] * batches_per_epoch,
        total_steps=total_steps,
    )
    log.info(
        f"Model [{cfg['model']['architecture']}]: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
    )

    run_dir = output_dir / run_name
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        dirpath=run_dir / "checkpoints", filename="best-model",
    )
    logger = TensorBoardLogger(str(output_dir / "tb_logs"), name=run_name)

    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", patience=20, mode="min"),
        RichProgressBar(),
    ]
    if cfg["train"].get("enable_example_plots", False):
        callbacks.append(ExamplePlotsCallback(num_samples=8))

    trainer = pl.Trainer(
        **cfg["trainer"],
        default_root_dir=run_dir,
        logger=logger,
        callbacks=callbacks,
    )

    log.info("--- Starting Training ---")
    train_start = time.perf_counter()
    trainer.fit(model, datamodule=datamodule)
    train_time_s = time.perf_counter() - train_start

    log.info("--- Testing on Best Checkpoint ---")
    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_start = time.perf_counter()
    trainer.test(
        model, datamodule=datamodule,
        ckpt_path=best_ckpt if best_ckpt and Path(best_ckpt).exists() else None,
    )
    test_time_s = time.perf_counter() - test_start

    # Persist test predictions for downstream diagnostics/parity plotting.
    if model.all_test_preds_np is not None and model.all_test_trues_np is not None:
        preds_path = run_dir / "test_predictions.npz"
        np.savez_compressed(
            preds_path,
            y_pred=model.all_test_preds_np.astype(np.float32, copy=False),
            y_true=model.all_test_trues_np.astype(np.float32, copy=False),
            v_slice=(
                model.all_test_v_slices_np.astype(np.float32, copy=False)
                if model.all_test_v_slices_np is not None
                else np.array([], dtype=np.float32)
            ),
            run_name=run_name,
        )
        log.info(f"Saved test predictions to {preds_path}")
    else:
        log.warning("No test predictions found to save.")

    final_stats = model.all_test_stats or {}
    final_stats["run_name"] = run_name
    final_stats["architecture"] = cfg["model"]["architecture"]
    final_stats["use_attention"] = bool(cfg["model"]["use_attention"])
    final_stats["use_dilated_conv"] = bool(cfg["model"]["use_dilated_conv"])
    final_stats["train_time_s"] = float(train_time_s)
    final_stats["test_time_s"] = float(test_time_s)
    final_stats["total_time_s"] = float(time.perf_counter() - total_start)
    final_stats["train_steps"] = int(trainer.global_step)
    final_stats["max_epochs"] = int(cfg["trainer"]["max_epochs"])
    final_stats["batch_size"] = int(cfg["dataset"]["dataloader"]["batch_size"])

    stats_path = run_dir / "test_stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in final_stats.items()},
            f, indent=2,
        )
    log.info(f"Saved test stats to {stats_path}")
    log.info(f"--- Experiment Finished: {run_name} ---")
    return final_stats


def generate_summary_plots(results: list[dict], output_dir: Path):
    if not results:
        return
    log.info("--- Generating Summary Plots ---")
    df = pd.DataFrame([r for r in results if "error" not in r])
    if df.empty:
        return

    metrics_to_plot = {
        "Median R² Score": "r2_median",
        "Median Abs. MAE": "mae_median",
        "Median Voc Error (V)": "voc_error_abs_median",
        "Median Isc Error (mA/cm²)": "isc_error_abs_median",
    }
    plot_keys = [v for v in metrics_to_plot.values() if v in df.columns]
    if not plot_keys:
        return

    nrows = (len(plot_keys) + 1) // 2
    fig, axes = plt.subplots(
        nrows, 2, figsize=(16, 6 * nrows), constrained_layout=True
    )
    axes = axes.flatten()

    for i, key in enumerate(plot_keys):
        ax = axes[i]
        sns.boxplot(data=df, x=key, y="run_name", ax=ax, orient="h", palette="viridis")
        title = [k for k, v in metrics_to_plot.items() if v == key][0]
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Experiment", fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

    for j in range(len(plot_keys), len(axes)):
        fig.delaxes(axes[j])

    summary_path = output_dir / "cnn_experiment_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved summary plot to {summary_path}")


# ──────────────────────────────────────────────────────────────────────────────
#   CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Physics-Informed I-V Curve Reconstruction (SLURM/CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Data paths ──
    p.add_argument("--params", required=True, help="Path to LHS_parameters_m.txt")
    p.add_argument("--iv", required=True, help="Path to IV curves txt file")
    p.add_argument("--params-extra", default=None, help="Extra params file (e.g. 300k)")
    p.add_argument("--iv-extra", default=None, help="Extra IV file (e.g. 300k)")
    p.add_argument(
        "--scalar-files", nargs="*", default=None,
        help="Scalar txt files for primary data (e.g. voc_100k.txt vmpp_100k.txt)",
    )
    p.add_argument(
        "--scalar-files-extra", nargs="*", default=None,
        help="Scalar txt files for extra data (must match --scalar-files order)",
    )

    # ── Output ──
    p.add_argument("--output-dir", default="./lightning_output", help="Output root")
    p.add_argument(
        "--data-dir", default="./data/cnn_processed",
        help="Directory for preprocessed data cache",
    )

    # ── Experiment ──
    p.add_argument("--run-name", default="DilatedConv-full", help="Experiment run name")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--enable-example-plots", action="store_true",
        help="Enable expensive qualitative plotting callback at test time",
    )

    # ── Architecture ──
    p.add_argument(
        "--architecture", choices=["cnn", "conv", "pointwise"], default="conv",
        help="Backbone architecture: cnn (causal dilated), conv (non-causal 1D), pointwise (1x1)",
    )
    p.add_argument(
        "--use-attention", action="store_true", default=False,
        help="Enable self-attention block",
    )
    p.add_argument(
        "--no-attention", dest="use_attention", action="store_false",
        help="Disable self-attention",
    )
    p.add_argument(
        "--use-dilated", action="store_true", default=True,
        help="Enable dilated convolutions (CNN architecture only)",
    )
    p.add_argument(
        "--no-dilated", dest="use_dilated", action="store_false",
        help="Disable dilated convolutions",
    )

    # ── Physics features & Jacobian ──
    p.add_argument(
        "--use-physics-features", action="store_true", default=False,
        help="Compute and use 71 physics-derived features from raw parameters",
    )
    p.add_argument(
        "--physics-feature-selection", action="store_true", default=False,
        help="Enable train-only physics feature filtering (multicollinearity + relevance)",
    )
    p.add_argument(
        "--physics-weak-threshold", type=float, default=0.30,
        help="Minimum max |corr| with curve-derived targets to keep a physics feature",
    )
    p.add_argument(
        "--physics-corr-threshold", type=float, default=0.85,
        help="Pairwise |corr| threshold for multicollinearity pruning",
    )
    p.add_argument(
        "--physics-max-features", type=int, default=None,
        help="Optional cap on number of selected physics features",
    )
    p.add_argument(
        "--jacobian-weight", type=float, default=0.0,
        help="Weight for Jacobian norm regularization (0 = disabled). Recommended: 1e-4 to 1e-2",
    )

    # ── Training ──
    p.add_argument("--max-epochs", type=int, default=100, help="Maximum epochs")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size")
    p.add_argument(
        "--num-workers", type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="DataLoader workers",
    )
    p.add_argument(
        "--prefetch-factor", type=int, default=4,
        help="DataLoader prefetch factor (only used when num_workers > 0)",
    )
    p.add_argument(
        "--force-preprocess", action="store_true",
        help="Delete cached preprocessed data and re-run",
    )

    return p.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)
    cfg["_output_dir"] = str(Path(args.output_dir).resolve())
    cfg["train"]["enable_example_plots"] = bool(args.enable_example_plots)

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.force_preprocess:
        npz_path = Path(cfg["dataset"]["paths"]["preprocessed_npz"])
        if npz_path.exists():
            npz_path.unlink()
            log.info(f"Removed cached data: {npz_path}")

    results = run_experiment(cfg)
    generate_summary_plots([results], Path(args.output_dir))

    log.info("=" * 60)
    log.info("Done.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
