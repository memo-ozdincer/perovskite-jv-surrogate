#!/usr/bin/env python3
"""
Generate scalar TXT files (Voc, Vmpp) from IV curves.

This is intentionally separated from preprocessing/training so that
Voc/Vmpp can be swapped with outputs from other ML models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import V_GRID
from data import extract_targets_gpu


def generate_scalars(iv_file: str, output_dir: str, tag: str, suffix: str = "_clean") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iv_data = np.loadtxt(iv_file, delimiter=',', dtype=np.float32)
    if iv_data.ndim == 1:
        iv_data = iv_data.reshape(1, -1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v_grid = torch.from_numpy(V_GRID.astype(np.float32)).to(device)
    iv_tensor = torch.from_numpy(iv_data).to(device)

    targets = extract_targets_gpu(iv_tensor, v_grid)
    voc = targets['Voc'].cpu().numpy()
    vmpp = targets['Vmpp'].cpu().numpy()

    voc_out = output_path / f"voc{suffix}_{tag}.txt"
    vmpp_out = output_path / f"vmpp{suffix}_{tag}.txt"

    np.savetxt(voc_out, voc, delimiter=',', fmt='%.6f', header='Voc', comments='')
    np.savetxt(vmpp_out, vmpp, delimiter=',', fmt='%.6f', header='Vmpp', comments='')

    print(f"Generated scalar files:")
    print(f"  Voc:  {voc_out}")
    print(f"  Vmpp: {vmpp_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Voc/Vmpp TXT files from IV curves')
    parser.add_argument('--iv', type=str, required=True, help='Path to IV curves file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--tag', type=str, required=True, help='Dataset tag (e.g., 100k, 300k)')
    parser.add_argument('--suffix', type=str, default='_clean', help='Filename suffix (default: _clean)')
    args = parser.parse_args()

    generate_scalars(args.iv, args.output_dir, args.tag, args.suffix)


if __name__ == '__main__':
    main()
