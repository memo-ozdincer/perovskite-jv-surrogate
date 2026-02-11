#!/usr/bin/env python3
"""
Prepare scalar TXT files (Voc, Vmpp) from externally predicted sources.

This script does NOT compute scalars from true IV curves.
It only reformats externally predicted scalar files into the naming
convention expected by the Stage-2 curve model pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_scalar_column(path: str) -> np.ndarray:
    """Load one scalar column from csv/txt with optional header."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        first = f.readline().strip()

    skiprows = 0
    if first:
        token = first.split(",")[0].strip()
        try:
            float(token)
        except ValueError:
            skiprows = 1

    values = np.loadtxt(p, delimiter=",", skiprows=skiprows, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return values


def prepare_scalars(
    voc_input: str, vmpp_input: str, output_dir: str, tag: str, suffix: str = "_clean"
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    voc = _load_scalar_column(voc_input)
    vmpp = _load_scalar_column(vmpp_input)
    if len(voc) != len(vmpp):
        raise ValueError(f"Length mismatch: voc={len(voc)} vmpp={len(vmpp)}")

    voc_out = output_path / f"voc{suffix}_{tag}.txt"
    vmpp_out = output_path / f"vmpp{suffix}_{tag}.txt"

    np.savetxt(voc_out, voc, delimiter=",", fmt="%.6f", header="Voc", comments="")
    np.savetxt(vmpp_out, vmpp, delimiter=",", fmt="%.6f", header="Vmpp", comments="")

    print("Prepared scalar files from external predictions:")
    print(f"  Source Voc:  {voc_input}")
    print(f"  Source Vmpp: {vmpp_input}")
    print(f"  Voc:  {voc_out}")
    print(f"  Vmpp: {vmpp_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Voc/Vmpp TXT files from external scalar predictions"
    )
    parser.add_argument(
        "--voc-input",
        type=str,
        required=True,
        help="Path to externally predicted Voc file",
    )
    parser.add_argument(
        "--vmpp-input",
        type=str,
        required=True,
        help="Path to externally predicted Vmpp file",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--tag", type=str, required=True, help="Dataset tag (e.g., 100k, 300k)")
    parser.add_argument(
        "--suffix", type=str, default="_clean", help="Filename suffix (default: _clean)"
    )
    args = parser.parse_args()

    prepare_scalars(
        args.voc_input, args.vmpp_input, args.output_dir, args.tag, args.suffix
    )


if __name__ == "__main__":
    main()
