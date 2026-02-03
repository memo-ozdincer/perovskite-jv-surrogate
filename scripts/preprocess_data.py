#!/usr/bin/env python3
"""
Data Preprocessing Script for PV Curve Prediction.

This script performs quality filtering on the raw IV curve dataset,
removing samples with anomalous characteristics that are difficult
to model accurately.

Usage:
    python preprocess_data.py --params LHS_parameters_m.txt --iv IV_m.txt --output-dir preprocessed/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLNAMES, V_GRID
from data import extract_targets_gpu


def preprocess_dataset(
    params_file: str,
    iv_file: str,
    output_dir: str,
    min_ff: float = 0.30,
    min_vmpp: float = 0.30,
    min_pce_quantile: float = 0.0,
    suffix: str = "_clean"
) -> dict:
    """
    Preprocess and filter a PV dataset.

    Quality filtering removes samples with:
    - Low fill factor (< min_ff): indicates S-shaped or kinked IV curves
    - Low Vmpp (≤ min_vmpp): indicates extreme operating conditions
    - Low PCE (< percentile): optional removal of lowest efficiency samples

    Args:
        params_file: Path to parameters file
        iv_file: Path to IV curves file
        output_dir: Directory to save preprocessed files
        min_ff: Minimum fill factor threshold
        min_vmpp: Minimum Vmpp threshold
        min_pce_quantile: Minimum PCE percentile (0 = no filter)
        suffix: Suffix for output filenames

    Returns:
        Statistics dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load raw data
    print(f"\nLoading data from:")
    print(f"  Parameters: {params_file}")
    print(f"  IV curves:  {iv_file}")

    params_df = pd.read_csv(params_file, header=None, names=COLNAMES)
    iv_data = np.loadtxt(iv_file, delimiter=',', dtype=np.float32)

    if iv_data.ndim == 1:
        iv_data = iv_data.reshape(1, -1)

    n_original = len(params_df)
    print(f"\nOriginal dataset: {n_original} samples")

    # Extract IV curve characteristics
    v_grid = torch.from_numpy(V_GRID.astype(np.float32)).to(device)
    iv_tensor = torch.from_numpy(iv_data).to(device)
    targets = extract_targets_gpu(iv_tensor, v_grid)

    ff = targets['FF'].cpu().numpy()
    vmpp = targets['Vmpp'].cpu().numpy()
    pce = targets['PCE'].cpu().numpy()
    jsc = targets['Jsc'].cpu().numpy()
    voc = targets['Voc'].cpu().numpy()

    # Print distribution statistics
    print(f"\nOriginal data statistics:")
    print(f"  FF:   min={ff.min():.3f}, max={ff.max():.3f}, mean={ff.mean():.3f}")
    print(f"  Vmpp: min={vmpp.min():.3f}, max={vmpp.max():.3f}, mean={vmpp.mean():.3f}")
    print(f"  PCE:  min={pce.min():.4f}, max={pce.max():.4f}, mean={pce.mean():.4f}")
    print(f"  Jsc:  min={jsc.min():.1f}, max={jsc.max():.1f}, mean={jsc.mean():.1f}")
    print(f"  Voc:  min={voc.min():.3f}, max={voc.max():.3f}, mean={voc.mean():.3f}")

    # Build quality filter mask
    mask = np.ones(n_original, dtype=bool)

    # Filter by fill factor
    ff_mask = ff >= min_ff
    n_ff_removed = (~ff_mask).sum()
    mask &= ff_mask

    # Filter by Vmpp
    vmpp_mask = vmpp > min_vmpp
    n_vmpp_removed = (~vmpp_mask).sum()
    mask &= vmpp_mask

    # Filter by PCE quantile (optional)
    pce_threshold = 0.0
    n_pce_removed = 0
    if min_pce_quantile > 0:
        pce_threshold = np.quantile(pce, min_pce_quantile)
        pce_mask = pce >= pce_threshold
        n_pce_removed = (~pce_mask & mask).sum()
        mask &= pce_mask

    # Apply filter
    params_clean = params_df.iloc[mask].reset_index(drop=True)
    iv_clean = iv_data[mask]
    kept_indices = np.where(mask)[0].astype(np.int64)

    n_kept = mask.sum()
    n_removed = n_original - n_kept

    print(f"\nQuality filtering results:")
    print(f"  Removed for low FF (<{min_ff}):     {n_ff_removed:,}")
    print(f"  Removed for low Vmpp (≤{min_vmpp}): {n_vmpp_removed:,}")
    if min_pce_quantile > 0:
        print(f"  Removed for low PCE (<{pce_threshold:.4f}): {n_pce_removed:,}")
    print(f"  Total removed: {n_removed:,} ({100*n_removed/n_original:.1f}%)")
    print(f"  Samples retained: {n_kept:,} ({100*n_kept/n_original:.1f}%)")

    # Compute statistics for cleaned data
    ff_clean = ff[mask]
    vmpp_clean = vmpp[mask]
    pce_clean = pce[mask]

    print(f"\nCleaned data statistics:")
    print(f"  FF:   min={ff_clean.min():.3f}, max={ff_clean.max():.3f}, mean={ff_clean.mean():.3f}")
    print(f"  Vmpp: min={vmpp_clean.min():.3f}, max={vmpp_clean.max():.3f}, mean={vmpp_clean.mean():.3f}")
    print(f"  PCE:  min={pce_clean.min():.4f}, max={pce_clean.max():.4f}, mean={pce_clean.mean():.4f}")

    # Save preprocessed data
    params_file_base = Path(params_file).stem
    iv_file_base = Path(iv_file).stem

    params_out = output_path / f"{params_file_base}{suffix}.txt"
    iv_out = output_path / f"{iv_file_base}{suffix}.txt"

    # Save parameters (no header, comma-separated)
    params_clean.to_csv(params_out, header=False, index=False)

    # Save IV curves
    np.savetxt(iv_out, iv_clean, delimiter=',', fmt='%.6f')

    # Extract and save anchor values for the CLEANED data
    # These can be used as auxiliary inputs to the curve model
    targets_clean = extract_targets_gpu(
        torch.from_numpy(iv_clean).to(device),
        v_grid
    )

    jsc_clean = targets_clean['Jsc'].cpu().numpy()
    voc_clean = targets_clean['Voc'].cpu().numpy()
    vmpp_clean = targets_clean['Vmpp'].cpu().numpy()
    jmpp_clean = targets_clean['Jmpp'].cpu().numpy()
    ff_clean_arr = targets_clean['FF'].cpu().numpy()
    pce_clean_arr = targets_clean['PCE'].cpu().numpy()
    pmpp_clean = targets_clean['Pmpp'].cpu().numpy()

    # Stack anchors: [Jsc, Voc, Vmpp, Jmpp, FF, PCE, Pmpp]
    anchors = np.column_stack([
        jsc_clean, voc_clean, vmpp_clean, jmpp_clean,
        ff_clean_arr, pce_clean_arr, pmpp_clean
    ])

    # Save anchors to txt file
    anchors_out = output_path / f"anchors{suffix}_{params_file_base.replace('LHS_parameters_m', '')}.txt"
    if anchors_out.name == f"anchors{suffix}_.txt":
        anchors_out = output_path / f"anchors{suffix}_100k.txt"

    np.savetxt(anchors_out, anchors, delimiter=',', fmt='%.6f',
               header='Jsc,Voc,Vmpp,Jmpp,FF,PCE,Pmpp', comments='')

    print(f"\nSaved preprocessed data:")
    print(f"  Parameters: {params_out}")
    print(f"  IV curves:  {iv_out}")
    print(f"  Anchors:    {anchors_out}")

    # Save Voc values separately (for decoupled inference inputs)
    voc_out = output_path / f"voc{suffix}_{params_file_base.replace('LHS_parameters_m', '')}.txt"
    if voc_out.name == f"voc{suffix}_.txt":
        voc_out = output_path / f"voc{suffix}_100k.txt"
    np.savetxt(voc_out, voc_clean, delimiter=',', fmt='%.6f', header='Voc', comments='')
    print(f"  Voc file:   {voc_out}")

    # Save kept indices for alignment checks across files
    indices_out = output_path / f"kept_indices{suffix}_{params_file_base.replace('LHS_parameters_m', '')}.txt"
    if indices_out.name == f"kept_indices{suffix}_.txt":
        indices_out = output_path / f"kept_indices{suffix}_100k.txt"
    np.savetxt(indices_out, kept_indices, fmt='%d', header='idx', comments='')
    print(f"  Kept idx:   {indices_out}")

    # Save statistics
    stats = {
        'original_samples': n_original,
        'retained_samples': int(n_kept),
        'removed_samples': int(n_removed),
        'removal_percentage': float(100 * n_removed / n_original),
        'quality_thresholds': {
            'min_ff': min_ff,
            'min_vmpp': min_vmpp,
            'min_pce_quantile': min_pce_quantile,
            'pce_threshold': float(pce_threshold)
        },
        'removed_by_criterion': {
            'low_ff': int(n_ff_removed),
            'low_vmpp': int(n_vmpp_removed),
            'low_pce': int(n_pce_removed)
        },
        'original_stats': {
            'ff_mean': float(ff.mean()),
            'ff_std': float(ff.std()),
            'vmpp_mean': float(vmpp.mean()),
            'pce_mean': float(pce.mean())
        },
        'cleaned_stats': {
            'ff_mean': float(ff_clean.mean()),
            'ff_std': float(ff_clean.std()),
            'vmpp_mean': float(vmpp_clean.mean()),
            'pce_mean': float(pce_clean.mean())
        },
        'alignment': {
            'kept_indices_file': str(indices_out),
            'params_out': str(params_out),
            'iv_out': str(iv_out),
            'anchors_out': str(anchors_out),
            'voc_out': str(voc_out)
        },
        'output_files': {
            'params': str(params_out),
            'iv': str(iv_out),
            'anchors': str(anchors_out),
            'voc': str(voc_out),
            'kept_indices': str(indices_out)
        }
    }

    stats_file = output_path / f"preprocessing_stats{suffix}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics: {stats_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess PV dataset with quality filtering'
    )
    parser.add_argument('--params', type=str, required=True,
                        help='Path to parameters file')
    parser.add_argument('--iv', type=str, required=True,
                        help='Path to IV curves file')
    parser.add_argument('--params-extra', type=str, nargs='*', default=[],
                        help='Additional parameter files to process')
    parser.add_argument('--iv-extra', type=str, nargs='*', default=[],
                        help='Additional IV files to process')
    parser.add_argument('--output-dir', type=str, default='preprocessed',
                        help='Output directory for preprocessed files')
    parser.add_argument('--min-ff', type=float, default=0.30,
                        help='Minimum fill factor threshold (default: 0.30)')
    parser.add_argument('--min-vmpp', type=float, default=0.30,
                        help='Minimum Vmpp threshold (default: 0.30)')
    parser.add_argument('--min-pce-quantile', type=float, default=0.0,
                        help='Minimum PCE quantile (default: 0.0, no filter)')
    parser.add_argument('--suffix', type=str, default='_clean',
                        help='Suffix for output filenames (default: _clean)')

    args = parser.parse_args()

    print("=" * 60)
    print("PV Dataset Preprocessing")
    print("=" * 60)
    print(f"\nQuality thresholds:")
    print(f"  Min FF:   {args.min_ff}")
    print(f"  Min Vmpp: {args.min_vmpp}")
    if args.min_pce_quantile > 0:
        print(f"  Min PCE quantile: {args.min_pce_quantile}")

    # Process primary files
    all_params = [args.params] + args.params_extra
    all_iv = [args.iv] + args.iv_extra

    if len(all_params) != len(all_iv):
        raise ValueError("Number of params and IV files must match")

    all_stats = []
    for params_file, iv_file in zip(all_params, all_iv):
        print(f"\n{'=' * 60}")
        stats = preprocess_dataset(
            params_file=params_file,
            iv_file=iv_file,
            output_dir=args.output_dir,
            min_ff=args.min_ff,
            min_vmpp=args.min_vmpp,
            min_pce_quantile=args.min_pce_quantile,
            suffix=args.suffix
        )
        all_stats.append(stats)

    # Print summary
    print(f"\n{'=' * 60}")
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    total_original = sum(s['original_samples'] for s in all_stats)
    total_retained = sum(s['retained_samples'] for s in all_stats)
    print(f"Total samples: {total_original:,} -> {total_retained:,} ({100*total_retained/total_original:.1f}% retained)")


if __name__ == '__main__':
    main()
