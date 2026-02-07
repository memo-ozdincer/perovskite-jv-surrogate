#!/usr/bin/env python3
"""
Collect and aggregate all TCN experiment results into a single CSV.

Usage:
    python tcn_collect_results.py --results-dir outputs/tcn_icml/results --output-base outputs/tcn_icml
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def collect_results(results_dir: str, output_base: str):
    results_dir = Path(results_dir)
    output_base = Path(output_base)

    results = []
    for f in sorted(results_dir.glob("*_stats.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
            name = f.stem.replace("_stats", "")
            # Parse exp_id and seed from name like "T0-1-DilatedTCN_seed42"
            parts = name.rsplit("_seed", 1)
            data["exp_id"] = parts[0] if len(parts) == 2 else name
            data["seed"] = int(parts[1]) if len(parts) == 2 else -1
            data["run_file"] = str(f)
            results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    raw_path = output_base / "all_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved {len(df)} results to {raw_path}")

    # ── Aggregate across seeds (mean ± std) ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "seed"]

    if not numeric_cols:
        print("No numeric columns to aggregate.")
        return

    agg_funcs = {col: ["mean", "std", "count"] for col in numeric_cols}
    grouped = df.groupby("exp_id").agg(agg_funcs)
    grouped.columns = ["_".join(c) for c in grouped.columns]

    # Add name and architecture
    meta = df.groupby("exp_id").first()[
        [c for c in ["architecture", "run_name"] if c in df.columns]
    ]
    agg = pd.concat([meta, grouped], axis=1).reset_index()
    agg_path = output_base / "results_aggregated.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated results to {agg_path}")

    # ── Print summary ──
    print()
    print("=" * 72)
    print("RESULTS SUMMARY  (mean ± std across seeds)")
    print("=" * 72)

    key_metrics = ["r2_median", "r2_mean", "mae_median", "mae_mean",
                   "voc_error_abs_median", "isc_error_abs_median"]
    avail = [m for m in key_metrics if m in df.columns]

    for exp_id in sorted(df["exp_id"].unique()):
        sub = df[df["exp_id"] == exp_id]
        arch = sub["architecture"].iloc[0] if "architecture" in sub.columns else "?"
        print(f"\n{exp_id}  [{arch}]  (n={len(sub)} seeds)")
        for m in avail:
            vals = sub[m].dropna()
            if len(vals) > 0:
                print(f"  {m:30s}: {vals.mean():.6f} ± {vals.std():.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output-base", required=True)
    args = p.parse_args()
    collect_results(args.results_dir, args.output_base)


if __name__ == "__main__":
    main()
