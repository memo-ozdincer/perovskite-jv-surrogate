#!/usr/bin/env python3
"""
Experiment-level analysis for the dilated-convolution I-V reconstruction model.

Runs:
  1. Collect all available test statistics across experiments
  2. Generate ranked comparison plots
  3. Export consolidated analysis tables

Usage:
    python tcn_analysis.py --results-dir outputs/tcn_icml/ \\
                           --output-dir outputs/tcn_icml/analysis \\
                           --main-model-dir outputs/tcn_icml/T0-1-Conv-Dilated/seed_42
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def run_analysis(
    results_dir: str,
    output_dir: str,
    main_model_dir: Optional[str] = None,
):
    """Run aggregate analysis modules."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DILATED-CONV AGGREGATE ANALYSIS")
    print("=" * 60)

    # ── Collect test_stats from all experiments for cross-comparison ──
    print(f"\n[1] Collecting test statistics from {results_dir}")
    stats_files = list(Path(results_dir).rglob("test_stats.json"))
    all_stats = []
    for sf in stats_files:
        try:
            with open(sf) as f:
                s = json.load(f)
            s["source_file"] = str(sf)
            all_stats.append(s)
        except Exception:
            pass

    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(out / "all_test_statistics.csv", index=False)
        print(f"  Saved {len(stats_df)} test statistics")

        # Generate comparison plots
        if "r2_median" in stats_df.columns and "run_name" in stats_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats_df_sorted = stats_df.sort_values("r2_median", ascending=False)
            ax.barh(range(len(stats_df_sorted)),
                    stats_df_sorted["r2_median"],
                    color="#2E86AB", alpha=0.8)
            ax.set_yticks(range(len(stats_df_sorted)))
            ax.set_yticklabels(stats_df_sorted["run_name"], fontsize=8)
            ax.set_xlabel("$R^2$ (median)")
            ax.set_title("All Experiments — Ranked by $R^2$")
            ax.invert_yaxis()
            plt.tight_layout()
            fig.savefig(out / "ranking_r2.pdf", dpi=150)
            plt.close(fig)
            print(f"  saved: {out / 'ranking_r2.pdf'}")

    print(f"\n{'='*60}")
    print(f"Analysis complete. Results in {out}/")
    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--main-model-dir", default=None)
    args = p.parse_args()
    run_analysis(args.results_dir, args.output_dir, args.main_model_dir)


if __name__ == "__main__":
    main()
