#!/usr/bin/env python3
"""
Generate ALL publication figures and LaTeX tables for the ICML paper
from architecture ablation experiment results.

Usage:
    python tcn_generate_figures.py --results all_results.csv --output figures/

Output:
    figures/
    ├── main_paper/
    │   ├── fig_architecture_comparison.pdf      # Bar chart: R² across architectures
    │   ├── fig_jv_overlays.pdf                  # 3×3 true vs predicted
    │   ├── fig_ablation_table.pdf               # Heatmap of ablation results
    │   ├── fig_error_distribution.pdf           # Histogram of per-curve errors
    │   └── fig_training_curves.pdf              # Loss vs epoch (if TB logs)
    ├── appendix/
    │   ├── fig_batch_size_sweep.pdf
    │   ├── fig_epoch_sweep.pdf
    │   ├── fig_data_scaling.pdf
    │   ├── fig_attention_impact.pdf
    │   ├── fig_runtime_comparison.pdf
    │   └── fig_error_vs_voltage.pdf
    └── tables/
        ├── table_main_results.tex
        └── table_ablations.tex
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#2E8B57", "#6C5CE7",
          "#00B894", "#FDCB6E", "#E17055", "#636E72"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} results  ({df['exp_id'].nunique()} experiments)")
    return df


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")


# ============================================================================
# MAIN-PAPER FIGURES
# ============================================================================

def fig_architecture_comparison(df: pd.DataFrame, out: Path):
    """Bar chart comparing R² across architecture variants."""
    # Select Tier 0 architecture experiments
    arch_exps = [e for e in df["exp_id"].unique() if e.startswith("T0-")]
    if not arch_exps:
        print("  skipped (no T0- experiments)")
        return
    df_plot = df[df["exp_id"].isin(arch_exps)].copy()

    # Pick best metric column available
    for metric in ["r2_median", "r2_mean"]:
        if metric in df_plot.columns:
            break
    else:
        print("  skipped (no R² column)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    order = sorted(df_plot["exp_id"].unique())
    palette = COLORS[: len(order)]
    sns.barplot(data=df_plot, x="exp_id", y=metric, order=order,
                palette=palette, capsize=0.08, errorbar="sd", ax=ax)
    ax.set_ylabel("Curve $R^2$ (median)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.axhline(0.999, ls="--", color="gray", alpha=0.5, label="0.999 target")
    ax.legend()
    ax.set_title("Architecture Comparison — Curve Reconstruction Accuracy")
    _save(fig, out / "fig_architecture_comparison.pdf")


def fig_error_distribution(df: pd.DataFrame, out: Path):
    """Histogram of per-curve MAE for the main model."""
    main = df[df["exp_id"].str.contains("T0-1-Conv-Dilated")]
    if main.empty or "mae_mean" not in main.columns:
        print("  skipped (no main-model MAE)")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(main["mae_mean"].dropna(), bins=30, color=COLORS[0], alpha=0.8,
            edgecolor="white")
    ax.set_xlabel("Mean Absolute Error (mA/cm²)")
    ax.set_ylabel("Count (across seeds)")
    ax.set_title("Error Distribution — Dilated Conv (Main Model)")
    _save(fig, out / "fig_error_distribution.pdf")


def fig_ablation_heatmap(df: pd.DataFrame, out: Path):
    """Heatmap of metrics across ablation experiments."""
    tier0 = df[df["exp_id"].str.startswith("T0-")].copy()
    if tier0.empty:
        print("  skipped (no T0 data)")
        return

    metrics = ["r2_median", "r2_mean", "mae_median", "voc_error_abs_median",
               "isc_error_abs_median"]
    avail = [m for m in metrics if m in tier0.columns]
    if not avail:
        print("  skipped (no metrics)")
        return

    agg = tier0.groupby("exp_id")[avail].mean()
    agg = agg.reindex(sorted(agg.index))

    fig, ax = plt.subplots(figsize=(max(8, len(avail) * 2), max(4, len(agg) * 0.6)))
    sns.heatmap(agg, annot=True, fmt=".4f", cmap="RdYlGn", ax=ax, linewidths=0.5)
    ax.set_title("Ablation Study — Metric Heatmap")
    ax.set_ylabel("")
    _save(fig, out / "fig_ablation_heatmap.pdf")


def fig_attention_impact(df: pd.DataFrame, out: Path):
    """Paired comparison: with vs without attention."""
    pairs = [
        ("T0-1-Conv-Dilated", "T0-5-Conv-Attn"),
        ("T0-3-TCN-Dilated", "T0-6-TCN-Attn"),
    ]
    metric = "r2_median" if "r2_median" in df.columns else "r2_mean"
    if metric not in df.columns:
        print("  skipped (no R² column)")
        return

    rows = []
    for no_attn, with_attn in pairs:
        d_no = df[df["exp_id"] == no_attn]
        d_yes = df[df["exp_id"] == with_attn]
        if d_no.empty or d_yes.empty:
            continue
        label = "Conv" if "Conv" in no_attn else "TCN"
        for _, r in d_no.iterrows():
            rows.append({"arch": label, "attention": "No", metric: r[metric]})
        for _, r in d_yes.iterrows():
            rows.append({"arch": label, "attention": "Yes", metric: r[metric]})

    if not rows:
        print("  skipped (no matching pairs)")
        return

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=plot_df, x="arch", y=metric, hue="attention",
                palette=[COLORS[0], COLORS[1]], capsize=0.08, ax=ax)
    ax.set_ylabel("Curve $R^2$ (median)")
    ax.set_xlabel("Architecture")
    ax.set_title("Impact of Self-Attention")
    ax.legend(title="Attention")
    _save(fig, out / "fig_attention_impact.pdf")


# ============================================================================
# APPENDIX FIGURES
# ============================================================================

def fig_batch_size_sweep(df: pd.DataFrame, out: Path):
    """Line plot: R² vs batch size."""
    bs_exps = df[df["exp_id"].str.contains("BS", case=False)]
    if bs_exps.empty:
        print("  skipped (no BS experiments)")
        return

    metric = "r2_median" if "r2_median" in df.columns else "r2_mean"
    if metric not in bs_exps.columns:
        return

    bs_exps = bs_exps.copy()
    bs_exps["batch_size"] = bs_exps["exp_id"].str.extract(r"BS(\d+)").astype(float)
    bs_exps = bs_exps.dropna(subset=["batch_size"])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=bs_exps, x="batch_size", y=metric, marker="o", ax=ax)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("$R^2$ (median)")
    ax.set_title("Effect of Batch Size")
    _save(fig, out / "fig_batch_size_sweep.pdf")


def fig_epoch_sweep(df: pd.DataFrame, out: Path):
    """Line plot: R² vs training epochs."""
    ep_exps = df[df["exp_id"].str.contains("EP|epoch", case=False)]
    if ep_exps.empty:
        print("  skipped (no epoch experiments)")
        return

    metric = "r2_median" if "r2_median" in df.columns else "r2_mean"
    if metric not in ep_exps.columns:
        return

    ep_exps = ep_exps.copy()
    ep_exps["epochs"] = ep_exps["exp_id"].str.extract(r"(\d+)").astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=ep_exps, x="epochs", y=metric, marker="o", ax=ax)
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("$R^2$ (median)")
    ax.set_title("Effect of Training Duration")
    _save(fig, out / "fig_epoch_sweep.pdf")


def fig_data_scaling(df: pd.DataFrame, out: Path):
    """Compare 100k-only vs 100k+300k training."""
    exps = ["T0-1-Conv-Dilated", "T0-8-100kOnly"]
    sub = df[df["exp_id"].isin(exps)]
    if sub.empty:
        print("  skipped (no data scaling experiments)")
        return

    metric = "r2_median" if "r2_median" in sub.columns else "r2_mean"
    if metric not in sub.columns:
        return

    sub = sub.copy()
    sub["Data"] = sub["exp_id"].map({
        "T0-1-Conv-Dilated": "100k + 300k",
        "T0-8-100kOnly": "100k only",
    })

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=sub, x="Data", y=metric, palette=COLORS[:2], capsize=0.1, ax=ax)
    ax.set_ylabel("$R^2$ (median)")
    ax.set_title("Effect of Training Data Size")
    _save(fig, out / "fig_data_scaling.pdf")


def fig_runtime_comparison(df: pd.DataFrame, out: Path):
    """Bar chart of training runtimes by architecture."""
    # Check if runtime columns exist
    for col in ["training_time_s", "elapsed_seconds"]:
        if col in df.columns:
            runtime_col = col
            break
    else:
        print("  skipped (no runtime column)")
        return

    tier0 = df[df["exp_id"].str.startswith("T0-")]
    if tier0.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    order = sorted(tier0["exp_id"].unique())
    sns.barplot(data=tier0, x="exp_id", y=runtime_col, order=order,
                palette=COLORS[:len(order)], ax=ax)
    ax.set_ylabel("Training Time (s)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Training Runtime by Configuration")
    _save(fig, out / "fig_runtime_comparison.pdf")


# ============================================================================
# TABLES
# ============================================================================

def table_main_results(df: pd.DataFrame, out: Path):
    """LaTeX table: main architecture comparison."""
    tier0 = df[df["exp_id"].str.startswith("T0-")]
    if tier0.empty:
        print("  skipped (no T0 data)")
        return

    metrics = ["r2_median", "mae_median", "voc_error_abs_median", "isc_error_abs_median"]
    avail = [m for m in metrics if m in tier0.columns]

    agg = tier0.groupby("exp_id")[avail].agg(["mean", "std"])

    col_labels = {
        "r2_median": "$R^2$ median",
        "mae_median": "MAE median",
        "voc_error_abs_median": "Voc err (V)",
        "isc_error_abs_median": "Isc err (mA)",
    }

    lines = [
        "\\begin{tabular}{l" + "c" * len(avail) + "}",
        "\\toprule",
        "Configuration & " + " & ".join(col_labels.get(m, m) for m in avail) + " \\\\",
        "\\midrule",
    ]

    for exp_id in sorted(agg.index):
        label = exp_id.replace("T0-", "").replace("-", " ")
        vals = []
        for m in avail:
            mean = agg.loc[exp_id, (m, "mean")]
            std = agg.loc[exp_id, (m, "std")]
            if "r2" in m:
                vals.append(f"{mean:.4f} $\\pm$ {std:.4f}")
            else:
                vals.append(f"{mean:.4f} $\\pm$ {std:.4f}")
        lines.append(f"{label} & " + " & ".join(vals) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]

    tex_path = out / "table_main_results.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  saved: {tex_path}")


def table_ablations(df: pd.DataFrame, out: Path):
    """LaTeX table: ablation study (what each component contributes)."""
    ablation_map = {
        "T0-1-Conv-Dilated": "Full Model (dilated Conv)",
        "T0-2-Conv-NoDilation": "w/o dilation (Conv)",
        "T0-3-TCN-Dilated": "TCN baseline (dilated)",
        "T0-4-Pointwise": "w/o spatial mixing (1×1)",
        "T0-5-Conv-Attn": "Full + self-attention",
        "T0-6-TCN-Attn": "TCN + self-attention",
        "T0-7-NoScalars": "w/o scalar conditioning",
        "T0-8-100kOnly": "w/o extra data (100k only)",
    }

    sub = df[df["exp_id"].isin(ablation_map.keys())]
    if sub.empty:
        print("  skipped (no ablation data)")
        return

    metrics = ["r2_median", "mae_median"]
    avail = [m for m in metrics if m in sub.columns]

    agg = sub.groupby("exp_id")[avail].agg(["mean", "std"])

    lines = [
        "\\begin{tabular}{l" + "c" * len(avail) + "}",
        "\\toprule",
        "Ablation & " + " & ".join(["$R^2$ med.", "MAE med."][:len(avail)]) + " \\\\",
        "\\midrule",
    ]

    order = [k for k in ablation_map if k in agg.index]
    for exp_id in order:
        label = ablation_map[exp_id]
        vals = []
        for m in avail:
            mean = agg.loc[exp_id, (m, "mean")]
            std = agg.loc[exp_id, (m, "std")]
            vals.append(f"{mean:.4f} $\\pm$ {std:.4f}")
        prefix = "\\textbf{" if "Full" in label else ""
        suffix = "}" if "Full" in label else ""
        lines.append(f"{prefix}{label}{suffix} & " + " & ".join(vals) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]

    tex_path = out / "table_ablations.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  saved: {tex_path}")


# ============================================================================
# MAIN
# ============================================================================

def generate_all(results_path: str, output_base: str,
                 analysis_dir: Optional[str] = None):
    df = load_results(results_path)

    main_dir = Path(output_base) / "main_paper"
    app_dir = Path(output_base) / "appendix"
    tab_dir = Path(output_base) / "tables"
    for d in (main_dir, app_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("\n--- Main Paper Figures ---")
    fig_architecture_comparison(df, main_dir)
    fig_error_distribution(df, main_dir)
    fig_ablation_heatmap(df, main_dir)

    print("\n--- Appendix Figures ---")
    fig_batch_size_sweep(df, app_dir)
    fig_epoch_sweep(df, app_dir)
    fig_data_scaling(df, app_dir)
    fig_attention_impact(df, app_dir)
    fig_runtime_comparison(df, app_dir)

    print("\n--- Tables ---")
    table_main_results(df, tab_dir)
    table_ablations(df, tab_dir)

    print(f"\nDone. Figures in {output_base}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to all_results.csv")
    p.add_argument("--output", default="figures", help="Output directory")
    p.add_argument("--analysis-dir", default=None, help="Analysis directory (optional)")
    args = p.parse_args()
    generate_all(args.results, args.output, args.analysis_dir)


if __name__ == "__main__":
    main()
