#!/usr/bin/env python3
"""Analyze curve_error_analysis.csv for bad-curve cohorts and patterns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_NUM_COLS = [
    "curve_mse",
    "curve_r2",
    "mse_region1",
    "mse_region2",
    "ff_abs_err",
    "ff_true",
    "ff_pred",
    "jsc_true",
    "voc_true",
    "vmpp_true",
    "jmpp_true",
    "jsc_pred",
    "voc_pred",
    "vmpp_pred",
    "jmpp_pred",
    "jsc_ratio",
    "voc_ratio",
    "pce_true",
    "pmpp_true",
]


def _safe_quantile(series: pd.Series, q: float) -> float | None:
    if series is None or series.empty:
        return None
    return float(series.quantile(q))


def _group_summary(df: pd.DataFrame) -> dict:
    stats = {}
    for col in df.columns:
        if col in DEFAULT_NUM_COLS and col in df:
            s = df[col].dropna()
            if s.empty:
                continue
            stats[col] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p10": float(s.quantile(0.10)),
                "p90": float(s.quantile(0.90)),
            }
    return stats


def _corr_table(df: pd.DataFrame, target: str) -> list[dict]:
    if target not in df:
        return []
    rows = []
    for col in DEFAULT_NUM_COLS:
        if col == target or col not in df:
            continue
        s = df[[target, col]].dropna()
        if len(s) < 10:
            continue
        corr = float(s[target].corr(s[col]))
        rows.append({"feature": col, "corr": corr, "n": int(len(s))})
    rows.sort(key=lambda r: abs(r["corr"]), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to curve_error_analysis.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    parser.add_argument("--high-error-quantile", type=float, default=0.90)
    parser.add_argument("--low-pce-quantile", type=float, default=0.20)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if "curve_mse" not in df:
        raise ValueError("curve_mse column is required in curve_error_analysis.csv")

    high_err_thr = _safe_quantile(df["curve_mse"], args.high_error_quantile)
    low_pce_thr = _safe_quantile(df["pce_true"], args.low_pce_quantile) if "pce_true" in df else None

    df["high_error"] = df["curve_mse"] >= (high_err_thr if high_err_thr is not None else df["curve_mse"].max())
    if low_pce_thr is not None:
        df["low_pce"] = df["pce_true"] <= low_pce_thr
    else:
        df["low_pce"] = False

    df["bad_curve"] = df["high_error"] | df["low_pce"]

    summary = {
        "n_samples": int(len(df)),
        "high_error_quantile": args.high_error_quantile,
        "low_pce_quantile": args.low_pce_quantile,
        "high_error_threshold": float(high_err_thr) if high_err_thr is not None else None,
        "low_pce_threshold": float(low_pce_thr) if low_pce_thr is not None else None,
        "counts": {
            "high_error": int(df["high_error"].sum()),
            "low_pce": int(df["low_pce"].sum()),
            "bad_curve": int(df["bad_curve"].sum()),
        },
        "overall_stats": _group_summary(df),
        "bad_curve_stats": _group_summary(df[df["bad_curve"]]),
        "good_curve_stats": _group_summary(df[~df["bad_curve"]]),
        "corr_with_curve_mse": _corr_table(df, "curve_mse"),
        "corr_with_pce": _corr_table(df, "pce_true") if "pce_true" in df else [],
    }

    summary_path = output_dir / "curve_error_analysis_report.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save cohort CSVs for manual inspection
    df[df["high_error"]].to_csv(output_dir / "cohort_high_error.csv", index=False)
    df[df["low_pce"]].to_csv(output_dir / "cohort_low_pce.csv", index=False)
    df[df["bad_curve"]].to_csv(output_dir / "cohort_bad_curve.csv", index=False)
    df[~df["bad_curve"]].to_csv(output_dir / "cohort_good_curve.csv", index=False)

    # Minimal markdown report
    md_lines = [
        "# Curve Error Analysis Report",
        f"- Samples: {summary['n_samples']}",
        f"- High-error threshold (q={args.high_error_quantile}): {summary['high_error_threshold']}",
        f"- Low-PCE threshold (q={args.low_pce_quantile}): {summary['low_pce_threshold']}",
        f"- High-error count: {summary['counts']['high_error']}",
        f"- Low-PCE count: {summary['counts']['low_pce']}",
        f"- Bad-curve count: {summary['counts']['bad_curve']}",
        "",
        "## Top correlations with curve_mse",
    ]
    for row in summary["corr_with_curve_mse"][:10]:
        md_lines.append(f"- {row['feature']}: corr={row['corr']:.3f} (n={row['n']})")
    if summary["corr_with_pce"]:
        md_lines.append("")
        md_lines.append("## Top correlations with pce_true")
        for row in summary["corr_with_pce"][:10]:
            md_lines.append(f"- {row['feature']}: corr={row['corr']:.3f} (n={row['n']})")

    report_path = output_dir / "curve_error_analysis_report.md"
    report_path.write_text("\n".join(md_lines))

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote cohorts to: {output_dir}")


if __name__ == "__main__":
    main()
