#!/usr/bin/env python3
"""
Inference CLI for the dilated-conv Stage-2 curve model.

Inputs:
  - 31-parameter table (txt/csv)
  - scalar txt files: Voc and Vmpp
  - trained checkpoint (.ckpt)
  - preprocessing cache dir from training (transformers + npz metadata)

Outputs:
  - 8-point normalized curve predictions
  - optional 45-point normalized PCHIP reconstructions
  - optional absolute-current outputs when Jsc is provided
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from scipy.interpolate import PchipInterpolator
    from config import V_GRID
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # allows --help even if env is incomplete
    joblib = None
    np = None
    pd = None
    torch = None
    PchipInterpolator = None
    V_GRID = None
    _IMPORT_ERROR = exc


COLNAMES = [
    "lH", "lP", "lE",
    "muHh", "muPh", "muPe", "muEe",
    "NvH", "NcH", "NvE", "NcE", "NvP", "NcP",
    "chiHh", "chiHe", "chiPh", "chiPe", "chiEh", "chiEe",
    "Wlm", "Whm",
    "epsH", "epsP", "epsE",
    "Gavg", "Aug", "Brad", "Taue", "Tauh", "vII", "vIII",
]


def load_params(params_path: Path) -> pd.DataFrame:
    """Load 31-parameter table in either named-column or raw format."""
    try:
        df_named = pd.read_csv(params_path)
        if set(COLNAMES).issubset(df_named.columns):
            return df_named[COLNAMES].astype(np.float32)
    except Exception:
        pass

    df_raw = pd.read_csv(params_path, header=None)
    if df_raw.shape[1] != len(COLNAMES):
        raise ValueError(
            f"{params_path} has {df_raw.shape[1]} columns; expected {len(COLNAMES)}."
        )
    df_raw.columns = COLNAMES
    return df_raw.astype(np.float32)


def load_scalar_txt(path: Path) -> tuple[str, np.ndarray]:
    """Load scalar txt with optional one-line header."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()

    skiprows = 0
    name = path.stem
    if first:
        token = first.split(",")[0].strip()
        try:
            float(token)
        except ValueError:
            skiprows = 1
            if token:
                name = token

    values = np.loadtxt(path, delimiter=",", skiprows=skiprows, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return name, values


def build_query_voltages(voc: np.ndarray, vmpp: np.ndarray, n_pre: int, n_post: int) -> np.ndarray:
    """Construct 8-point voltage query grid from Voc/Vmpp (sample-wise)."""
    seq_len = n_pre + 1 + n_post
    all_v = np.zeros((len(voc), seq_len), dtype=np.float32)

    for i in range(len(voc)):
        voc_i = float(max(voc[i], 1e-6))
        vmpp_i = float(np.clip(vmpp[i], 0.0, voc_i))

        v_pre = np.linspace(0.0, vmpp_i, n_pre + 2, endpoint=True)[:-1]
        v_post = np.linspace(vmpp_i, voc_i, n_post + 2, endpoint=True)[1:]
        v_mpp_grid = np.unique(np.concatenate([v_pre, v_post]))

        if len(v_mpp_grid) < seq_len:
            all_v[i] = np.linspace(0.0, voc_i, seq_len, dtype=np.float32)
        else:
            all_v[i] = np.interp(
                np.linspace(0.0, 1.0, seq_len),
                np.linspace(0.0, 1.0, len(v_mpp_grid)),
                v_mpp_grid,
            ).astype(np.float32)

    return all_v


def build_scalar_dataframe(
    scalar_names: list[str], voc: np.ndarray, vmpp: np.ndarray
) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = {}
    for name in scalar_names:
        lname = str(name).lower()
        if "voc" in lname:
            cols[str(name)] = voc
        elif "vmpp" in lname:
            cols[str(name)] = vmpp
        else:
            raise ValueError(
                f"Unsupported scalar '{name}'. This CLI currently supports Voc/Vmpp only."
            )
    return pd.DataFrame(cols)


def reconstruct_45pt(v_query: np.ndarray, j_query_norm: np.ndarray) -> np.ndarray:
    """PCHIP 8-point normalized predictions back to the fixed 45-point grid."""
    full = np.empty((len(j_query_norm), len(V_GRID)), dtype=np.float32)
    for i in range(len(j_query_norm)):
        pi = PchipInterpolator(v_query[i], j_query_norm[i], extrapolate=False)
        y = pi(V_GRID).astype(np.float32)
        y[~np.isfinite(y)] = -1.0
        full[i] = y
    return full


def run_inference(model, x_combined: np.ndarray, v_query: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        x_t = torch.from_numpy(x_combined).to(device)
        v_t = torch.from_numpy(v_query).to(device)
        pred = model(x_t, v_t).detach().cpu().numpy().astype(np.float32)
    return pred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference for the Stage-2 dilated-conv model (31 params + Voc/Vmpp)."
    )
    p.add_argument("--params", required=True, help="31-parameter input txt/csv")
    p.add_argument("--voc", required=True, help="Voc scalar txt file")
    p.add_argument("--vmpp", required=True, help="Vmpp scalar txt file")
    p.add_argument("--checkpoint", required=True, help="Path to best-model.ckpt")
    p.add_argument(
        "--cache-dir",
        required=True,
        help="Training cache dir containing atcn_* transformers/npz",
    )
    p.add_argument(
        "--output-dir",
        default="./inference_outputs",
        help="Output directory for predictions",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    p.add_argument(
        "--jsc",
        default=None,
        help="Optional Jsc txt file for denormalized absolute-current outputs",
    )
    p.add_argument(
        "--skip-45pt",
        action="store_true",
        help="Skip 45-point PCHIP reconstruction outputs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            f"Missing dependency: {_IMPORT_ERROR}. "
            "Install runtime deps first (e.g., pip install -r requirements.txt "
            "pytorch-lightning scipy seaborn pillow tqdm)."
        )

    params_path = Path(args.params).resolve()
    voc_path = Path(args.voc).resolve()
    vmpp_path = Path(args.vmpp).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    params_df = load_params(params_path)
    _, voc_vals = load_scalar_txt(voc_path)
    _, vmpp_vals = load_scalar_txt(vmpp_path)

    n = len(params_df)
    if len(voc_vals) != n or len(vmpp_vals) != n:
        raise ValueError(
            f"Length mismatch: params={n}, voc={len(voc_vals)}, vmpp={len(vmpp_vals)}"
        )

    jsc_vals = None
    if args.jsc:
        _, jsc_vals_raw = load_scalar_txt(Path(args.jsc).resolve())
        if len(jsc_vals_raw) != n:
            raise ValueError(f"Jsc length mismatch: expected {n}, got {len(jsc_vals_raw)}")
        jsc_vals = jsc_vals_raw.astype(np.float32)

    param_tf_path = cache_dir / "atcn_param_transformer.joblib"
    scalar_tf_path = cache_dir / "atcn_scalar_transformer.joblib"
    physics_tf_path = cache_dir / "atcn_physics_transformer.joblib"
    pre_npz_path = cache_dir / "atcn_preprocessed.npz"

    if not param_tf_path.exists():
        raise FileNotFoundError(f"Missing transformer: {param_tf_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    param_tf = joblib.load(param_tf_path)
    x_param = param_tf.transform(params_df).astype(np.float32)

    scalar_names: list[str] = []
    physics_feature_names: list[str] = []
    if pre_npz_path.exists():
        cached = np.load(pre_npz_path, allow_pickle=True)
        scalar_names = [str(s) for s in cached.get("scalar_names", np.array([]))]
        physics_feature_names = [
            str(s) for s in cached.get("physics_feature_names", np.array([]))
        ]

    x_parts = [x_param]

    if scalar_tf_path.exists():
        scalar_tf = joblib.load(scalar_tf_path)
        if not scalar_names:
            scalar_names = ["Voc", "Vmpp"]
        scalar_df = build_scalar_dataframe(scalar_names, voc_vals, vmpp_vals)
        x_scalar = scalar_tf.transform(scalar_df).astype(np.float32)
        x_parts.append(x_scalar)

    if physics_tf_path.exists():
        from features import compute_all_physics_features, get_feature_names

        physics_tf = joblib.load(physics_tf_path)
        raw_params = torch.from_numpy(params_df.values.astype(np.float32))
        with torch.no_grad():
            physics_all = compute_all_physics_features(raw_params).cpu().numpy()
        physics_df = pd.DataFrame(physics_all, columns=get_feature_names())

        if not physics_feature_names:
            physics_feature_names = list(physics_df.columns)
        physics_df = physics_df[physics_feature_names]
        x_physics = physics_tf.transform(physics_df).astype(np.float32)
        x_parts.append(x_physics)

    x_combined = np.concatenate(x_parts, axis=1).astype(np.float32)

    from train_attention_tcn import PhysicsIVSystem

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = PhysicsIVSystem.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = model.to(device).eval()

    expected_dim = int(model.hparams.model["param_dim"])
    if x_combined.shape[1] != expected_dim:
        raise ValueError(
            f"Input feature dimension mismatch: got {x_combined.shape[1]}, expected {expected_dim}. "
            "Check that cache-dir matches the checkpoint and includes the correct transformers."
        )

    n_pre = int(model.hparams.dataset["pchip"]["n_pre_mpp"])
    n_post = int(model.hparams.dataset["pchip"]["n_post_mpp"])
    v_query = build_query_voltages(voc_vals, vmpp_vals, n_pre=n_pre, n_post=n_post)

    j8_norm = run_inference(model, x_combined, v_query, device=device)

    cols_v = [f"Vq_{i}" for i in range(v_query.shape[1])]
    cols_j8 = [f"Jnorm8_{i}" for i in range(j8_norm.shape[1])]
    out_8 = pd.DataFrame(
        np.concatenate([v_query, j8_norm], axis=1),
        columns=cols_v + cols_j8,
    )
    out_8.to_csv(out_dir / "predictions_8pt_normalized.csv", index=False)

    out_files: dict[str, str] = {
        "predictions_8pt_normalized": str(out_dir / "predictions_8pt_normalized.csv"),
    }

    if jsc_vals is not None:
        j8_abs = (j8_norm + 1.0) * jsc_vals[:, None] / 2.0
        cols_j8_abs = [f"Jabs8_{i}" for i in range(j8_abs.shape[1])]
        out_8_abs = pd.DataFrame(
            np.concatenate([v_query, j8_abs], axis=1),
            columns=cols_v + cols_j8_abs,
        )
        out_8_abs.to_csv(out_dir / "predictions_8pt_absolute.csv", index=False)
        out_files["predictions_8pt_absolute"] = str(out_dir / "predictions_8pt_absolute.csv")

    if not args.skip_45pt:
        j45_norm = reconstruct_45pt(v_query, j8_norm)
        cols_j45 = [f"Jnorm45_{i}" for i in range(j45_norm.shape[1])]
        pd.DataFrame(j45_norm, columns=cols_j45).to_csv(
            out_dir / "predictions_45pt_normalized.csv", index=False
        )
        pd.DataFrame([V_GRID], columns=[f"V45_{i}" for i in range(len(V_GRID))]).to_csv(
            out_dir / "voltage_grid_45pt.csv", index=False
        )
        out_files["predictions_45pt_normalized"] = str(out_dir / "predictions_45pt_normalized.csv")
        out_files["voltage_grid_45pt"] = str(out_dir / "voltage_grid_45pt.csv")

        if jsc_vals is not None:
            j45_abs = (j45_norm + 1.0) * jsc_vals[:, None] / 2.0
            cols_j45_abs = [f"Jabs45_{i}" for i in range(j45_abs.shape[1])]
            pd.DataFrame(j45_abs, columns=cols_j45_abs).to_csv(
                out_dir / "predictions_45pt_absolute.csv", index=False
            )
            out_files["predictions_45pt_absolute"] = str(
                out_dir / "predictions_45pt_absolute.csv"
            )

    metadata = {
        "n_samples": n,
        "checkpoint": str(ckpt_path),
        "cache_dir": str(cache_dir),
        "device": str(device),
        "input_dim": int(x_combined.shape[1]),
        "expected_input_dim": expected_dim,
        "scalar_features_used": scalar_names,
        "physics_features_used": len(physics_feature_names),
        "outputs": out_files,
    }
    with open(out_dir / "inference_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Inference complete for {n} samples.")
    for k, v in out_files.items():
        print(f"  {k}: {v}")
    print(f"  metadata: {out_dir / 'inference_metadata.json'}")


if __name__ == "__main__":
    main()
