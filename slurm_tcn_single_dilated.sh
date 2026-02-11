#!/bin/bash
#SBATCH --job-name=conv_dilated_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/tcn_single_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/tcn_single_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# SINGLE DILATED-CONV RUN  —  Minimal diagnostics pipeline
# ============================================================================
# Runs ONE seed of the dilated conv architecture and produces:
#   - Training curves (loss components per epoch)
#   - Parity plots (predicted vs true)
#   - Residual analysis (per-voltage-point error distribution)
#   - Correlation matrix (parameter → metric)
#   - Timings (wall-clock seconds in JSON + timing.log)
#
# Usage:
#   sbatch slurm_tcn_single_dilated.sh
#   sbatch slurm_tcn_single_dilated.sh --skip-preprocessing
#   sbatch slurm_tcn_single_dilated.sh --figures-only
#   sbatch slurm_tcn_single_dilated.sh --dry-run
# ============================================================================

set -e

echo "=============================================="
echo "SINGLE DILATED-CONV DIAGNOSTICS"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "=============================================="

# ── CLI flags ────────────────────────────────────────────────────────────────
FIGURES_ONLY=false
SKIP_PREPROCESSING=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --figures-only) FIGURES_ONLY=true ;;
        --skip-preprocessing) SKIP_PREPROCESSING=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# ── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors"
PREPROCESS_DIR="$WORK_DIR/preprocessed"
OUTPUT_BASE="$WORK_DIR/outputs/tcn_single_$(date +%Y%m%d)"
ATCN_DATA_DIR="$OUTPUT_BASE/data_cache"
RESULTS_DIR="$OUTPUT_BASE/results"
DIAGNOSTICS_DIR="$OUTPUT_BASE/diagnostics"
LOGS_DIR="$WORK_DIR/logs"
SCALAR_DIR="${SCALAR_DIR:-$WORK_DIR/scalars_external}"

# Raw data
PARAMS_PRIMARY="$WORK_DIR/LHS_parameters_m.txt"
IV_PRIMARY="$WORK_DIR/IV_m.txt"
PARAMS_EXTRA="$WORK_DIR/LHS_parameters_m_300k.txt"
IV_EXTRA="$WORK_DIR/IV_m_300k.txt"

SEED=42
MIN_FF=0.30
MIN_VMPP=0.00
MAX_EPOCHS=100
BATCH_SIZE=128

# Physics features & Jacobian regularization
USE_PHYSICS_FEATURES=true
USE_PHYSICS_FEATURE_SELECTION=true
PHYSICS_MAX_FEATURES=5
JACOBIAN_WEIGHT=0.0

# ── Environment setup ────────────────────────────────────────────────────────
cd $WORK_DIR
mkdir -p $LOGS_DIR $OUTPUT_BASE $ATCN_DATA_DIR $RESULTS_DIR $DIAGNOSTICS_DIR

module purge
module load gcc/12.3 cuda/12.2 python/3.11
source ../venv/bin/activate
pip install --quiet pytorch_lightning rich seaborn scipy pillow tqdm 2>/dev/null || true

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

TIMING_LOG="$OUTPUT_BASE/timing.log"
echo "Pipeline started: $(date)" > $TIMING_LOG

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================

if [ "$FIGURES_ONLY" = false ] && [ "$SKIP_PREPROCESSING" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 1: Data Preprocessing (100k + 300k)"
    echo "=============================================="
    STEP_START=$(date +%s)

    mkdir -p $PREPROCESS_DIR

    python scripts/preprocess_data.py \
        --params "$PARAMS_PRIMARY" --iv "$IV_PRIMARY" \
        --output-dir "$PREPROCESS_DIR" \
        --min-ff $MIN_FF --min-vmpp $MIN_VMPP --suffix "_clean"

    python scripts/preprocess_data.py \
        --params "$PARAMS_EXTRA" --iv "$IV_EXTRA" \
        --output-dir "$PREPROCESS_DIR" \
        --min-ff $MIN_FF --min-vmpp $MIN_VMPP --suffix "_clean"

    STEP_END=$(date +%s)
    echo "Preprocessing: $((STEP_END - STEP_START))s" >> $TIMING_LOG
fi

# Preprocessed paths
PARAMS_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_clean.txt"
IV_CLEAN="$PREPROCESS_DIR/IV_m_clean.txt"
PARAMS_EXTRA_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_300k_clean.txt"
IV_EXTRA_CLEAN="$PREPROCESS_DIR/IV_m_300k_clean.txt"
VOC_100K="$SCALAR_DIR/voc_clean_100k.txt"
VMPP_100K="$SCALAR_DIR/vmpp_clean_100k.txt"
VOC_300K="$SCALAR_DIR/voc_clean_300k.txt"
VMPP_300K="$SCALAR_DIR/vmpp_clean_300k.txt"

# Verify files
echo ""
echo "Input files:"
for f in "$PARAMS_CLEAN" "$IV_CLEAN" "$PARAMS_EXTRA_CLEAN" "$IV_EXTRA_CLEAN" \
         "$VOC_100K" "$VMPP_100K" "$VOC_300K" "$VMPP_300K"; do
    if [ -f "$f" ]; then
        echo "  [OK]      $(basename $f) ($(wc -l < "$f") lines)"
    else
        echo "  [MISSING] $f"
        if [ "$DRY_RUN" = false ] && [ "$FIGURES_ONLY" = false ]; then
            echo "ERROR: Required input file missing. Run preprocessing first."
            exit 1
        fi
    fi
done

# ============================================================================
# STEP 2: TRAIN DILATED CONV (single seed)
# ============================================================================

EXP_ID="DilatedConv"
EXP_OUT="$OUTPUT_BASE/$EXP_ID/seed_$SEED"
EXP_DATA="$ATCN_DATA_DIR/${EXP_ID}_seed${SEED}"
RUN_NAME="${EXP_ID}_seed${SEED}"

if [ "$FIGURES_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 2: Training Dilated Conv (seed=$SEED)"
    echo "=============================================="
    STEP_START=$(date +%s)

    mkdir -p "$EXP_OUT" "$EXP_DATA"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] train_dilated_conv.py --architecture conv --no-attention --use-dilated"
        echo "            physics_features=$USE_PHYSICS_FEATURES  jacobian_weight=$JACOBIAN_WEIGHT"
    else
        python train_dilated_conv.py \
            --params "$PARAMS_CLEAN" \
            --iv "$IV_CLEAN" \
            --params-extra "$PARAMS_EXTRA_CLEAN" \
            --iv-extra "$IV_EXTRA_CLEAN" \
            --scalar-files "$VOC_100K" "$VMPP_100K" \
            --scalar-files-extra "$VOC_300K" "$VMPP_300K" \
            --output-dir "$EXP_OUT" \
            --data-dir "$EXP_DATA" \
            --run-name "$RUN_NAME" \
            --seed $SEED \
            --max-epochs $MAX_EPOCHS \
            --batch-size $BATCH_SIZE \
            --num-workers $((SLURM_CPUS_PER_TASK / 2)) \
            --enable-example-plots \
            --architecture conv --no-attention --use-dilated \
            ${USE_PHYSICS_FEATURES:+--use-physics-features} \
            ${USE_PHYSICS_FEATURE_SELECTION:+--physics-feature-selection --physics-max-features $PHYSICS_MAX_FEATURES} \
            --jacobian-weight $JACOBIAN_WEIGHT \
            --force-preprocess \
            2>&1 | tee "$EXP_OUT/train.log"

        cp -f "$EXP_OUT/$RUN_NAME/test_stats.json" \
              "$RESULTS_DIR/${RUN_NAME}_stats.json" 2>/dev/null || true
    fi

    STEP_END=$(date +%s)
    TRAIN_SECS=$((STEP_END - STEP_START))
    echo "Training: ${TRAIN_SECS}s" >> $TIMING_LOG
fi

# ============================================================================
# STEP 3: DIAGNOSTICS  (training curves, parity, residuals, correlations)
# ============================================================================

echo ""
echo "=============================================="
echo "STEP 3: Generating Diagnostics"
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    STEP_START=$(date +%s)

    python - "$EXP_OUT" "$RUN_NAME" "$DIAGNOSTICS_DIR" <<'PYEOF'
import sys, os, json, glob
import numpy as np

out_dir   = sys.argv[1]           # e.g. .../DilatedConv/seed_42
run_name  = sys.argv[2]           # DilatedConv_seed42
diag_dir  = sys.argv[3]           # .../diagnostics
os.makedirs(diag_dir, exist_ok=True)

# ── locate outputs ──────────────────────────────────────────────────────
run_dir   = os.path.join(out_dir, run_name)
stats_path = os.path.join(run_dir, "test_stats.json")

# Try to find tensorboard events (for training curves)
tb_root = os.path.join(out_dir, "tb_logs", run_name)
if not os.path.isdir(tb_root):
    tb_root = os.path.join(out_dir, "tb_logs")

# ── 0.  Print test stats summary ────────────────────────────────────────
if os.path.isfile(stats_path):
    with open(stats_path) as f:
        stats = json.load(f)
    print("\n=== Test Statistics ===")
    for k in sorted(stats):
        print(f"  {k:30s}  {stats[k]}")
    # copy to diagnostics for easy access
    import shutil
    shutil.copy2(stats_path, os.path.join(diag_dir, "test_stats.json"))
else:
    print(f"WARNING: {stats_path} not found")
    stats = {}

# ── lazy matplotlib import ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfbfb",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── 1.  Training curves from TensorBoard ────────────────────────────────
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_files = glob.glob(os.path.join(tb_root, "**", "events.out.tfevents.*"), recursive=True)
    if event_files:
        ea = EventAccumulator(os.path.dirname(event_files[0]))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        print(f"\nTensorBoard tags found: {tags}")

        # Collect all scalar series
        curves = {}
        for tag in tags:
            events = ea.Scalars(tag)
            steps  = [e.step for e in events]
            vals   = [e.value for e in events]
            curves[tag] = (steps, vals)

        # Save raw training curves as CSV
        import csv
        csv_path = os.path.join(diag_dir, "training_curves.csv")
        all_steps = set()
        for steps, _ in curves.values():
            all_steps.update(steps)
        all_steps = sorted(all_steps)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step"] + list(curves.keys())
            writer.writerow(header)
            for s in all_steps:
                row = [s]
                for tag in curves:
                    steps, vals = curves[tag]
                    idx = None
                    for i, st in enumerate(steps):
                        if st == s:
                            idx = i
                            break
                    row.append(vals[idx] if idx is not None else "")
                writer.writerow(row)
        print(f"  Saved training_curves.csv ({len(all_steps)} steps, {len(curves)} metrics)")

        # Plot training curves — group train and val losses
        train_tags = [t for t in tags if "train" in t.lower()]
        val_tags   = [t for t in tags if "val" in t.lower()]
        loss_tags  = train_tags + val_tags if train_tags or val_tags else list(curves.keys())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: main loss (train + val)
        ax = axes[0]
        for tag in sorted(loss_tags):
            if "loss" in tag.lower() or "mse" in tag.lower():
                steps, vals = curves[tag]
                ax.plot(steps, vals, label=tag, linewidth=1.5)
        ax.set_xlabel("Step / Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Right: physics components (mono, convex, excurv)
        ax = axes[1]
        phys_tags = [t for t in tags if any(k in t.lower() for k in ("mono", "convex", "excurv", "curv"))]
        if phys_tags:
            for tag in sorted(phys_tags):
                steps, vals = curves[tag]
                ax.plot(steps, vals, label=tag, linewidth=1.5)
            ax.set_xlabel("Step / Epoch")
            ax.set_ylabel("Penalty")
            ax.set_title("Physics Loss Components")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No physics tags found", transform=ax.transAxes,
                    ha="center", va="center")

        plt.tight_layout()
        fig.savefig(os.path.join(diag_dir, "training_curves.pdf"), dpi=150)
        fig.savefig(os.path.join(diag_dir, "training_curves.png"), dpi=150)
        plt.close(fig)
        print("  Saved training_curves.pdf/.png")
    else:
        print("  No TensorBoard event files found — skipping training curves plot")
except ImportError:
    print("  tensorboard not installed — skipping training curves from events")
except Exception as e:
    print(f"  Training curves error: {e}")

# ── 2.  Parity plots  (predicted vs true)  ──────────────────────────────
# Look for test predictions saved by the training script
pred_files = glob.glob(os.path.join(run_dir, "**", "*.npz"), recursive=True) + \
             glob.glob(os.path.join(out_dir, "**", "*.npz"), recursive=True)

# Also check for cached preprocessed data that has test indices
data_cache_npz = glob.glob(os.path.join(out_dir, "..", "..", "data_cache", "**", "*.npz"), recursive=True)
pred_files.extend(data_cache_npz)

pred_files = sorted(set(pred_files), key=os.path.getmtime, reverse=True)
pred_files = sorted(
    pred_files,
    key=lambda p: (
        "pred" not in os.path.basename(p).lower(),
        "test" not in os.path.basename(p).lower(),
    ),
)
print(f"\nPrediction npz candidates found: {len(pred_files)}")

def _extract_pred_arrays(npz_data):
    for pred_key, true_key in [
        ("y_pred", "y_true"),
        ("test_pred", "test_true"),
        ("pred", "true"),
        ("preds", "trues"),
        ("predictions", "targets"),
    ]:
        if pred_key in npz_data and true_key in npz_data:
            return npz_data[pred_key], npz_data[true_key], pred_key, true_key
    return None, None, None, None

# The trainer stores test predictions in the checkpoint callback or we can
# reconstruct from the test loop.  Check if test_stats has the arrays:
parity_done = False
try:
    # Try loading from any npz that contains predictions
    for npz_path in pred_files:
        with np.load(npz_path, allow_pickle=True) as data:
            y_pred, y_true, pred_key, true_key = _extract_pred_arrays(data)
        if y_pred is not None and y_true is not None:
            print(f"  Using predictions from: {npz_path} ({pred_key}, {true_key})")
            break
    else:
        # Check if trainer saved per-sample R² values we can use
        raise FileNotFoundError("No prediction arrays found in npz files")

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_pred.ndim != 2 or y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y_pred={y_pred.shape}, y_true={y_true.shape}")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {y_pred.shape} vs {y_true.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 2a. Curve-level R² parity
    r2_per_sample = []
    for i in range(len(y_true)):
        ss_res = np.sum((y_true[i] - y_pred[i])**2)
        ss_tot = np.sum((y_true[i] - np.mean(y_true[i]))**2)
        r2_per_sample.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
    r2_per_sample = np.array(r2_per_sample)

    # 2b. Flatten for point-level parity
    yt_flat = y_true.flatten()
    yp_flat = y_pred.flatten()
    ss_res = np.sum((yt_flat - yp_flat) ** 2)
    ss_tot = np.sum((yt_flat - np.mean(yt_flat)) ** 2)
    global_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    ax = axes[0]
    hb = ax.hexbin(yt_flat, yp_flat, gridsize=90, cmap="viridis", mincnt=1, bins="log")
    lims = [min(yt_flat.min(), yp_flat.min()), max(yt_flat.max(), yp_flat.max())]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Ideal")
    ax.set_xlabel("True current (norm)")
    ax.set_ylabel("Predicted current (norm)")
    ax.set_title(f"Point-level Parity (R²={global_r2:.4f})")
    ax.set_aspect("equal")
    fig.colorbar(hb, ax=ax, label="log10(count)")

    # 2c. R² histogram
    ax = axes[1]
    ax.hist(r2_per_sample, bins=60, edgecolor="black", alpha=0.85, color="#4C72B0")
    ax.axvline(np.median(r2_per_sample), color="red", linestyle="--",
               label=f"median = {np.median(r2_per_sample):.4f}")
    ax.axvline(np.mean(r2_per_sample), color="#1b9e77", linestyle="-.",
               label=f"mean = {np.mean(r2_per_sample):.4f}")
    ax.set_xlabel("Per-curve R²")
    ax.set_ylabel("Count")
    ax.set_title("R² Distribution")
    ax.legend()

    # 2d. ISc parity (first current value ~ short-circuit)
    ax = axes[2]
    isc_true = y_true[:, 0]
    isc_pred = y_pred[:, 0]
    ax.scatter(isc_true, isc_pred, s=5, alpha=0.25, color="#dd8452", edgecolors="none")
    lims = [min(isc_true.min(), isc_pred.min()), max(isc_true.max(), isc_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.2)
    ax.set_xlabel("True ISc (norm)")
    ax.set_ylabel("Predicted ISc (norm)")
    ax.set_title("Short-circuit Current Parity")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(os.path.join(diag_dir, "parity_plots.pdf"), dpi=150)
    fig.savefig(os.path.join(diag_dir, "parity_plots.png"), dpi=150)
    plt.close(fig)
    print("  Saved parity_plots.pdf/.png")
    parity_done = True

except Exception as e:
    print(f"  Parity plots: {e}")
    print("  (Will be generated if test predictions are saved as npz)")

# ── 3.  Residual analysis ───────────────────────────────────────────────
if parity_done:
    try:
        residuals = y_true - y_pred  # shape: (N, seq_len)
        n_samples, seq_len = residuals.shape

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 3a. Mean absolute error per voltage point
        ax = axes[0]
        mae_per_point = np.mean(np.abs(residuals), axis=0)
        ax.plot(range(seq_len), mae_per_point, linewidth=1.5)
        ax.set_xlabel("Voltage index")
        ax.set_ylabel("MAE")
        ax.set_title("MAE per Voltage Point")
        ax.grid(True, alpha=0.3)

        # 3b. Residual distribution (histogram)
        ax = axes[1]
        ax.hist(residuals.flatten(), bins=100, edgecolor="black", alpha=0.8, density=True)
        ax.set_xlabel("Residual (true − pred)")
        ax.set_ylabel("Density")
        ax.set_title(f"Residual Distribution (std={np.std(residuals):.4f})")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)

        # 3c. Residual heatmap (samples × voltage)
        ax = axes[2]
        # Sort by R² for visualization
        sort_idx = np.argsort(r2_per_sample)
        subset = sort_idx[::max(1, len(sort_idx)//200)]  # ~200 rows
        heat = residuals[subset]
        vmax = np.percentile(np.abs(heat), 98)
        vmax = max(vmax, 1e-6)
        im = ax.imshow(residuals[subset], aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("Voltage index")
        ax.set_ylabel("Sample (sorted by R²)")
        ax.set_title("Residual Heatmap")
        plt.colorbar(im, ax=ax, label="Residual")

        plt.tight_layout()
        fig.savefig(os.path.join(diag_dir, "residual_analysis.pdf"), dpi=150)
        fig.savefig(os.path.join(diag_dir, "residual_analysis.png"), dpi=150)
        plt.close(fig)
        print("  Saved residual_analysis.pdf/.png")

        # Save residual stats as CSV
        import csv
        csv_path = os.path.join(diag_dir, "residual_per_voltage.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["voltage_idx", "mae", "rmse", "bias", "std"])
            for i in range(seq_len):
                col = residuals[:, i]
                writer.writerow([i,
                                 np.mean(np.abs(col)),
                                 np.sqrt(np.mean(col**2)),
                                 np.mean(col),
                                 np.std(col)])
        print(f"  Saved residual_per_voltage.csv ({seq_len} points)")

    except Exception as e:
        print(f"  Residual analysis error: {e}")

# ── 4.  Correlation: device parameters → metrics ────────────────────────
try:
    # Load the preprocessed data to get parameters
    data_cache_parent = os.path.join(out_dir, "..", "..", "data_cache")
    npz_candidates = glob.glob(os.path.join(data_cache_parent, "**", "atcn_preprocessed.npz"), recursive=True)
    if not npz_candidates:
        npz_candidates = glob.glob(os.path.join(out_dir, "**", "atcn_preprocessed.npz"), recursive=True)

    if npz_candidates and parity_done:
        cache = np.load(npz_candidates[0], allow_pickle=True)
        cache_keys = list(cache.keys())
        print(f"\n  Preprocessed data keys: {cache_keys}")

        # Try to get test parameters
        params = None
        if "params_test" in cache_keys:
            params = cache["params_test"]
        elif "X_test" in cache_keys:
            params = cache["X_test"]

        if params is not None and len(params) == len(r2_per_sample):
            # Build correlation matrix: each param col vs R², MAE
            mae_per_sample = np.mean(np.abs(residuals), axis=1)
            n_params = params.shape[1]

            # Param names (31 device parameters — generic if not stored)
            param_names = [f"p{i}" for i in range(n_params)]
            if "param_names" in cache_keys:
                param_names = list(cache["param_names"])

            from scipy.stats import pearsonr
            corr_r2  = []
            corr_mae = []
            for j in range(n_params):
                r, _ = pearsonr(params[:, j], r2_per_sample)
                corr_r2.append(r)
                r, _ = pearsonr(params[:, j], mae_per_sample)
                corr_mae.append(r)
            corr_r2  = np.array(corr_r2)
            corr_mae = np.array(corr_mae)

            # Save as CSV
            csv_path = os.path.join(diag_dir, "param_metric_correlations.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["parameter", "corr_with_R2", "corr_with_MAE"])
                for j in range(n_params):
                    writer.writerow([param_names[j], f"{corr_r2[j]:.4f}", f"{corr_mae[j]:.4f}"])
            print(f"  Saved param_metric_correlations.csv ({n_params} params)")

            # Bar plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            ax = axes[0]
            order = np.argsort(np.abs(corr_r2))[::-1]
            colors = ["#d73027" if c < 0 else "#4575b4" for c in corr_r2[order]]
            ax.barh(range(n_params), corr_r2[order], color=colors, edgecolor="black", linewidth=0.3)
            ax.set_yticks(range(n_params))
            ax.set_yticklabels([param_names[i] for i in order], fontsize=7)
            ax.set_xlabel("Pearson r")
            ax.set_title("Parameter ↔ R² Correlation")
            ax.axvline(0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.2, axis="x")
            ax.invert_yaxis()

            ax = axes[1]
            order = np.argsort(np.abs(corr_mae))[::-1]
            colors = ["#d73027" if c > 0 else "#4575b4" for c in corr_mae[order]]
            ax.barh(range(n_params), corr_mae[order], color=colors, edgecolor="black", linewidth=0.3)
            ax.set_yticks(range(n_params))
            ax.set_yticklabels([param_names[i] for i in order], fontsize=7)
            ax.set_xlabel("Pearson r")
            ax.set_title("Parameter ↔ MAE Correlation")
            ax.axvline(0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.2, axis="x")
            ax.invert_yaxis()

            plt.tight_layout()
            fig.savefig(os.path.join(diag_dir, "correlations.pdf"), dpi=150)
            fig.savefig(os.path.join(diag_dir, "correlations.png"), dpi=150)
            plt.close(fig)
            print("  Saved correlations.pdf/.png")
        else:
            if params is None:
                print("  No test parameters found in preprocessed data — skipping correlations")
            else:
                print(f"  Parameter/prediction length mismatch ({len(params)} vs {len(r2_per_sample)}) — skipping correlations")
    else:
        if not npz_candidates:
            print("  Preprocessed npz not found — skipping correlations")
        if not parity_done:
            print("  No predictions available — skipping correlations")
except Exception as e:
    print(f"  Correlation analysis error: {e}")

# ── 5.  Timing summary ──────────────────────────────────────────────────
timing_summary = {
    "seed": 42,
    "architecture": "conv_dilated_noattn",
    "max_epochs": stats.get("max_epochs", ""),
    "batch_size": stats.get("batch_size", ""),
    "train_time_s": stats.get("train_time_s", ""),
    "test_time_s": stats.get("test_time_s", ""),
    "total_time_s": stats.get("total_time_s", ""),
    "train_steps": stats.get("train_steps", ""),
}
timing_path = os.path.join(diag_dir, "timing_summary.json")
with open(timing_path, "w") as f:
    json.dump(timing_summary, f, indent=2)
print(f"\n  Saved timing_summary.json")

# ── Done ────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"All diagnostics written to: {diag_dir}")
print(f"{'='*50}")
PYEOF

    STEP_END=$(date +%s)
    echo "Diagnostics: $((STEP_END - STEP_START))s" >> $TIMING_LOG
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=============================================="
echo "SINGLE DILATED-CONV PIPELINE COMPLETE"
echo "=============================================="
echo "End: $(date)"
echo ""
echo "Outputs:"
echo "  Model:        $EXP_OUT/$RUN_NAME/"
echo "  Test stats:   $RESULTS_DIR/${RUN_NAME}_stats.json"
echo "  Diagnostics:  $DIAGNOSTICS_DIR/"
echo "  Timing:       $TIMING_LOG"
echo ""
echo "Diagnostics contents:"
ls -lh $DIAGNOSTICS_DIR/ 2>/dev/null || echo "  (empty)"
echo ""
echo "Timing:"
cat $TIMING_LOG
echo "=============================================="
