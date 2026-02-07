#!/bin/bash
#SBATCH --job-name=conv_icml_final_fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/tcn_final_fast_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/tcn_final_fast_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# FAST MINIMAL CONV-DILATED PIPELINE (FINAL FIGURES)
# ============================================================================
# Purpose:
#   - Run ONLY the base model (Conv + Dilated, no attention)
#   - Keep runtime low on a single GPU (e.g., L40S)
#   - Still produce final CSV + figures/tables
#
# Usage:
#   sbatch slurm_tcn_master_pipeline.sh
#   sbatch slurm_tcn_master_pipeline.sh --figures-only
#   sbatch slurm_tcn_master_pipeline.sh --skip-preprocessing
#   sbatch slurm_tcn_master_pipeline.sh --dry-run
# ============================================================================

set -euo pipefail

echo "=============================================="
echo "FAST MINIMAL CONV-DILATED PIPELINE"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   ${SLURM_NODELIST:-local}"
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
OUTPUT_BASE="$WORK_DIR/outputs/tcn_icml_final_fast_$(date +%Y%m%d)"
DATA_CACHE_DIR="$OUTPUT_BASE/data_cache"
RESULTS_DIR="$OUTPUT_BASE/results"
FIGURES_DIR="$OUTPUT_BASE/figures"
LOGS_DIR="$WORK_DIR/logs"

# Raw data
PARAMS_PRIMARY="$WORK_DIR/LHS_parameters_m.txt"
IV_PRIMARY="$WORK_DIR/IV_m.txt"
PARAMS_EXTRA="$WORK_DIR/LHS_parameters_m_300k.txt"
IV_EXTRA="$WORK_DIR/IV_m_300k.txt"

# Preprocessed paths
PARAMS_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_clean.txt"
IV_CLEAN="$PREPROCESS_DIR/IV_m_clean.txt"
PARAMS_EXTRA_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_300k_clean.txt"
IV_EXTRA_CLEAN="$PREPROCESS_DIR/IV_m_300k_clean.txt"
VOC_100K="$PREPROCESS_DIR/voc_clean_100k.txt"
VMPP_100K="$PREPROCESS_DIR/vmpp_clean_100k.txt"
VOC_300K="$PREPROCESS_DIR/voc_clean_300k.txt"
VMPP_300K="$PREPROCESS_DIR/vmpp_clean_300k.txt"

# Base run configuration (fast defaults)
BASE_EXP_ID="T0-1-Conv-Dilated"
BASE_SEED=42
MAX_EPOCHS=60
BATCH_SIZE=512
MIN_FF=0.30
MIN_VMPP=0.00

# ── Environment setup ────────────────────────────────────────────────────────
cd "$WORK_DIR"
mkdir -p "$LOGS_DIR" "$OUTPUT_BASE" "$DATA_CACHE_DIR" "$RESULTS_DIR" "$FIGURES_DIR"

module purge
module load gcc/12.3 cuda/12.2 python/3.11
source ../venv/bin/activate
pip install --quiet pytorch_lightning rich seaborn scipy pillow tqdm 2>/dev/null || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export CUDA_VISIBLE_DEVICES=0

# Conservative, performant dataloader worker count
NUM_WORKERS=8
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ "$SLURM_CPUS_PER_TASK" -gt 0 ]; then
    NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 2))
    if [ "$NUM_WORKERS" -lt 4 ]; then NUM_WORKERS=4; fi
    if [ "$NUM_WORKERS" -gt 12 ]; then NUM_WORKERS=12; fi
fi

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "  Workers: $NUM_WORKERS"
echo "  Epochs: $MAX_EPOCHS"
echo "  Batch: $BATCH_SIZE"
echo ""

TIMING_LOG="$OUTPUT_BASE/timing.log"
echo "Pipeline started: $(date)" > "$TIMING_LOG"

# ============================================================================
# STEP 1: DATA PREPROCESSING (SKIPS IF OUTPUTS EXIST)
# ============================================================================
if [ "$FIGURES_ONLY" = false ] && [ "$SKIP_PREPROCESSING" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 1: Data Preprocessing"
    echo "=============================================="
    STEP_START=$(date +%s)

    mkdir -p "$PREPROCESS_DIR"

    need_preprocess=false
    for f in "$PARAMS_CLEAN" "$IV_CLEAN" "$PARAMS_EXTRA_CLEAN" "$IV_EXTRA_CLEAN" \
             "$VOC_100K" "$VMPP_100K" "$VOC_300K" "$VMPP_300K"; do
        if [ ! -f "$f" ]; then
            need_preprocess=true
            break
        fi
    done

    if [ "$need_preprocess" = true ]; then
        echo "Preprocessed files missing. Running preprocessing..."

        python scripts/preprocess_data.py \
            --params "$PARAMS_PRIMARY" --iv "$IV_PRIMARY" \
            --output-dir "$PREPROCESS_DIR" \
            --min-ff $MIN_FF --min-vmpp $MIN_VMPP --suffix "_clean"

        python scripts/generate_scalar_txt.py \
            --iv "$PREPROCESS_DIR/IV_m_clean.txt" \
            --output-dir "$PREPROCESS_DIR" --tag 100k --suffix "_clean"

        python scripts/preprocess_data.py \
            --params "$PARAMS_EXTRA" --iv "$IV_EXTRA" \
            --output-dir "$PREPROCESS_DIR" \
            --min-ff $MIN_FF --min-vmpp $MIN_VMPP --suffix "_clean"

        python scripts/generate_scalar_txt.py \
            --iv "$PREPROCESS_DIR/IV_m_300k_clean.txt" \
            --output-dir "$PREPROCESS_DIR" --tag 300k --suffix "_clean"
    else
        echo "All preprocessed/scalar files already exist. Skipping preprocessing."
    fi

    STEP_END=$(date +%s)
    echo "Preprocessing: $((STEP_END - STEP_START))s" >> "$TIMING_LOG"
fi

# Verify required inputs
for f in "$PARAMS_CLEAN" "$IV_CLEAN" "$PARAMS_EXTRA_CLEAN" "$IV_EXTRA_CLEAN" \
         "$VOC_100K" "$VMPP_100K" "$VOC_300K" "$VMPP_300K"; do
    if [ ! -f "$f" ]; then
        echo "Missing required input: $f"
        exit 1
    fi
done

# ============================================================================
# STEP 2: TRAIN ONLY BASE MODEL (CONV + DILATED, NO ATTENTION)
# ============================================================================
if [ "$FIGURES_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 2: Base Training (only $BASE_EXP_ID)"
    echo "=============================================="
    STEP_START=$(date +%s)

    EXP_OUT="$OUTPUT_BASE/$BASE_EXP_ID/seed_$BASE_SEED"
    EXP_DATA="$DATA_CACHE_DIR/${BASE_EXP_ID}_seed${BASE_SEED}"
    RUN_NAME="${BASE_EXP_ID}_seed${BASE_SEED}"
    mkdir -p "$EXP_OUT" "$EXP_DATA"

    if [ "$DRY_RUN" = false ]; then
        python train_attention_tcn.py \
            --params "$PARAMS_CLEAN" \
            --iv "$IV_CLEAN" \
            --params-extra "$PARAMS_EXTRA_CLEAN" \
            --iv-extra "$IV_EXTRA_CLEAN" \
            --scalar-files "$VOC_100K" "$VMPP_100K" \
            --scalar-files-extra "$VOC_300K" "$VMPP_300K" \
            --output-dir "$EXP_OUT" \
            --data-dir "$EXP_DATA" \
            --run-name "$RUN_NAME" \
            --seed "$BASE_SEED" \
            --max-epochs "$MAX_EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --prefetch-factor 4 \
            --architecture conv --no-attention --use-dilated \
            2>&1 | tee "$EXP_OUT/train.log"

        cp -f "$EXP_OUT/$RUN_NAME/test_stats.json" \
              "$RESULTS_DIR/${RUN_NAME}_stats.json" 2>/dev/null || true
    else
        echo "[DRY RUN] Would run: $RUN_NAME"
    fi

    STEP_END=$(date +%s)
    echo "Base training: $((STEP_END - STEP_START))s" >> "$TIMING_LOG"
fi

# ============================================================================
# STEP 3: COLLECT RESULTS
# ============================================================================
echo ""
echo "=============================================="
echo "STEP 3: Collecting Results"
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    python tcn_collect_results.py \
        --results-dir "$RESULTS_DIR" \
        --output-base "$OUTPUT_BASE" \
        2>&1 || true
fi

# ============================================================================
# STEP 4: GENERATE FIGURES & TABLES
# ============================================================================
echo ""
echo "=============================================="
echo "STEP 4: Generating Figures & Tables"
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    RESULTS_CSV="$OUTPUT_BASE/all_results.csv"
    if [ -f "$RESULTS_CSV" ]; then
        python tcn_generate_figures.py \
            --results "$RESULTS_CSV" \
            --output "$FIGURES_DIR" \
            2>&1 | tee "$FIGURES_DIR/generation.log"
    else
        echo "Warning: $RESULTS_CSV not found. Skipping figure generation."
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=============================================="
echo "FAST MINIMAL PIPELINE COMPLETE"
echo "=============================================="
echo "End:  $(date)"
echo ""
echo "Outputs:"
echo "  Results CSV:  $OUTPUT_BASE/all_results.csv"
echo "  Figures:      $FIGURES_DIR/"
echo "  Timing:       $TIMING_LOG"
echo ""
echo "Timing:"
cat "$TIMING_LOG"
echo ""

N_COMPLETED=$(find "$RESULTS_DIR" -name "*_stats.json" 2>/dev/null | wc -l)
echo "Completed experiments: $N_COMPLETED"
echo "=============================================="
