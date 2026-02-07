#!/bin/bash
#SBATCH --job-name=conv_icml_master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/tcn_master_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/tcn_master_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# CONV/TCN ICML MASTER PIPELINE
# ============================================================================
# Complete pipeline for ICML paper using the dilated convolution backbone.
#
# Runs:
#   STEP 1  Data preprocessing (100k + 300k, quality filtering)
#   STEP 2  Tier 0 ablation experiments (10 configs × 3 seeds = 30 runs)
#   STEP 3  Tier 1 architecture sweeps (extra configs × 3 seeds)
#   STEP 4  Physics analysis (Jacobian sensitivity, parameter sensitivity)
#   STEP 5  Collect & aggregate all results
#   STEP 6  Generate every figure, table, and statistic for the paper
#
# All scalars loaded from external TXT files — no data leakage.
# Both 100k and 300k datasets are used for training.
#
# Usage:
#   sbatch slurm_tcn_master_pipeline.sh                  # Full pipeline
#   sbatch slurm_tcn_master_pipeline.sh --tier0-only     # Only main ablations
#   sbatch slurm_tcn_master_pipeline.sh --figures-only   # Regen figures
#   sbatch slurm_tcn_master_pipeline.sh --dry-run        # Show commands
# ============================================================================

set -e

echo "=============================================="
echo "CONV/TCN ICML MASTER PIPELINE"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "=============================================="

# ── CLI flags ────────────────────────────────────────────────────────────────
TIER0_ONLY=false
TIER1_ONLY=false
FIGURES_ONLY=false
SKIP_PREPROCESSING=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --tier0-only) TIER0_ONLY=true ;;
        --tier1-only) TIER1_ONLY=true ;;
        --figures-only) FIGURES_ONLY=true ;;
        --skip-preprocessing) SKIP_PREPROCESSING=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# ── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors"
PREPROCESS_DIR="$WORK_DIR/preprocessed"
OUTPUT_BASE="$WORK_DIR/outputs/tcn_icml_$(date +%Y%m%d)"
ATCN_DATA_DIR="$OUTPUT_BASE/data_cache"
RESULTS_DIR="$OUTPUT_BASE/results"
FIGURES_DIR="$OUTPUT_BASE/figures"
LOGS_DIR="$WORK_DIR/logs"

# Raw data
PARAMS_PRIMARY="$WORK_DIR/LHS_parameters_m.txt"
IV_PRIMARY="$WORK_DIR/IV_m.txt"
PARAMS_EXTRA="$WORK_DIR/LHS_parameters_m_300k.txt"
IV_EXTRA="$WORK_DIR/IV_m_300k.txt"

# Experiment seeds
SEEDS=(42 123 456)

# Preprocessing thresholds
MIN_FF=0.30
MIN_VMPP=0.00

# Training
MAX_EPOCHS=100
BATCH_SIZE=128

# ── Environment setup ────────────────────────────────────────────────────────
cd $WORK_DIR
mkdir -p $LOGS_DIR $OUTPUT_BASE $ATCN_DATA_DIR $RESULTS_DIR $FIGURES_DIR

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
echo "  PL: $(python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)')"
echo ""

TIMING_LOG="$OUTPUT_BASE/timing.log"
echo "Pipeline started: $(date)" > $TIMING_LOG

# ============================================================================
# STEP 1: DATA PREPROCESSING (100k + 300k)
# ============================================================================

if [ "$FIGURES_ONLY" = false ] && [ "$SKIP_PREPROCESSING" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 1: Data Preprocessing (100k + 300k)"
    echo "=============================================="
    STEP_START=$(date +%s)

    mkdir -p $PREPROCESS_DIR

    # 100k
    python scripts/preprocess_data.py \
        --params "$PARAMS_PRIMARY" --iv "$IV_PRIMARY" \
        --output-dir "$PREPROCESS_DIR" \
        --min-ff $MIN_FF --min-vmpp $MIN_VMPP --suffix "_clean"

    python scripts/generate_scalar_txt.py \
        --iv "$PREPROCESS_DIR/IV_m_clean.txt" \
        --output-dir "$PREPROCESS_DIR" --tag 100k --suffix "_clean"

    # 300k
    python scripts/preprocess_data.py \
        --params "$PARAMS_EXTRA" --iv "$IV_EXTRA" \
        --output-dir "$PREPROCESS_DIR" \
        --min-ff $MIN_FF --min-vmpp $MIN_VMPP --suffix "_clean"

    python scripts/generate_scalar_txt.py \
        --iv "$PREPROCESS_DIR/IV_m_300k_clean.txt" \
        --output-dir "$PREPROCESS_DIR" --tag 300k --suffix "_clean"

    STEP_END=$(date +%s)
    echo "Preprocessing: $((STEP_END - STEP_START))s" >> $TIMING_LOG
fi

# Preprocessed paths
PARAMS_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_clean.txt"
IV_CLEAN="$PREPROCESS_DIR/IV_m_clean.txt"
PARAMS_EXTRA_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_300k_clean.txt"
IV_EXTRA_CLEAN="$PREPROCESS_DIR/IV_m_300k_clean.txt"

# Scalar txt files (true scalars — swap for predicted later)
VOC_100K="$PREPROCESS_DIR/voc_clean_100k.txt"
VMPP_100K="$PREPROCESS_DIR/vmpp_clean_100k.txt"
VOC_300K="$PREPROCESS_DIR/voc_clean_300k.txt"
VMPP_300K="$PREPROCESS_DIR/vmpp_clean_300k.txt"

# Verify files exist
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

# ── Helper: run one architecture experiment ──────────────────────────────────
run_tcn() {
    local EXP_ID="$1"
    local SEED="$2"
    local FLAGS="$3"
    local EXP_OUT="$OUTPUT_BASE/$EXP_ID/seed_$SEED"
    local EXP_DATA="$ATCN_DATA_DIR/${EXP_ID}_seed${SEED}"
    local RUN_NAME="${EXP_ID}_seed${SEED}"
    mkdir -p "$EXP_OUT" "$EXP_DATA"

    echo ""
    echo ">>> $EXP_ID  seed=$SEED"
    echo "    flags: $FLAGS"
    echo "    out:   $EXP_OUT"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN]"
        return
    fi

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
        --seed $SEED \
        --max-epochs $MAX_EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-workers $((SLURM_CPUS_PER_TASK / 2)) \
        $FLAGS \
        2>&1 | tee "$EXP_OUT/train.log"

    # Copy stats to central results dir
    cp -f "$EXP_OUT/$RUN_NAME/test_stats.json" \
          "$RESULTS_DIR/${RUN_NAME}_stats.json" 2>/dev/null || true
}

# ============================================================================
# STEP 2: TIER 0 — Main-paper ablations (10 experiments)
# ============================================================================
# These 10 experiments establish the core paper claims.
# Each run × 3 seeds = 30 jobs total.

if [ "$FIGURES_ONLY" = false ] && [ "$TIER1_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 2: Tier 0 — Main Ablation Experiments"
    echo "=============================================="
    STEP_START=$(date +%s)

    # ── T0-1  Main model: dilated Conv, no attention ─────────────────────────
    for S in "${SEEDS[@]}"; do
        run_tcn "T0-1-Conv-Dilated" $S \
            "--architecture conv --no-attention --use-dilated"
    done

    # ── T0-2  Conv without dilation (isolates dilation contribution) ─────────
    for S in "${SEEDS[@]}"; do
        run_tcn "T0-2-Conv-NoDilation" $S \
            "--architecture conv --no-attention --no-dilated"
    done

    # ── T0-3  Dilated TCN, no attention (temporal baseline) ──────────────────
    for S in "${SEEDS[@]}"; do
        run_tcn "T0-3-TCN-Dilated" $S \
            "--architecture tcn --no-attention --use-dilated"
    done

    # ── T0-4  Pointwise (1×1 conv, position-independent baseline) ────────────
    for S in "${SEEDS[@]}"; do
        run_tcn "T0-4-Pointwise" $S \
            "--architecture pointwise --no-attention"
    done

    # ── T0-5  Conv WITH self-attention (minimal negative control) ────────────
    for S in "${SEEDS[@]}"; do
        run_tcn "T0-5-Conv-Attn" $S \
            "--architecture conv --use-attention --use-dilated"
    done

    # ── T0-6  TCN WITH self-attention (minimal negative control) ─────────────
    for S in "${SEEDS[@]}"; do
        run_tcn "T0-6-TCN-Attn" $S \
            "--architecture tcn --use-attention --use-dilated"
    done

    # ── T0-7  No scalars (params only — no Voc/Vmpp external input) ──────────
    # Tests the value of scalar conditioning
    for S in "${SEEDS[@]}"; do
        EXP_ID="T0-7-NoScalars"
        EXP_OUT="$OUTPUT_BASE/$EXP_ID/seed_$S"
        EXP_DATA="$ATCN_DATA_DIR/${EXP_ID}_seed${S}"
        RUN_NAME="${EXP_ID}_seed${S}"
        mkdir -p "$EXP_OUT" "$EXP_DATA"

        if [ "$DRY_RUN" = false ]; then
            python train_attention_tcn.py \
                --params "$PARAMS_CLEAN" \
                --iv "$IV_CLEAN" \
                --params-extra "$PARAMS_EXTRA_CLEAN" \
                --iv-extra "$IV_EXTRA_CLEAN" \
                --output-dir "$EXP_OUT" \
                --data-dir "$EXP_DATA" \
                --run-name "$RUN_NAME" \
                --seed $S \
                --max-epochs $MAX_EPOCHS \
                --batch-size $BATCH_SIZE \
                --num-workers $((SLURM_CPUS_PER_TASK / 2)) \
                --architecture conv --no-attention --use-dilated \
                2>&1 | tee "$EXP_OUT/train.log"
            cp -f "$EXP_OUT/$RUN_NAME/test_stats.json" \
                  "$RESULTS_DIR/${RUN_NAME}_stats.json" 2>/dev/null || true
        else
            echo ">>> T0-7-NoScalars seed=$S [DRY RUN — no scalar files passed]"
        fi
    done

    # ── T0-8  100k only (no 300k extra data) ─────────────────────────────────
    for S in "${SEEDS[@]}"; do
        EXP_ID="T0-8-100kOnly"
        EXP_OUT="$OUTPUT_BASE/$EXP_ID/seed_$S"
        EXP_DATA="$ATCN_DATA_DIR/${EXP_ID}_seed${S}"
        RUN_NAME="${EXP_ID}_seed${S}"
        mkdir -p "$EXP_OUT" "$EXP_DATA"

        if [ "$DRY_RUN" = false ]; then
            python train_attention_tcn.py \
                --params "$PARAMS_CLEAN" \
                --iv "$IV_CLEAN" \
                --scalar-files "$VOC_100K" "$VMPP_100K" \
                --output-dir "$EXP_OUT" \
                --data-dir "$EXP_DATA" \
                --run-name "$RUN_NAME" \
                --seed $S \
                --max-epochs $MAX_EPOCHS \
                --batch-size $BATCH_SIZE \
                --num-workers $((SLURM_CPUS_PER_TASK / 2)) \
                --architecture conv --no-attention --use-dilated \
                2>&1 | tee "$EXP_OUT/train.log"
            cp -f "$EXP_OUT/$RUN_NAME/test_stats.json" \
                  "$RESULTS_DIR/${RUN_NAME}_stats.json" 2>/dev/null || true
        else
            echo ">>> T0-8-100kOnly seed=$S [DRY RUN]"
        fi
    done

    # ── T0-9  Longer training (200 epochs) ───────────────────────────────────
    for S in "${SEEDS[@]}"; do
        EXP_ID="T0-9-200epochs"
        EXP_OUT="$OUTPUT_BASE/$EXP_ID/seed_$S"
        EXP_DATA="$ATCN_DATA_DIR/${EXP_ID}_seed${S}"
        RUN_NAME="${EXP_ID}_seed${S}"
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
                --seed $S \
                --max-epochs 200 \
                --batch-size $BATCH_SIZE \
                --num-workers $((SLURM_CPUS_PER_TASK / 2)) \
                --architecture conv --no-attention --use-dilated \
                2>&1 | tee "$EXP_OUT/train.log"
            cp -f "$EXP_OUT/$RUN_NAME/test_stats.json" \
                  "$RESULTS_DIR/${RUN_NAME}_stats.json" 2>/dev/null || true
        else
            echo ">>> T0-9-200epochs seed=$S [DRY RUN]"
        fi
    done

    # ── T0-10  Larger batch size (512) ───────────────────────────────────────
    for S in "${SEEDS[@]}"; do
        run_tcn "T0-10-BS512" $S \
            "--architecture conv --no-attention --use-dilated --batch-size 512"
    done

    STEP_END=$(date +%s)
    echo "Tier 0: $((STEP_END - STEP_START))s" >> $TIMING_LOG
fi

# ============================================================================
# STEP 3: TIER 1 — Extended sweeps (optional)
# ============================================================================

if [ "$FIGURES_ONLY" = false ] && [ "$TIER0_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 3: Tier 1 — Extended Sweeps"
    echo "=============================================="
    STEP_START=$(date +%s)

    # ── Batch size sweep ─────────────────────────────────────────────────────
    for BS in 64 256 512 1024; do
        for S in "${SEEDS[@]}"; do
            run_tcn "T1-BS${BS}" $S \
                "--architecture conv --no-attention --use-dilated --batch-size $BS"
        done
    done

    # ── Epoch sweep ──────────────────────────────────────────────────────────
    for EP in 50 150 200; do
        for S in "${SEEDS[@]}"; do
            run_tcn "T1-EP${EP}" $S \
                "--architecture conv --no-attention --use-dilated --max-epochs $EP"
        done
    done

    STEP_END=$(date +%s)
    echo "Tier 1: $((STEP_END - STEP_START))s" >> $TIMING_LOG
fi

# ============================================================================
# STEP 4: PHYSICS ANALYSIS
# ============================================================================

if [ "$FIGURES_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 4: Physics Analysis"
    echo "=============================================="
    STEP_START=$(date +%s)

    MAIN_MODEL="$OUTPUT_BASE/T0-1-Conv-Dilated/seed_42"
    ANALYSIS_DIR="$OUTPUT_BASE/analysis"
    mkdir -p "$ANALYSIS_DIR"

    if [ -d "$MAIN_MODEL" ] && [ "$DRY_RUN" = false ]; then
        python tcn_analysis.py \
            --results-dir "$OUTPUT_BASE" \
            --output-dir "$ANALYSIS_DIR" \
            --main-model-dir "$MAIN_MODEL" \
            2>&1 | tee "$ANALYSIS_DIR/analysis.log" || true
    fi

    STEP_END=$(date +%s)
    echo "Analysis: $((STEP_END - STEP_START))s" >> $TIMING_LOG
fi

# ============================================================================
# STEP 5: COLLECT & AGGREGATE RESULTS
# ============================================================================

echo ""
echo "=============================================="
echo "STEP 5: Collecting Results"
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    python tcn_collect_results.py \
        --results-dir "$RESULTS_DIR" \
        --output-base "$OUTPUT_BASE" \
        2>&1 || true
fi

# ============================================================================
# STEP 6: GENERATE FIGURES, TABLES, STATISTICS
# ============================================================================

echo ""
echo "=============================================="
echo "STEP 6: Generating Figures & Tables"
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    RESULTS_CSV="$OUTPUT_BASE/all_results.csv"
    if [ -f "$RESULTS_CSV" ]; then
        python tcn_generate_figures.py \
            --results "$RESULTS_CSV" \
            --output "$FIGURES_DIR" \
            --analysis-dir "$OUTPUT_BASE/analysis" \
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
echo "CONV/TCN ICML PIPELINE COMPLETE"
echo "=============================================="
echo "End:  $(date)"
echo ""
echo "Outputs:"
echo "  Results CSV:  $OUTPUT_BASE/all_results.csv"
echo "  Figures:      $FIGURES_DIR/"
echo "  Analysis:     $OUTPUT_BASE/analysis/"
echo "  Timing:       $TIMING_LOG"
echo ""
echo "Timing:"
cat $TIMING_LOG
echo ""

N_COMPLETED=$(find $RESULTS_DIR -name "*_stats.json" 2>/dev/null | wc -l)
echo "Completed experiments: $N_COMPLETED"
echo "=============================================="