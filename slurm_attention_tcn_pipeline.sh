#!/bin/bash
#SBATCH --job-name=atcn_iv_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=7:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/atcn_pipeline_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/atcn_pipeline_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# ATTENTION-TCN I-V RECONSTRUCTION PIPELINE
# ============================================================================
# Parallel pipeline to slurm_master_pipeline.sh.
# Trains 1 architecture variant × 1 seed = 1 run on 100k+300k data.
# All scalar features loaded from external txt files (no data leakage).
#
# Experiments:
#   Pointwise-NoAttn        : 1x1 conv (position-independent baseline)
#
# Usage:
#   sbatch slurm_attention_tcn_pipeline.sh
#   sbatch slurm_attention_tcn_pipeline.sh --skip-preprocessing
#   sbatch slurm_attention_tcn_pipeline.sh --dry-run
# ============================================================================

set -e

echo "=============================================="
echo "ATTENTION-TCN I-V RECONSTRUCTION PIPELINE"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# ============================================================================
# CONFIGURATION
# ============================================================================

SKIP_PREPROCESSING=false
EXPERIMENTS_ONLY=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --skip-preprocessing) SKIP_PREPROCESSING=true ;;
        --experiments-only) EXPERIMENTS_ONLY=true; SKIP_PREPROCESSING=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# Paths (same work directory as master pipeline)
WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors"
PREPROCESS_DIR="$WORK_DIR/preprocessed"
OUTPUT_BASE="$WORK_DIR/outputs/atcn_experiments_$(date +%Y%m%d)"
ATCN_DATA_DIR="$OUTPUT_BASE/atcn_processed"
LOGS_DIR="$WORK_DIR/logs"

# Raw data files (100k + 300k)
PARAMS_PRIMARY="$WORK_DIR/LHS_parameters_m.txt"
IV_PRIMARY="$WORK_DIR/IV_m.txt"
PARAMS_EXTRA="$WORK_DIR/LHS_parameters_m_300k.txt"
IV_EXTRA="$WORK_DIR/IV_m_300k.txt"

# Preprocessing thresholds (shared with master pipeline)
MIN_FF=0.30
MIN_VMPP=0.00

# Experiment seeds
SEEDS=(42)

# Training configuration
MAX_EPOCHS=100
BATCH_SIZE=128

# ============================================================================
# SETUP
# ============================================================================

cd $WORK_DIR
mkdir -p $LOGS_DIR $OUTPUT_BASE $ATCN_DATA_DIR

# Load modules
module purge
module load gcc/12.3 cuda/12.2 python/3.11

# Activate environment
source ../venv/bin/activate

# Install additional dependencies needed by the Attention-TCN pipeline
echo "Checking Attention-TCN dependencies..."
pip install --quiet pytorch_lightning rich seaborn scipy pillow tqdm 2>/dev/null || true

# Verify environment
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "  PyTorch Lightning: $(python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)')"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

TIMING_LOG="$OUTPUT_BASE/timing.log"
echo "ATCN Pipeline started: $(date)" > $TIMING_LOG

# ============================================================================
# STEP 1: DATA PREPROCESSING (100k + 300k)
# ============================================================================

if [ "$SKIP_PREPROCESSING" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 1: Data Preprocessing (100k + 300k)"
    echo "=============================================="
    STEP_START=$(date +%s)

    mkdir -p $PREPROCESS_DIR

    # --- 100k dataset ---
    echo "Processing primary dataset (100k)..."
    python scripts/preprocess_data.py \
        --params "$PARAMS_PRIMARY" \
        --iv "$IV_PRIMARY" \
        --output-dir "$PREPROCESS_DIR" \
        --min-ff $MIN_FF \
        --min-vmpp $MIN_VMPP \
        --suffix "_clean"

    echo "Generating scalar txt files (100k)..."
    python scripts/generate_scalar_txt.py \
        --iv "$PREPROCESS_DIR/IV_m_clean.txt" \
        --output-dir "$PREPROCESS_DIR" \
        --tag 100k --suffix "_clean"

    # --- 300k dataset ---
    echo "Processing extra dataset (300k)..."
    python scripts/preprocess_data.py \
        --params "$PARAMS_EXTRA" \
        --iv "$IV_EXTRA" \
        --output-dir "$PREPROCESS_DIR" \
        --min-ff $MIN_FF \
        --min-vmpp $MIN_VMPP \
        --suffix "_clean"

    echo "Generating scalar txt files (300k)..."
    python scripts/generate_scalar_txt.py \
        --iv "$PREPROCESS_DIR/IV_m_300k_clean.txt" \
        --output-dir "$PREPROCESS_DIR" \
        --tag 300k --suffix "_clean"

    STEP_END=$(date +%s)
    echo "Preprocessing time: $((STEP_END - STEP_START))s" >> $TIMING_LOG
    echo "Preprocessing complete: $(date)"
fi

# Preprocessed data paths
PARAMS_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_clean.txt"
IV_CLEAN="$PREPROCESS_DIR/IV_m_clean.txt"
PARAMS_EXTRA_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_300k_clean.txt"
IV_EXTRA_CLEAN="$PREPROCESS_DIR/IV_m_300k_clean.txt"

# Scalar txt files (true scalars from preprocessing)
VOC_100K="$PREPROCESS_DIR/voc_clean_100k.txt"
VMPP_100K="$PREPROCESS_DIR/vmpp_clean_100k.txt"
VOC_300K="$PREPROCESS_DIR/voc_clean_300k.txt"
VMPP_300K="$PREPROCESS_DIR/vmpp_clean_300k.txt"

# Verify input files exist
echo ""
echo "Input files:"
for f in "$PARAMS_CLEAN" "$IV_CLEAN" "$PARAMS_EXTRA_CLEAN" "$IV_EXTRA_CLEAN" \
         "$VOC_100K" "$VMPP_100K" "$VOC_300K" "$VMPP_300K"; do
    if [ -f "$f" ]; then
        echo "  [OK] $(basename $f) ($(wc -l < "$f") lines)"
    else
        echo "  [MISSING] $f"
        if [ "$DRY_RUN" = false ]; then
            echo "ERROR: Required input file not found. Run preprocessing first."
            exit 1
        fi
    fi
done

# ============================================================================
# STEP 2: ARCHITECTURE EXPERIMENTS (1 arch × 1 seed = 1 run)
# ============================================================================

echo ""
echo "=============================================="
echo "STEP 2: Architecture Experiments"
echo "=============================================="
STEP_START=$(date +%s)

# All experiments share the same preprocessed PCHIP cache (same seed → same split).
# Only the model architecture differs, so we can reuse one data cache per seed.

# Define experiments: NAME -> FLAGS
declare -A EXP_FLAGS
EXP_FLAGS["Pointwise-NoAttn"]="--architecture pointwise --no-attention"

RESULTS_DIR="$OUTPUT_BASE/results"
mkdir -p "$RESULTS_DIR"

for EXP_ID in "Pointwise-NoAttn"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${EXP_ID}_seed${SEED}"
        EXP_OUT="$OUTPUT_BASE/$EXP_ID/seed_$SEED"
        EXP_DATA="$ATCN_DATA_DIR/seed_${SEED}"
        mkdir -p "$EXP_OUT" "$EXP_DATA"

        echo ""
        echo "Running: $EXP_ID (seed=$SEED)"
        echo "  Output: $EXP_OUT"
        echo "  Flags:  ${EXP_FLAGS[$EXP_ID]}"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] python train_attention_tcn.py ... ${EXP_FLAGS[$EXP_ID]} --seed $SEED"
        else
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
                ${EXP_FLAGS[$EXP_ID]} \
                2>&1 | tee "$EXP_OUT/train.log"

            # Copy test stats to results directory
            cp -f "$EXP_OUT/$RUN_NAME/test_stats.json" \
                   "$RESULTS_DIR/${RUN_NAME}_stats.json" 2>/dev/null || true
        fi
    done
done

STEP_END=$(date +%s)
echo "Experiments time: $((STEP_END - STEP_START))s" >> $TIMING_LOG

# ============================================================================
# STEP 3: COLLECT AND AGGREGATE RESULTS
# ============================================================================

echo ""
echo "=============================================="
echo "STEP 3: Collecting Results"
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    echo "Aggregating all experiment results..."
    python -c "
import json
import pandas as pd
from pathlib import Path

results = []
for f in sorted(Path('$RESULTS_DIR').glob('*_stats.json')):
    try:
        with open(f) as fp:
            data = json.load(fp)
        name = f.stem.replace('_stats', '')
        parts = name.rsplit('_seed', 1)
        data['exp_id'] = parts[0] if len(parts) == 2 else name
        data['seed'] = parts[1] if len(parts) == 2 else 'unknown'
        results.append(data)
    except Exception as e:
        print(f'Error loading {f}: {e}')

if results:
    df = pd.DataFrame(results)
    out_csv = '$OUTPUT_BASE/atcn_all_results.csv'
    df.to_csv(out_csv, index=False)
    print(f'Saved {len(df)} results to {out_csv}')
    print()
    print('Summary by experiment:')
    for exp_id in df['exp_id'].unique():
        sub = df[df['exp_id'] == exp_id]
        r2 = sub.get('r2_median')
        mae = sub.get('mae_median')
        arch = sub.get('architecture', pd.Series(['?'])).iloc[0]
        if r2 is not None:
            print(f'  {exp_id} [{arch}]: R2_med={r2.mean():.4f}+/-{r2.std():.4f}  MAE_med={mae.mean():.6f}+/-{mae.std():.6f}')
else:
    print('No results found.')
"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=============================================="
echo "ATCN PIPELINE COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  Results CSV:  $OUTPUT_BASE/atcn_all_results.csv"
echo "  Experiments:  $OUTPUT_BASE/{TCN-DilatedConv-NoAttn,Conv-NoAttn,Pointwise-NoAttn}/"
echo "  TB Logs:      $OUTPUT_BASE/*/seed_*/tb_logs/"
echo "  Timing:       $TIMING_LOG"
echo ""
echo "Timing summary:"
cat $TIMING_LOG
echo ""

# Count completed experiments
N_COMPLETED=$(find $OUTPUT_BASE -name "test_stats.json" 2>/dev/null | wc -l)
N_EXPECTED=$((3 * ${#SEEDS[@]}))
echo "Completed experiments: $N_COMPLETED / $N_EXPECTED"
echo "=============================================="
