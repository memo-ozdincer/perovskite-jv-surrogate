#!/bin/bash
#SBATCH --job-name=direct_no_hpo_filtered
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_filtered_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_filtered_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Direct Curve Pipeline with Preprocessed Data
# ============================================================================
# This version uses preprocessed (quality-filtered) data for training.
# The preprocessing step removes anomalous samples to improve model accuracy.
# ============================================================================

echo "==========================================="
echo "PREPROCESSED DATA PIPELINE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "==========================================="

# Load modules
module purge
module load gcc/12.3 cuda/12.2 python/3.11

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Working directory
WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors"
cd $WORK_DIR

mkdir -p /scratch/memoozd/ts-tools-scratch/dbe/logs
source ../venv/bin/activate

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================
PREPROCESS_DIR="$WORK_DIR/preprocessed"
mkdir -p "$PREPROCESS_DIR"

echo "==========================================="
echo "STEP 1: Data Preprocessing"
echo "==========================================="

# Preprocess primary dataset (100k)
python scripts/preprocess_data.py \
    --params "$WORK_DIR/LHS_parameters_m.txt" \
    --iv "$WORK_DIR/IV_m.txt" \
    --output-dir "$PREPROCESS_DIR" \
    --min-ff 0.30 \
    --min-vmpp 0.30 \
    --suffix "_clean"

# Preprocess extra dataset (300k)
python scripts/preprocess_data.py \
    --params "$WORK_DIR/LHS_parameters_m_300k.txt" \
    --iv "$WORK_DIR/IV_m_300k.txt" \
    --output-dir "$PREPROCESS_DIR" \
    --min-ff 0.30 \
    --min-vmpp 0.30 \
    --suffix "_clean"

echo ""
echo "Preprocessing complete. Using cleaned datasets for training."
echo ""

# ============================================================================
# STEP 2: MODEL TRAINING (NO HPO)
# ============================================================================
OUT_DIR="$WORK_DIR/outputs_preprocessed_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$WORK_DIR/results"

CTRL_POINTS=8

echo "==========================================="
echo "STEP 2: Model Training"
echo "==========================================="
echo ""
echo "Configuration:"
echo "  CTRL_POINTS: $CTRL_POINTS"
echo "  HPO: DISABLED"
echo ""
echo "Data files (preprocessed):"
echo "  Primary: $PREPROCESS_DIR/LHS_parameters_m_clean.txt"
echo "  Extra:   $PREPROCESS_DIR/LHS_parameters_m_300k_clean.txt"
echo ""

CMD="python train.py \
    --params \"$PREPROCESS_DIR/LHS_parameters_m_clean.txt\" \
    --iv \"$PREPROCESS_DIR/IV_m_clean.txt\" \
    --params-extra \"$PREPROCESS_DIR/LHS_parameters_m_300k_clean.txt\" \
    --iv-extra \"$PREPROCESS_DIR/IV_m_300k_clean.txt\" \
    --output \"$OUT_DIR\" \
    --device cuda \
    --train-curves \
    --direct-curve \
    --drop-weak-features \
    --drop-multicollinear \
    --no-hpo \
    --ctrl-points $CTRL_POINTS \
    --report-trimmed-metrics"

echo "Running command:"
echo "$CMD"
echo ""
mkdir -p "$RESULTS_DIR"

eval $CMD

RUN_TAG=$(basename "$OUT_DIR")
cp -f "$OUT_DIR/metrics.json" "$RESULTS_DIR/${RUN_TAG}_metrics.json" 2>/dev/null
cp -f "$OUT_DIR/training_summary.json" "$RESULTS_DIR/${RUN_TAG}_training_summary.json" 2>/dev/null
cp -f "$OUT_DIR/curve_error_analysis.csv" "$RESULTS_DIR/${RUN_TAG}_curve_error_analysis.csv" 2>/dev/null
cp -f "$OUT_DIR/curve_error_analysis_summary.json" "$RESULTS_DIR/${RUN_TAG}_curve_error_analysis_summary.json" 2>/dev/null
cp -f "$OUT_DIR/outlier_detection.csv" "$RESULTS_DIR/${RUN_TAG}_outlier_detection.csv" 2>/dev/null
cp -f "$OUT_DIR/model_comparison.md" "$RESULTS_DIR/${RUN_TAG}_model_comparison.md" 2>/dev/null

# Copy preprocessing stats
cp -f "$PREPROCESS_DIR/preprocessing_stats_clean.json" "$RESULTS_DIR/${RUN_TAG}_preprocessing_stats.json" 2>/dev/null

echo ""
echo "==========================================="
echo "PIPELINE COMPLETE"
echo "End time: $(date)"
echo "==========================================="
echo ""
echo "Output: $OUT_DIR"
echo "Preprocessing stats: $PREPROCESS_DIR/preprocessing_stats_clean.json"
