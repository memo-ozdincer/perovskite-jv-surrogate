#!/bin/bash
#SBATCH --job-name=direct_hpo_filtered
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_filtered_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_filtered_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Direct Curve Pipeline - FILTERED VERSION
# ============================================================================
# This version filters out outlier samples before training to improve
# model accuracy on well-behaved IV curves.
#
# Filtering criteria (based on IV curve analysis):
# - Remove samples with Fill Factor < 0.30 (abnormal IV curve shapes)
# - Remove samples with Vmpp <= 0.30 (extreme operating conditions)
#
# These filters remove ~15% of samples but improve R² by ~4-5%.
# ============================================================================

echo "==========================================="
echo "FILTERED VERSION - Outlier Removal"
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

# Output directory
OUT_DIR="$WORK_DIR/outputs_filtered_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$WORK_DIR/results"

# ============================================================================
# CONFIGURATION - FILTERED
# ============================================================================
HPO_TRIALS_NN=5
HPO_TRIALS_LGBM=100
HPO_TIMEOUT=14400      # 4 hours per model
CTRL_POINTS=8

# Filtering thresholds (based on curve error analysis)
FILTER_MIN_FF=0.30      # Drop samples with FF < 0.30
FILTER_MIN_VMPP=0.30    # Drop samples with Vmpp <= 0.30

# Additional data files
PARAMS_EXTRA="$WORK_DIR/LHS_parameters_m_300k.txt"
IV_EXTRA="$WORK_DIR/IV_m_300k.txt"

echo ""
echo "Configuration (FILTERED):"
echo "  HPO_TRIALS_NN: $HPO_TRIALS_NN"
echo "  HPO_TRIALS_LGBM: $HPO_TRIALS_LGBM"
echo "  HPO_TIMEOUT: $HPO_TIMEOUT"
echo "  CTRL_POINTS: $CTRL_POINTS"
echo "  curve-hpo: ENABLED"
echo ""
echo "Outlier Filtering:"
echo "  FILTER_MIN_FF: $FILTER_MIN_FF"
echo "  FILTER_MIN_VMPP: $FILTER_MIN_VMPP"
echo "  Expected drop: ~15% of samples"
echo ""
echo "Data files:"
echo "  Primary: $WORK_DIR/LHS_parameters_m.txt, $WORK_DIR/IV_m.txt"
echo "  Extra: $PARAMS_EXTRA, $IV_EXTRA"
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================
CMD="python train.py \
    --params \"$WORK_DIR/LHS_parameters_m.txt\" \
    --iv \"$WORK_DIR/IV_m.txt\" \
    --params-extra \"$PARAMS_EXTRA\" \
    --iv-extra \"$IV_EXTRA\" \
    --output \"$OUT_DIR\" \
    --device cuda \
    --train-curves \
    --direct-curve \
    --curve-hpo \
    --drop-weak-features \
    --drop-multicollinear \
    --hpo-trials-nn $HPO_TRIALS_NN \
    --hpo-trials-lgbm $HPO_TRIALS_LGBM \
    --hpo-timeout $HPO_TIMEOUT \
    --ctrl-points $CTRL_POINTS \
    --filter-outliers \
    --filter-min-ff $FILTER_MIN_FF \
    --filter-min-vmpp $FILTER_MIN_VMPP \
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


echo ""
echo "==========================================="
echo "FILTERED VERSION - Complete"
echo "End time: $(date)"
echo "==========================================="
echo ""
echo "Output: $OUT_DIR"
