#!/bin/bash
#SBATCH --job-name=curve_no_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/curve_no_hpo_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/curve_no_hpo_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Curve Reconstruction Pipeline - NO HPO (Fast Training)
# ============================================================================
# Uses default/fixed hyperparameters for quick iteration:
# - Skips HPO entirely (saves ~3 hours)
# - Uses sensible default configs
# - Good for debugging and testing changes
# ============================================================================
#
# USAGE:
#   sbatch slurm_no_hpo.sh
#
#   # Or with direct curve model:
#   sbatch slurm_no_hpo.sh --direct-curve
#
# OUTPUT:
#   - $OUT_DIR/metrics.json     : Final test metrics
#   - $OUT_DIR/models/          : Trained model weights
#
# ============================================================================

echo "==========================================="
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

# CUDA settings
export CUDA_VISIBLE_DEVICES=0

# Working directory
WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors"
cd $WORK_DIR

# Create logs directory if needed
mkdir -p /scratch/memoozd/ts-tools-scratch/dbe/logs

# Activate virtual environment
source ../venv/bin/activate

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"
echo ""

# Output directory with timestamp
OUT_DIR="$WORK_DIR/outputs_no_hpo_$(date +%Y%m%d_%H%M%S)"

echo "Running NO-HPO Pipeline..."
echo "Output directory: $OUT_DIR"

# ============================================================================
# CONFIGURATION - Fixed hyperparameters (no HPO)
# ============================================================================
CONTINUITY_WEIGHT=0.1
CTRL_POINTS=6

# Parse command line arguments
USE_DIRECT_CURVE=""
for arg in "$@"; do
    if [ "$arg" == "--direct-curve" ]; then
        USE_DIRECT_CURVE="--direct-curve"
        echo "Using DIRECT curve model (no scalar predictor dependencies)"
    fi
done

echo ""
echo "Configuration (FIXED - no HPO):"
echo "  CONTINUITY_WEIGHT: $CONTINUITY_WEIGHT"
echo "  CTRL_POINTS: $CTRL_POINTS"
echo "  DIRECT_CURVE: ${USE_DIRECT_CURVE:-'(no - using split-spline)'}"
echo ""

# ============================================================================
# BUILD COMMAND - Skip HPO entirely with --no-hpo flag
# ============================================================================
CMD="python train.py \
    --params \"$WORK_DIR/LHS_parameters_m.txt\" \"$WORK_DIR/LHS_parameters_m_300k.txt\" \
    --iv \"$WORK_DIR/IV_m.txt\" \"$WORK_DIR/IV_m_300k.txt\" \
    --output \"$OUT_DIR\" \
    --device cuda \
    --train-curves \
    --drop-weak-features \
    --no-hpo \
    --continuity-weight $CONTINUITY_WEIGHT \
    --ctrl-points $CTRL_POINTS"

# Add direct curve flag if specified
if [ -n "$USE_DIRECT_CURVE" ]; then
    CMD="$CMD $USE_DIRECT_CURVE"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Run the command
eval $CMD

# ============================================================================
# OUTPUT FILES
# ============================================================================
# The following files will be generated:
# - $OUT_DIR/training_summary.json    : Overall training summary
# - $OUT_DIR/multitask_losses.csv     : Per-epoch sigma values
# - $OUT_DIR/constraint_violations.csv: Per-epoch constraint violations
# - $OUT_DIR/metrics.json             : Final test metrics
# - $OUT_DIR/models/                  : Trained model weights

echo ""
echo "==========================================="
echo "End time: $(date)"
echo "==========================================="
echo ""
echo "Key output files:"
echo "  - $OUT_DIR/training_summary.json"
echo "  - $OUT_DIR/metrics.json"
echo "  - $OUT_DIR/models/"
