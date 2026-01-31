#!/bin/bash
#SBATCH --job-name=direct_curve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_no_hpo_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_no_hpo_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Direct Curve Pipeline - NO HPO (Fast Training)
# ============================================================================
# Uses simplified direct curve model (no Vmpp split):
# - Takes Jsc from pretrained LGBM (accurate: R²=0.965)
# - Predicts Voc + control points jointly
# - Single-region PCHIP interpolation
# - Avoids cascade errors from Voc/Vmpp/Jmpp predictions
#
# NO HPO - uses default hyperparameters for fast iteration
# ============================================================================
#
# USAGE:
#   sbatch slurm_direct_no_hpo.sh
#
#   # Or load HPO results from a previous run:
#   sbatch slurm_direct_no_hpo.sh /path/to/hpo_results.json
#
# OUTPUT:
#   - $OUT_DIR/metrics.json         : Final test metrics
#   - $OUT_DIR/models/              : Trained model weights
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
OUT_DIR="$WORK_DIR/outputs_direct_$(date +%Y%m%d_%H%M%S)"

echo "Running Direct Curve Pipeline (NO HPO)..."
echo "Output directory: $OUT_DIR"

# ============================================================================
# CONFIGURATION
# ============================================================================
CTRL_POINTS=8  # Shape-only model uses at least 8 for better knee capture

# Check for HPO file argument
LOAD_HPO_FLAG=""
if [ -n "$1" ] && [ -f "$1" ]; then
    LOAD_HPO_FLAG="--load-hpo \"$1\""
    echo "Loading HPO results from: $1"
fi

echo ""
echo "Configuration (FIXED - no HPO):"
echo "  CTRL_POINTS: $CTRL_POINTS"
echo "  MODEL: Direct Curve (no Vmpp split)"
echo "  LOAD_HPO: ${LOAD_HPO_FLAG:-'(none - using defaults)'}"
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================
CMD="python train.py \
    --params \"$WORK_DIR/LHS_parameters_m.txt\" \"$WORK_DIR/LHS_parameters_m_300k.txt\" \
    --iv \"$WORK_DIR/IV_m.txt\" \"$WORK_DIR/IV_m_300k.txt\" \
    --output \"$OUT_DIR\" \
    --device cuda \
    --train-curves \
    --direct-curve \
    --drop-weak-features \
    --no-hpo \
    --ctrl-points $CTRL_POINTS"

# Add load HPO flag if provided
if [ -n "$LOAD_HPO_FLAG" ]; then
    CMD="$CMD $LOAD_HPO_FLAG"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Run the command
eval $CMD

# ============================================================================
# OUTPUT FILES
# ============================================================================
echo ""
echo "==========================================="
echo "End time: $(date)"
echo "==========================================="
echo ""
echo "Key output files:"
echo "  - $OUT_DIR/training_summary.json"
echo "  - $OUT_DIR/metrics.json"
echo "  - $OUT_DIR/models/"
