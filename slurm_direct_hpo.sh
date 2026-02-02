#!/bin/bash
#SBATCH --job-name=direct_curve_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Direct Curve Pipeline - WITH HPO
# ============================================================================
# Uses simplified direct curve model (no Vmpp split):
# - Takes Jsc from pretrained LGBM (accurate: R²=0.965)
# - Predicts Voc + control points jointly
# - Single-region PCHIP interpolation
# - Avoids cascade errors from Voc/Vmpp/Jmpp predictions
# ============================================================================
#
# USAGE:
#   sbatch slurm_direct_hpo.sh
#
# OUTPUT:
#   - $OUT_DIR/hpo_results.json     : HPO results (for scalar models)
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
OUT_DIR="$WORK_DIR/outputs_direct_hpo_$(date +%Y%m%d_%H%M%S)"

echo "Running Direct Curve Pipeline WITH HPO..."
echo "Output directory: $OUT_DIR"

# ============================================================================
# CONFIGURATION
# ============================================================================
HPO_TRIALS_NN=5        # HPO trials for Voc NN (used by shape model)
HPO_TRIALS_LGBM=100       # HPO trials for LightGBM models
HPO_TIMEOUT=7200          # 2 hours per model
CTRL_POINTS=8             # Shape-only model uses at least 8 for better knee capture

echo ""
echo "Configuration:"
echo "  HPO_TRIALS_NN: $HPO_TRIALS_NN"
echo "  HPO_TRIALS_LGBM: $HPO_TRIALS_LGBM"
echo "  CTRL_POINTS: $CTRL_POINTS"
echo "  MODEL: Direct Curve (no Vmpp split)"
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================
CMD="python train.py \
    --params \"$WORK_DIR/LHS_parameters_m.txt\" \
    --iv \"$WORK_DIR/IV_m.txt\" \
    --output \"$OUT_DIR\" \
    --device cuda \
    --train-curves \
    --direct-curve \
    --drop-weak-features \
    --drop-multicollinear \
    --hpo-trials-nn $HPO_TRIALS_NN \
    --hpo-trials-lgbm $HPO_TRIALS_LGBM \
    --hpo-timeout $HPO_TIMEOUT \
    --ctrl-points $CTRL_POINTS"

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
echo "  - $OUT_DIR/hpo_results.json (REUSE FOR FUTURE RUNS)"
echo "  - $OUT_DIR/training_summary.json"
echo "  - $OUT_DIR/metrics.json"
echo "  - $OUT_DIR/models/"
echo ""
echo "To skip HPO in future runs:"
echo "  Use slurm_direct_no_hpo.sh or pass --load-hpo $OUT_DIR/hpo_results.json"
