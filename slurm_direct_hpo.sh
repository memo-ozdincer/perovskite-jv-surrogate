#!/bin/bash
#SBATCH --job-name=direct_curve_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=18:00:00  # Increased for longer HPO runs (was 20:00:00)
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_%j.err
#SBATCH --account=rrg-aspuru

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
# CONFIGURATION - UPDATED v2.0 for improved HPO
# ============================================================================
HPO_TRIALS_NN=200         # HPO trials for Voc NN (WAS 5, increased for thorough search)
HPO_TRIALS_LGBM=300       # HPO trials for LightGBM models (WAS 100)
HPO_TIMEOUT=14400         # 4 hours per model (WAS 7200 = 2h)
CTRL_POINTS=8             # Shape-only model uses at least 8 for better knee capture

# Additional data files (loaded on top of primary data)
PARAMS_EXTRA="$WORK_DIR/LHS_parameters_m_300k.txt"
IV_EXTRA="$WORK_DIR/IV_m_300k.txt"

echo ""
echo "Configuration:"
echo "  HPO_TRIALS_NN: $HPO_TRIALS_NN"
echo "  HPO_TRIALS_LGBM: $HPO_TRIALS_LGBM"
echo "  HPO_TIMEOUT: $HPO_TIMEOUT"
echo "  CTRL_POINTS: $CTRL_POINTS"
echo "  MODEL: Direct Curve (no Vmpp split)"
echo ""
echo "Data files:"
echo "  Primary: $WORK_DIR/LHS_parameters_m.txt, $WORK_DIR/IV_m.txt"
echo "  Extra: $PARAMS_EXTRA, $IV_EXTRA"
echo ""

# ============================================================================
# BUILD COMMAND - UPDATED v2.0 with multi-file loading
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
