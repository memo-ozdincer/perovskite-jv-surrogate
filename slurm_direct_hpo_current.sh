#!/bin/bash
#SBATCH --job-name=direct_hpo_current
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_current_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_current_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Direct Curve Pipeline - CURRENT VERSION (Full HPO)
# ============================================================================
# This is the current implementation with full HPO.
# Used for comparison against simplified version.
#
# Key settings:
# - Full HPO: 200 NN trials, 300 LGBM trials
# - Includes curve HPO
# - Uses both 100k and 300k datasets
# ============================================================================

echo "==========================================="
echo "CURRENT VERSION - Full HPO"
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
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"
echo ""

# Output directory
OUT_DIR="$WORK_DIR/outputs_current_$(date +%Y%m%d_%H%M%S)"

# ============================================================================
# CONFIGURATION - CURRENT (Full HPO)
# ============================================================================
HPO_TRIALS_NN=200
HPO_TRIALS_LGBM=300
HPO_TIMEOUT=14400      # 4 hours per model
CTRL_POINTS=8

# Additional data files
PARAMS_EXTRA="$WORK_DIR/LHS_parameters_m_300k.txt"
IV_EXTRA="$WORK_DIR/IV_m_300k.txt"

echo ""
echo "Configuration (CURRENT - Full HPO):"
echo "  HPO_TRIALS_NN: $HPO_TRIALS_NN"
echo "  HPO_TRIALS_LGBM: $HPO_TRIALS_LGBM"
echo "  HPO_TIMEOUT: $HPO_TIMEOUT"
echo "  CTRL_POINTS: $CTRL_POINTS"
echo "  curve-hpo: ENABLED"
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
    --ctrl-points $CTRL_POINTS"

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "==========================================="
echo "CURRENT VERSION - Complete"
echo "End time: $(date)"
echo "==========================================="
echo ""
echo "Output: $OUT_DIR"
