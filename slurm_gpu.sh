#!/bin/bash
#SBATCH --job-name=scalar_pred_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/scalar_pred_hpo_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/scalar_pred_hpo_%j.err
#SBATCH --account=rrg-aspuru

# ============================================================================
# Scalar PV Predictors - GPU Training with HPO
# 1x H100 80GB is sufficient - LightGBM GPU + PyTorch both saturate single GPU
# ============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules (adjust for your cluster)
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

# Activate virtual environment (must run setup_env.sh first on login node)
source ../venv/bin/activate

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Run training with full HPO
python train.py \
    --params $WORK_DIR/LHS_parameters_m.txt \
    --iv $WORK_DIR/IV_m.txt \
    --output $WORK_DIR/outputs_$(date +%Y%m%d_%H%M%S) \
    --device cuda \
    --hpo-trials-nn 300 \
    --hpo-trials-lgbm 500 \
    --hpo-timeout 7200

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
