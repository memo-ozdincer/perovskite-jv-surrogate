#!/bin/bash
#SBATCH --job-name=scalar_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/scalar_debug_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/scalar_debug_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Scalar PV Predictors - Debug Training (No HPO)
# Quick runs for debugging inf/nan issues and testing individual models
# ============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

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

# Create logs directory
mkdir -p /scratch/memoozd/ts-tools-scratch/dbe/logs

# Activate virtual environment
source ../venv/bin/activate

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# ============================================================================
# DEBUG MODE OPTIONS (uncomment one)
# ============================================================================

# Option 1: Data validation only (fastest - checks for inf/nan)
# echo "Running data validation..."
# python debug_train.py \
#     --params $WORK_DIR/LHS_parameters_m.txt \
#     --iv $WORK_DIR/IV_m.txt \
#     --device cuda \
#     --check-data-only \
#     --verbose

# Option 2: Quick test with subset (1000 samples, all models)
# echo "Running quick test with 1000 samples..."
# python debug_train.py \
#     --params $WORK_DIR/LHS_parameters_m.txt \
#     --iv $WORK_DIR/IV_m.txt \
#     --device cuda \
#     --n-samples 1000 \
#     --all \
#     --epochs 10 \
#     --n-estimators 50

# Option 3: Single model test (Voc NN)
# echo "Training Voc NN..."
# python debug_train.py \
#     --params $WORK_DIR/LHS_parameters_m.txt \
#     --iv $WORK_DIR/IV_m.txt \
#     --device cuda \
#     --model voc_nn \
#     --epochs 20

# Option 4: Single model test (Jsc LGBM)
echo "Training Jsc LGBM..."
python debug_train.py \
    --params $WORK_DIR/LHS_parameters_m.txt \
    --iv $WORK_DIR/IV_m.txt \
    --device cuda \
    --model jsc_lgbm \
    --n-estimators 200

# Option 5: Full debug run (all models, full data, minimal epochs)
echo "Running full debug (all models, minimal epochs)..."
python debug_train.py \
    --params $WORK_DIR/LHS_parameters_m.txt \
    --iv $WORK_DIR/IV_m.txt \
    --device cuda \
    --all \
    --epochs 20 \
    --n-estimators 100

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
