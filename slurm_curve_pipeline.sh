#!/bin/bash
#SBATCH --job-name=curve_recon_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/curve_recon_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/curve_recon_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Curve Reconstruction Pipeline
# Runs full pipeline:
# 1. Feature Engineering & Validation
# 2. Scalar Predictor Training (Voc, Jsc, FF, etc.)
# 3. Curve Model Training (Unified Split-Spline)
# 4. CVAE Baseline Training
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
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Output directory with timestamp
OUT_DIR="$WORK_DIR/outputs_curve_$(date +%Y%m%d_%H%M%S)"

echo "Running Curve Reconstruction Pipeline..."
echo "Output directory: $OUT_DIR"

# Run training
# - --train-curves: Enables the Split-Spline Model
# - --train-cvae: Enables the CVAE baseline
# - --drop-weak-features: Automatically removes redundant inputs
python train.py \
    --params "$WORK_DIR/LHS_parameters_m.txt" \
    --iv "$WORK_DIR/IV_m.txt" \
    --output "$OUT_DIR" \
    --device cuda \
    --train-curves \
    --train-cvae \
    --drop-weak-features \
    --hpo-trials-nn 5 \
    --hpo-trials-lgbm 5 \
    --hpo-timeout 7200

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
