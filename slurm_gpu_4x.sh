#!/bin/bash
#SBATCH --job-name=scalar_pred_hpo_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/scalar_pred_hpo_4gpu_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/scalar_pred_hpo_4gpu_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Scalar PV Predictors - 4x GPU Training
# Runs 4 independent HPO studies in parallel (one per GPU)
# Each GPU runs full HPO for different model types simultaneously
# ============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

module purge
module load gcc/12.3 cuda/12.2 python/3.11

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24

BASE_DIR="/scratch/memoozd/ts-tools-scratch/dbe"
WORK_DIR="$BASE_DIR/scalar_predictors"
cd $WORK_DIR
mkdir -p $BASE_DIR/logs

# Activate virtual environment (must run setup_env.sh first on login node)
source $BASE_DIR/venv/bin/activate

echo "Python: $(which python)"
echo "GPUs available: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create output directory with timestamp
OUTPUT_DIR="$WORK_DIR/outputs_4gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Run 4 parallel training jobs, each on a different GPU
# This is useful if you want to try different random seeds or configurations
for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --params $WORK_DIR/LHS_parameters_m.txt $WORK_DIR/LHS_parameters_m_300k.txt \
        --iv $WORK_DIR/IV_m.txt $WORK_DIR/IV_m_300k.txt \
        --output ${OUTPUT_DIR}/gpu${GPU_ID} \
        --device cuda \
        --hpo-trials-nn 300 \
        --hpo-trials-lgbm 500 \
        --hpo-timeout 7200 \
        > ${OUTPUT_DIR}/gpu${GPU_ID}.log 2>&1 &
done

# Wait for all background jobs to complete
wait

echo ""
echo "=========================================="
echo "All 4 GPU jobs completed"
echo "End time: $(date)"
echo "=========================================="
