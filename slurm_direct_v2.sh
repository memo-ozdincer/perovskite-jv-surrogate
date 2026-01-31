#!/bin/bash
#SBATCH --job-name=direct_curve_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_v2_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_v2_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# DirectCurveNetV2 Pipeline - HIGH-ACCURACY CURVE PREDICTION
# ============================================================================
# Key improvements over V1 (inspired by Zbinden et al. 2026 & Toprak 2025):
#
# 1. DIRECT 45-POINT PREDICTION (no control point bottleneck)
#    - Removes information bottleneck of K control points + PCHIP
#    - Each curve point is predicted directly
#
# 2. PARAMETER RECONSTRUCTION LOSS (Zbinden's "latent loss")
#    - Network also reconstructs 6 key physics parameters
#    - Forces encoder to learn physically meaningful representations
#    - Parameters: mobility_e, mobility_h, Gen_rate, Aug, Brad, SRV
#
# 3. HARD MONOTONICITY via cumulative decrements
#    - Network predicts positive decrements
#    - Curve = Jsc - cumsum(decrements) * Jsc / sum(decrements)
#    - Guarantees monotonically decreasing J-V curve
#
# 4. REGION-WEIGHTED LOSS
#    - Higher weight on knee region (~0.75 * Voc)
#    - Endpoint constraints (Jsc at V=0, ~0 at Voc)
#    - Smoothness penalty for non-physical oscillations
#
# TARGET: >99% R² (following Zbinden's 99.6% and Toprak's 99.96%)
# ============================================================================
#
# USAGE:
#   sbatch slurm_direct_v2.sh
#
#   # With HPO from previous run:
#   sbatch slurm_direct_v2.sh /path/to/hpo_results.json
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
OUT_DIR="$WORK_DIR/outputs_direct_v2_$(date +%Y%m%d_%H%M%S)"

echo "Running DirectCurveNetV2 Pipeline (High-Accuracy)..."
echo "Output directory: $OUT_DIR"

# ============================================================================
# CONFIGURATION
# ============================================================================
HPO_TRIALS_NN=5          # HPO trials for Voc NN (endpoints still from pretrained)
HPO_TRIALS_LGBM=100      # HPO trials for LightGBM (Jsc, Vmpp, Jmpp)
HPO_TIMEOUT=7200         # 2 hours per model

# Check for HPO file argument
LOAD_HPO_FLAG=""
if [ -n "$1" ] && [ -f "$1" ]; then
    LOAD_HPO_FLAG="--load-hpo \"$1\""
    echo "Loading HPO results from: $1"
fi

echo ""
echo "Configuration:"
echo "  MODEL: DirectCurveNetV2 (high-accuracy, direct 45-point prediction)"
echo "  HPO_TRIALS_NN: $HPO_TRIALS_NN"
echo "  HPO_TRIALS_LGBM: $HPO_TRIALS_LGBM"
echo "  Features: Parameter reconstruction (latent loss)"
echo "  LOAD_HPO: ${LOAD_HPO_FLAG:-'(none - will run HPO)'}"
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
    --direct-curve-v2 \
    --drop-weak-features \
    --drop-multicollinear \
    --hpo-trials-nn $HPO_TRIALS_NN \
    --hpo-trials-lgbm $HPO_TRIALS_LGBM \
    --hpo-timeout $HPO_TIMEOUT"

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
echo "  - $OUT_DIR/hpo_results.json (REUSE FOR FUTURE RUNS)"
echo "  - $OUT_DIR/training_summary.json"
echo "  - $OUT_DIR/metrics.json"
echo "  - $OUT_DIR/models/direct_curve_v2_model.pt"
echo ""
echo "To skip HPO in future runs:"
echo "  sbatch slurm_direct_v2.sh $OUT_DIR/hpo_results.json"
