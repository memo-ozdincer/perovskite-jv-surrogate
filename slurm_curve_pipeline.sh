#!/bin/bash
#SBATCH --job-name=curve_recon_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/curve_recon_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/curve_recon_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Curve Reconstruction Pipeline - WITH HPO PERSISTENCE
# ============================================================================
# Key improvements:
# - Hard clamping during training (fixes train-test mismatch)
# - Simplified control points (4 instead of 6)
# - Multicollinearity check on features
# - Comprehensive logging (sigma values, constraint violations)
# - Comparison table output (Split-Spline vs CVAE)
# - HPO results saved to JSON for reuse
# - Can skip scalar HPO and load from previous run
# - Curve reconstruction HPO support
# ============================================================================
#
# USAGE MODES:
#   1. Full HPO (default): Run all HPO including curve model
#      sbatch slurm_curve_pipeline.sh
#
#   2. Load HPO from file (skip scalar HPO, still runs curve HPO):
#      sbatch slurm_curve_pipeline.sh /path/to/hpo_results.json
#
#   3. Load HPO, skip curve HPO too (just train with existing params):
#      sbatch slurm_curve_pipeline.sh /path/to/hpo_results.json --no-curve-hpo
#
# OUTPUT:
#   - $OUT_DIR/hpo_results.json : HPO results (can be reused in future runs)
#   - $OUT_DIR/metrics.json     : Final test metrics
#   - $OUT_DIR/models/          : Trained model weights
#
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

# ============================================================================
# CONFIGURATION - Adjust these for your run
# ============================================================================
HPO_TRIALS_NN=300           # Number of HPO trials for NN and curve model
HPO_TRIALS_LGBM=50        # Number of HPO trials for LGBM
HPO_TIMEOUT=7200           # 2 hours per model
CONTINUITY_WEIGHT=0.1      # Try 0.1, 0.5, or 1.0 (may be overridden by curve HPO)
CTRL_POINTS=4              # Simplified from 6 (may be overridden by curve HPO)

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================
LOAD_HPO_PATH=""
RUN_CURVE_HPO="--curve-hpo"  # Default: run curve HPO

# Check for HPO file argument
if [ -n "$1" ] && [ -f "$1" ]; then
    LOAD_HPO_PATH="$1"
    echo "Will load HPO results from: $LOAD_HPO_PATH"
    shift
fi

# Check for --no-curve-hpo flag
for arg in "$@"; do
    if [ "$arg" == "--no-curve-hpo" ]; then
        RUN_CURVE_HPO=""
        echo "Skipping curve HPO"
    fi
done

echo ""
echo "Configuration:"
echo "  HPO_TRIALS_NN: $HPO_TRIALS_NN"
echo "  HPO_TRIALS_LGBM: $HPO_TRIALS_LGBM"
echo "  CONTINUITY_WEIGHT: $CONTINUITY_WEIGHT"
echo "  CTRL_POINTS: $CTRL_POINTS"
echo "  LOAD_HPO_PATH: ${LOAD_HPO_PATH:-'(none - will run HPO)'}"
echo "  RUN_CURVE_HPO: ${RUN_CURVE_HPO:-'(no)'}"
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
    --train-cvae \
    --drop-weak-features \
    --drop-multicollinear \
    --hpo-trials-nn $HPO_TRIALS_NN \
    --hpo-trials-lgbm $HPO_TRIALS_LGBM \
    --hpo-timeout $HPO_TIMEOUT \
    --continuity-weight $CONTINUITY_WEIGHT \
    --ctrl-points $CTRL_POINTS"

# Add curve HPO flag if enabled
if [ -n "$RUN_CURVE_HPO" ]; then
    CMD="$CMD $RUN_CURVE_HPO"
fi

# Add load HPO flag if provided
if [ -n "$LOAD_HPO_PATH" ]; then
    CMD="$CMD --load-hpo \"$LOAD_HPO_PATH\""
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
# - $OUT_DIR/hpo_results.json         : HPO results (REUSE THIS FOR FUTURE RUNS)
# - $OUT_DIR/training_summary.json    : Overall training summary
# - $OUT_DIR/multitask_losses.csv     : Per-epoch sigma values
# - $OUT_DIR/constraint_violations.csv: Per-epoch constraint violations
# - $OUT_DIR/multicollinearity.json   : Feature multicollinearity report
# - $OUT_DIR/model_comparison.json    : Split-Spline vs CVAE metrics
# - $OUT_DIR/model_comparison.md      : Markdown comparison table
# - $OUT_DIR/metrics.json             : Final test metrics
# - $OUT_DIR/models/                  : Trained model weights

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Key output files:"
echo "  - $OUT_DIR/hpo_results.json (REUSE THIS FOR FUTURE RUNS)"
echo "  - $OUT_DIR/training_summary.json"
echo "  - $OUT_DIR/model_comparison.md"
echo "  - $OUT_DIR/metrics.json"
echo ""
echo "To skip scalar HPO in future runs, use:"
echo "  sbatch slurm_curve_pipeline.sh $OUT_DIR/hpo_results.json"
