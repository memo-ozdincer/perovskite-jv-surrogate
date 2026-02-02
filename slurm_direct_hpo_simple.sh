#!/bin/bash
#SBATCH --job-name=direct_hpo_simple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_simple_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/direct_hpo_simple_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# Direct Curve Pipeline - SIMPLIFIED DEBUG VERSION
# ============================================================================
# GOAL: Verify the model trains correctly before running full HPO.
#
# Key changes from current:
# - MINIMAL HPO: Only 20 NN trials, 50 LGBM trials
# - NO curve HPO: Use default config (faster iteration)
# - SINGLE DATASET: Only 100k (simpler debugging)
# - SHORTER TIMEOUT: 1 hour per model
#
# This should complete in ~2-3 hours and tell us if:
# 1. VOC NN is learning (not stuck at 0.266)
# 2. HPO is actually exploring different configs
# 3. Basic pipeline works end-to-end
#
# WATCH FOR in output:
# - "voc_target_mean" should be ~0.9-1.1V (not 0.266!)
# - "Predicted Voc range" should NOT be constant
# - "[OK] Model output range looks reasonable"
# ============================================================================

echo "==========================================="
echo "SIMPLIFIED DEBUG VERSION - Minimal HPO"
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
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Output directory
OUT_DIR="$WORK_DIR/outputs_simple_$(date +%Y%m%d_%H%M%S)"

# ============================================================================
# CONFIGURATION - SIMPLIFIED (Minimal HPO for debugging)
# ============================================================================
HPO_TRIALS_NN=20       # REDUCED: Just enough to verify HPO works
HPO_TRIALS_LGBM=50     # REDUCED: Fast iteration
HPO_TIMEOUT=3600       # 1 hour per model (shorter)
CTRL_POINTS=6          # REDUCED: Simpler curve model

echo ""
echo "Configuration (SIMPLIFIED - Debug Mode):"
echo "  HPO_TRIALS_NN: $HPO_TRIALS_NN (reduced from 200)"
echo "  HPO_TRIALS_LGBM: $HPO_TRIALS_LGBM (reduced from 300)"
echo "  HPO_TIMEOUT: $HPO_TIMEOUT seconds (1 hour)"
echo "  CTRL_POINTS: $CTRL_POINTS (reduced from 8)"
echo "  curve-hpo: DISABLED (use defaults)"
echo ""
echo "Data files:"
echo "  Primary ONLY: $WORK_DIR/LHS_parameters_m.txt, $WORK_DIR/IV_m.txt"
echo "  (300k extra data SKIPPED for faster debugging)"
echo ""

# ============================================================================
# IMPORTANT: Data sanity check before training
# ============================================================================
echo "============================================"
echo "PRE-FLIGHT DATA CHECK"
echo "============================================"
python -c "
import numpy as np
import sys

# Load data
params = np.loadtxt('$WORK_DIR/LHS_parameters_m.txt', delimiter=',')
iv = np.loadtxt('$WORK_DIR/IV_m.txt', delimiter=',')

print(f'Parameters shape: {params.shape}')
print(f'IV curves shape: {iv.shape}')

# Quick target extraction (Voc is where J crosses zero)
# This is a rough check - actual extraction is more sophisticated
v_grid = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525,
                   0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75,
                   0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975,
                   1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2,
                   1.225, 1.25, 1.275, 1.3, 1.325, 1.35, 1.375, 1.4])

# Find Voc (voltage where J first goes negative)
voc_list = []
for i in range(min(1000, len(iv))):
    curve = iv[i]
    # Find first index where J <= 0
    neg_idx = np.where(curve <= 0)[0]
    if len(neg_idx) > 0:
        voc_list.append(v_grid[neg_idx[0]])
    else:
        voc_list.append(v_grid[-1])

voc_arr = np.array(voc_list)
print(f'')
print(f'CRITICAL - Voc Statistics (first 1000 samples):')
print(f'  Voc min:  {voc_arr.min():.4f} V')
print(f'  Voc max:  {voc_arr.max():.4f} V')
print(f'  Voc mean: {voc_arr.mean():.4f} V')
print(f'  Voc std:  {voc_arr.std():.4f} V')

# Sanity check
if voc_arr.mean() < 0.5:
    print(f'')
    print(f'*** WARNING: Voc mean ({voc_arr.mean():.4f}V) seems LOW! ***')
    print(f'Typical perovskite Voc should be 0.9-1.1V')
    print(f'Check your data or target extraction!')
elif voc_arr.mean() > 1.3:
    print(f'')
    print(f'*** WARNING: Voc mean ({voc_arr.mean():.4f}V) seems HIGH! ***')
else:
    print(f'')
    print(f'[OK] Voc range looks reasonable for perovskite cells')

# Check Jsc
jsc_arr = iv[:1000, 0]
print(f'')
print(f'Jsc Statistics (first 1000 samples):')
print(f'  Jsc min:  {jsc_arr.min():.2f} mA/cm2')
print(f'  Jsc max:  {jsc_arr.max():.2f} mA/cm2')
print(f'  Jsc mean: {jsc_arr.mean():.2f} mA/cm2')
"
echo "============================================"
echo ""

# ============================================================================
# BUILD COMMAND - NO curve-hpo, NO extra data
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

echo "Running command (NO curve-hpo, NO extra data):"
echo "$CMD"
echo ""

eval $CMD

# ============================================================================
# POST-RUN DIAGNOSTICS
# ============================================================================
echo ""
echo "==========================================="
echo "POST-RUN DIAGNOSTICS"
echo "==========================================="

if [ -f "$OUT_DIR/metrics.json" ]; then
    echo "Metrics found. Key results:"
    python -c "
import json
with open('$OUT_DIR/metrics.json') as f:
    m = json.load(f)

print('VOC Model:')
if 'voc' in m:
    print(f\"  RMSE: {m['voc'].get('RMSE', 'N/A')}\")
    print(f\"  R2:   {m['voc'].get('R2', 'N/A')}\")
    print(f\"  MAPE: {m['voc'].get('MAPE', 'N/A')}%\")

print('')
print('JSC Model:')
if 'jsc' in m:
    print(f\"  RMSE: {m['jsc'].get('RMSE', 'N/A')}\")
    print(f\"  R2:   {m['jsc'].get('R2', 'N/A')}\")

print('')
print('Curve Model:')
if 'curve' in m:
    print(f\"  MSE Full: {m['curve'].get('mse_full_curve', 'N/A')}\")
    print(f\"  FF MAPE:  {m['curve'].get('mape_ff', 'N/A')}%\")
"
else
    echo "WARNING: metrics.json not found - training may have failed"
fi

echo ""
echo "==========================================="
echo "SIMPLIFIED DEBUG VERSION - Complete"
echo "End time: $(date)"
echo "==========================================="
echo ""
echo "Output: $OUT_DIR"
echo ""
echo "NEXT STEPS:"
echo "1. Check if VOC NN learned (look for 'Model output range looks reasonable')"
echo "2. If VOC is stuck at constant, check voc_target_mean in the logs"
echo "3. If working, run full HPO with slurm_direct_hpo_current.sh"
