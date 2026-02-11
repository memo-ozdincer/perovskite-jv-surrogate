#!/bin/bash
#SBATCH --job-name=pinn_icml_master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=7:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/master_pipeline_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/master_pipeline_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# PINN ICML MASTER PIPELINE
# ============================================================================
# This script runs the COMPLETE pipeline for ICML paper preparation:
#   1. Data preprocessing
#   2. Tier 0 experiments (main ablations - 7 configs × 3 seeds = 21 runs)
#   3. Tier 1 experiments (sweeps - optional)
#   4. Physics analysis (Jacobian, sensitivity)
#   5. Figure and table generation
#   6. Results aggregation
#
# Output: Complete figures/, tables/, and results/ for paper integration
#
# Usage:
#   sbatch slurm_master_pipeline.sh                    # Run full pipeline
#   sbatch slurm_master_pipeline.sh --tier0-only      # Only Tier 0
#   sbatch slurm_master_pipeline.sh --figures-only    # Only generate figures
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "PINN ICML MASTER PIPELINE"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parse command line arguments
TIER0_ONLY=false
TIER1_ONLY=false
FIGURES_ONLY=false
SKIP_PREPROCESSING=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --tier0-only) TIER0_ONLY=true ;;
        --tier1-only) TIER1_ONLY=true ;;
        --figures-only) FIGURES_ONLY=true ;;
        --skip-preprocessing) SKIP_PREPROCESSING=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# Paths
WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors"
PREPROCESS_DIR="$WORK_DIR/preprocessed"
OUTPUT_BASE="$WORK_DIR/outputs/icml_experiments_$(date +%Y%m%d)"
FIGURES_DIR="$OUTPUT_BASE/figures"
RESULTS_DIR="$OUTPUT_BASE/results"
LOGS_DIR="$WORK_DIR/logs"
SCALAR_DIR="${SCALAR_DIR:-$WORK_DIR/scalars_external}"

# Data files
PARAMS_PRIMARY="$WORK_DIR/LHS_parameters_m.txt"
IV_PRIMARY="$WORK_DIR/IV_m.txt"
PARAMS_EXTRA="$WORK_DIR/LHS_parameters_m_300k.txt"
IV_EXTRA="$WORK_DIR/IV_m_300k.txt"

# Experiment seeds
SEEDS=(42 123 456)

# Preprocessing thresholds
MIN_FF=0.30
MIN_VMPP=0.00

# Training configuration
CTRL_POINTS=8
CONTINUITY_WEIGHT=0.1

# ============================================================================
# SETUP
# ============================================================================

cd $WORK_DIR
mkdir -p $LOGS_DIR $OUTPUT_BASE $FIGURES_DIR $RESULTS_DIR

# Load modules
module purge
module load gcc/12.3 cuda/12.2 python/3.11

# Activate environment
source ../venv/bin/activate

# Verify environment
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Create timing log
TIMING_LOG="$OUTPUT_BASE/timing.log"
echo "Pipeline started: $(date)" > $TIMING_LOG

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================

if [ "$FIGURES_ONLY" = false ] && [ "$SKIP_PREPROCESSING" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 1: Data Preprocessing"
    echo "=============================================="
    STEP_START=$(date +%s)

    mkdir -p $PREPROCESS_DIR

    # Preprocess primary dataset (100k)
    echo "Processing primary dataset (100k)..."
    python scripts/preprocess_data.py \
        --params "$PARAMS_PRIMARY" \
        --iv "$IV_PRIMARY" \
        --output-dir "$PREPROCESS_DIR" \
        --min-ff $MIN_FF \
        --min-vmpp $MIN_VMPP \
        --suffix "_clean"

    # Preprocess extra dataset (300k)
    echo "Processing extra dataset (300k)..."
    python scripts/preprocess_data.py \
        --params "$PARAMS_EXTRA" \
        --iv "$IV_EXTRA" \
        --output-dir "$PREPROCESS_DIR" \
        --min-ff $MIN_FF \
        --min-vmpp $MIN_VMPP \
        --suffix "_clean"

    STEP_END=$(date +%s)
    echo "Preprocessing time: $((STEP_END - STEP_START))s" >> $TIMING_LOG
    echo "Preprocessing complete: $(date)"
fi

# Use preprocessed data
PARAMS_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_clean.txt"
IV_CLEAN="$PREPROCESS_DIR/IV_m_clean.txt"
PARAMS_EXTRA_CLEAN="$PREPROCESS_DIR/LHS_parameters_m_300k_clean.txt"
IV_EXTRA_CLEAN="$PREPROCESS_DIR/IV_m_300k_clean.txt"

# Anchor files
ANCHORS="$PREPROCESS_DIR/anchors_clean_100k.txt"
ANCHORS_EXTRA="$PREPROCESS_DIR/anchors_clean_300k.txt"
VOC="$SCALAR_DIR/voc_clean_100k.txt"
VOC_EXTRA="$SCALAR_DIR/voc_clean_300k.txt"
VMPP="$SCALAR_DIR/vmpp_clean_100k.txt"
VMPP_EXTRA="$SCALAR_DIR/vmpp_clean_300k.txt"

# ============================================================================
# STEP 2: TIER 0 EXPERIMENTS (Main Ablations)
# ============================================================================

if [ "$FIGURES_ONLY" = false ] && [ "$TIER1_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 2: Tier 0 Experiments (Main Ablations)"
    echo "=============================================="
    STEP_START=$(date +%s)

    # Define experiments
    declare -A EXPERIMENTS
    EXPERIMENTS["T0-1-main"]="--train-curves --drop-weak-features --drop-multicollinear --ctrl-points $CTRL_POINTS --continuity-weight $CONTINUITY_WEIGHT --no-hpo"
    EXPERIMENTS["T0-2-no-split"]="--train-curves --drop-weak-features --drop-multicollinear --ctrl-points 12 --no-split --no-hpo"
    EXPERIMENTS["T0-3-no-anchors"]="--train-curves --direct-curve --drop-weak-features --drop-multicollinear --no-hpo"
    EXPERIMENTS["T0-4-no-projection"]="--train-curves --drop-weak-features --drop-multicollinear --ctrl-points $CTRL_POINTS --no-physics-projection --no-hpo"
    EXPERIMENTS["T0-5-no-physics-features"]="--train-curves --ctrl-points $CTRL_POINTS --no-physics-features --no-hpo"
    EXPERIMENTS["T0-6-cvae-baseline"]="--train-cvae --drop-weak-features --drop-multicollinear --no-hpo"
    EXPERIMENTS["T0-7-mlp-baseline"]="--train-curves --direct-mlp --no-hpo"

    # Run experiments
    for EXP_ID in "${!EXPERIMENTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            EXP_OUT="$OUTPUT_BASE/$EXP_ID/seed_$SEED"
            mkdir -p "$EXP_OUT"

            echo ""
            echo "Running: $EXP_ID (seed=$SEED)"
            echo "Output: $EXP_OUT"

            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] python train.py --params $PARAMS_CLEAN --iv $IV_CLEAN ... ${EXPERIMENTS[$EXP_ID]} --seed $SEED"
            else
                python train.py \
                    --params "$PARAMS_CLEAN" \
                    --iv "$IV_CLEAN" \
                    --params-extra "$PARAMS_EXTRA_CLEAN" \
                    --iv-extra "$IV_EXTRA_CLEAN" \
                    --anchors "$ANCHORS" \
                    --anchors-extra "$ANCHORS_EXTRA" \
                    --voc-anchors "$VOC" \
                    --voc-anchors-extra "$VOC_EXTRA" \
                    --vmpp-anchors "$VMPP" \
                    --vmpp-anchors-extra "$VMPP_EXTRA" \
                    --output "$EXP_OUT" \
                    --device cuda \
                    --seed $SEED \
                    ${EXPERIMENTS[$EXP_ID]} \
                    2>&1 | tee "$EXP_OUT/train.log"

                # Copy key results
                cp -f "$EXP_OUT/metrics.json" "$RESULTS_DIR/${EXP_ID}_seed${SEED}_metrics.json" 2>/dev/null || true
            fi
        done
    done

    STEP_END=$(date +%s)
    echo "Tier 0 time: $((STEP_END - STEP_START))s" >> $TIMING_LOG
fi

# ============================================================================
# STEP 3: TIER 1 EXPERIMENTS (Sweeps)
# ============================================================================

if [ "$FIGURES_ONLY" = false ] && [ "$TIER0_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 3: Tier 1 Experiments (Parameter Sweeps)"
    echo "=============================================="
    STEP_START=$(date +%s)

    # Control points sweep
    for K in 2 4 6 8 10 12; do
        for SEED in "${SEEDS[@]}"; do
            EXP_ID="T1-ctrl_points-$K"
            EXP_OUT="$OUTPUT_BASE/$EXP_ID/seed_$SEED"
            mkdir -p "$EXP_OUT"

            echo "Running: $EXP_ID (seed=$SEED)"

            if [ "$DRY_RUN" = false ]; then
                python train.py \
                    --params "$PARAMS_CLEAN" \
                    --iv "$IV_CLEAN" \
                    --params-extra "$PARAMS_EXTRA_CLEAN" \
                    --iv-extra "$IV_EXTRA_CLEAN" \
                    --anchors "$ANCHORS" \
                    --anchors-extra "$ANCHORS_EXTRA" \
                    --voc-anchors "$VOC" \
                    --voc-anchors-extra "$VOC_EXTRA" \
                    --vmpp-anchors "$VMPP" \
                    --vmpp-anchors-extra "$VMPP_EXTRA" \
                    --output "$EXP_OUT" \
                    --device cuda \
                    --seed $SEED \
                    --train-curves \
                    --drop-weak-features \
                    --drop-multicollinear \
                    --ctrl-points $K \
                    --continuity-weight $CONTINUITY_WEIGHT \
                    --no-hpo \
                    2>&1 | tee "$EXP_OUT/train.log"

                cp -f "$EXP_OUT/metrics.json" "$RESULTS_DIR/${EXP_ID}_seed${SEED}_metrics.json" 2>/dev/null || true
            fi
        done
    done

    # Continuity weight sweep
    for LAMBDA in 0.0 0.01 0.05 0.1 0.5 1.0; do
        for SEED in "${SEEDS[@]}"; do
            EXP_ID="T1-continuity-$LAMBDA"
            EXP_OUT="$OUTPUT_BASE/$EXP_ID/seed_$SEED"
            mkdir -p "$EXP_OUT"

            echo "Running: $EXP_ID (seed=$SEED)"

            if [ "$DRY_RUN" = false ]; then
                python train.py \
                    --params "$PARAMS_CLEAN" \
                    --iv "$IV_CLEAN" \
                    --params-extra "$PARAMS_EXTRA_CLEAN" \
                    --iv-extra "$IV_EXTRA_CLEAN" \
                    --anchors "$ANCHORS" \
                    --anchors-extra "$ANCHORS_EXTRA" \
                    --voc-anchors "$VOC" \
                    --voc-anchors-extra "$VOC_EXTRA" \
                    --vmpp-anchors "$VMPP" \
                    --vmpp-anchors-extra "$VMPP_EXTRA" \
                    --output "$EXP_OUT" \
                    --device cuda \
                    --seed $SEED \
                    --train-curves \
                    --drop-weak-features \
                    --drop-multicollinear \
                    --ctrl-points $CTRL_POINTS \
                    --continuity-weight $LAMBDA \
                    --no-hpo \
                    2>&1 | tee "$EXP_OUT/train.log"

                cp -f "$EXP_OUT/metrics.json" "$RESULTS_DIR/${EXP_ID}_seed${SEED}_metrics.json" 2>/dev/null || true
            fi
        done
    done

    STEP_END=$(date +%s)
    echo "Tier 1 time: $((STEP_END - STEP_START))s" >> $TIMING_LOG
fi

# ============================================================================
# STEP 4: PHYSICS ANALYSIS
# ============================================================================

if [ "$FIGURES_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "STEP 4: Physics Analysis"
    echo "=============================================="
    STEP_START=$(date +%s)

    # Run analysis on the main model
    MAIN_MODEL_DIR="$OUTPUT_BASE/T0-1-main/seed_42"
    ANALYSIS_DIR="$OUTPUT_BASE/analysis"

    if [ -d "$MAIN_MODEL_DIR" ] && [ "$DRY_RUN" = false ]; then
        echo "Running physics analysis..."
        python physics_analysis.py \
            --model-dir "$MAIN_MODEL_DIR" \
            --output-dir "$ANALYSIS_DIR" \
            --analysis all \
            2>&1 | tee "$ANALYSIS_DIR/analysis.log" || true
    fi

    STEP_END=$(date +%s)
    echo "Analysis time: $((STEP_END - STEP_START))s" >> $TIMING_LOG
fi

# ============================================================================
# STEP 5: COLLECT RESULTS
# ============================================================================

echo ""
echo "=============================================="
echo "STEP 5: Collecting Results"
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    python run_all_experiments.py \
        --config ablation_configs.yaml \
        --output-base "$OUTPUT_BASE" \
        --collect-only

    # Also collect all metrics into a single CSV
    echo "Aggregating all metrics..."
    python -c "
import json
import pandas as pd
from pathlib import Path
import glob

results = []
for f in glob.glob('$RESULTS_DIR/*_metrics.json'):
    try:
        with open(f) as fp:
            data = json.load(fp)
        parts = Path(f).stem.split('_')
        exp_id = '_'.join(parts[:-2])  # Remove seed and metrics
        seed = parts[-2].replace('seed', '')
        data['exp_id'] = exp_id
        data['seed'] = seed
        results.append(data)
    except Exception as e:
        print(f'Error loading {f}: {e}')

if results:
    df = pd.DataFrame(results)
    df.to_csv('$OUTPUT_BASE/all_results.csv', index=False)
    print(f'Saved {len(df)} results to $OUTPUT_BASE/all_results.csv')
"
fi

# ============================================================================
# STEP 6: GENERATE FIGURES AND TABLES
# ============================================================================

echo ""
echo "=============================================="
echo "STEP 6: Generating Figures and Tables"
echo "=============================================="
STEP_START=$(date +%s)

if [ "$DRY_RUN" = false ]; then
    # Check if results exist
    if [ -f "$OUTPUT_BASE/all_results.csv" ]; then
        python generate_paper_figures.py \
            --results "$OUTPUT_BASE/all_results.csv" \
            --output "$FIGURES_DIR" \
            --log-dir "$OUTPUT_BASE/T0-1-main/seed_42" \
            2>&1 | tee "$FIGURES_DIR/generation.log"
    else
        echo "Warning: No results file found. Skipping figure generation."
    fi
fi

STEP_END=$(date +%s)
echo "Figures time: $((STEP_END - STEP_START))s" >> $TIMING_LOG

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  Results: $OUTPUT_BASE/all_results.csv"
echo "  Figures: $FIGURES_DIR/"
echo "  Logs: $LOGS_DIR/"
echo ""
echo "Timing summary:"
cat $TIMING_LOG
echo ""

# List generated files
echo "Generated figures:"
ls -la $FIGURES_DIR/main_paper/ 2>/dev/null || echo "  (none)"
echo ""
echo "Generated tables:"
ls -la $FIGURES_DIR/tables/ 2>/dev/null || echo "  (none)"
echo ""

# Count completed experiments
N_COMPLETED=$(find $OUTPUT_BASE -name "metrics.json" | wc -l)
echo "Completed experiments: $N_COMPLETED"

echo "=============================================="
