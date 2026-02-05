#!/bin/bash
#SBATCH --job-name=pinn_figures
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/figures_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/figures_%j.err
#SBATCH --account=aip-aspuru

# ============================================================================
# FIGURE GENERATION ONLY
# ============================================================================
# Run this after experiments complete to (re)generate all figures and tables.
#
# Usage:
#   sbatch slurm_figures_only.sh [RESULTS_DIR]
#
# Example:
#   sbatch slurm_figures_only.sh /scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors/outputs/icml_experiments_20240115
# ============================================================================

echo "=============================================="
echo "PINN ICML FIGURE GENERATION"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=============================================="

# Setup
WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors"
cd $WORK_DIR

module purge
module load gcc/12.3 cuda/12.2 python/3.11
source ../venv/bin/activate

# Get results directory from argument or find latest
if [ -n "$1" ]; then
    RESULTS_DIR="$1"
else
    RESULTS_DIR=$(ls -td $WORK_DIR/outputs/icml_experiments_* 2>/dev/null | head -1)
fi

if [ -z "$RESULTS_DIR" ] || [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: No results directory found. Provide path as argument."
    exit 1
fi

echo "Results directory: $RESULTS_DIR"

FIGURES_DIR="$RESULTS_DIR/figures"
mkdir -p $FIGURES_DIR

# ============================================================================
# STEP 1: Collect and aggregate results
# ============================================================================

echo ""
echo "Step 1: Collecting results..."

python run_all_experiments.py \
    --config ablation_configs.yaml \
    --output-base "$RESULTS_DIR" \
    --collect-only

# Also create consolidated CSV
python -c "
import json
import pandas as pd
from pathlib import Path
import glob

results_dir = '$RESULTS_DIR'
results = []

# Find all metrics.json files
for metrics_path in Path(results_dir).rglob('metrics.json'):
    try:
        with open(metrics_path) as f:
            data = json.load(f)

        # Extract experiment info from path
        parts = metrics_path.parent.parts
        exp_id = parts[-2] if 'seed' in parts[-1] else parts[-1]
        seed = parts[-1].replace('seed_', '') if 'seed' in parts[-1] else '42'

        data['exp_id'] = exp_id
        data['seed'] = int(seed)
        data['output_dir'] = str(metrics_path.parent)
        results.append(data)

    except Exception as e:
        print(f'Error loading {metrics_path}: {e}')

if results:
    df = pd.DataFrame(results)
    output_path = f'{results_dir}/all_results.csv'
    df.to_csv(output_path, index=False)
    print(f'Collected {len(df)} results')
    print(f'Experiments: {df[\"exp_id\"].unique().tolist()}')
    print(f'Saved to: {output_path}')
else:
    print('No results found!')
"

# ============================================================================
# STEP 2: Generate main paper figures
# ============================================================================

echo ""
echo "Step 2: Generating main paper figures..."

RESULTS_CSV="$RESULTS_DIR/all_results.csv"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "Error: Results CSV not found at $RESULTS_CSV"
    exit 1
fi

# Find a model directory for curve predictions and logs
MODEL_DIR=$(find $RESULTS_DIR -name "T0-1-main" -type d | head -1)
if [ -n "$MODEL_DIR" ]; then
    MODEL_DIR="$MODEL_DIR/seed_42"
fi

python generate_paper_figures.py \
    --results "$RESULTS_CSV" \
    --output "$FIGURES_DIR" \
    --log-dir "$MODEL_DIR"

# ============================================================================
# STEP 3: Generate additional analysis figures
# ============================================================================

echo ""
echo "Step 3: Generating analysis figures..."

# Run physics analysis if model exists
if [ -d "$MODEL_DIR" ]; then
    python physics_analysis.py \
        --model-dir "$MODEL_DIR" \
        --output-dir "$RESULTS_DIR/analysis" \
        --analysis all || echo "Physics analysis skipped (may require more setup)"
fi

# ============================================================================
# STEP 4: Generate summary statistics
# ============================================================================

echo ""
echo "Step 4: Generating summary statistics..."

python -c "
import pandas as pd
import numpy as np

df = pd.read_csv('$RESULTS_CSV')

print('='*60)
print('RESULTS SUMMARY')
print('='*60)

# Aggregate by experiment
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'seed']

summary = df.groupby('exp_id')[numeric_cols].agg(['mean', 'std'])

# Print key metrics
key_metrics = ['r2_mean', 'mape_mean', 'ff_mape']
available_metrics = [m for m in key_metrics if m in df.columns]

if available_metrics:
    print(f'\nKey metrics (mean ± std across seeds):')
    print('-'*60)
    for exp_id in df['exp_id'].unique():
        exp_data = df[df['exp_id'] == exp_id]
        print(f'\n{exp_id}:')
        for metric in available_metrics:
            if metric in exp_data.columns:
                mean = exp_data[metric].mean()
                std = exp_data[metric].std()
                print(f'  {metric}: {mean:.4f} ± {std:.4f}')

# Save full summary
summary.to_csv('$RESULTS_DIR/summary_statistics.csv')
print(f'\nFull summary saved to: $RESULTS_DIR/summary_statistics.csv')
"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=============================================="
echo "FIGURE GENERATION COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Generated outputs:"

echo ""
echo "Main paper figures:"
ls -la $FIGURES_DIR/main_paper/ 2>/dev/null || echo "  (not generated)"

echo ""
echo "Appendix figures:"
ls -la $FIGURES_DIR/appendix/ 2>/dev/null || echo "  (not generated)"

echo ""
echo "Tables:"
ls -la $FIGURES_DIR/tables/ 2>/dev/null || echo "  (not generated)"

echo ""
echo "Summary files:"
ls -la $RESULTS_DIR/*.csv 2>/dev/null | head -5

echo ""
echo "=============================================="
