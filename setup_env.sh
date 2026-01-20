#!/bin/bash
# ============================================================================
# Environment Setup Script (Run on login node BEFORE sbatch)
# Uses uv for fast package installation
# ============================================================================

set -e  # Exit on error

WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe"
PROJ_DIR="$WORK_DIR/scalar_predictors"
cd $WORK_DIR

echo "Setting up environment in $WORK_DIR"

# Load modules
module purge
module load gcc/12.3 cuda/12.2 python/3.11

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create venv with uv
echo "Creating virtual environment..."
uv venv venv --python 3.11

# Activate
source venv/bin/activate

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install LightGBM with GPU support
echo "Installing LightGBM (GPU)..."
uv pip install lightgbm --config-settings=cmake.define.USE_GPU=ON

# Install remaining dependencies
echo "Installing other dependencies..."
uv pip install \
    numpy \
    pandas \
    optuna \
    scikit-learn \
    joblib

# Create logs directory
mkdir -p $WORK_DIR/logs
mkdir -p $PROJ_DIR

# Verify installation
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"
python -c "
import torch
import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'LightGBM: {lgb.__version__}')
print(f'Optuna: {optuna.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print()
print('All dependencies installed successfully!')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo "Now run: cd $PROJ_DIR && sbatch slurm_gpu.sh"
echo "============================================"
