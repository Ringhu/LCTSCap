#!/usr/bin/env bash
# setup_env.sh - Create conda environment and install dependencies for LCTSCap.
#
# Usage:
#   bash scripts/setup_env.sh
#
# This script:
#   1. Creates a conda environment named "lctscap" with Python 3.12
#   2. Installs PyTorch and core dependencies via pip
#   3. Downloads required NLTK data
#   4. Installs the lctscap package in editable (development) mode

set -euo pipefail

ENV_NAME="${LCTSCAP_ENV_NAME:-lctscap}"
PYTHON_VERSION="3.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  LCTSCap Environment Setup"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Environment:  $ENV_NAME"
echo "Python:       $PYTHON_VERSION"
echo ""

# Step 1: Create conda environment
if conda info --envs | grep -qw "$ENV_NAME"; then
    echo "[INFO] Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "[INFO] Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate
echo "[INFO] Activating environment '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Step 2: Install PyTorch (CUDA 12.1 by default; override with TORCH_CUDA_VERSION)
CUDA_VERSION="${TORCH_CUDA_VERSION:-cu121}"
echo "[INFO] Installing PyTorch with CUDA=$CUDA_VERSION..."
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# Step 3: Install project dependencies
echo "[INFO] Installing project dependencies..."
cd "$PROJECT_ROOT"
pip install -e ".[dev]"

# Step 4: Install optional MOMENT dependency
echo "[INFO] Installing optional MOMENT dependency..."
pip install momentfm || echo "[WARN] momentfm installation failed; MOMENT encoder will not be available."

# Step 5: Download NLTK data
echo "[INFO] Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('NLTK data downloaded successfully.')
"

# Step 6: Verify installation
echo ""
echo "[INFO] Verifying installation..."
python -c "
import lctscap
print(f'lctscap version: {lctscap.__version__}')
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate with:  conda activate $ENV_NAME"
echo "============================================"
