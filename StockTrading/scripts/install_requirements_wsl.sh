#!/bin/bash

# Exit on error
set -e

echo "Starting installation of Python requirements using conda in WSL..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if DeepLearningFinance environment exists
if ! conda env list | grep -q "DeepLearningFinance"; then
    echo "Error: DeepLearningFinance conda environment not found"
    echo "Please create it first with: conda create -n DeepLearningFinance python=3.9"
    exit 1
fi

echo "Activating DeepLearningFinance conda environment..."
eval "$(conda shell.bash hook)"
conda activate DeepLearningFinance

# Upgrade pip in conda environment
echo "Upgrading pip..."
pip install --upgrade pip

# Install conda packages first (better performance)
echo "Installing conda packages..."
conda install -y numpy pandas scikit-learn scipy matplotlib seaborn plotly requests pyyaml tqdm joblib

# Install pip packages that are not available via conda or have better versions
echo "Installing remaining pip packages (non-PyTorch)..."
pip install python-dotenv yfinance hydra-core mlflow tensorboard numba pyarrow fastparquet statsmodels

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo -e "\nInstallation complete! All requirements have been installed in the DeepLearningFinance conda environment."
echo -e "\nTo activate the environment in the future, run:"
echo "conda activate DeepLearningFinance"
echo -e "\nTo deactivate the environment, simply run:"
echo "conda deactivate"
