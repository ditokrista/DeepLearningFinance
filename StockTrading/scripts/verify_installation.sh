#!/bin/bash

# Exit on error
set -e

echo "Checking package installation in DeepLearningFinance conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if DeepLearningFinance environment exists
if ! conda env list | grep -q "DeepLearningFinance"; then
    echo "Error: DeepLearningFinance conda environment not found"
    exit 1
fi

echo "Activating DeepLearningFinance conda environment..."
eval "$(conda shell.bash hook)"
conda activate DeepLearningFinance

echo "Listing all installed packages..."
echo "================================"
conda list

echo ""
echo "Checking specific packages from requirements.txt..."
echo "================================"

python3 << 'EOF'
import pkg_resources
import sys

required_packages = [
    'torch', 'numpy', 'pandas', 'scikit-learn', 'requests', 
    'python-dotenv', 'yfinance', 'pyyaml', 'hydra-core', 
    'mlflow', 'tensorboard', 'matplotlib', 'seaborn', 'plotly',
    'joblib', 'tqdm', 'numba', 'scipy', 'statsmodels',
    'pyarrow', 'fastparquet'
]

missing = []
installed = []

for package in required_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        installed.append(f"{package}=={version}")
        print(f"✓ {package}=={version}")
    except pkg_resources.DistributionNotFound:
        missing.append(package)
        print(f"✗ {package} - MISSING")

print(f"\nSummary:")
print(f"Installed: {len(installed)} packages")
print(f"Missing: {len(missing)} packages")

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\n✓ All packages installed successfully!")
EOF

echo ""
echo "Environment verification complete!"
