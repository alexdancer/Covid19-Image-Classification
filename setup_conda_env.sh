#!/bin/bash

# Setup script for Covid19-env conda environment with GPU support
# This script creates a conda environment and installs all necessary dependencies

set -e  # Exit on error

ENV_NAME="Covid19-env"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "Creating Conda Environment: $ENV_NAME"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

# Create new conda environment with Python 3.10
echo "Creating conda environment with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install conda packages (faster and more reliable for some packages)
echo "Installing conda packages..."
conda install -y \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyter \
    ipykernel \
    -c conda-forge

# Install pip packages (TensorFlow and OpenCV are better via pip)
echo "Installing pip packages with GPU support..."
pip install --upgrade pip

# Install TensorFlow with GPU support
# For Python 3.10, use tensorflow[and-cuda] or regular tensorflow
# tensorflow[and-cuda] bundles CUDA libraries, making it easier
echo "Installing TensorFlow with GPU support..."
pip install "tensorflow[and-cuda]>=2.15.0"

# Install other required packages
echo "Installing additional dependencies..."
pip install \
    opencv-python>=4.11.0.86 \
    scikit-learn>=1.2.2 \
    seaborn>=0.13.2 \
    matplotlib>=3.10.0 \
    numpy>=1.26.4 \
    pandas>=2.2.2 \
    jupyter \
    ipykernel

# Register the environment as a Jupyter kernel
echo "Registering environment as Jupyter kernel..."
python -m ipykernel install --user --name $ENV_NAME --display-name "Python ($ENV_NAME)"

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify GPU support, run:"
echo "  python -c \"import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)\""
echo ""
echo "To use this environment in Jupyter:"
echo "  1. Activate the environment: conda activate $ENV_NAME"
echo "  2. Start Jupyter: jupyter notebook"
echo "  3. Select kernel: Python ($ENV_NAME)"
echo ""

