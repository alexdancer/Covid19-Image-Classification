# Conda Environment Setup Instructions

This project includes several ways to set up the `Covid19-env` conda environment with GPU support.

## Quick Start

### Option 1: Using the Shell Script (Recommended for Linux/Mac/WSL)
```bash
./setup_conda_env.sh
```

### Option 2: Using the Python Script
```bash
python setup_conda_env.py
```

### Option 3: Using environment.yml (Most Portable)
```bash
conda env create -f environment.yml
conda activate Covid19-env
```

### Option 4: Manual Setup
```bash
# Create environment
conda create -n Covid19-env python=3.10 -y
conda activate Covid19-env

# Install packages
conda install -y numpy pandas matplotlib seaborn scikit-learn jupyter ipykernel -c conda-forge
pip install "tensorflow[and-cuda]>=2.15.0" opencv-python>=4.11.0.86

# Register as Jupyter kernel
python -m ipykernel install --user --name Covid19-env --display-name "Python (Covid19-env)"
```

## Verify GPU Support

After setup, verify that TensorFlow can detect your GPU:

```bash
conda activate Covid19-env
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

Or in Python:
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpus) > 0}")
if len(gpus) > 0:
    for gpu in gpus:
        print(f"  - {gpu}")
```

## Using the Environment

### Activate the environment:
```bash
conda activate Covid19-env
```

### Run Jupyter Notebooks:
```bash
conda activate Covid19-env
jupyter notebook
```

Then select the kernel: **Python (Covid19-env)**

***Enter your local username and password if you have it setup*** 

## Troubleshooting

### GPU Not Detected

1. **Restart the kernel** after installing TensorFlow
2. **Check NVIDIA drivers**: Run `nvidia-smi` in terminal
3. **For WSL2**: Ensure GPU passthrough is enabled
4. **Alternative**: If `tensorflow[and-cuda]` doesn't work, try:
   ```bash
   pip install tensorflow>=2.13.0
   ```
   Then install CUDA/cuDNN separately from NVIDIA website

### Package Installation Issues

If you encounter conflicts:
- Try installing packages one at a time
- Use `conda install` for packages available in conda-forge
- Use `pip install` for TensorFlow and OpenCV

### Environment Already Exists

If the environment already exists:
```bash
conda env remove -n Covid19-env
# Then run setup again
```

## Dependencies Included

- **TensorFlow** (with GPU/CUDA support)
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Plotting
- **Seaborn** - Statistical visualization
- **Scikit-learn** - Machine learning utilities
- **OpenCV** - Image processing
- **Jupyter** - Notebook environment
- **IPykernel** - Jupyter kernel support

## System Requirements

- **Python**: 3.10
- **CUDA**: 11.8 or 12.x (for GPU support)
- **cuDNN**: 8.x (matching CUDA version)
- **NVIDIA GPU** with compatible drivers (for GPU support)

Note: CPU-only training is also supported, but will be much slower.

