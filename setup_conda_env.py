#!/usr/bin/env python3
"""
Setup script to create a conda environment for the COVID-19 project with GPU support.
This script can be run directly: python setup_conda_env.py
"""

import subprocess
import sys
import os

ENV_NAME = "Covid19-env"
PYTHON_VERSION = "3.10"

def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    return result

def check_conda():
    """Check if conda is installed"""
    try:
        run_command(["conda", "--version"], check=False)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: conda is not installed or not in PATH")
        print("Please install Anaconda or Miniconda first")
        return False

def create_environment():
    """Create the conda environment"""
    print("=" * 60)
    print(f"Creating Conda Environment: {ENV_NAME}")
    print("=" * 60)
    
    # Check if environment already exists
    result = run_command(["conda", "env", "list"], check=False)
    if ENV_NAME in result.stdout:
        print(f"Environment {ENV_NAME} already exists.")
        response = input("Do you want to remove and recreate it? (y/n): ")
        if response.lower() == 'y':
            print(f"Removing existing environment {ENV_NAME}...")
            run_command(["conda", "env", "remove", "-n", ENV_NAME, "-y"], check=False)
        else:
            print("Keeping existing environment. Exiting.")
            return False
    
    # Create environment using environment.yml
    print(f"Creating environment from environment.yml...")
    if os.path.exists("environment.yml"):
        run_command(["conda", "env", "create", "-f", "environment.yml"])
    else:
        # Fallback: create manually
        print("environment.yml not found, creating environment manually...")
        run_command(["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"])
        
        # Install packages
        print("Installing conda packages...")
        run_command([
            "conda", "install", "-n", ENV_NAME, "-y",
            "numpy", "pandas", "matplotlib", "seaborn", 
            "scikit-learn", "jupyter", "ipykernel",
            "-c", "conda-forge"
        ])
        
        # Install pip packages
        print("Installing pip packages...")
        pip_cmd = [
            "conda", "run", "-n", ENV_NAME, "pip", "install",
            "tensorflow[and-cuda]>=2.15.0",
            "opencv-python>=4.11.0.86",
            "--upgrade"
        ]
        run_command(pip_cmd)
    
    # Register as Jupyter kernel
    print("Registering environment as Jupyter kernel...")
    kernel_cmd = [
        "conda", "run", "-n", ENV_NAME, "python", "-m", "ipykernel", "install",
        "--user", "--name", ENV_NAME, "--display-name", f"Python ({ENV_NAME})"
    ]
    run_command(kernel_cmd, check=False)  # Don't fail if kernel already exists
    
    return True

def verify_installation():
    """Verify that TensorFlow and GPU are working"""
    print("\n" + "=" * 60)
    print("Verifying Installation")
    print("=" * 60)
    
    verify_cmd = [
        "conda", "run", "-n", ENV_NAME, "python", "-c",
        "import tensorflow as tf; "
        "gpus = tf.config.list_physical_devices('GPU'); "
        "print(f'TensorFlow version: {tf.__version__}'); "
        "print(f'GPU Available: {len(gpus) > 0}'); "
        "print(f'GPU Devices: {len(gpus)}'); "
        "for gpu in gpus: print(f'  - {gpu}')"
    ]
    
    try:
        run_command(verify_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not verify GPU support: {e}")
        print("You may need to restart your terminal or install CUDA separately.")

def main():
    """Main setup function"""
    if not check_conda():
        sys.exit(1)
    
    if create_environment():
        verify_installation()
        
        print("\n" + "=" * 60)
        print("Environment setup complete!")
        print("=" * 60)
        print(f"\nTo activate the environment, run:")
        print(f"  conda activate {ENV_NAME}")
        print(f"\nTo use in Jupyter:")
        print(f"  1. conda activate {ENV_NAME}")
        print(f"  2. jupyter notebook")
        print(f"  3. Select kernel: Python ({ENV_NAME})")
    else:
        print("Setup cancelled or failed.")

if __name__ == "__main__":
    main()

