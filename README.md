# COVID-19 Chest X-Ray Image Classification

A deep learning project for classifying COVID-19 chest X-ray images into four categories using Convolutional Neural Networks (CNNs) and Transfer Learning techniques.

## Overview

This project implements and compares two deep learning approaches for COVID-19 chest X-ray classification:

1. **Custom CNN Model**: A baseline convolutional neural network built from scratch with residual connections and SE (Squeeze-and-Excitation) attention mechanisms
2. **Transfer Learning Model**: A ResNet50-based model using pre-trained weights from ImageNet with fine-tuning

### Classification Categories

The models classify chest X-ray images into four categories:
- **Negative for Pneumonia** (Class 0)
- **Typical Appearance** (Class 1)
- **Indeterminate Appearance** (Class 2)
- **Atypical Appearance** (Class 3)

## Features

- **Data Preparation Pipeline**: Automated data loading, preprocessing, and train/validation splitting
- **Custom CNN Architecture**: Baseline model with residual connections and attention mechanisms
- **Transfer Learning**: ResNet50-based model with two-phase training (feature extraction + fine-tuning)
- **Model Comparison Tools**: Comprehensive comparison scripts with visualizations and metrics
- **Model Serialization**: Save and load trained models with metadata
- **GPU Support**: Optimized for CUDA-enabled GPUs with TensorFlow
- **Class Imbalance Handling**: Built-in support for class weights and balanced loss functions

## Dataset

**SIIM COVID-19 Dataset** - 256px JPG Images
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/shanmukh05/siim-covid19-dataset-256px-jpg/data)
- **Image Size**: 256x256 pixels
- **Format**: JPG

### Dataset Structure

```
data/
├── 256px/
│   ├── train/
│   │   └── train/          # Training images
│   └── test/
│       └── test/           # Test images
├── train.csv               # Training labels
├── meta_train.csv          # Training metadata
├── meta_test.csv           # Test metadata
├── train_split.csv         # Generated train split
└── val_split.csv           # Generated validation split
```

## Quick Start

### Prerequisites

- Python 3.10
- Conda (recommended) or pip
- NVIDIA GPU with CUDA support (optional, but recommended for faster training)
- CUDA 11.8 or 12.x (for GPU support)
- cuDNN 8.x (matching CUDA version)

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Covid19-Project

# Create and activate conda environment
conda env create -f environment.yml
conda activate Covid19-env

# Register as Jupyter kernel
python -m ipykernel install --user --name Covid19-env --display-name "Python (Covid19-env)"
```

#### Option 2: Using Setup Scripts

**Linux/Mac/WSL:**
```bash
chmod +x setup_conda_env.sh
./setup_conda_env.sh
```

**Python Script:**
```bash
python setup_conda_env.py
```

#### Option 3: Using pip

```bash
pip install -r requirements.txt
```

### Verify GPU Support

After installation, verify TensorFlow can detect your GPU:

```bash
conda activate Covid19-env
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## Usage

### 1. Data Preparation

Run the data preparation notebook to prepare and split the dataset:

```bash
jupyter notebook Covid19_DataPrep.ipynb
```

This notebook will:
- Load the dataset from the `data/` directory
- Create train/validation splits
- Generate `train_split.csv` and `val_split.csv`

### 2. Train Custom CNN Model

Train the baseline custom CNN model:

```bash
jupyter notebook Covid19_Image_Modeling_1.ipynb
```

**Key Features:**
- Custom CNN architecture with residual connections
- SE (Squeeze-and-Excitation) attention mechanisms
- Image augmentation (rotation, shifts, zoom, flip)
- Model checkpointing and early stopping
- Comprehensive evaluation metrics

### 3. Train Transfer Learning Model

Train the ResNet50-based transfer learning model:

```bash
jupyter notebook Covid19_Transfer_Learning.ipynb
```

**Training Strategy:**
- **Phase 1**: Freeze ResNet50 base, train only classification head (30 epochs)
- **Phase 2**: Unfreeze top layers, fine-tune with lower learning rate (20 epochs)
- Class weight balancing for imbalanced datasets

### 4. Compare Models

Compare the performance of both models:

```bash
# Using the comparison script
python compare_models.py

# Or in Python
python -c "
from compare_models import ModelComparator, create_default_custom_cnn_metrics, create_default_transfer_learning_metrics

custom_cnn_metrics = create_default_custom_cnn_metrics()
transfer_learning_metrics = create_default_transfer_learning_metrics()

class_labels = {
    0: 'Negative for Pneumonia',
    1: 'Typical Appearance',
    2: 'Indeterminate Appearance',
    3: 'Atypical Appearance'
}

comparator = ModelComparator(
    custom_cnn_metrics=custom_cnn_metrics,
    transfer_learning_metrics=transfer_learning_metrics,
    class_labels=class_labels,
    output_dir='comparison_results'
)

comparator.run_full_comparison()
"
```

The comparison generates:
- Overall metrics comparison (accuracy, precision, recall, F1-score)
- Per-class metrics comparison
- Visualizations (bar charts)
- Summary report

## Project Structure

```
Covid19-Project/
├── data/                           # Dataset directory
│   ├── 256px/                     # Image files
│   ├── train.csv                  # Training labels
│   ├── meta_train.csv             # Training metadata
│   ├── meta_test.csv              # Test metadata
│   ├── train_split.csv            # Generated train split
│   └── val_split.csv              # Generated validation split
│
├── models/                         # Model directory
│   ├── saved_models/              # Saved model packages
│   └── features/                  # Extracted features
│
├── comparison_results/             # Model comparison outputs
│   ├── overall_metrics_comparison.csv
│   ├── per_class_metrics_comparison.csv
│   ├── overall_metrics_comparison.png
│   ├── per_class_metrics_comparison.png
│   └── comparison_summary.txt
│
├── notebooks/
│   ├── Covid19_DataPrep.ipynb                    # Data preparation
│   ├── Covid19_Image_Modeling_1.ipynb            # Custom CNN training
│   ├── Covid19_Transfer_Learning.ipynb           # Transfer learning training
│   └── Model_Comparison.ipynb                    # Model comparison
│
├── compare_models.py              # Model comparison script
├── model_serialization.py         # Model save/load utilities
├── setup_conda_env.py             # Python setup script
├── setup_conda_env.sh             # Shell setup script
├── read_logs.py                   # Log reading utilities
│
├── environment.yml                # Conda environment file
├── requirements.txt               # pip requirements
├── SETUP_INSTRUCTIONS.md          # Detailed setup guide
├── COMPARISON_README.md           # Comparison tool documentation
└── README.md                      # This file
```

## Configuration

### Model Hyperparameters

**Custom CNN:**
- Image Size: 256x256
- Batch Size: 64
- Learning Rate: 0.001
- Epochs: 50
- Number of Classes: 4

**Transfer Learning:**
- Image Size: 256x256
- Batch Size: 32
- Learning Rate (Phase 1): 0.001
- Learning Rate (Phase 2): 1e-5
- Epochs (Phase 1): 30
- Epochs (Phase 2): 20
- Base Model: ResNet50

### Data Augmentation

Both models use the following augmentation:
- Rotation: ±15 degrees
- Width/Height Shift: 0.1
- Shear: 0.1
- Zoom: 0.1
- Horizontal Flip: Enabled

## Model Performance

The project includes tools to compare model performance across multiple metrics:

- **Overall Metrics**: Accuracy, Precision (weighted/macro), Recall (weighted/macro), F1-Score (weighted/macro)
- **Per-Class Metrics**: Precision, Recall, F1-Score for each class
- **Visualizations**: Bar charts comparing both models

## Dependencies

### Core Dependencies

- **TensorFlow** (>=2.15.0) - Deep learning framework with GPU support
- **NumPy** (>=1.26.4) - Numerical computing
- **Pandas** (>=2.2.2) - Data manipulation
- **Scikit-learn** (>=1.2.2) - Machine learning utilities
- **Matplotlib** (>=3.10.0) - Plotting
- **Seaborn** (>=0.13.2) - Statistical visualization
- **OpenCV** (>=4.11.0.86) - Image processing
- **Jupyter** - Notebook environment
- **Joblib** (>=1.3.0) - Model serialization

See `requirements.txt` or `environment.yml` for complete dependency list.

## Notebooks

### Covid19_DataPrep.ipynb
- Loads and explores the dataset
- Creates train/validation splits
- Generates data statistics and visualizations

### Covid19_Image_Modeling_1.ipynb
- Implements custom CNN architecture
- Trains baseline model from scratch
- Evaluates model performance
- Saves model and metadata

### Covid19_Transfer_Learning.ipynb
- Implements ResNet50 transfer learning
- Two-phase training strategy
- Handles class imbalance
- Evaluates and saves model

### Model_Comparison.ipynb
- Compares both models
- Generates comparison visualizations
- Creates summary reports

## Troubleshooting

### GPU Not Detected

1. **Restart the kernel** after installing TensorFlow
2. **Check NVIDIA drivers**: Run `nvidia-smi` in terminal
3. **For WSL2**: Ensure GPU passthrough is enabled
4. **Alternative**: Install TensorFlow CPU-only version if GPU is unavailable

### Environment Issues

If you encounter package conflicts:
```bash
# Remove existing environment
conda env remove -n Covid19-env

# Recreate environment
conda env create -f environment.yml
conda activate Covid19-env
```

### Dataset Issues

Ensure the dataset is properly downloaded and extracted:
- Download from [Kaggle](https://www.kaggle.com/datasets/shanmukh05/siim-covid19-dataset-256px-jpg/data)
- Extract to `data/` directory
- Verify CSV files are present

## Acknowledgments

- **Dataset**: [SIIM COVID-19 Dataset](https://www.kaggle.com/datasets/shanmukh05/siim-covid19-dataset-256px-jpg/data) on Kaggle
- **Base Model**: ResNet50 from Keras Applications
- **Framework**: TensorFlow/Keras