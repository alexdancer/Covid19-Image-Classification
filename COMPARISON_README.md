# Model Comparison Script

This directory contains a standalone script for comparing Custom CNN and Transfer Learning model performance.

## Files

- `compare_models.py` - Main comparison script with ModelComparator class
- `run_comparison.py` - Example usage script
- `COMPARISON_README.md` - This file

## Quick Start

### Option 1: Run with default metrics (from notebook)

```python
from compare_models import ModelComparator, create_default_custom_cnn_metrics, create_default_transfer_learning_metrics

# Load default metrics
custom_cnn_metrics = create_default_custom_cnn_metrics()
transfer_learning_metrics = create_default_transfer_learning_metrics()

# Class labels
class_labels = {
    0: 'Negative for Pneumonia',
    1: 'Typical Appearance',
    2: 'Indeterminate Appearance',
    3: 'Atypical Appearance'
}

# Create comparator
comparator = ModelComparator(
    custom_cnn_metrics=custom_cnn_metrics,
    transfer_learning_metrics=transfer_learning_metrics,
    class_labels=class_labels,
    output_dir=Path('comparison_results')
)

# Run comparison
comparator.run_full_comparison()
```

### Option 2: Run from command line

```bash
# Using default metrics
python compare_models.py

# Using metrics from JSON files
python compare_models.py \
    --custom-cnn-json path/to/custom_cnn_metrics.json \
    --transfer-learning-json path/to/transfer_learning_metrics.json \
    --output-dir comparison_results
```

### Option 3: Load from saved model packages

```python
from model_serialization import ModelSerializer
from compare_models import ModelComparator

serializer = ModelSerializer()

# Load models
custom_cnn_package = serializer.load_model(Path('models/saved_models/custom_cnn_model'))
tl_package = serializer.load_model(Path('models/saved_models/transfer_learning_model'))

# Extract metrics
custom_cnn_metrics = custom_cnn_package['validation_metrics']
transfer_learning_metrics = tl_package['validation_metrics']
class_labels = custom_cnn_package['class_labels']

# Run comparison
comparator = ModelComparator(
    custom_cnn_metrics=custom_cnn_metrics,
    transfer_learning_metrics=transfer_learning_metrics,
    class_labels=class_labels
)
comparator.run_full_comparison()
```

## Metrics Format

The metrics dictionaries should have the following structure:

```python
metrics = {
    'accuracy': 0.6093,
    'precision_weighted': 0.4581,
    'recall_weighted': 0.6093,
    'f1_weighted': 0.5183,
    'precision_macro': 0.31,
    'recall_macro': 0.39,
    'f1_macro': 0.34,
    'per_class': {
        'Negative for Pneumonia': {
            'precision': 0.63,
            'recall': 0.64,
            'f1': 0.64
        },
        'Typical Appearance': {
            'precision': 0.60,
            'recall': 0.91,
            'f1': 0.72
        },
        # ... etc for all classes
    }
}
```

## Output

The comparison script generates:

1. **Console Output**: 
   - Overall metrics comparison table
   - Per-class metrics comparison table
   - Summary of improvements

2. **Saved Files** (in `comparison_results/` directory):
   - `overall_metrics_comparison.csv` - Overall metrics comparison table
   - `per_class_metrics_comparison.csv` - Per-class metrics comparison table
   - `overall_metrics_comparison.png` - Visualization of overall metrics
   - `per_class_metrics_comparison.png` - Visualization of per-class metrics
   - `comparison_summary.txt` - Text summary of improvements

## Features

- **Overall Metrics Comparison**: Accuracy, Precision, Recall, F1-Score (weighted and macro)
- **Per-Class Metrics Comparison**: Precision, Recall, F1-Score for each class
- **Visualizations**: Bar charts comparing both models
- **Improvement Analysis**: Calculates absolute and percentage improvements
- **Summary Report**: Key findings and recommendations

## Updating Metrics

To use your own metrics:

1. **From Notebook**: After running evaluation, extract metrics:
   ```python
   custom_cnn_metrics = {
       'accuracy': accuracy,
       'precision_weighted': precision,
       'recall_weighted': recall,
       'f1_weighted': f1,
       'precision_macro': precision_macro,
       'recall_macro': recall_macro,
       'f1_macro': f1_macro,
       'per_class': {
           'Negative for Pneumonia': {'precision': ..., 'recall': ..., 'f1': ...},
           # ... etc
       }
   }
   ```

2. **From Saved Models**: Load using ModelSerializer (see Option 3 above)

3. **From JSON**: Save metrics to JSON and load:
   ```python
   import json
   with open('metrics.json', 'w') as f:
       json.dump(metrics, f, indent=2)
   ```

## Integration with Notebook

You can import and use this in your notebook:

```python
# In your notebook, after evaluation
from compare_models import ModelComparator

# Build metrics dictionaries from your evaluation results
custom_cnn_metrics = {...}  # Your custom CNN metrics
transfer_learning_metrics = {...}  # Your transfer learning metrics

# Run comparison
comparator = ModelComparator(
    custom_cnn_metrics=custom_cnn_metrics,
    transfer_learning_metrics=transfer_learning_metrics,
    class_labels=CLASS_LABELS
)
comparator.run_full_comparison()
```

This allows you to keep the comparison code separate from your training/evaluation notebook.

