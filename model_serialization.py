"""
Model Serialization using joblib
Saves and loads complete model packages including model, config, and metadata
"""

import joblib
import json
from pathlib import Path
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import warnings


class ModelSerializer:
    """
    Serialize and deserialize complete model packages using joblib
    """
    
    def __init__(self, base_dir: Path = Path('models/saved_models')):
        """
        Initialize ModelSerializer
        
        Args:
            base_dir: Base directory for saving models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: tf.keras.Model,
        hyperparameters: Dict[str, Any],
        class_labels: Dict[int, str],
        preprocessing_func: callable,
        augmentation_config: Dict[str, Any],
        training_history: Dict[str, list],
        validation_metrics: Dict[str, float],
        model_version: str = "1.0",
        description: Optional[str] = None,
        compress: bool = True
    ) -> Path:
        """
        Save complete model package using joblib
        
        Args:
            model: Trained Keras model
            hyperparameters: Dictionary of hyperparameters (IMG_SIZE, BATCH_SIZE, etc.)
            class_labels: Dictionary mapping class indices to labels
            preprocessing_func: Function for preprocessing images
            augmentation_config: Data augmentation configuration
            training_history: Training history dictionary
            validation_metrics: Validation metrics (accuracy, loss, etc.)
            model_version: Version string (e.g., "1.0", "1.1")
            description: Optional description of the model
            compress: Whether to compress the saved file
            
        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"covid19_model_v{model_version}_{timestamp}"
        model_path = self.base_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        model_package = {
            'model': model,
            'model_architecture': model.to_json(),
            'hyperparameters': hyperparameters,
            'class_labels': class_labels,
            'preprocessing_func': preprocessing_func,
            'augmentation_config': augmentation_config,
            'training_history': training_history,
            'validation_metrics': validation_metrics,
            'metadata': {
                'version': model_version,
                'created_at': timestamp,
                'created_date': datetime.now().isoformat(),
                'tensorflow_version': tf.__version__,
                'description': description,
                'model_type': 'Custom CNN with Residual Connections and SE Attention',
                'num_classes': len(class_labels),
                'input_shape': list(model.input_shape[1:]) if model.input_shape else None,
            }
        }
        
        model_file = model_path / f"{model_name}.joblib"
        if compress:
            joblib.dump(model_package, model_file, compress=('gzip', 3))
        else:
            joblib.dump(model_package, model_file)
        
        # Also save human-readable metadata as JSON
        metadata_file = model_path / "metadata.json"
        readable_metadata = {
            'version': model_version,
            'created_at': timestamp,
            'created_date': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'description': description,
            'model_type': 'Custom CNN with Residual Connections and SE Attention',
            'num_classes': len(class_labels),
            'input_shape': list(model.input_shape[1:]) if model.input_shape else None,
            'hyperparameters': hyperparameters,
            'class_labels': class_labels,
            'validation_metrics': validation_metrics,
            'augmentation_config': augmentation_config,
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(readable_metadata, f, indent=2)
        
        if training_history:
            # Handle training histories with different lengths
            # Check if all arrays have the same length
            lengths = [len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in training_history.values()]
            if len(set(lengths)) > 1:
                # Pad shorter arrays with NaN
                max_length = max(lengths)
                padded_history = {}
                for key, values in training_history.items():
                    if isinstance(values, (list, np.ndarray)):
                        current_length = len(values)
                        if current_length < max_length:
                            padding = [np.nan] * (max_length - current_length)
                            padded_history[key] = list(values) + padding
                        else:
                            padded_history[key] = values
                    else:
                        padded_history[key] = values
                training_history = padded_history
            
            history_df = pd.DataFrame(training_history)
            history_file = model_path / "training_history.csv"
            history_df.to_csv(history_file, index=False)
        
        print(f"   Model saved successfully!")
        print(f"   Location: {model_file}")
        print(f"   Size: {model_file.stat().st_size / (1024*1024):.2f} MB")
        print(f"   Version: {model_version}")
        if 'accuracy' in validation_metrics:
            print(f"   Validation Accuracy: {validation_metrics['accuracy']:.4f}")
        if 'loss' in validation_metrics:
            print(f"   Validation Loss: {validation_metrics['loss']:.4f}")
        
        return model_file
    
    def load_model(self, model_path: Path) -> Dict[str, Any]:
        """
        Load complete model package from joblib file
        
        Args:
            model_path: Path to .joblib file or model directory
            
        Returns:
            Dictionary containing:
            - 'model': Keras model
            - 'config': Configuration dictionary
            - 'metadata': Model metadata
            - 'preprocessing_func': Preprocessing function
            - 'class_labels': Class labels mapping
        """
        if model_path.is_dir():
            joblib_files = list(model_path.glob("*.joblib"))
            if not joblib_files:
                raise FileNotFoundError(f"No .joblib file found in {model_path}")
            model_file = joblib_files[0]
        else:
            model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        print(f"Loading model from: {model_file}")
        
        model_package = joblib.load(model_file)
        
        if 'model' in model_package:
            model = model_package['model']
        elif 'model_architecture' in model_package:
            model = tf.keras.models.model_from_json(model_package['model_architecture'])
            if 'model_weights_path' in model_package:
                model.load_weights(model_package['model_weights_path'])
        else:
            raise ValueError("Model not found in package")
        
        loaded_package = {
            'model': model,
            'config': {
                'hyperparameters': model_package.get('hyperparameters', {}),
                'class_labels': model_package.get('class_labels', {}),
                'augmentation_config': model_package.get('augmentation_config', {}),
            },
            'metadata': model_package.get('metadata', {}),
            'preprocessing_func': model_package.get('preprocessing_func'),
            'training_history': model_package.get('training_history', {}),
            'validation_metrics': model_package.get('validation_metrics', {}),
        }
        
        print(f" Model loaded successfully!")
        print(f"   Version: {loaded_package['metadata'].get('version', 'Unknown')}")
        print(f"   Created: {loaded_package['metadata'].get('created_date', 'Unknown')}")
        print(f"   Input shape: {loaded_package['metadata'].get('input_shape', 'Unknown')}")
        
        return loaded_package
    
    def list_saved_models(self) -> list:
        model_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        return sorted(model_dirs, reverse=True)  # Most recent first
    
    def get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """
        Get metadata for a saved model without loading the full model
        
        Args:
            model_path: Path to model directory or .joblib file
            
        Returns:
            Dictionary with model metadata
        """
        if model_path.is_dir():
            metadata_file = model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        
        # If no metadata.json, load just metadata from joblib
        if model_path.is_dir():
            joblib_files = list(model_path.glob("*.joblib"))
            if joblib_files:
                model_file = joblib_files[0]
            else:
                raise FileNotFoundError(f"No .joblib file found in {model_path}")
        else:
            model_file = model_path
        
        # Load only metadata (this still loads the whole file, but we can optimize)
        model_package = joblib.load(model_file)
        return model_package.get('metadata', {})


def predict_with_preprocessing(
    model_package: Dict[str, Any],
    image_path: str,
    return_proba: bool = False
) -> Tuple[int, str, Optional[np.ndarray]]:
    """
    Convenience function to predict on a single image using loaded model package
    
    Args:
        model_package: Loaded model package from load_model()
        image_path: Path to image file
        return_proba: Whether to return probability distribution
        
    Returns:
        Tuple of (predicted_class_index, predicted_class_label, probabilities)
    """
    model = model_package['model']
    preprocessing_func = model_package['preprocessing_func']
    class_labels = model_package['config']['class_labels']
    
    img = preprocessing_func(image_path)
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_label = class_labels.get(predicted_class_idx, f"Class {predicted_class_idx}")
    
    if return_proba:
        return predicted_class_idx, predicted_class_label, predictions[0]
    else:
        return predicted_class_idx, predicted_class_label, None


# Example usage functions
# def save_model_example(
#     model: tf.keras.Model,
#     IMG_SIZE: int,
#     BATCH_SIZE: int,
#     LEARNING_RATE: float,
#     NUM_CLASSES: int,
#     CLASS_LABELS: Dict[int, str],
#     load_and_preprocess_image: callable,
#     train_datagen_config: Dict[str, Any],
#     history: tf.keras.callbacks.History,
#     val_accuracy: float,
#     val_loss: float
# ):
#     """
#     Example function showing how to use ModelSerializer
    
#     This would be called after training in your notebook
    
#     Args:
#         model: Trained Keras model
#         IMG_SIZE: Image size used for training
#         BATCH_SIZE: Batch size used for training
#         LEARNING_RATE: Learning rate used for training
#         NUM_CLASSES: Number of classes
#         CLASS_LABELS: Dictionary mapping class indices to labels
#         load_and_preprocess_image: Preprocessing function
#         train_datagen_config: Data augmentation configuration
#         history: Training history from model.fit()
#         val_accuracy: Validation accuracy
#         val_loss: Validation loss
        
#     Returns:
#         Path to saved model file
#     """
#     serializer = ModelSerializer()
    
#     augmentation_config = {
#         'rotation_range': train_datagen_config.get('rotation_range', 15),
#         'width_shift_range': train_datagen_config.get('width_shift_range', 0.1),
#         'height_shift_range': train_datagen_config.get('height_shift_range', 0.1),
#         'shear_range': train_datagen_config.get('shear_range', 0.1),
#         'zoom_range': train_datagen_config.get('zoom_range', 0.1),
#         'horizontal_flip': train_datagen_config.get('horizontal_flip', True),
#         'fill_mode': train_datagen_config.get('fill_mode', 'nearest'),
#     }
    
#     model_file = serializer.save_model(
#         model=model,
#         hyperparameters={
#             'IMG_SIZE': IMG_SIZE,
#             'BATCH_SIZE': BATCH_SIZE,
#             'LEARNING_RATE': LEARNING_RATE,
#             'NUM_CLASSES': NUM_CLASSES,
#         },
#         class_labels=CLASS_LABELS,
#         preprocessing_func=load_and_preprocess_image,
#         augmentation_config=augmentation_config,
#         training_history=history.history if hasattr(history, 'history') else {},
#         validation_metrics={
#             'accuracy': val_accuracy,
#             'loss': val_loss,
#         },
#         model_version="1.0",
#         description="Custom CNN with Residual Connections and SE Attention for COVID-19 classification",
#         compress=True
#     )
    
#     return model_file


# def load_model_example(model_path: str):
#     """
#     Example function showing how to load a saved model
    
#     Args:
#         model_path: Path to model directory or .joblib file
        
#     Returns:
#         Model package dictionary
#     """
#     serializer = ModelSerializer()
#     model_package = serializer.load_model(Path(model_path))
    
#     model = model_package['model']
#     config = model_package['config']
#     metadata = model_package['metadata']
#     preprocessing_func = model_package['preprocessing_func']
    
#     return model_package

