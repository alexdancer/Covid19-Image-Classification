"""
TensorBoard Log Reader Script
Reads and analyzes TensorBoard event files (.v2) from model training
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import argparse
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None


def extract_tensor_value(tensor_proto):
    """
    Extract scalar value from tensor proto
    
    Args:
        tensor_proto: TensorProto object from TensorBoard
        
    Returns:
        Scalar value (float)
    """
    if tf is None:
        # Fallback: try to extract from tensor proto directly
        # TensorProto has a float_val field for scalar tensors
        if hasattr(tensor_proto, 'float_val') and len(tensor_proto.float_val) > 0:
            return float(tensor_proto.float_val[0])
        elif hasattr(tensor_proto, 'double_val') and len(tensor_proto.double_val) > 0:
            return float(tensor_proto.double_val[0])
        elif hasattr(tensor_proto, 'int_val') and len(tensor_proto.int_val) > 0:
            return float(tensor_proto.int_val[0])
        else:
            # Try to parse tensor content
            import struct
            if hasattr(tensor_proto, 'tensor_content'):
                # For float32 scalars
                if tensor_proto.dtype == 1:  # DT_FLOAT
                    return struct.unpack('f', tensor_proto.tensor_content[:4])[0]
                elif tensor_proto.dtype == 2:  # DT_DOUBLE
                    return struct.unpack('d', tensor_proto.tensor_content[:8])[0]
    else:
        # Use TensorFlow to convert
        tensor = tf.make_ndarray(tensor_proto)
        if tensor.size == 1:
            return float(tensor.item())
        else:
            return float(tensor.flatten()[0])
    
    return 0.0


def read_tensorboard_logs(log_dir):
    """
    Read TensorBoard event files and return all metrics as a dictionary of DataFrames
    
    Args:
        log_dir: Path to directory containing TensorBoard event files
        
    Returns:
        Dictionary where keys are metric names and values are DataFrames with columns:
        ['step', 'value', 'wall_time']
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    metrics = {}
    
    # Get all scalar metrics
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = pd.DataFrame([
            {
                'step': event.step,
                'value': event.value,
                'wall_time': event.wall_time
            }
            for event in events
        ])
    
    # Get all tensor metrics (these are often used for metrics in newer TensorFlow versions)
    for tag in ea.Tags()['tensors']:
        try:
            tensor_events = ea.Tensors(tag)
            values = []
            for event in tensor_events:
                try:
                    value = extract_tensor_value(event.tensor_proto)
                    values.append({
                        'step': event.step,
                        'value': value,
                        'wall_time': event.wall_time
                    })
                except Exception as e:
                    # Skip events that can't be parsed
                    continue
            
            if values:
                metrics[tag] = pd.DataFrame(values)
        except Exception as e:
            # Skip tags that can't be read
            continue
    
    return metrics


def inspect_logs(log_dir):
    """
    Inspect TensorBoard logs and print summary information
    
    Args:
        log_dir: Path to directory containing TensorBoard event files
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    print(f"\n{'='*60}")
    print(f"Log Directory: {log_dir}")
    print(f"{'='*60}")
    
    # Read all metrics (scalars and tensors)
    metrics = read_tensorboard_logs(log_dir)
    
    print(f"\nAvailable Metrics ({len(metrics)}):")
    
    if not metrics:
        print("  No metrics found!")
        print(f"  Scalars: {ea.Tags()['scalars']}")
        print(f"  Tensors: {ea.Tags()['tensors']}")
        return ea
    
    for tag, df in metrics.items():
        if len(df) > 0:
            values = df['value'].values
            print(f"  - {tag}:")
            print(f"    Data points: {len(df)}")
            print(f"    Step range: {df['step'].min()} to {df['step'].max()}")
            print(f"    Value range: {values.min():.4f} to {values.max():.4f}")
            print(f"    Final value: {df['value'].iloc[-1]:.4f}")
    
    return ea


def plot_metric(train_dir, val_dir, metric_name, save_path=None):
    """
    Plot a specific metric for both training and validation
    
    Args:
        train_dir: Path to training logs directory
        val_dir: Path to validation logs directory
        metric_name: Name of the metric to plot (e.g., 'accuracy', 'loss')
        save_path: Optional path to save the plot
    """
    # Load logs
    train_metrics = read_tensorboard_logs(train_dir)
    val_metrics = read_tensorboard_logs(val_dir)
    
    if metric_name not in train_metrics and metric_name not in val_metrics:
        print(f"Metric '{metric_name}' not found in logs")
        print(f"Available training metrics: {list(train_metrics.keys())}")
        print(f"Available validation metrics: {list(val_metrics.keys())}")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Overlay view
    if metric_name in train_metrics:
        train_df = train_metrics[metric_name]
        axes[0].plot(train_df['step'], train_df['value'], 
                    label='Training', marker='o', markersize=2, alpha=0.7)
    
    if metric_name in val_metrics:
        val_df = val_metrics[metric_name]
        axes[0].plot(val_df['step'], val_df['value'], 
                    label='Validation', marker='s', markersize=2, alpha=0.7)
    
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel(metric_name.title(), fontsize=12)
    axes[0].set_title(f'{metric_name.title()} Over Time', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Side by side
    if metric_name in train_metrics:
        train_df = train_metrics[metric_name]
        axes[1].plot(train_df['step'], train_df['value'], 
                    label='Training', marker='o', markersize=2)
    
    if metric_name in val_metrics:
        val_df = val_metrics[metric_name]
        axes[1].plot(val_df['step'], val_df['value'], 
                    label='Validation', marker='s', markersize=2)
    
    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel(metric_name.title(), fontsize=12)
    axes[1].set_title('Detailed View', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def compare_all_metrics(train_dir, val_dir, save_dir=None):
    """
    Compare all available metrics between training and validation
    
    Args:
        train_dir: Path to training logs directory
        val_dir: Path to validation logs directory
        save_dir: Optional directory to save plots
    """
    train_metrics = read_tensorboard_logs(train_dir)
    val_metrics = read_tensorboard_logs(val_dir)
    
    # Find common metrics
    common_metrics = set(train_metrics.keys()) & set(val_metrics.keys())
    
    if not common_metrics:
        print("No common metrics found between training and validation logs")
        return
    
    num_metrics = len(common_metrics)
    cols = 2
    rows = (num_metrics + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric_name in enumerate(sorted(common_metrics)):
        train_df = train_metrics[metric_name]
        val_df = val_metrics[metric_name]
        
        axes[idx].plot(train_df['step'], train_df['value'], 
                      label='Training', marker='o', markersize=2, alpha=0.7)
        axes[idx].plot(val_df['step'], val_df['value'], 
                      label='Validation', marker='s', markersize=2, alpha=0.7)
        axes[idx].set_xlabel('Step', fontsize=10)
        axes[idx].set_ylabel(metric_name.title(), fontsize=10)
        axes[idx].set_title(metric_name.title(), fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'all_metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def export_to_csv(log_dir, output_path):
    """
    Export all metrics from TensorBoard logs to CSV files
    
    Args:
        log_dir: Path to directory containing TensorBoard event files
        output_path: Path to output CSV file (or directory for multiple files)
    """
    metrics = read_tensorboard_logs(log_dir)
    
    if not metrics:
        print(f"No metrics found in {log_dir}")
        return
    
    # If output_path is a directory, save separate CSV for each metric
    if os.path.isdir(output_path) or output_path.endswith('/'):
        os.makedirs(output_path, exist_ok=True)
        for metric_name, df in metrics.items():
            csv_path = os.path.join(output_path, f"{metric_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Exported {metric_name} to {csv_path}")
    else:
        # Single CSV with all metrics (wide format)
        # Combine all metrics into one DataFrame
        combined = None
        for metric_name, df in metrics.items():
            df_renamed = df.rename(columns={'value': metric_name})
            if combined is None:
                combined = df_renamed[['step', metric_name]]
            else:
                combined = combined.merge(
                    df_renamed[['step', metric_name]], 
                    on='step', 
                    how='outer'
                )
        
        combined = combined.sort_values('step')
        combined.to_csv(output_path, index=False)
        print(f"Exported all metrics to {output_path}")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Read and analyze TensorBoard event files (.v2)'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default='models/logs/train',
        help='Path to training logs directory (default: models/logs/train)'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default='models/logs/validation',
        help='Path to validation logs directory (default: models/logs/validation)'
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect logs and print summary'
    )
    parser.add_argument(
        '--plot',
        type=str,
        help='Plot a specific metric (e.g., accuracy, loss)'
    )
    parser.add_argument(
        '--plot-all',
        action='store_true',
        help='Plot all available metrics'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export metrics to CSV (provide output path)'
    )
    parser.add_argument(
        '--export-dir',
        type=str,
        help='Export each metric to separate CSV files in directory'
    )
    
    args = parser.parse_args()
    
    # Inspect logs
    if args.inspect:
        print("\n" + "="*60)
        print("TRAINING LOGS")
        print("="*60)
        inspect_logs(args.train_dir)
        
        print("\n" + "="*60)
        print("VALIDATION LOGS")
        print("="*60)
        inspect_logs(args.val_dir)
    
    # Plot specific metric
    if args.plot:
        plot_metric(args.train_dir, args.val_dir, args.plot)
    
    # Plot all metrics
    if args.plot_all:
        compare_all_metrics(args.train_dir, args.val_dir)
    
    # Export to CSV
    if args.export:
        export_to_csv(args.train_dir, args.export)
        export_to_csv(args.val_dir, args.export.replace('.csv', '_val.csv'))
    
    if args.export_dir:
        export_to_csv(args.train_dir, os.path.join(args.export_dir, 'train'))
        export_to_csv(args.val_dir, os.path.join(args.export_dir, 'validation'))
    
    # If no arguments, show help
    if not any([args.inspect, args.plot, args.plot_all, args.export, args.export_dir]):
        parser.print_help()
        print("\n" + "="*60)
        print("QUICK START EXAMPLES")
        print("="*60)
        print("\n1. Inspect logs:")
        print("   python read_logs.py --inspect")
        print("\n2. Plot accuracy:")
        print("   python read_logs.py --plot accuracy")
        print("\n3. Plot all metrics:")
        print("   python read_logs.py --plot-all")
        print("\n4. Export to CSV:")
        print("   python read_logs.py --export metrics.csv")
        print("\n5. Export to directory (separate files):")
        print("   python read_logs.py --export-dir exported_metrics/")


if __name__ == "__main__":
    main()

