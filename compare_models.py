"""
Model Comparison Script
Compare Custom CNN baseline with Transfer Learning model performance

Generates comprehensive comparison visualizations and reports from model metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def normalize_metrics(metrics: Dict, class_labels: Dict) -> Dict:
    """
    Normalize metrics dictionary to standard format
    
    Handles different metric key formats:
    - Old format: 'precision', 'recall', 'f1_score'
    - New format: 'precision_weighted', 'recall_weighted', 'f1_weighted', etc.
    
    Args:
        metrics: Dictionary with metrics (may be in old or new format)
        class_labels: Dictionary mapping class indices to labels (can have int or str keys)
        
    Returns:
        Normalized metrics dictionary
    """
    normalized = metrics.copy() if metrics else {}
    
    # Normalize old format to new format (always ensure weighted metrics exist)
    if 'precision_weighted' not in normalized:
        normalized['precision_weighted'] = normalized.get('precision', 0.0)
    if 'recall_weighted' not in normalized:
        normalized['recall_weighted'] = normalized.get('recall', 0.0)
    if 'f1_weighted' not in normalized:
        normalized['f1_weighted'] = normalized.get('f1_score', 0.0)
    
    # Debug: verify keys were created
    if 'precision_weighted' not in normalized:
        raise ValueError(f"Failed to create precision_weighted. Available keys: {list(normalized.keys())}")
    
    # Set default macro metrics if missing (use weighted as fallback)
    if 'precision_macro' not in normalized:
        normalized['precision_macro'] = normalized.get('precision_weighted', 0.0)
    if 'recall_macro' not in normalized:
        normalized['recall_macro'] = normalized.get('recall_weighted', 0.0)
    if 'f1_macro' not in normalized:
        normalized['f1_macro'] = normalized.get('f1_weighted', 0.0)
    
    # Ensure accuracy exists
    if 'accuracy' not in normalized:
        normalized['accuracy'] = 0.0
    
    # Ensure per_class metrics exist (create empty structure if missing)
    if 'per_class' not in normalized:
        normalized['per_class'] = {}
    
    # Handle class_labels with string or int keys
    # Convert to list of class names
    if class_labels:
        # Try to get class names - handle both string and int keys
        class_names = []
        for i in range(len(class_labels)):
            # Try int key first, then string key
            class_name = class_labels.get(i) or class_labels.get(str(i))
            if class_name:
                class_names.append(class_name)
        
        # If that didn't work, just get all values
        if not class_names:
            class_names = list(class_labels.values())
        
        # Ensure all classes have per_class metrics
        for class_name in class_names:
            if class_name not in normalized['per_class']:
                normalized['per_class'][class_name] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
    
    return normalized


class ModelComparator:
    """Compare performance metrics between two models"""
    
    def __init__(self, custom_cnn_metrics: Dict, transfer_learning_metrics: Dict, 
                 class_labels: Dict, output_dir: Optional[Path] = None):
        """
        Initialize ModelComparator
        
        Args:
            custom_cnn_metrics: Dictionary with custom CNN metrics
            transfer_learning_metrics: Dictionary with transfer learning metrics
            class_labels: Dictionary mapping class indices to labels
            output_dir: Optional directory to save comparison results
        """
        # Normalize metrics to handle different formats
        self.custom_cnn_metrics = normalize_metrics(custom_cnn_metrics, class_labels)
        self.transfer_learning_metrics = normalize_metrics(transfer_learning_metrics, class_labels)
        
        # Verify normalization worked (debug check)
        required_keys = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 
                         'precision_macro', 'recall_macro', 'f1_macro']
        missing_cnn = [k for k in required_keys if k not in self.custom_cnn_metrics]
        missing_tl = [k for k in required_keys if k not in self.transfer_learning_metrics]
        
        if missing_cnn:
            raise ValueError(f"Custom CNN metrics missing required keys after normalization: {missing_cnn}. "
                           f"Available keys: {list(self.custom_cnn_metrics.keys())}")
        if missing_tl:
            raise ValueError(f"Transfer Learning metrics missing required keys after normalization: {missing_tl}. "
                           f"Available keys: {list(self.transfer_learning_metrics.keys())}")
        
        self.class_labels = class_labels
        self.output_dir = output_dir or Path('comparison_results')
        self.output_dir.mkdir(exist_ok=True)
        
        # Extract class names (handle both string and int keys)
        if class_labels:
            self.class_names = []
            for i in range(len(class_labels)):
                class_name = class_labels.get(i) or class_labels.get(str(i))
                if class_name:
                    self.class_names.append(class_name)
            # Fallback: just get all values if above didn't work
            if not self.class_names:
                self.class_names = list(class_labels.values())
        else:
            self.class_names = []
    
    def compare_overall_metrics(self) -> pd.DataFrame:
        """
        Compare overall metrics between models
        
        Returns:
            DataFrame with comparison results
        """
        print("=" * 60)
        print("OVERALL METRICS COMPARISON")
        print("=" * 60)
        
        # Use .get() with defaults for safety (normalization should have created these, but be defensive)
        comparison_data = {
            'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)',
                       'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
            'Custom CNN': [
                self.custom_cnn_metrics.get('accuracy', 0.0),
                self.custom_cnn_metrics.get('precision_weighted', 0.0),
                self.custom_cnn_metrics.get('recall_weighted', 0.0),
                self.custom_cnn_metrics.get('f1_weighted', 0.0),
                self.custom_cnn_metrics.get('precision_macro', 0.0),
                self.custom_cnn_metrics.get('recall_macro', 0.0),
                self.custom_cnn_metrics.get('f1_macro', 0.0)
            ],
            'Transfer Learning': [
                self.transfer_learning_metrics.get('accuracy', 0.0),
                self.transfer_learning_metrics.get('precision_weighted', 0.0),
                self.transfer_learning_metrics.get('recall_weighted', 0.0),
                self.transfer_learning_metrics.get('f1_weighted', 0.0),
                self.transfer_learning_metrics.get('precision_macro', 0.0),
                self.transfer_learning_metrics.get('recall_macro', 0.0),
                self.transfer_learning_metrics.get('f1_macro', 0.0)
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['Improvement'] = comparison_df['Transfer Learning'] - comparison_df['Custom CNN']
        comparison_df['Improvement %'] = (comparison_df['Improvement'] / comparison_df['Custom CNN'] * 100).round(2)
        
        print(comparison_df.to_string(index=False))
        print("=" * 60)
        
        # Save to CSV
        csv_path = self.output_dir / 'overall_metrics_comparison.csv'
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison saved to: {csv_path}")
        
        return comparison_df
    
    def visualize_overall_metrics(self, comparison_df: pd.DataFrame):
        """
        Visualize overall metrics comparison
        
        Args:
            comparison_df: DataFrame with comparison results
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)']
        
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            models = ['Custom CNN', 'Transfer Learning']
            
            # Get values from comparison_df
            row = comparison_df[comparison_df['Metric'] == metric_name.replace(' (Weighted)', '')]
            if len(row) == 0:
                row = comparison_df[comparison_df['Metric'] == metric_name]
            
            values = [
                row.iloc[0]['Custom CNN'],
                row.iloc[0]['Transfer Learning']
            ]
            
            colors = ['#d62728', '#2ca02c']
            bars = ax.bar(models, values, color=colors, alpha=0.7)
            ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Add improvement arrow
            improvement = values[1] - values[0]
            if improvement != 0:
                ax.annotate(f'{improvement:+.4f}', 
                           xy=(1, values[1]), xytext=(0.5, max(values) + 0.05),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                           fontsize=10, fontweight='bold',
                           ha='center')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'overall_metrics_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {fig_path}")
        
        plt.show()
    
    def compare_per_class_metrics(self) -> pd.DataFrame:
        """
        Compare per-class metrics between models
        
        Returns:
            DataFrame with per-class comparison results
        """
        print("\n" + "=" * 60)
        print("PER-CLASS METRICS COMPARISON")
        print("=" * 60)
        
        per_class_comparison = []
        for class_name in self.class_names:
            # Get per_class metrics with safe defaults
            cnn_per_class = self.custom_cnn_metrics.get('per_class', {}).get(class_name, {})
            tl_per_class = self.transfer_learning_metrics.get('per_class', {}).get(class_name, {})
            
            cnn_precision = cnn_per_class.get('precision', 0.0)
            cnn_recall = cnn_per_class.get('recall', 0.0)
            cnn_f1 = cnn_per_class.get('f1', 0.0)
            
            tl_precision = tl_per_class.get('precision', 0.0)
            tl_recall = tl_per_class.get('recall', 0.0)
            tl_f1 = tl_per_class.get('f1', 0.0)
            
            per_class_comparison.append({
                'Class': class_name,
                'CNN Precision': cnn_precision,
                'TL Precision': tl_precision,
                'Precision Δ': tl_precision - cnn_precision,
                'CNN Recall': cnn_recall,
                'TL Recall': tl_recall,
                'Recall Δ': tl_recall - cnn_recall,
                'CNN F1': cnn_f1,
                'TL F1': tl_f1,
                'F1 Δ': tl_f1 - cnn_f1
            })
        
        per_class_df = pd.DataFrame(per_class_comparison)
        print(per_class_df.to_string(index=False))
        print("=" * 60)
        
        # Save to CSV
        csv_path = self.output_dir / 'per_class_metrics_comparison.csv'
        per_class_df.to_csv(csv_path, index=False)
        print(f"\n✓ Per-class comparison saved to: {csv_path}")
        
        return per_class_df
    
    def visualize_per_class_metrics(self, per_class_df: pd.DataFrame):
        """
        Visualize per-class metrics comparison
        
        Args:
            per_class_df: DataFrame with per-class comparison results
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['Precision', 'Recall', 'F1']
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            x = np.arange(len(self.class_names))
            width = 0.35
            
            cnn_col = f'CNN {metric}'
            tl_col = f'TL {metric}'
            
            cnn_values = per_class_df[cnn_col].values
            tl_values = per_class_df[tl_col].values
            
            bars1 = ax.bar(x - width/2, cnn_values, width, label='Custom CNN', 
                          color='#d62728', alpha=0.7)
            bars2 = ax.bar(x + width/2, tl_values, width, label='Transfer Learning', 
                          color='#2ca02c', alpha=0.7)
            
            ax.set_xlabel('Class', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f'Per-Class {metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'per_class_metrics_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {fig_path}")
        
        plt.show()
    
    def generate_summary_report(self, comparison_df: pd.DataFrame, per_class_df: pd.DataFrame):
        """
        Generate summary report of improvements
        
        Args:
            comparison_df: DataFrame with overall comparison
            per_class_df: DataFrame with per-class comparison
        """
        print("\n" + "=" * 60)
        print("SUMMARY OF IMPROVEMENTS")
        print("=" * 60)
        
        print("\nOverall Metrics:")
        print(f"  Accuracy improvement: {comparison_df.iloc[0]['Improvement']:+.4f} ({comparison_df.iloc[0]['Improvement %']:+.2f}%)")
        print(f"  Weighted F1 improvement: {comparison_df.iloc[3]['Improvement']:+.4f} ({comparison_df.iloc[3]['Improvement %']:+.2f}%)")
        print(f"  Macro F1 improvement: {comparison_df.iloc[6]['Improvement']:+.4f} ({comparison_df.iloc[6]['Improvement %']:+.2f}%)")
        
        if len(per_class_df) > 0:
            print("\nPer-Class Improvements:")
            for _, row in per_class_df.iterrows():
                print(f"\n  {row['Class']}:")
                print(f"    Precision: {row['Precision Δ']:+.4f}")
                print(f"    Recall:    {row['Recall Δ']:+.4f}")
                print(f"    F1-Score:  {row['F1 Δ']:+.4f}")
        else:
            print("\nPer-Class Improvements: Not available")
        
        # Key findings
        print("\n" + "=" * 60)
        print("KEY FINDINGS")
        print("=" * 60)
        
        # Check if transfer learning improved minority classes (if per_class data available)
        if len(per_class_df) > 0:
            try:
                indeterminate_row = per_class_df[per_class_df['Class'] == 'Indeterminate Appearance']
                atypical_row = per_class_df[per_class_df['Class'] == 'Atypical Appearance']
                
                if len(indeterminate_row) > 0 and len(atypical_row) > 0:
                    indeterminate_improved = indeterminate_row['F1 Δ'].values[0] > 0
                    atypical_improved = atypical_row['F1 Δ'].values[0] > 0
                    
                    if indeterminate_improved or atypical_improved:
                        print("\n✓ Transfer learning improved performance on minority classes:")
                        if indeterminate_improved:
                            tl_f1 = self.transfer_learning_metrics.get('per_class', {}).get('Indeterminate Appearance', {}).get('f1', 0.0)
                            cnn_f1 = self.custom_cnn_metrics.get('per_class', {}).get('Indeterminate Appearance', {}).get('f1', 0.0)
                            print(f"  - Indeterminate Appearance: Improved from {cnn_f1:.4f} to {tl_f1:.4f} F1-Score")
                        if atypical_improved:
                            tl_f1 = self.transfer_learning_metrics.get('per_class', {}).get('Atypical Appearance', {}).get('f1', 0.0)
                            cnn_f1 = self.custom_cnn_metrics.get('per_class', {}).get('Atypical Appearance', {}).get('f1', 0.0)
                            print(f"  - Atypical Appearance: Improved from {cnn_f1:.4f} to {tl_f1:.4f} F1-Score")
                    else:
                        print("\nTransfer learning did not significantly improve minority class performance")
            except (KeyError, IndexError):
                print("\nNote: Per-class metrics comparison limited due to missing data")
        
        accuracy_improvement = comparison_df.iloc[0]['Improvement']
        if accuracy_improvement > 0:
            print(f"\nOverall accuracy improved by {accuracy_improvement:+.4f}")
        else:
            print(f"\nOverall accuracy decreased by {abs(accuracy_improvement):.4f}")
        
        print("=" * 60)
        
        # Save summary to text file
        summary_path = self.output_dir / 'comparison_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy improvement: {comparison_df.iloc[0]['Improvement']:+.4f} ({comparison_df.iloc[0]['Improvement %']:+.2f}%)\n")
            f.write(f"  Weighted F1 improvement: {comparison_df.iloc[3]['Improvement']:+.4f} ({comparison_df.iloc[3]['Improvement %']:+.2f}%)\n")
            f.write(f"  Macro F1 improvement: {comparison_df.iloc[6]['Improvement']:+.4f} ({comparison_df.iloc[6]['Improvement %']:+.2f}%)\n\n")
            if len(per_class_df) > 0:
                f.write("Per-Class Improvements:\n")
                for _, row in per_class_df.iterrows():
                    f.write(f"\n  {row['Class']}:\n")
                    f.write(f"    Precision: {row['Precision Δ']:+.4f}\n")
                    f.write(f"    Recall:    {row['Recall Δ']:+.4f}\n")
                    f.write(f"    F1-Score:  {row['F1 Δ']:+.4f}\n")
            else:
                f.write("Per-Class Improvements: Not available\n")
        
        print(f"\n✓ Summary saved to: {summary_path}")
    
    def run_full_comparison(self):
        """Run complete comparison analysis"""
        print("=" * 60)
        print("MODEL COMPARISON: TRANSFER LEARNING vs CUSTOM CNN")
        print("=" * 60)
        
        # Overall metrics comparison
        comparison_df = self.compare_overall_metrics()
        self.visualize_overall_metrics(comparison_df)
        
        # Per-class metrics comparison (if available)
        has_per_class = (
            self.custom_cnn_metrics.get('per_class') and 
            self.transfer_learning_metrics.get('per_class') and
            len(self.custom_cnn_metrics.get('per_class', {})) > 0
        )
        
        if has_per_class:
            per_class_df = self.compare_per_class_metrics()
            self.visualize_per_class_metrics(per_class_df)
        else:
            print("\n" + "=" * 60)
            print("PER-CLASS METRICS COMPARISON")
            print("=" * 60)
            print("Note: Per-class metrics not available for one or both models.")
            print("Skipping per-class comparison.")
            print("=" * 60)
            # Create empty DataFrame for summary report
            per_class_df = pd.DataFrame(columns=['Class', 'CNN Precision', 'TL Precision', 
                                                  'Precision Δ', 'CNN Recall', 'TL Recall', 
                                                  'Recall Δ', 'CNN F1', 'TL F1', 'F1 Δ'])
        
        # Generate summary
        self.generate_summary_report(comparison_df, per_class_df)
        
        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE!")
        print(f"All results saved to: {self.output_dir}")
        print("=" * 60)
