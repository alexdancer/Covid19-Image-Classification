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


class ModelComparator:
    """Compare performance metrics between two models"""
    
    def __init__(self, custom_cnn_metrics: Dict, transfer_learning_metrics: Dict, 
                 class_labels: Dict[int, str], output_dir: Optional[Path] = None):
        """
        Initialize ModelComparator
        
        Args:
            custom_cnn_metrics: Dictionary with custom CNN metrics
            transfer_learning_metrics: Dictionary with transfer learning metrics
            class_labels: Dictionary mapping class indices to labels
            output_dir: Optional directory to save comparison results
        """
        self.custom_cnn_metrics = custom_cnn_metrics
        self.transfer_learning_metrics = transfer_learning_metrics
        self.class_labels = class_labels
        self.output_dir = output_dir or Path('comparison_results')
        self.output_dir.mkdir(exist_ok=True)
        
        # Extract class names
        self.class_names = [class_labels[i] for i in range(len(class_labels))]
    
    def compare_overall_metrics(self) -> pd.DataFrame:
        """
        Compare overall metrics between models
        
        Returns:
            DataFrame with comparison results
        """
        print("=" * 60)
        print("OVERALL METRICS COMPARISON")
        print("=" * 60)
        
        comparison_data = {
            'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)',
                       'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
            'Custom CNN': [
                self.custom_cnn_metrics['accuracy'],
                self.custom_cnn_metrics['precision_weighted'],
                self.custom_cnn_metrics['recall_weighted'],
                self.custom_cnn_metrics['f1_weighted'],
                self.custom_cnn_metrics['precision_macro'],
                self.custom_cnn_metrics['recall_macro'],
                self.custom_cnn_metrics['f1_macro']
            ],
            'Transfer Learning': [
                self.transfer_learning_metrics['accuracy'],
                self.transfer_learning_metrics['precision_weighted'],
                self.transfer_learning_metrics['recall_weighted'],
                self.transfer_learning_metrics['f1_weighted'],
                self.transfer_learning_metrics['precision_macro'],
                self.transfer_learning_metrics['recall_macro'],
                self.transfer_learning_metrics['f1_macro']
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
            cnn_metrics = self.custom_cnn_metrics['per_class'][class_name]
            tl_metrics = self.transfer_learning_metrics['per_class'][class_name]
            
            per_class_comparison.append({
                'Class': class_name,
                'CNN Precision': cnn_metrics['precision'],
                'TL Precision': tl_metrics['precision'],
                'Precision Δ': tl_metrics['precision'] - cnn_metrics['precision'],
                'CNN Recall': cnn_metrics['recall'],
                'TL Recall': tl_metrics['recall'],
                'Recall Δ': tl_metrics['recall'] - cnn_metrics['recall'],
                'CNN F1': cnn_metrics['f1'],
                'TL F1': tl_metrics['f1'],
                'F1 Δ': tl_metrics['f1'] - cnn_metrics['f1']
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
        
        print("\nPer-Class Improvements:")
        for _, row in per_class_df.iterrows():
            print(f"\n  {row['Class']}:")
            print(f"    Precision: {row['Precision Δ']:+.4f}")
            print(f"    Recall:    {row['Recall Δ']:+.4f}")
            print(f"    F1-Score:  {row['F1 Δ']:+.4f}")
        
        # Key findings
        print("\n" + "=" * 60)
        print("KEY FINDINGS")
        print("=" * 60)
        
        # Check if transfer learning improved minority classes
        indeterminate_improved = per_class_df[per_class_df['Class'] == 'Indeterminate Appearance']['F1 Δ'].values[0] > 0
        atypical_improved = per_class_df[per_class_df['Class'] == 'Atypical Appearance']['F1 Δ'].values[0] > 0
        
        if indeterminate_improved or atypical_improved:
            print("\n✓ Transfer learning improved performance on minority classes:")
            if indeterminate_improved:
                tl_f1 = self.transfer_learning_metrics['per_class']['Indeterminate Appearance']['f1']
                print(f"  - Indeterminate Appearance: Improved from 0.00 to {tl_f1:.4f} F1-Score")
            if atypical_improved:
                tl_f1 = self.transfer_learning_metrics['per_class']['Atypical Appearance']['f1']
                print(f"  - Atypical Appearance: Improved from 0.00 to {tl_f1:.4f} F1-Score")
        else:
            print("\n⚠ Transfer learning did not significantly improve minority class performance")
        
        accuracy_improvement = comparison_df.iloc[0]['Improvement']
        if accuracy_improvement > 0:
            print(f"\n✓ Overall accuracy improved by {accuracy_improvement:+.4f}")
        else:
            print(f"\n⚠ Overall accuracy decreased by {abs(accuracy_improvement):.4f}")
        
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
            f.write("Per-Class Improvements:\n")
            for _, row in per_class_df.iterrows():
                f.write(f"\n  {row['Class']}:\n")
                f.write(f"    Precision: {row['Precision Δ']:+.4f}\n")
                f.write(f"    Recall:    {row['Recall Δ']:+.4f}\n")
                f.write(f"    F1-Score:  {row['F1 Δ']:+.4f}\n")
        
        print(f"\n✓ Summary saved to: {summary_path}")
    
    def run_full_comparison(self):
        """Run complete comparison analysis"""
        print("=" * 60)
        print("MODEL COMPARISON: TRANSFER LEARNING vs CUSTOM CNN")
        print("=" * 60)
        
        # Overall metrics comparison
        comparison_df = self.compare_overall_metrics()
        self.visualize_overall_metrics(comparison_df)
        
        # Per-class metrics comparison
        per_class_df = self.compare_per_class_metrics()
        self.visualize_per_class_metrics(per_class_df)
        
        # Generate summary
        self.generate_summary_report(comparison_df, per_class_df)
        
        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE!")
        print(f"All results saved to: {self.output_dir}")
        print("=" * 60)
