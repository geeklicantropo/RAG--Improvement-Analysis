import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np

class CategoriesPlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.plots_dir = results_dir / "plots" / "categories"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_results(self, results: List[Dict]):
        metrics_df = pd.DataFrame([r['metrics'] for r in results if r['success']])
        
        self._plot_category_accuracies(metrics_df)
        self._plot_category_distributions(metrics_df)
        self._plot_confidence_thresholds(metrics_df)
        self._plot_category_impact(metrics_df)
        
    def _plot_category_accuracies(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        category_cols = [col for col in df.columns if 'category_accuracy' in col]
        df_melted = pd.melt(df,
                           value_vars=category_cols,
                           var_name='Category',
                           value_name='Accuracy')
        
        sns.barplot(data=df_melted, x='Category', y='Accuracy')
        plt.title('Accuracy by Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'category_accuracies.png', dpi=300)
        plt.close()
        
    def _plot_category_distributions(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        dist_cols = [col for col in df.columns if 'category_size' in col]
        df_melted = pd.melt(df,
                           value_vars=dist_cols,
                           var_name='Category',
                           value_name='Size')
        
        sns.boxplot(data=df_melted, x='Category', y='Size')
        plt.title('Category Size Distributions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'category_distributions.png', dpi=300)
        plt.close()
        
    def _plot_confidence_thresholds(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        threshold_cols = [col for col in df.columns if 'confidence_threshold' in col]
        df_melted = pd.melt(df,
                           value_vars=threshold_cols,
                           var_name='Threshold',
                           value_name='Value')
        
        sns.lineplot(data=df_melted, x='Threshold', y='Value')
        plt.title('Confidence Threshold Impact')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confidence_thresholds.png', dpi=300)
        plt.close()
        
    def _plot_category_impact(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        impact_cols = ['accuracy_with_categories', 'accuracy_without_categories']
        df_melted = pd.melt(df,
                           value_vars=impact_cols,
                           var_name='Configuration',
                           value_name='Accuracy')
        
        sns.boxplot(data=df_melted, x='Configuration', y='Accuracy')
        plt.title('Impact of Category Organization')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'category_impact.png', dpi=300)
        plt.close()