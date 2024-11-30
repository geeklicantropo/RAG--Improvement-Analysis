import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np
import logging

class BaselinePlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.plots_dir = results_dir / "plots" / "baseline"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_results(self, results: List[Dict]):
        if not results:
            logging.warning("No results to plot for baseline experiment")
            return

        processed_results = []
        for result in results:
            if result.get('success', False):
                metrics = result.get('metrics', {})
                config = result.get('config', {})
                
                processed_result = {
                    'accuracy': metrics.get('accuracy', 0),
                    'total_examples': metrics.get('total_examples', 0),
                    'avg_response_time': metrics.get('avg_response_time', 0),
                    'avg_context_length': metrics.get('avg_context_length', 0),
                    'noise_ratio': metrics.get('noise_ratio', 0),
                    'noise_type': config.get('args', {}).get('noise_type', 'none'),
                    'retriever': config.get('args', {}).get('retriever', 'Unknown')
                }
                processed_results.append(processed_result)

        if not processed_results:
            logging.warning("No valid results for plotting")
            return

        metrics_df = pd.DataFrame(processed_results)
        self._plot_accuracy_comparison(metrics_df)
        self._plot_noise_impact(metrics_df)
        self._plot_performance_metrics(metrics_df)
        self._plot_comparative_analysis(metrics_df)

    def _plot_accuracy_comparison(self, df: pd.DataFrame):
        try:
            plt.figure(figsize=(10, 6))
            if 'retriever' in df.columns:
                plot_data = df[['retriever', 'accuracy']].copy()
                sns.barplot(data=plot_data, x='retriever', y='accuracy')
                plt.title('Accuracy by Retriever Type')
                plt.ylabel('Accuracy')
                plt.xlabel('Retriever')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'accuracy_comparison.png', dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting accuracy comparison: {str(e)}")
            plt.close()

    def _plot_noise_impact(self, df: pd.DataFrame):
        try:
            plt.figure(figsize=(12, 6))
            if all(col in df.columns for col in ['noise_ratio', 'accuracy', 'noise_type']):
                sns.scatterplot(data=df, x='noise_ratio', y='accuracy', hue='noise_type', style='retriever')
                plt.title('Impact of Noise on Accuracy')
                plt.xlabel('Noise Ratio')
                plt.ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'noise_impact.png', dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting noise impact: {str(e)}")
            plt.close()

    def _plot_performance_metrics(self, df: pd.DataFrame):
        try:
            plt.figure(figsize=(10, 6))
            metrics = ['avg_response_time', 'avg_context_length']
            available_metrics = [m for m in metrics if m in df.columns]

            if available_metrics and 'retriever' in df.columns:
                df_melted = pd.melt(df,
                                  id_vars=['retriever'],
                                  value_vars=available_metrics,
                                  var_name='Metric',
                                  value_name='Value')

                sns.boxplot(data=df_melted, x='retriever', y='Value', hue='Metric')
                plt.title('Performance Metrics by Retriever Type')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'performance_metrics.png', dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting performance metrics: {str(e)}")
            plt.close()

    def _plot_comparative_analysis(self, df: pd.DataFrame):
        try:
            plt.figure(figsize=(12, 8))
            if all(col in df.columns for col in ['retriever', 'accuracy', 'noise_type']):
                g = sns.FacetGrid(df, col='noise_type', height=6)
                g.map_dataframe(sns.barplot, x='retriever', y='accuracy')
                g.fig.suptitle('Comparative Analysis: Retriever Performance by Noise Type')
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'comparative_analysis.png', dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting comparative analysis: {str(e)}")
            plt.close()