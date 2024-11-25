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
        """Plot results with error handling."""
        if not results:
            logging.warning("No results to plot for baseline experiment")
            return

        # Add default metrics if missing
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
                    'description': config.get('description', 'Unknown'),
                    'retriever': config.get('args', {}).get('retriever', 'Unknown')
                }
                processed_results.append(processed_result)

        if not processed_results:
            logging.warning("No valid results for plotting")
            return

        metrics_df = pd.DataFrame(processed_results)
        self._plot_accuracy_comparison(metrics_df)
        self._plot_retrieval_statistics(metrics_df)
        self._plot_performance_metrics(metrics_df)

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

    def _plot_retrieval_statistics(self, df: pd.DataFrame):
        try:
            plt.figure(figsize=(12, 6))
            metrics = ['accuracy', 'total_examples']
            available_metrics = [m for m in metrics if m in df.columns]
            
            if available_metrics and 'retriever' in df.columns:
                df_melted = pd.melt(df,
                                  id_vars=['retriever'],
                                  value_vars=available_metrics,
                                  var_name='Metric',
                                  value_name='Score')
                
                sns.barplot(data=df_melted, x='retriever', y='Score', hue='Metric')
                plt.title('Retrieval Metrics Comparison')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'retrieval_metrics.png', dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting retrieval statistics: {str(e)}")
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