import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np
import logging

class FusionPlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.plots_dir = results_dir / "plots" / "fusion"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_results(self, results: List[Dict]):
        """Plot fusion experiment results with robust error handling."""
        if not results:
            logging.warning("No results to plot for fusion experiment") 
            return
            
        # Extract and validate metrics
        processed_results = []
        for result in results:
            if result.get('success', False):
                metrics = result.get('metrics', {})
                config = result.get('config', {})
                
                # Extract strategy from config
                strategy = config.get('args', {}).get('strategy', 'unknown')
                
                processed_result = {
                    'accuracy': metrics.get('accuracy', 0),
                    'fusion_strategy': strategy,
                    'total_examples': metrics.get('total_examples', 0),
                    'avg_response_time': metrics.get('avg_response_time', 0),
                    'description': config.get('description', 'Unknown')
                }
                processed_results.append(processed_result)
                
        if not processed_results:
            logging.warning("No valid metrics found for plotting")
            return
            
        metrics_df = pd.DataFrame(processed_results)
        
        try:
            self._plot_fusion_accuracy(metrics_df) 
            self._plot_retriever_contributions(metrics_df)
            self._plot_score_distributions(metrics_df)
        except Exception as e:
            logging.error(f"Error plotting fusion results: {str(e)}")

    def _plot_fusion_accuracy(self, df: pd.DataFrame):
        """Plot fusion accuracy with error handling for missing columns."""
        plt.figure(figsize=(10, 6))
        try:
            if 'fusion_strategy' in df.columns and 'accuracy' in df.columns:
                sns.barplot(data=df, x='fusion_strategy', y='accuracy')
                plt.title('Accuracy by Fusion Strategy')
                plt.ylabel('Accuracy') 
                plt.xlabel('Strategy')
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'fusion_accuracy.png', dpi=300)
            else:
                logging.warning("Missing required columns for fusion accuracy plot")
        except Exception as e:
            logging.error(f"Error creating fusion accuracy plot: {str(e)}")
        finally:
            plt.close()
        
    def _plot_retriever_contributions(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        contribution_cols = ['contriever_contribution', 'bm25_contribution']
        df_melted = pd.melt(df,
                           id_vars=['fusion_strategy'],
                           value_vars=contribution_cols,
                           var_name='Retriever',
                           value_name='Contribution')
        
        sns.barplot(data=df_melted, x='fusion_strategy', y='Contribution', hue='Retriever')
        plt.title('Retriever Contributions by Strategy')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'retriever_contributions.png', dpi=300)
        plt.close()
        
    def _plot_score_distributions(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        score_cols = ['fused_scores', 'individual_scores']
        df_melted = pd.melt(df,
                           id_vars=['fusion_strategy'],
                           value_vars=score_cols,
                           var_name='Score Type',
                           value_name='Score')
        
        sns.boxplot(data=df_melted, x='fusion_strategy', y='Score', hue='Score Type')
        plt.title('Score Distributions')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'score_distributions.png', dpi=300)
        plt.close()