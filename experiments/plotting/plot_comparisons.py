import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Dict
import numpy as np
import logging

class ComparativePlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.plots_dir = results_dir / "plots" / "comparisons"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_results(self, baseline_results: List[Dict], clustering_results: List[Dict]) -> None:
        """
        Plot comparative analysis between baseline and clustering experiments.

        Args:
            baseline_results: List of baseline experiment results.
            clustering_results: List of clustering experiment results.
        """
        try:
            self._plot_accuracy_comparison(baseline_results, clustering_results)
            self._plot_noise_impact(baseline_results, clustering_results)
        except Exception as e:
            logging.error(f"Error plotting comparative analysis: {str(e)}")

    def _plot_accuracy_comparison(self, baseline_results: List[Dict], clustering_results: List[Dict]) -> None:
        """
        Plot the accuracy comparison between baseline and clustering experiments.
        """
        try:
            plt.figure(figsize=(10, 6))

            baseline_df = pd.DataFrame([r['metrics'] for r in baseline_results if r.get('success', False)])
            clustering_df = pd.DataFrame([r['metrics'] for r in clustering_results if r.get('success', False)])

            accuracy_data = pd.DataFrame({
                'Baseline': baseline_df['accuracy'],
                'Clustering': clustering_df['accuracy']
            })

            sns.barplot(data=accuracy_data)
            plt.title('Accuracy Comparison: Baseline vs Clustering')
            plt.xlabel('Experiment')
            plt.ylabel('Accuracy')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'accuracy_comparison.png', dpi=300)
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting accuracy comparison: {str(e)}")
            plt.close()

    def _plot_noise_impact(self, baseline_results: List[Dict], clustering_results: List[Dict]) -> None:
        """
        Plot the impact of noise injection on accuracy for baseline and clustering experiments.
        """
        try:
            plt.figure(figsize=(12, 6))

            baseline_df = pd.DataFrame([r['metrics'] for r in baseline_results if r.get('success', False)])
            clustering_df = pd.DataFrame([r['metrics'] for r in clustering_results if r.get('success', False)])

            # Assuming 'noise_ratio' and 'accuracy' are available in the metrics
            baseline_data = baseline_df[['noise_ratio', 'accuracy']]
            baseline_data['Experiment'] = 'Baseline'

            clustering_data = clustering_df[['noise_ratio', 'accuracy']]
            clustering_data['Experiment'] = 'Clustering'

            combined_data = pd.concat([baseline_data, clustering_data], ignore_index=True)

            sns.scatterplot(data=combined_data, x='noise_ratio', y='accuracy', hue='Experiment')
            plt.title('Impact of Noise on Accuracy')
            plt.xlabel('Noise Ratio')
            plt.ylabel('Accuracy')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'noise_impact.png', dpi=300)
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting noise impact: {str(e)}")
            plt.close()