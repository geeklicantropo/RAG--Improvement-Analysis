import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np
import logging

class ClusteringPlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.plots_dir = results_dir / "plots" / "clustering"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_results(self, results: List[Dict]):
        if not results:
            logging.warning("No results to plot for clustering experiment")
            return
            
        metrics_df = pd.DataFrame([r['metrics'] for r in results if r['success']])
        
        if metrics_df.empty:
            logging.warning("No valid metrics found for plotting")
            return
            
        self._plot_cluster_accuracies(metrics_df)
        self._plot_cluster_sizes(metrics_df)
        self._plot_silhouette_scores(metrics_df)
        self._plot_cluster_coherence(metrics_df)
        
    def _plot_cluster_accuracies(self, df: pd.DataFrame):
        try:
            plt.figure(figsize=(10, 6))
            # Use raw string for regex pattern and handle missing retriever_type
            if 'description' in df.columns:
                df['num_clusters'] = df['description'].str.extract(r'Clustering-(\d+)').astype(float)
            elif 'num_clusters' in df.columns:
                pass  # Already has num_clusters
            else:
                logging.warning("No cluster information found for accuracy plot")
                return
                
            df_clean = df[df['num_clusters'].notna()]
            
            if not df_clean.empty and 'accuracy' in df_clean.columns:
                sns.barplot(data=df_clean, x='num_clusters', y='accuracy')
                plt.title('Accuracy vs Number of Clusters')
                plt.ylabel('Accuracy')
                plt.xlabel('Number of Clusters')
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'cluster_accuracies.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting cluster accuracies: {str(e)}")
            plt.close()
            
    def _plot_cluster_sizes(self, df: pd.DataFrame):
        try:
            plt.figure(figsize=(12, 6))
            size_cols = [col for col in df.columns if 'cluster_size' in col]
            
            if size_cols and 'num_clusters' in df.columns:
                df_melted = pd.melt(df,
                                  id_vars=['num_clusters'],
                                  value_vars=size_cols,
                                  var_name='Cluster',
                                  value_name='Size')
                
                sns.boxplot(data=df_melted, x='num_clusters', y='Size')
                plt.title('Cluster Size Distributions')
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'cluster_sizes.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting cluster sizes: {str(e)}")
            plt.close()
        
    def _plot_silhouette_scores(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='num_clusters', y='silhouette_score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'silhouette_scores.png', dpi=300)
        plt.close()
        
    def _plot_cluster_coherence(self, df: pd.DataFrame):
        try:
            plt.figure(figsize=(12, 6))
            coherence_cols = ['intra_cluster_similarity', 'inter_cluster_distance']
            valid_cols = [col for col in coherence_cols if col in df.columns]
            
            if valid_cols and 'num_clusters' in df.columns:
                df_melted = pd.melt(df,
                                  id_vars=['num_clusters'],
                                  value_vars=valid_cols,
                                  var_name='Metric',
                                  value_name='Value')
                
                sns.boxplot(data=df_melted, x='num_clusters', y='Value', hue='Metric')
                plt.title('Cluster Coherence Metrics')
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'cluster_coherence.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting cluster coherence: {str(e)}")
            plt.close()