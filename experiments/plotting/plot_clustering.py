import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import logging
from tqdm import tqdm
from scipy.stats import ttest_ind

class ClusteringPlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.plots_dir = results_dir / "plots" / "clustering"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        plt.style.use('ggplot')

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ClusteringPlotter")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(self.plots_dir / 'plotting.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def _plot_optimal_cluster_analysis(self, df: pd.DataFrame):
        plt.figure(figsize=(15, 8))
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            # Elbow curve
            if 'inertia' in df.columns:
                sns.lineplot(data=df, x='num_clusters', y='inertia', marker='o', ax=ax1)
                ax1.set_title('Elbow Curve Analysis')
                
            # Silhouette analysis
            if 'silhouette_score' in df.columns:
                sns.lineplot(data=df, x='num_clusters', y='silhouette_score', marker='o', ax=ax2)
                ax2.set_title('Silhouette Score Analysis')
                
                # Highlight optimal k
                optimal_k = df.loc[df['silhouette_score'].idxmax(), 'num_clusters']
                ax2.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
                
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'optimal_cluster_analysis.png', dpi=300)
        finally:
            plt.close()

    def _plot_cluster_quality_metrics(self, df: pd.DataFrame):
        quality_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        available_metrics = [m for m in quality_metrics if m in df.columns]
        
        if not available_metrics:
            return
            
        plt.figure(figsize=(15, 6))
        try:
            df_melted = pd.melt(df, 
                              id_vars=['num_clusters'],
                              value_vars=available_metrics,
                              var_name='Metric',
                              value_name='Score')
            
            sns.lineplot(data=df_melted, x='num_clusters', y='Score', 
                        hue='Metric', style='Metric', markers=True)
            
            plt.title('Cluster Quality Metrics Analysis')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Score')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'cluster_quality_metrics.png', dpi=300)
        finally:
            plt.close()

    def _plot_baseline_comparison(self, cluster_df: pd.DataFrame, baseline_df: pd.DataFrame):
        plt.figure(figsize=(15, 10))
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy comparison
            accuracy_data = pd.DataFrame({
                'Clustering': cluster_df['accuracy'],
                'Baseline': baseline_df['accuracy']
            })
            
            sns.boxplot(data=pd.melt(accuracy_data), x='variable', y='value', ax=axes[0,0])
            axes[0,0].set_title('Accuracy Comparison')
            
            # Statistical test
            t_stat, p_val = ttest_ind(cluster_df['accuracy'], baseline_df['accuracy'])
            axes[0,0].text(0.05, 0.95, f'p-value: {p_val:.4f}', 
                          transform=axes[0,0].transAxes)
            
            # Response time comparison
            if all(col in cluster_df.columns for col in ['response_time', 'retriever_type']):
                sns.boxplot(data=cluster_df, x='retriever_type', y='response_time', ax=axes[0,1])
                axes[0,1].set_title('Response Time by Retriever')
            
            # Document relevance
            if 'relevance_score' in cluster_df.columns:
                sns.histplot(data=cluster_df, x='relevance_score', ax=axes[1,0])
                axes[1,0].set_title('Document Relevance Distribution')
            
            # Memory usage
            if 'memory_usage' in cluster_df.columns:
                sns.scatterplot(data=cluster_df, x='num_clusters', y='memory_usage', ax=axes[1,1])
                axes[1,1].set_title('Memory Usage vs Clusters')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'baseline_comparison.png', dpi=300)
        finally:
            plt.close()

    def plot_results(self, results: List[Dict], baseline_results: Optional[List[Dict]] = None):
        if not results:
            self.logger.warning("No results to plot for clustering experiment")
            return
            
        metrics_df = pd.DataFrame([r['metrics'] for r in results if r['success']])
        
        if metrics_df.empty:
            self.logger.warning("No valid metrics found for plotting")
            return

        plot_functions = [
            self._plot_cluster_metrics,
            self._plot_cluster_sizes,
            self._plot_silhouette_analysis,
            self._plot_comparative_metrics,
            self._plot_optimal_cluster_analysis,
            self._plot_cluster_quality_metrics,
            self._plot_baseline_comparison
        ]

        for plot_func in tqdm(plot_functions, desc="Generating plots"):
            try:
                if plot_func.__name__ == '_plot_baseline_comparison' and baseline_results:
                    plot_func(metrics_df, pd.DataFrame([r['metrics'] for r in baseline_results if r['success']]))
                else:
                    plot_func(metrics_df)
            except Exception as e:
                self.logger.error(f"Error in {plot_func.__name__}: {str(e)}")

    def _plot_cluster_metrics(self, df: pd.DataFrame):
        if 'num_clusters' not in df.columns:
            return
            
        plt.figure(figsize=(12, 8))
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Plot accuracy vs number of clusters
            if 'accuracy' in df.columns:
                sns.lineplot(
                    data=df,
                    x='num_clusters',
                    y='accuracy',
                    marker='o',
                    ax=ax1
                )
                ax1.set_title('Accuracy vs Number of Clusters')
                ax1.set_xlabel('Number of Clusters')
                ax1.set_ylabel('Accuracy')
                
            # Plot cluster quality metrics
            quality_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
            available_metrics = [m for m in quality_metrics if m in df.columns]
            
            if available_metrics:
                df_melted = pd.melt(
                    df,
                    id_vars=['num_clusters'],
                    value_vars=available_metrics,
                    var_name='Metric',
                    value_name='Score'
                )
                
                sns.lineplot(
                    data=df_melted,
                    x='num_clusters',
                    y='Score',
                    hue='Metric',
                    marker='o',
                    ax=ax2
                )
                ax2.set_title('Cluster Quality Metrics')
                ax2.set_xlabel('Number of Clusters')
                ax2.legend(title='Metric')
            
            plt.tight_layout()
            save_path = self.plots_dir / 'cluster_metrics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved cluster metrics plot to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting cluster metrics: {str(e)}")
        finally:
            plt.close()

    def _plot_cluster_sizes(self, df: pd.DataFrame):
        cluster_size_cols = [col for col in df.columns if 'cluster_size' in col]
        if not cluster_size_cols:
            return
            
        plt.figure(figsize=(12, 6))
        try:
            df_sizes = pd.melt(
                df,
                value_vars=cluster_size_cols,
                var_name='Cluster',
                value_name='Size'
            )
            
            sns.boxplot(data=df_sizes, x='Cluster', y='Size')
            plt.title('Distribution of Cluster Sizes')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            save_path = self.plots_dir / 'cluster_sizes.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved cluster sizes plot to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting cluster sizes: {str(e)}")
        finally:
            plt.close()

    def _plot_silhouette_analysis(self, df: pd.DataFrame):
        if 'silhouette_score' not in df.columns or 'num_clusters' not in df.columns:
            return
            
        plt.figure(figsize=(10, 6))
        try:
            sns.scatterplot(
                data=df,
                x='num_clusters',
                y='silhouette_score',
                s=100
            )
            
            # Add trend line
            z = np.polyfit(df['num_clusters'], df['silhouette_score'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(df['num_clusters'].min(), df['num_clusters'].max(), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.8)
            
            plt.title('Silhouette Score Analysis')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.grid(True, alpha=0.3)
            
            save_path = self.plots_dir / 'silhouette_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved silhouette analysis plot to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting silhouette analysis: {str(e)}")
        finally:
            plt.close()

    def _plot_comparative_metrics(self, df: pd.DataFrame):
        if 'accuracy' not in df.columns or 'num_clusters' not in df.columns:
            return
            
        plt.figure(figsize=(15, 8))
        try:
            # Create subplot grid
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy and cluster count correlation
            sns.scatterplot(
                data=df,
                x='num_clusters',
                y='accuracy',
                ax=axes[0,0]
            )
            axes[0,0].set_title('Accuracy vs Number of Clusters')
            
            # Cluster size distribution
            if 'avg_cluster_size' in df.columns:
                sns.boxplot(
                    data=df,
                    x='num_clusters',
                    y='avg_cluster_size',
                    ax=axes[0,1]
                )
                axes[0,1].set_title('Cluster Size Distribution')
            
            # Cluster quality metrics comparison
            quality_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
            available_metrics = [m for m in quality_metrics if m in df.columns]
            
            if available_metrics:
                df_melted = pd.melt(
                    df,
                    id_vars=['num_clusters'],
                    value_vars=available_metrics,
                    var_name='Metric',
                    value_name='Score'
                )
                
                sns.lineplot(
                    data=df_melted,
                    x='num_clusters',
                    y='Score',
                    hue='Metric',
                    marker='o',
                    ax=axes[1,0]
                )
                axes[1,0].set_title('Quality Metrics Comparison')
            
            # Performance metrics
            perf_metrics = ['response_time', 'memory_usage']
            available_perf = [m for m in perf_metrics if m in df.columns]
            
            if available_perf:
                df_perf = pd.melt(
                    df,
                    id_vars=['num_clusters'],
                    value_vars=available_perf,
                    var_name='Metric',
                    value_name='Value'
                )
                
                sns.lineplot(
                    data=df_perf,
                    x='num_clusters',
                    y='Value',
                    hue='Metric',
                    marker='o',
                    ax=axes[1,1]
                )
                axes[1,1].set_title('Performance Metrics')
            
            plt.tight_layout()
            save_path = self.plots_dir / 'comparative_metrics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comparative metrics plot to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting comparative metrics: {str(e)}")
        finally:
            plt.close()

    def save_summary_metrics(self, df: pd.DataFrame):
        """Save statistical summary of clustering metrics to a file."""
        try:
            summary = {
                'accuracy': {
                    'mean': df['accuracy'].mean(),
                    'std': df['accuracy'].std(),
                    'min': df['accuracy'].min(),
                    'max': df['accuracy'].max()
                },
                'cluster_metrics': {}
            }
            
            # Add clustering-specific metrics
            quality_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
            for metric in quality_metrics:
                if metric in df.columns:
                    summary['cluster_metrics'][metric] = {
                        'mean': df[metric].mean(),
                        'std': df[metric].std(),
                        'best_k': df.loc[df[metric].idxmax(), 'num_clusters']
                    }
            
            # Save to file
            summary_path = self.plots_dir / 'clustering_summary.json'
            with open(summary_path, 'w') as f:
                pd.json.dumps(summary, indent=2)
            
            self.logger.info(f"Saved clustering summary to {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving summary metrics: {str(e)}")