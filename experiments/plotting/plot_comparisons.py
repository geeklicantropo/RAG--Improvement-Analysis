import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import logging
from tqdm import tqdm
import json
from scipy import stats

class ComparativePlotter:
    """Generates comparative visualizations across different experiment types."""

    def __init__(self, results_dir: Path, logger: Optional[logging.Logger] = None):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots" / "comparisons"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or self._setup_logger()

        # Set style configurations
        #plt.style.use('seaborn-v0_8') 
        sns.set_theme(style="whitegrid", palette="husl") 

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ComparativePlotter")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(self.plots_dir / 'plotting.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger
    
    def _plot_baseline_comparison(self, baseline_results: pd.DataFrame):
        plt.figure(figsize=(8, 6))
        sns.barplot(data=baseline_results, x='experiment_type', y='accuracy')
        plt.title("Baseline: Gold vs Random vs Adversarial")
        plt.xlabel("Experiment Type") 
        plt.ylabel("Accuracy")
        plt.savefig(self.plots_dir / "baseline_comparison.png")
        plt.close()

    def _plot_clustering_comparison(self, clustering_results: pd.DataFrame):
        plt.figure(figsize=(8, 6))  
        sns.barplot(data=clustering_results, x='experiment_type', y='accuracy')
        plt.title("Clustering: Gold vs Random vs Adversarial")
        plt.xlabel("Experiment Type")
        plt.ylabel("Accuracy") 
        plt.savefig(self.plots_dir / "clustering_comparison.png")
        plt.close()

    def _plot_baseline_vs_clustering(self, results_data: Dict[str, pd.DataFrame]):
        best_baseline = results_data['baseline'].loc[results_data['baseline']['accuracy'].idxmax()]
        best_clustering = results_data['clustering'].loc[results_data['clustering']['accuracy'].idxmax()]
        
        comparison_df = pd.DataFrame([
            {'Experiment': 'Baseline', 'Accuracy': best_baseline['accuracy'], 'Type': best_baseline['experiment_type']},
            {'Experiment': 'Clustering', 'Accuracy': best_clustering['accuracy'], 'Type': best_clustering['experiment_type']}
        ])
        
        plt.figure(figsize=(8, 6))
        sns.barplot(data=comparison_df, x='Experiment', y='Accuracy', hue='Type')  
        plt.title("Best Baseline vs Best Clustering")
        plt.savefig(self.plots_dir / "baseline_vs_clustering.png")
        plt.close()

    def plot_experiment_results(
    self, 
    baseline_results: List[Dict], 
    clustering_results: List[Dict],
    fusion_results: Optional[List[Dict]] = None,
    category_results: Optional[List[Dict]] = None
    ) -> None:
        """
        Generate and save plots for experiment results.
        """
        # Prepare results data for plotting
        results_data = self._prepare_results_data(
            baseline_results, 
            clustering_results,
            fusion_results if fusion_results else [],
            category_results if category_results else []
        )
        
        # Compare baseline gold vs random vs adversarial
        self._plot_baseline_comparison(results_data['baseline'])
        
        # Compare clustering gold vs random vs adversarial
        self._plot_clustering_comparison(results_data['clustering']) 

        # Compare best baseline vs best clustering
        self._plot_baseline_vs_clustering(results_data)

    def _prepare_results_data(
    self,
    baseline_results: List[Dict], 
    clustering_results: List[Dict],
    fusion_results: Optional[List[Dict]] = None,
    category_results: Optional[List[Dict]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Process raw results from experiments and convert them into a dictionary of DataFrames.

        Args:
            baseline_results (List[Dict]): Results from baseline experiments.
            clustering_results (List[Dict]): Results from clustering experiments.
            fusion_results (List[Dict], optional): Results from fusion experiments.
            category_results (List[Dict], optional): Results from category experiments.

        Returns:
            Dict[str, pd.DataFrame]: Processed results data organized by experiment type.
        """
        data = {}
        
        # Process and store baseline and clustering results
        data['baseline'] = self._process_baseline_results(baseline_results)
        data['clustering'] = self._process_clustering_results(clustering_results)
        
        # Optionally process fusion and category results if provided
        if fusion_results:
            data['fusion'] = self._process_fusion_results(fusion_results)
        else:
            data['fusion'] = pd.DataFrame()  # Return an empty DataFrame if no fusion results
        
        if category_results:
            data['category'] = self._process_category_results(category_results)
        else:
            data['category'] = pd.DataFrame()  # Return an empty DataFrame if no category results
            
        return data

    def _plot_accuracy_comparison(self, results_data: Dict[str, pd.DataFrame]) -> None:
        """Plot accuracy comparisons across experiments."""
        try:
            plt.figure(figsize=(12, 6))

            accuracy_data = []
            for exp_type, df in results_data.items():
                if 'accuracy' in df.columns:
                    accuracy_data.append({
                        'Experiment': exp_type,
                        'Accuracy': df['accuracy'].mean(),
                        'Std': df['accuracy'].std()
                    })

            plot_df = pd.DataFrame(accuracy_data)

            sns.barplot(data=plot_df, x='Experiment', y='Accuracy')
            plt.errorbar(
                x=range(len(plot_df)),
                y=plot_df['Accuracy'],
                yerr=plot_df['Std'],
                fmt='none',
                color='black',
                capsize=5
            )

            plt.title('Accuracy Comparison Across Experiments')
            plt.ylabel('Accuracy')
            plt.xlabel('Experiment Type')
            plt.tight_layout()

            save_path = self.plots_dir / 'accuracy_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved accuracy comparison plot to {save_path}")

        except Exception as e:
            self.logger.error(f"Error in _plot_accuracy_comparison: {str(e)}")
            plt.close()

    def _plot_noise_impact(self, results_data: Dict[str, pd.DataFrame]) -> None:
        """Plot impact of noise injection across experiments."""
        try:
            plt.figure(figsize=(12, 6))

            noise_data = []
            for exp_type, df in results_data.items():
                if 'noise_ratio' in df.columns and 'accuracy' in df.columns:
                    df_noise = df.groupby('noise_ratio')['accuracy'].agg(['mean', 'std']).reset_index()
                    df_noise['Experiment'] = exp_type
                    noise_data.append(df_noise)

            if noise_data:
                plot_df = pd.concat(noise_data, ignore_index=True)

                sns.lineplot(
                    data=plot_df,
                    x='noise_ratio',
                    y='mean',
                    hue='Experiment',
                    style='Experiment',
                    markers=True,
                    dashes=False
                )

                for exp_type in plot_df['Experiment'].unique():
                    exp_data = plot_df[plot_df['Experiment'] == exp_type]
                    plt.fill_between(
                        exp_data['noise_ratio'],
                        exp_data['mean'] - exp_data['std'],
                        exp_data['mean'] + exp_data['std'],
                        alpha=0.2
                    )

                plt.title('Impact of Noise Injection on Accuracy')
                plt.xlabel('Noise Ratio')
                plt.ylabel('Accuracy')
                plt.tight_layout()

                save_path = self.plots_dir / 'noise_impact.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                self.logger.info(f"Saved noise impact plot to {save_path}")
            else:
                self.logger.warning("No noise impact data available for plotting")
                plt.close()

        except Exception as e:
            self.logger.error(f"Error in _plot_noise_impact: {str(e)}")
            plt.close()

    def _plot_cluster_vs_rag_comparison(self, results_data: Dict[str, pd.DataFrame]) -> None:
        """Plot comparison of clustering accuracy vs naive RAG for each number of clusters."""
        try:
            plt.figure(figsize=(12, 6))

            cluster_data = results_data.get('clustering')
            baseline_data = results_data.get('baseline')

            if cluster_data is not None and 'num_clusters' in cluster_data.columns:
                cluster_accuracy = cluster_data.groupby('num_clusters')['accuracy'].mean().reset_index()
                cluster_accuracy['Experiment'] = 'Clustering'

                if baseline_data is not None and 'accuracy' in baseline_data.columns:
                    baseline_mean = baseline_data['accuracy'].mean()
                    cluster_accuracy = pd.concat([
                        cluster_accuracy,
                        pd.DataFrame({'num_clusters': [0], 'accuracy': [baseline_mean], 'Experiment': ['Naive RAG']})
                    ], ignore_index=True)

                sns.lineplot(
                    data=cluster_accuracy,
                    x='num_clusters',
                    y='accuracy',
                    hue='Experiment',
                    style='Experiment',
                    markers=True,
                    dashes=False
                )

                plt.title('Clustering vs Naive RAG Accuracy by Number of Clusters')
                plt.xlabel('Number of Clusters')
                plt.ylabel('Accuracy')
                plt.tight_layout()

                save_path = self.plots_dir / 'cluster_vs_rag_comparison.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                self.logger.info(f"Saved cluster vs RAG comparison plot to {save_path}")
            else:
                self.logger.warning("No clustering or baseline data available for cluster vs RAG comparison")
                plt.close()

        except Exception as e:
            self.logger.error(f"Error in _plot_cluster_vs_rag_comparison: {str(e)}")
            plt.close()

    def _plot_retrieval_analysis(self, results_data: Dict[str, pd.DataFrame]) -> None:
        """Plot retrieval performance analysis."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot retrieval precision
            precision_data = []
            for exp_type, df in results_data.items():
                if 'retrieval_precision' in df.columns:
                    precision_data.append({
                        'Experiment': exp_type,
                        'Precision': df['retrieval_precision'].mean(),
                        'Std': df['retrieval_precision'].std()
                    })

            if precision_data:
                prec_df = pd.DataFrame(precision_data)
                sns.barplot(data=prec_df, x='Experiment', y='Precision', ax=ax1)
                ax1.set_title('Retrieval Precision')

            # Plot document relevance distribution
            relevance_data = []
            for exp_type, df in results_data.items():
                if 'doc_relevance' in df.columns:
                    relevance_data.extend([{
                        'Experiment': exp_type,
                        'Relevance': rel
                    } for rel in df['doc_relevance']])

            if relevance_data:
                rel_df = pd.DataFrame(relevance_data)
                sns.boxplot(data=rel_df, x='Experiment', y='Relevance', ax=ax2)
                ax2.set_title('Document Relevance Distribution')

            plt.tight_layout()
            save_path = self.plots_dir / 'retrieval_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved retrieval analysis plot to {save_path}")

        except Exception as e:
            self.logger.error(f"Error in _plot_retrieval_analysis: {str(e)}")
            plt.close()

    def _plot_performance_metrics(self, results_data: Dict[str, pd.DataFrame]) -> None:
        """Plot performance metrics comparison."""
        try:
            metrics = ['response_time', 'memory_usage', 'context_length']
            available_metrics = [m for m in metrics if any(m in df.columns for df in results_data.values())]

            if available_metrics:
                fig, axes = plt.subplots(
                    len(available_metrics), 1,
                    figsize=(10, 5 * len(available_metrics))
                )
                if len(available_metrics) == 1:
                    axes = [axes]

                for ax, metric in zip(axes, available_metrics):
                    metric_data = []
                    for exp_type, df in results_data.items():
                        if metric in df.columns:
                            metric_data.extend([{
                                'Experiment': exp_type,
                                'Value': val,
                                'Metric': metric.replace('_', ' ').title()
                            } for val in df[metric]])

                    if metric_data:
                        metric_df = pd.DataFrame(metric_data)
                        sns.violinplot(data=metric_df, x='Experiment', y='Value', ax=ax)
                        ax.set_title(f'{metric_df["Metric"].iloc[0]} Distribution')

                plt.tight_layout()
                save_path = self.plots_dir / 'performance_metrics.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                self.logger.info(f"Saved performance metrics plot to {save_path}")
            else:
                self.logger.warning("No performance metrics available for plotting")

        except Exception as e:
            self.logger.error(f"Error in _plot_performance_metrics: {str(e)}")
            plt.close()

    def _process_baseline_results(self, results: List[Dict]) -> pd.DataFrame:
        """Process baseline experiment results."""
        processed_data = []
        for result in results:
            if result.get('success', False):
                metrics = result.get('metrics', {})
                processed_data.append({
                    'accuracy': metrics.get('accuracy', 0),
                    'retrieval_precision': metrics.get('retrieval_precision', 0),
                    'response_time': metrics.get('avg_response_time', 0),
                    'context_length': metrics.get('avg_context_length', 0),
                    'noise_ratio': metrics.get('noise_ratio', 0)
                })
        return pd.DataFrame(processed_data)

    def _process_clustering_results(self, results: List[Dict]) -> pd.DataFrame:
        """Process clustering experiment results."""
        processed_data = []
        for result in results:
            if result.get('success', False):
                metrics = result.get('metrics', {})
                processed_data.append({
                    'accuracy': metrics.get('accuracy', 0),
                    'num_clusters': metrics.get('num_clusters', 0),
                    'silhouette_score': metrics.get('silhouette_score', 0),
                    'noise_ratio': metrics.get('noise_ratio', 0),
                    'response_time': metrics.get('avg_response_time', 0)
                })
        return pd.DataFrame(processed_data)

    def _process_fusion_results(self, results: List[Dict]) -> pd.DataFrame:
        """Process fusion experiment results."""
        if not results:
            return pd.DataFrame()

        processed_data = []
        for result in results:
            if result.get('success', False):
                metrics = result.get('metrics', {})
                processed_data.append({
                    'accuracy': metrics.get('accuracy', 0),
                    'fusion_strategy': metrics.get('fusion_strategy', 'unknown'),
                    'noise_ratio': metrics.get('noise_ratio', 0),
                    'response_time': metrics.get('avg_response_time', 0)
                })
        return pd.DataFrame(processed_data)

    def _process_category_results(self, results: List[Dict]) -> pd.DataFrame:
        """Process category experiment results."""
        if not results:
            return pd.DataFrame()

        processed_data = []
        for result in results:
            if result.get('success', False):
                metrics = result.get('metrics', {})
                processed_data.append({
                    'accuracy': metrics.get('accuracy', 0),
                    'category_type': metrics.get('category_type', 'unknown'),
                    'noise_ratio': metrics.get('noise_ratio', 0),
                    'response_time': metrics.get('avg_response_time', 0)
                })
        return pd.DataFrame(processed_data)

    def save_summary_report(self, results_data: Dict[str, pd.DataFrame]) -> None:
        """Save statistical summary of comparative results."""
        try:
            summary = {}
            for exp_type, df in results_data.items():
                summary[exp_type] = {
                    'accuracy': {
                        'mean': df['accuracy'].mean(),
                        'std': df['accuracy'].std(),
                        'min': df['accuracy'].min(),
                        'max': df['accuracy'].max()
                    }
                }

                if 'noise_ratio' in df.columns:
                    noise_impact = df.groupby('noise_ratio')['accuracy'].mean().to_dict()
                    summary[exp_type]['noise_impact'] = noise_impact

            report_path = self.plots_dir / 'summary_report.json'
            with open(report_path, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Saved summary report to {report_path}")

        except Exception as e:
            self.logger.error(f"Error in save_summary_report: {str(e)}")

    def plot_experiments_comparison(
    self,
    results: Dict[str, List[Dict]], 
    output_dir: Path,
    experiment_types: List[str] = ['baseline', 'random', 'adore']
    ) -> None:
        """Plot comparisons between baseline gold documents, random noise, and adversarial documents"""
        plots_dir = output_dir / "plots" / "comparisons"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Add new plots while keeping existing functionality
        try:
            baseline_results = [r for r in results.get('baseline', []) if r.get('experiment_type') == 'baseline']
            random_results = [r for r in results.get('baseline', []) if r.get('experiment_type') == 'random']
            adv_results = [r for r in results.get('baseline', []) if r.get('experiment_type') == 'adore']

            # New comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Accuracy comparison
            accuracies = {
                'Baseline (Gold)': np.mean([r['llm_evaluation']['correct'] for r in baseline_results]),
                'Random Noise': np.mean([r['llm_evaluation']['correct'] for r in random_results]),
                'Adversarial': np.mean([r['llm_evaluation']['correct'] for r in adv_results])
            }
            sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), ax=axes[0,0])
            axes[0,0].set_title('Accuracy by Document Type')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].tick_params(axis='x', rotation=45)

            # 2. Score distributions
            scores_data = []
            for exp_type, exp_results in [('Baseline', baseline_results), 
                                        ('Random', random_results),
                                        ('Adversarial', adv_results)]:
                scores = [r['llm_evaluation']['score'] for r in exp_results]
                scores_data.extend([(score, exp_type) for score in scores])
            
            df_scores = pd.DataFrame(scores_data, columns=['Score', 'Type'])
            sns.violinplot(data=df_scores, x='Type', y='Score', ax=axes[0,1])
            axes[0,1].set_title('LLM Evaluation Score Distributions')

            # 3. Performance degradation
            baseline_acc = accuracies['Baseline (Gold)']
            degradation = {
                'Random': baseline_acc - accuracies['Random Noise'],
                'Adversarial': baseline_acc - accuracies['Adversarial']
            }
            sns.barplot(x=list(degradation.keys()), y=list(degradation.values()), ax=axes[1,0])
            axes[1,0].set_title('Performance Degradation')
            axes[1,0].set_ylabel('Accuracy Drop')

            # 4. Statistical significance
            stats_results = {}
            for exp_type in ['Random', 'Adversarial']:
                exp_scores = ([r['llm_evaluation']['score'] for r in random_results] if exp_type == 'Random' 
                            else [r['llm_evaluation']['score'] for r in adv_results])
                baseline_scores = [r['llm_evaluation']['score'] for r in baseline_results]
                t_stat, p_val = stats.ttest_ind(baseline_scores, exp_scores)
                stats_results[f'vs_{exp_type}'] = {'t_stat': t_stat, 'p_value': p_val}

            axes[1,1].text(0.1, 0.5, f"Statistical Tests:\n" + \
                        "\n".join([f"{k}: p={v['p_value']:.3f}" for k,v in stats_results.items()]),
                        fontsize=10)
            axes[1,1].axis('off')

            plt.tight_layout()
            plt.savefig(plots_dir / 'doc_type_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Save numerical results
            results_summary = {
                'accuracies': accuracies,
                'degradation': degradation,
                'statistical_tests': stats_results,
                'sample_sizes': {
                    'baseline': len(baseline_results),
                    'random': len(random_results),
                    'adversarial': len(adv_results)
                }
            }
            
            with open(plots_dir / 'comparison_metrics.json', 'w') as f:
                json.dump(results_summary, f, indent=2)

        except Exception as e:
            print(f"Error generating comparison plots: {str(e)}")
