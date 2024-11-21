import logging
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import paths

class ExperimentVisualizer:
    """Handles visualization and comparison of experiment results."""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Create visualization directories
        self.viz_dir = paths.VISUALIZATION_DIR
        self.individual_dir = self.viz_dir / "individual"
        self.comparison_dir = self.viz_dir / "comparison"
        self.tables_dir = self.viz_dir / "tables"
        
        for dir_path in [self.individual_dir, self.comparison_dir, self.tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set style for visualizations
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = paths.LOGS_DIR / "visualization"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                ),
                logging.StreamHandler()
            ]
        )

    def load_experiment_results(self) -> Dict[str, Dict[str, Any]]:
        """Load results from all experiments."""
        experiment_results = {}
        experiment_dirs = {
            'baseline': paths.RESULTS_DIR / "baseline",
            'kmeans': paths.RESULTS_DIR / "kmeans",
            'fusion_kmeans': paths.RESULTS_DIR / "fusion",
            'fusion_categories': paths.RESULTS_DIR / "fusion_categories"
        }
        
        for exp_name, exp_dir in experiment_dirs.items():
            try:
                # Get most recent result directory
                result_dirs = sorted(exp_dir.glob("*"))
                if not result_dirs:
                    continue
                    
                latest_dir = result_dirs[-1]
                metrics_file = latest_dir / "metrics.json"
                results_file = latest_dir / "results_final.json"
                
                if metrics_file.exists() and results_file.exists():
                    with open(metrics_file) as f:
                        metrics = yaml.safe_load(f)
                    with open(results_file) as f:
                        results = yaml.safe_load(f)
                        
                    experiment_results[exp_name] = {
                        'metrics': metrics,
                        'results': results
                    }
                    
                    self.logger.info(f"Loaded results for {exp_name}")
                    
            except Exception as e:
                self.logger.error(f"Error loading results for {exp_name}: {e}")
        
        return experiment_results

    def generate_individual_visualizations(self, results: Dict[str, Dict[str, Any]]):
        """Generate visualizations for each experiment individually."""
        for exp_name, exp_data in results.items():
            try:
                # Performance over time
                self._plot_performance_over_time(exp_name, exp_data)
                
                # Score distributions
                self._plot_score_distributions(exp_name, exp_data)
                
                # Document usage patterns
                self._plot_document_usage(exp_name, exp_data)
                
                if 'kmeans' in exp_name:
                    self._plot_cluster_analysis(exp_name, exp_data)
                elif 'categories' in exp_name:
                    self._plot_category_analysis(exp_name, exp_data)
                    
                self.logger.info(f"Generated individual visualizations for {exp_name}")
                
            except Exception as e:
                self.logger.error(f"Error generating visualizations for {exp_name}: {e}")

    def generate_comparison_visualizations(self, results: Dict[str, Dict[str, Any]]):
        """Generate visualizations comparing all experiments."""
        try:
            # Performance metrics comparison
            self._plot_performance_comparison(results)
            
            # Retrieval effectiveness comparison
            self._plot_retrieval_comparison(results)
            
            # Time efficiency comparison
            self._plot_time_efficiency_comparison(results)
            
            # Generate summary tables
            self._generate_summary_tables(results)
            
            self.logger.info("Generated comparison visualizations")
            
        except Exception as e:
            self.logger.error(f"Error generating comparison visualizations: {e}")

    def _plot_performance_over_time(self, exp_name: str, exp_data: Dict[str, Any]):
        """Plot performance metrics over time for an experiment."""
        metrics = exp_data['metrics']
        results = exp_data['results']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot retrieval time
        retrieval_times = [r['metrics']['retrieval_time'] for r in results]
        ax1.plot(retrieval_times, label='Retrieval Time')
        ax1.set_title(f'{exp_name} - Retrieval Time Over Queries')
        ax1.set_xlabel('Query Number')
        ax1.set_ylabel('Time (seconds)')
        ax1.legend()
        
        # Plot document scores
        scores = [r['metrics']['retrieved_doc_scores'] for r in results]
        ax2.boxplot(scores)
        ax2.set_title(f'{exp_name} - Document Score Distribution')
        ax2.set_xlabel('Query Number')
        ax2.set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(self.individual_dir / f"{exp_name}_performance.png")
        plt.close()

    def _plot_score_distributions(self, exp_name: str, exp_data: Dict[str, Any]):
        """Plot score distributions for retrieved documents."""
        results = exp_data['results']
        
        all_scores = []
        for result in results:
            all_scores.extend(result['metrics']['retrieved_doc_scores'])
            
        plt.figure(figsize=(10, 6))
        sns.histplot(all_scores, bins=50)
        plt.title(f'{exp_name} - Document Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Count')
        
        plt.savefig(self.individual_dir / f"{exp_name}_scores.png")
        plt.close()

    def _plot_document_usage(self, exp_name: str, exp_data: Dict[str, Any]):
        """Plot document usage patterns."""
        results = exp_data['results']
        
        doc_usage = defaultdict(int)
        for result in results:
            for doc in result['fused_docs' if 'fused_docs' in result else 'retrieved_docs']:
                doc_usage[doc['id']] += 1
                
        usage_counts = list(doc_usage.values())
        
        plt.figure(figsize=(10, 6))
        sns.histplot(usage_counts, bins=30)
        plt.title(f'{exp_name} - Document Usage Distribution')
        plt.xlabel('Times Used')
        plt.ylabel('Number of Documents')
        
        plt.savefig(self.individual_dir / f"{exp_name}_doc_usage.png")
        plt.close()

    def _plot_cluster_analysis(self, exp_name: str, exp_data: Dict[str, Any]):
        """Plot cluster-specific analysis."""
        results = exp_data['results']
        
        cluster_metrics = defaultdict(list)
        for result in results:
            cluster = result['query_cluster']
            cluster_metrics[cluster].append(result['metrics']['retrieval_time'])
            
        plt.figure(figsize=(10, 6))
        plt.boxplot([times for times in cluster_metrics.values()],
                   labels=[f'Cluster {i}' for i in range(len(cluster_metrics))])
        plt.title(f'{exp_name} - Retrieval Time by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Retrieval Time (seconds)')
        
        plt.savefig(self.individual_dir / f"{exp_name}_cluster_analysis.png")
        plt.close()

    def _plot_category_analysis(self, exp_name: str, exp_data: Dict[str, Any]):
        """Plot category-specific analysis."""
        results = exp_data['results']
        
        category_metrics = defaultdict(list)
        for result in results:
            category = result['query_category']
            category_metrics[category].append(result['metrics']['retrieval_time'])
            
        plt.figure(figsize=(12, 6))
        plt.boxplot([times for times in category_metrics.values()],
                   labels=list(category_metrics.keys()))
        plt.title(f'{exp_name} - Retrieval Time by Category')
        plt.xlabel('Category')
        plt.ylabel('Retrieval Time (seconds)')
        plt.xticks(rotation=45)
        
        plt.savefig(self.individual_dir / f"{exp_name}_category_analysis.png")
        plt.close()

    def _plot_performance_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Plot performance comparison across experiments."""
        metrics_data = {
            name: {
                'avg_retrieval_time': np.mean([
                    r['metrics']['retrieval_time'] for r in data['results']
                ]),
                'avg_score': np.mean([
                    np.mean(r['metrics']['retrieved_doc_scores']) 
                    for r in data['results']
                ])
            }
            for name, data in results.items()
        }
        
        df = pd.DataFrame(metrics_data).T
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        df['avg_retrieval_time'].plot(kind='bar', ax=ax1)
        ax1.set_title('Average Retrieval Time')
        ax1.set_ylabel('Time (seconds)')
        
        df['avg_score'].plot(kind='bar', ax=ax2)
        ax2.set_title('Average Document Score')
        ax2.set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "performance_comparison.png")
        plt.close()

    def _plot_retrieval_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Plot retrieval effectiveness comparison."""
        retrieval_metrics = {
            name: {
                'precision': np.mean([
                    r['metrics'].get('precision', 0) for r in data['results']
                ]),
                'recall': np.mean([
                    r['metrics'].get('recall', 0) for r in data['results']
                ])
            }
            for name, data in results.items()
        }
        
        df = pd.DataFrame(retrieval_metrics).T
        
        plt.figure(figsize=(10, 6))
        df.plot(kind='bar')
        plt.title('Retrieval Effectiveness Comparison')
        plt.ylabel('Score')
        plt.legend(title='Metric')
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "retrieval_comparison.png")
        plt.close()

    def _plot_time_efficiency_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Plot time efficiency comparison."""
        time_metrics = {
            name: {
                'retrieval': data['metrics']['total_retrieval_time'],
                'processing': data['metrics'].get('total_fusion_time', 0),
                'generation': data['metrics']['total_generation_time']
            }
            for name, data in results.items()
        }
        
        df = pd.DataFrame(time_metrics).T
        
        plt.figure(figsize=(10, 6))
        df.plot(kind='bar', stacked=True)
        plt.title('Time Efficiency Comparison')
        plt.ylabel('Time (seconds)')
        plt.legend(title='Phase')
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "time_efficiency_comparison.png")
        plt.close()

    def _generate_summary_tables(self, results: Dict[str, Dict[str, Any]]):
        """Generate summary tables comparing all experiments."""
        # Overall performance summary
        performance_summary = {
            name: {
                'Total Time': data['metrics']['total_duration'],
                'Avg Retrieval Time': data['metrics']['avg_retrieval_time'],
                'Avg Generation Time': data['metrics']['avg_generation_time'],
                'Total Samples': data['metrics']['total_samples_processed']
            }
            for name, data in results.items()
        }
        
        df_performance = pd.DataFrame(performance_summary).T
        df_performance.to_csv(self.tables_dir / "performance_summary.csv")
        
        # Detailed metrics summary
        metrics_summary = {
            name: {
                'Retrieval Success Rate': np.mean([
                    1 if r['metrics']['num_retrieved_docs'] > 0 else 0 
                    for r in data['results']
                ]),
                'Avg Docs Retrieved': np.mean([
                    r['metrics']['num_retrieved_docs'] for r in data['results']
                ]),
                'Avg Doc Score': np.mean([
                    np.mean(r['metrics']['retrieved_doc_scores']) 
                    for r in data['results']
                ])
            }
            for name, data in results.items()
        }
        
        df_metrics = pd.DataFrame(metrics_summary).T
        df_metrics.to_csv(self.tables_dir / "metrics_summary.csv")

def main():
    """Main execution function."""
    try:
        visualizer = ExperimentVisualizer()
        
        # Load results
        results = visualizer.load_experiment_results()
        
        if not results:
            raise ValueError("No experiment results found")
            
        # Generate visualizations
        visualizer.generate_individual_visualizations(results)
        visualizer.generate_comparison_visualizations(results)
        
        visualizer.logger.info("Visualization generation completed successfully")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error in visualization generation: {e}")
        raise

if __name__ == "__main__":
    main()