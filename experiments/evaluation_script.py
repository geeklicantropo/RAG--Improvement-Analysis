import os
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from experiment_logger import ExperimentLogger
from utils import read_json, str2bool
from normalize_answers import are_answers_matching

class ExperimentEvaluator:
    """
    Evaluates and compares results across different experimental configurations.
    Handles clustering, fusion, and category-based results.
    """
    
    def __init__(self, base_results_dir: str, logger: ExperimentLogger):
        self.base_results_dir = base_results_dir
        self.logger = logger
        self.metrics = defaultdict(dict)
        self.experiment_configs = {}
        
    def load_experiment_results(
        self,
        experiment_name: str,
        results_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Load results from an experiment and its configuration."""
        try:
            self.logger.log_step_start(f"Loading {experiment_name} results")
            results = read_json(results_path)
            self.metrics[experiment_name]["results"] = results
            if config:
                self.experiment_configs[experiment_name] = config
            self.logger.log_step_end(f"Loading {experiment_name} results", time.time())
            
        except Exception as e:
            self.logger.log_error(e, f"Error loading results for {experiment_name}")
            raise
            
    def compute_metrics(self, experiment_name: str) -> Dict[str, float]:
        """Compute evaluation metrics for an experiment."""
        results = self.metrics[experiment_name]["results"]
        metrics = {}
        
        try:
            # Basic metrics
            total_examples = len(results)
            correct_answers = sum(1 for r in results if r['ans_match_after_norm'])
            accuracy = correct_answers / total_examples
            
            metrics.update({
                "accuracy": accuracy,
                "total_examples": total_examples,
                "correct_answers": correct_answers
            })
            
            # Clustering-specific metrics (if applicable)
            if "cluster_assignments" in results[0]:
                cluster_metrics = self._compute_clustering_metrics(results)
                metrics.update(cluster_metrics)
            
            # Fusion-specific metrics (if applicable)
            if "retriever_scores" in results[0]:
                fusion_metrics = self._compute_fusion_metrics(results)
                metrics.update(fusion_metrics)
            
            # Category-specific metrics (if applicable)
            if "category_info" in results[0]:
                category_metrics = self._compute_category_metrics(results)
                metrics.update(category_metrics)
            
            self.metrics[experiment_name]["computed_metrics"] = metrics
            return metrics
            
        except Exception as e:
            self.logger.log_error(e, f"Error computing metrics for {experiment_name}")
            raise
            
    def _compute_clustering_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute metrics specific to clustering experiments."""
        metrics = {}
        
        # Analyze per-cluster performance
        cluster_performance = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in results:
            clusters = result["cluster_assignments"]
            for doc_id, cluster_id in clusters.items():
                cluster_performance[cluster_id]["total"] += 1
                if result["ans_match_after_norm"]:
                    cluster_performance[cluster_id]["correct"] += 1
        
        # Calculate cluster-specific metrics
        for cluster_id, stats in cluster_performance.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            metrics[f"cluster_{cluster_id}_accuracy"] = accuracy
            
        # Calculate cluster diversity metrics
        unique_clusters = len(cluster_performance)
        metrics["num_clusters"] = unique_clusters
        
        return metrics
        
    def _compute_fusion_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute metrics specific to fusion experiments."""
        metrics = {}
        
        # Analyze retriever contribution
        retriever_contribution = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in results:
            scores = result["retriever_scores"]
            max_retriever = max(scores.items(), key=lambda x: x[1])[0]
            retriever_contribution[max_retriever]["total"] += 1
            if result["ans_match_after_norm"]:
                retriever_contribution[max_retriever]["correct"] += 1
        
        # Calculate retriever-specific metrics
        for retriever, stats in retriever_contribution.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            metrics[f"{retriever}_accuracy"] = accuracy
            
        return metrics
        
    def _compute_category_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute metrics specific to category-based experiments."""
        metrics = {}
        
        # Analyze per-category performance
        category_performance = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in results:
            categories = result["category_info"]
            for doc_id, category in categories.items():
                category_performance[category]["total"] += 1
                if result["ans_match_after_norm"]:
                    category_performance[category]["correct"] += 1
        
        # Calculate category-specific metrics
        for category, stats in category_performance.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            metrics[f"category_{category}_accuracy"] = accuracy
            
        return metrics
        
    def compare_experiments(
        self,
        experiment_names: List[str],
        metrics_to_compare: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare metrics across different experiments."""
        comparison_df = pd.DataFrame()
        
        for exp_name in experiment_names:
            if exp_name not in self.metrics:
                continue
                
            metrics = self.metrics[exp_name].get("computed_metrics", {})
            if not metrics:
                metrics = self.compute_metrics(exp_name)
                
            if metrics_to_compare:
                metrics = {k: v for k, v in metrics.items() if k in metrics_to_compare}
                
            comparison_df[exp_name] = pd.Series(metrics)
            
        return comparison_df
        
    def statistical_analysis(
        self,
        experiment_names: List[str],
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """Perform statistical analysis comparing experiments."""
        stats_results = {}
        
        # Collect results for each experiment
        experiment_results = {}
        for exp_name in experiment_names:
            results = self.metrics[exp_name]["results"]
            values = [1 if r["ans_match_after_norm"] else 0 for r in results]
            experiment_results[exp_name] = values
            
        # Perform pairwise statistical tests
        for i, exp1 in enumerate(experiment_names):
            for exp2 in experiment_names[i+1:]:
                # Chi-square test for independence
                contingency = pd.crosstab(
                    pd.Series(experiment_results[exp1]),
                    pd.Series(experiment_results[exp2])
                )
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
                
                stats_results[f"{exp1}_vs_{exp2}"] = {
                    "chi2": chi2,
                    "p_value": p_value
                }
                
        return stats_results
        
    def generate_visualizations(
        self,
        save_dir: str,
        experiment_names: Optional[List[str]] = None
    ) -> None:
        """Generate visualizations comparing experimental results."""
        os.makedirs(save_dir, exist_ok=True)
        
        if experiment_names is None:
            experiment_names = list(self.metrics.keys())
            
        # Accuracy comparison
        self._plot_accuracy_comparison(experiment_names, save_dir)
        
        # Experiment-specific visualizations
        for exp_name in experiment_names:
            results = self.metrics[exp_name]["results"]
            
            if "cluster_assignments" in results[0]:
                self._plot_cluster_analysis(exp_name, results, save_dir)
                
            if "retriever_scores" in results[0]:
                self._plot_fusion_analysis(exp_name, results, save_dir)
                
            if "category_info" in results[0]:
                self._plot_category_analysis(exp_name, results, save_dir)
                
    def _plot_accuracy_comparison(
        self,
        experiment_names: List[str],
        save_dir: str
    ) -> None:
        """Plot accuracy comparison across experiments."""
        accuracies = []
        for exp_name in experiment_names:
            metrics = self.metrics[exp_name].get("computed_metrics", {})
            accuracies.append(metrics.get("accuracy", 0))
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x=experiment_names, y=accuracies)
        plt.title("Accuracy Comparison Across Experiments")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "accuracy_comparison.png"))
        plt.close()
        
    def _plot_cluster_analysis(
        self,
        experiment_name: str,
        results: List[Dict],
        save_dir: str
    ) -> None:
        """Plot cluster-specific analysis."""
        cluster_accuracies = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in results:
            clusters = result["cluster_assignments"]
            for cluster_id in set(clusters.values()):
                cluster_accuracies[cluster_id]["total"] += 1
                if result["ans_match_after_norm"]:
                    cluster_accuracies[cluster_id]["correct"] += 1
                    
        accuracies = {
            k: v["correct"]/v["total"] 
            for k, v in cluster_accuracies.items()
        }
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
        plt.title(f"Cluster Performance - {experiment_name}")
        plt.xlabel("Cluster ID")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_cluster_analysis.png"))
        plt.close()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare experimental results.")
    
    parser.add_argument('--base_results_dir', type=str, required=True,
                        help='Base directory containing experiment results')
    parser.add_argument('--experiments', nargs='+', required=True,
                        help='List of experiment names to evaluate')
    parser.add_argument('--metrics', nargs='+', default=['accuracy'],
                        help='List of metrics to compare')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory for saving evaluation results')
    parser.add_argument('--generate_plots', type=str2bool, default=True,
                        help='Whether to generate visualization plots')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Initialize logger
    logger = ExperimentLogger(
        experiment_name="experiment_evaluation",
        base_log_dir="logs"
    )
    
    try:
        with logger:
            # Initialize evaluator
            evaluator = ExperimentEvaluator(args.base_results_dir, logger)
            
            # Load results for each experiment
            for exp_name in args.experiments:
                results_path = os.path.join(
                    args.base_results_dir,
                    exp_name,
                    "results.json"
                )
                evaluator.load_experiment_results(exp_name, results_path)
                
            # Compare experiments
            comparison_df = evaluator.compare_experiments(
                args.experiments,
                args.metrics
            )
            
            # Perform statistical analysis
            stats_results = evaluator.statistical_analysis(args.experiments)
            
            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            comparison_df.to_csv(
                os.path.join(args.output_dir, "metric_comparison.csv")
            )
            
            with open(os.path.join(args.output_dir, "statistical_analysis.json"), "w") as f:
                json.dump(stats_results, f, indent=2)
                
            # Generate visualizations
            if args.generate_plots:
                evaluator.generate_visualizations(
                    os.path.join(args.output_dir, "plots"),
                    args.experiments
                )
                
    except Exception as e:
        logger.log_error(e, "Error in main execution")
        raise

if __name__ == "__main__":
    main()