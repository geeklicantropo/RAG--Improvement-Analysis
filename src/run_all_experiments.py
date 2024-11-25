import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import subprocess
from tqdm import tqdm

from src.experiment_logger import ExperimentLogger

class BatchExperimentRunner:
    """Coordinates execution of all RAG experiments."""
    
    def __init__(self, base_output_dir: str = "experiments"):
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = ExperimentLogger(
            experiment_name=f"batch_run_{self.timestamp}",
            base_log_dir="logs"
        )

    def run_all_experiments(self):
        """Execute all experiments in sequence."""
        experiment_configs = [
            self._get_baseline_configs(),
            self._get_clustering_configs(),
            self._get_fusion_configs(),
            self._get_categories_configs()
        ]

        total_runs = sum(len(configs) for configs in experiment_configs)
        completed = 0

        try:
            with self.logger:
                progress_bar = tqdm(total=total_runs, desc="Overall Progress")
                
                # Run each experiment type
                for exp_idx, configs in enumerate(experiment_configs):
                    exp_name = [
                        "Baseline", "Clustering", "Fusion", "Categories"
                    ][exp_idx]
                    
                    self.logger.experiment_logger.info(f"\nStarting {exp_name} Experiments")
                    
                    for config in configs:
                        command = config['command']
                        desc = config['description']
                        
                        self.logger.experiment_logger.info(f"\nRunning: {desc}")
                        try:
                            result = subprocess.run(
                                command,
                                shell=True,
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            self.logger.experiment_logger.info("Completed successfully")
                            
                        except subprocess.CalledProcessError as e:
                            self.logger.log_error(e, f"Error in {desc}")
                            self.logger.experiment_logger.error(f"Output: {e.output}")
                            
                        completed += 1
                        progress_bar.update(1)
                        progress_bar.set_postfix({"Current": desc})
                        
                progress_bar.close()
                
                self.logger.experiment_logger.info("\nAll experiments completed")
                self.logger.log_metric("total_experiments", total_runs)
                self.logger.log_metric("successful_runs", completed)

        except Exception as e:
            self.logger.log_error(e, "Error in batch execution")
            raise

    def _get_baseline_configs(self) -> List[Dict[str, str]]:
        """Generate baseline experiment configurations."""
        return [
            {
                'command': 'python experiments/experiment0_baseline/main.py --retriever contriever',
                'description': 'Baseline-Contriever'
            },
            {
                'command': 'python experiments/experiment0_baseline/main.py --retriever bm25',
                'description': 'Baseline-BM25'
            },
            {
                'command': 'python experiments/experiment0_baseline/main.py --retriever contriever --use_random',
                'description': 'Baseline-Random'
            }
        ]

    def _get_clustering_configs(self) -> List[Dict[str, str]]:
        """Generate clustering experiment configurations."""
        configs = []
        for num_clusters in [3, 5, 7]:
            configs.append({
                'command': f'python experiments/experiment1_clustering/main.py --num_clusters {num_clusters}',
                'description': f'Clustering-{num_clusters}Clusters'
            })
        
        # Add random variant
        configs.append({
            'command': 'python experiments/experiment1_clustering/main.py --num_clusters 5 --use_random',
            'description': 'Clustering-Random'
        })
        return configs

    def _get_fusion_configs(self) -> List[Dict[str, str]]:
        """Generate fusion experiment configurations."""
        return [
            {
                'command': 'python experiments/experiment2_fusion/main.py --strategy rrf',
                'description': 'Fusion-RRF'
            },
            {
                'command': 'python experiments/experiment2_fusion/main.py --strategy linear',
                'description': 'Fusion-Linear'
            },
            {
                'command': 'python experiments/experiment2_fusion/main.py --strategy rrf --use_random',
                'description': 'Fusion-Random'
            }
        ]

    def _get_categories_configs(self) -> List[Dict[str, str]]:
        """Generate categories experiment configurations."""
        return [
            {
                'command': 'python -m experiments.experiment3_categories.main --config_type confidence',
                'description': 'Categories-Confidence'
            },
            {
                'command': 'python -m experiments.experiment3_categories.main --config_type fusion',
                'description': 'Categories-Fusion'
            },
            {
                'command': 'python -m experiments.experiment3_categories.main --config_type random',
                'description': 'Categories-Random'
            }
        ]

def main():
    parser = argparse.ArgumentParser(description="Run all RAG experiments")
    parser.add_argument('--output_dir', type=str, default='experiments',
                      help='Base output directory for experiments')
    args = parser.parse_args()

    runner = BatchExperimentRunner(args.output_dir)
    runner.run_all_experiments()

if __name__ == "__main__":
    main()