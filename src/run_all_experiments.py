import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import subprocess
from tqdm import tqdm
import pandas as pd
import importlib
import json
from typing import Dict, List, Any

project_root = Path(__file__).parent.parent
sys.path.extend([
    str(project_root),
    str(project_root / "experiments"),
    str(project_root / "experiments/plotting")
])

from experiments.plotting.plot_baseline import BaselinePlotter 
from experiments.plotting.plot_clustering import ClusteringPlotter
from experiments.plotting.plot_fusion import FusionPlotter
from experiments.plotting.plot_categories import CategoriesPlotter
from setup_experiments import ExperimentSetup
from experiment_logger import ExperimentLogger
from experiments.experiment0_baseline.main import BaselineExperiment
from experiments.experiment0_baseline.config import BaselineConfig
from experiments.experiment0_baseline.utils import save_metrics

class ExperimentPipeline:
    def __init__(self, base_output_dir: str = "experiments"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(base_output_dir)
        self.results_dir = self.base_output_dir / "results" / self.timestamp
        self.plots_dir = self.results_dir / "plots"
        self.setup = ExperimentSetup()
        self.logger = ExperimentLogger(
            experiment_name=f"experiment_pipeline_{self.timestamp}",
            base_log_dir="logs"
        )
        self.experiment_dir_mapping = {
            'baseline': 'experiment0_baseline',
            'clustering': 'experiment1_clustering',
            'fusion': 'experiment2_fusion', 
            'categories': 'experiment3_categories'
        }

    def run_pipeline(self):
        try:
            with self.logger:
                self.logger.log_step_start("Initializing experiment pipeline")
                
                if not self.setup.setup():
                    self.logger.log_error(None, "Setup failed")
                    return
                
                self.results_dir.mkdir(parents=True, exist_ok=True)
                self.plots_dir.mkdir(parents=True, exist_ok=True)

                results = {}
                
                # Run each experiment type
                experiment_types = ['baseline', 'clustering', 'fusion', 'categories']
                for exp_type in experiment_types:
                    self.logger.log_step_start(f"Running {exp_type} experiments")
                    run_method = getattr(self, f"run_{exp_type}_experiments")
                    results[exp_type] = run_method()
                    
                    # Plot results
                    plotter_class = globals()[f"{exp_type.capitalize()}Plotter"]
                    plotter_class(self.results_dir).plot_results(results[exp_type])
                
                self.generate_report(results)
                self.logger.log_step_end("Pipeline execution completed")
                
        except Exception as e:
            self.logger.log_error(e, "Error in experiment pipeline")
            raise

    def run_baseline_experiments(self) -> List[Dict]:
        configs = [
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment0_baseline/main.py',
                'args': {
                    'retriever': 'contriever',
                    'gold_position': None,
                    'num_documents': 7
                },
                'description': 'Baseline-Contriever'
            },
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment0_baseline/main.py',
                'args': {
                    'retriever': 'bm25',
                    'gold_position': None,
                    'num_documents': 7
                },
                'description': 'Baseline-BM25'
            },
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment0_baseline/main.py',
                'args': {
                    'retriever': 'contriever',
                    'use_random': True,
                    'num_documents': 7
                },
                'description': 'Baseline-Random'
            }
        ]
        return self.run_experiment_batch(configs, 'baseline')

    def run_clustering_experiments(self) -> List[Dict]:
        configs = [
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment1_clustering/main.py',
                'args': {
                    'num_clusters': k,
                    'use_random': False
                },
                'description': f'Clustering-{k}Clusters'
            }
            for k in [3, 5, 7]
        ]
        configs.append({
            'command': f'{sys.executable} {project_root}/experiments/experiment1_clustering/main.py',
            'args': {
                'num_clusters': 5,
                'use_random': True
            },
            'description': 'Clustering-Random'
        })
        return self.run_experiment_batch(configs, 'clustering')

    def run_fusion_experiments(self) -> List[Dict]:
        configs = [
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment2_fusion/main.py',
                'args': {
                    'strategy': 'rrf',
                    'use_random': False
                },
                'description': 'Fusion-RRF'
            },
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment2_fusion/main.py',
                'args': {
                    'strategy': 'linear',
                    'use_random': False
                },
                'description': 'Fusion-Linear'
            },
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment2_fusion/main.py',
                'args': {
                    'strategy': 'rrf',
                    'use_random': True
                },
                'description': 'Fusion-Random'
            }
        ]
        return self.run_experiment_batch(configs, 'fusion')

    def run_categories_experiments(self) -> List[Dict]:
        configs = [
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment3_categories/main.py',
                'args': {
                    'config_type': config_type
                },
                'description': f'Categories-{config_type.capitalize()}'
            }
            for config_type in ['confidence', 'fusion', 'random']
        ]
        return self.run_experiment_batch(configs, 'categories')

    def run_experiment_batch(self, configs: List[Dict], experiment_type: str) -> List[Dict]:
        """Run a batch of experiments with improved error handling."""
        results = []
        
        for config in tqdm(configs, desc=f"Running {experiment_type} experiments"):
            try:
                output_dir = self.results_dir / experiment_type / config['description']
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Load appropriate experiment module
                experiment_dir = self.experiment_dir_mapping.get(experiment_type)
                if not experiment_dir:
                    raise ValueError(f"Unknown experiment type: {experiment_type}")
                    
                module_path = f"experiments.{experiment_dir}.main"
                experiment_module = importlib.import_module(module_path)
                
                # Configure experiment
                experiment_args = config['args'].copy()
                experiment_args['output_dir'] = str(output_dir)
                
                # Run experiment
                experiment_results = experiment_module.main(experiment_args)
                
                if experiment_results:
                    results_dict, metrics_dict = experiment_results
                    
                    # Ensure all required metrics are present
                    if not isinstance(metrics_dict, dict):
                        metrics_dict = {}
                    
                    result = {
                        'config': config,
                        'metrics': {
                            'accuracy': metrics_dict.get('accuracy', 0),
                            'total_examples': metrics_dict.get('total_examples', 0),
                            'avg_response_time': metrics_dict.get('avg_response_time', 0),
                            'avg_context_length': metrics_dict.get('avg_context_length', 0),
                            'description': config['description']
                        },
                        'output_dir': str(output_dir),
                        'success': True
                    }
                    
                    # Save individual experiment results
                    results_file = output_dir / 'experiment_results.json'
                    with open(results_file, 'w') as f:
                        json.dump(result, f, indent=2)
                        
                    results.append(result)
                    self.logger.log_metric(f"{config['description']}_accuracy", 
                                        metrics_dict.get('accuracy', 0))
                    
                else:
                    raise ValueError(f"No results returned from {config['description']}")
                    
            except Exception as e:
                self.logger.log_error(e, f"Error running {config['description']}")
                results.append({
                    'config': config,
                    'error': str(e),
                    'success': False
                })
                
        return results

    def generate_report(self, all_results: Dict[str, List[Dict]]):
        report_path = self.results_dir / "report.md"
        
        experiment_metrics = {}
        for exp_type, results in all_results.items():
            exp_metrics = []
            for r in results:
                if r['success']:
                    metrics = r['metrics'].copy()
                    metrics['experiment'] = r['config']['description']
                    exp_metrics.append(metrics)
            experiment_metrics[exp_type] = pd.DataFrame(exp_metrics)
            
        with open(report_path, 'w') as f:
            f.write(f"# Experiment Results Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_experiments = sum(len(r) for r in all_results.values())
            successful = sum(sum(1 for r in exp_results if r['success']) 
                           for exp_results in all_results.values())
                           
            f.write("## Summary\n")
            f.write(f"Total Experiments: {total_experiments}\n")
            f.write(f"Successful Experiments: {successful}\n")
            f.write(f"Failed Experiments: {total_experiments - successful}\n\n")
            
            for exp_type, metrics_df in experiment_metrics.items():
                f.write(f"## {exp_type.capitalize()} Experiments\n")
                f.write("### Average Metrics\n")
                f.write(metrics_df.mean().to_string())
                f.write("\n\n### Individual Experiment Results\n")
                f.write(metrics_df.to_string())
                f.write("\n\n")

def main():
    parser = argparse.ArgumentParser(description="Run complete experiment pipeline")
    parser.add_argument('--output_dir', type=str, default='experiments',
                      help='Base output directory')
    args = parser.parse_args()

    pipeline = ExperimentPipeline(args.output_dir)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()