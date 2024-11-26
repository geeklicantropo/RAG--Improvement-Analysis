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
import logging
import contextlib
import io
import warnings

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
from experiments.experiment0_baseline.config import BaselineConfigFactory, BaselineConfig
from experiments.experiment0_baseline.utils import save_metrics

from experiments.experiment1_clustering.main import ClusteringExperiment
from experiments.experiment1_clustering.config import ClusteringConfig
#from experiments.experiment1_clustering.utils import ClusteringExperimentUtils

from experiments.experiment2_fusion.main import FusionExperiment
from experiments.experiment2_fusion.config import FusionConfigFactory, FusionConfig
from experiments.experiment2_fusion.utils import FusionExperimentUtils

#from experiments.experiment3_categories.main import CategoriesExperiment
from experiments.experiment3_categories.config import CategoriesConfigFactory, CategoriesConfig
from experiments.experiment3_categories.utils import CategoriesExperimentUtils

class OutputController:
    """Enhanced output controller with strict filtering"""
    def __init__(self, logger: ExperimentLogger):
        self.logger = logger
        self.stream = io.StringIO()
        self.output_buffer = []
        self.buffer_size = 100
        self.last_progress_update = 0
        self.min_progress_interval = 1.0  # seconds
        
    def write(self, text: str):
        """Write to buffer with enhanced filtering."""
        if self._should_log(text):
            self.output_buffer.append(text)
            if len(self.output_buffer) >= self.buffer_size:
                self.flush()
    
    def _should_log(self, text: str) -> bool:
        """More aggressive filter function to determine if text should be logged."""
        # Skip empty or whitespace-only lines
        if not text.strip():
            return False
            
        # Skip long text content (like corpus entries)
        if len(text) > 300:  # More restrictive length threshold
            return False
            
        # Expanded skip patterns
        skip_patterns = [
            # Original patterns
            'token indices sequence length is longer than',
            'Setting `pad_token_id`',
            'You are using',
            'Special tokens have been added',
            'full_corpus_idx',
            'title',
            '_idx',
            'corpus',
            # Additional patterns
            'Document [',
            'text',
            'idx:',
            'question',
            'answer',
            'retriever',
            'embedding',
            'loading',
            'processing',
            'computing',
            'creating'
        ]
        
        if any(pattern.lower() in text.lower() for pattern in skip_patterns):
            return False
            
        # Only allow specific patterns
        allow_patterns = [
            '%|',  # tqdm progress bar
            'Error:',
            'WARNING:',
            'Running experiment',
            'Completed experiment',
            'Starting',
            'Progress:',
            'Finished'
        ]
        
        return any(pattern in text for pattern in allow_patterns)

    def flush(self):
        """Flush buffer to logger."""
        if self.output_buffer:
            output = ''.join(self.output_buffer)
            if output.strip():
                # Only log important messages at INFO level
                if any(('Error:' in line or 'WARNING:' in line) for line in self.output_buffer):
                    self.logger.experiment_logger.warning(output)
                else:
                    self.logger.experiment_logger.debug(output)
            self.output_buffer.clear()
    
    def __enter__(self):
        # Suppress all warnings and debug output
        warnings.filterwarnings('ignore')
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Set strict logging levels
        for logger_name in [
            "transformers", 
            "torch", 
            "faiss", 
            "datasets", 
            "numpy",
            "pytorch_lightning",
            "accelerate"
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        # Flush existing output
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Save original stdout/stderr
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        
        # Replace with our controller
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout/stderr
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
        # Final flush
        self.flush()
        
        # Log any errors
        if exc_type is not None:
            self.logger.log_error(exc_val, "Error output:\n" + '\n'.join(self.output_buffer))

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
        self.enabled_experiments = {
            'baseline': True,
            'clustering': True,
            'fusion': True,
            'categories': True
        }
        self.plotting_enabled = True
        
        # Initialize output controller
        self.output_controller = OutputController(self.logger)

    def set_experiment_flags(self, flags: Dict[str, bool]):
        """Set which experiments to run."""
        self.enabled_experiments.update(flags)
        self.logger.log_metric("enabled_experiments", self.enabled_experiments)

    def set_plotting_enabled(self, enabled: bool):
        """Set whether to generate plots."""
        self.plotting_enabled = enabled
        self.logger.log_metric("plotting_enabled", enabled)

    @contextlib.contextmanager
    def managed_execution(self):
        """Context manager for controlled experiment execution."""
        with self.output_controller:
            with self.logger:
                try:
                    yield
                except Exception as e:
                    self.logger.log_error(e, "Experiment execution failed")
                    raise

    def run_pipeline(self):
        """Run the experiment pipeline with enhanced output control."""
        with self.managed_execution():
            self.logger.log_step_start("Initializing experiment pipeline")
            
            if not self.setup.setup():
                self.logger.log_error(None, "Setup failed")
                return
            
            self.results_dir.mkdir(parents=True, exist_ok=True)
            if self.plotting_enabled:
                self.plots_dir.mkdir(parents=True, exist_ok=True)

            results = {}
            
            # Run each enabled experiment type
            for exp_type in self.enabled_experiments:
                if self.enabled_experiments[exp_type]:
                    self.logger.log_step_start(f"Running {exp_type} experiments")
                    run_method = getattr(self, f"run_{exp_type}_experiments")
                    results[exp_type] = run_method()
                    
                    # Plot results if enabled
                    if self.plotting_enabled:
                        plotter_class = globals()[f"{exp_type.capitalize()}Plotter"]
                        plotter = plotter_class(self.results_dir)
                        plotter.plot_results(results[exp_type])
            
            self.generate_report(results)
            self.logger.log_step_end("Pipeline execution completed")

    # Running experimments methods
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
                    'strategy': strategy,
                    'use_random': False
                },
                'description': f'Fusion-{strategy}'
            }
            for strategy in ['rrf', 'linear']
        ]
        configs.append({
            'command': f'{sys.executable} {project_root}/experiments/experiment2_fusion/main.py',
            'args': {
                'strategy': 'rrf',
                'use_random': True
            },
            'description': 'Fusion-Random'
        })
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
        """Run a batch of experiments with enhanced output control."""
        results = []
        
        # Remove the OutputController here since we already have it in managed_execution
        for config in tqdm(configs, desc=f"Running {experiment_type} experiments", 
                        file=sys.stderr):
            try:
                output_dir = self.results_dir / experiment_type / config['description']
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Configure experiment
                experiment_args = config['args'].copy()
                experiment_args['output_dir'] = str(output_dir)
                
                # Run experiment without additional output control
                module_path = f"experiments.{self.experiment_dir_mapping[experiment_type]}.main"
                experiment_module = importlib.import_module(module_path)
                experiment_results = experiment_module.main(experiment_args)
                
                if experiment_results:
                    results_dict, metrics_dict = experiment_results
                    result = self._process_experiment_results(
                        config, metrics_dict, output_dir
                    )
                    results.append(result)
                    
                    # Only log essential information
                    self.logger.experiment_logger.info(
                        f"Completed {config['description']} with "
                        f"accuracy: {metrics_dict.get('accuracy', 0):.4f}"
                    )
                else:
                    raise ValueError(f"No results from {config['description']}")
                    
            except Exception as e:
                self.logger.log_error(e, f"Error in {config['description']}")
                results.append({
                    'config': config,
                    'error': str(e),
                    'success': False
                })
                
        return results

    def _process_experiment_results(
        self, 
        config: Dict, 
        metrics_dict: Dict, 
        output_dir: Path
    ) -> Dict:
        """Process and save individual experiment results."""
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
        
        # Save individual results
        results_file = output_dir / 'experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        self.logger.log_metric(
            f"{config['description']}_accuracy",
            metrics_dict.get('accuracy', 0)
        )
        
        return result

    # Your existing generate_report method remains unchanged
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

# Parser
def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run complete experiment pipeline")
    
    # Directory configurations
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='experiments',
        help='Base output directory'
    )
    parser.add_argument(
        '--log_dir', 
        type=str, 
        default='logs',
        help='Directory for experiment logs'
    )
    
    # Experiment selection
    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip baseline experiments'
    )
    parser.add_argument(
        '--skip_clustering',
        action='store_true',
        help='Skip clustering experiments'
    )
    parser.add_argument(
        '--skip_fusion',
        action='store_true',
        help='Skip fusion experiments'
    )
    parser.add_argument(
        '--skip_categories',
        action='store_true',
        help='Skip categories experiments'
    )
    
    # Visualization options
    parser.add_argument(
        '--save_plots',
        type=str2bool,
        default=True,
        help='Whether to generate plots (true/false)'
    )
    
    # Additional experiment-specific arguments
    parser.add_argument(
        '--cluster_sizes',
        nargs='+',
        type=int,
        default=[3, 5, 7],
        help='Cluster sizes to test'
    )
    parser.add_argument(
        '--fusion_strategies',
        nargs='+',
        type=str,
        default=['rrf', 'linear'],
        help='Fusion strategies to test'
    )
    parser.add_argument(
        '--category_types',
        nargs='+',
        type=str,
        default=['confidence', 'fusion', 'random'],
        help='Category types to test'
    )
    
    args = parser.parse_args()
    return args

def main():
    """Main entry point for running all experiments."""
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create experiment pipeline
    try:
        pipeline = ExperimentPipeline(args.output_dir)
        
        # Configure experiments based on arguments
        experiments_to_run = {
            'baseline': not args.skip_baseline,
            'clustering': not args.skip_clustering,
            'fusion': not args.skip_fusion,
            'categories': not args.skip_categories
        }
        
        # Update experiment configurations
        if not args.skip_clustering:
            pipeline.cluster_sizes = args.cluster_sizes
        if not args.skip_fusion:
            pipeline.fusion_strategies = args.fusion_strategies
        if not args.skip_categories:
            pipeline.category_types = args.category_types
            
        # Configure pipeline
        pipeline.set_experiment_flags(experiments_to_run)
        pipeline.set_plotting_enabled(args.save_plots)
        
        # Run pipeline with enhanced output control
        with pipeline.managed_execution():
            pipeline.run_pipeline()
            
    except Exception as e:
        logging.error(f"Error in experiment pipeline: {str(e)}", exc_info=True)
        raise
    finally:
        # Ensure all output is flushed
        sys.stdout.flush()
        sys.stderr.flush()

if __name__ == "__main__":
    main()