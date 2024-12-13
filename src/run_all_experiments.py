import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import warnings
import logging
from tqdm import tqdm
import json
import pandas as pd
import torch
import gc
import argparse
from collections import defaultdict
import numpy as np

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.extend([str(project_root)])

from experiments.experiment0_baseline.main import BaselineExperiment
from experiments.experiment0_baseline.config import BaselineConfigFactory
from experiments.plotting.plot_baseline import BaselinePlotter

from experiments.experiment1_clustering.main import ClusteringExperiment 
from experiments.experiment1_clustering.config import ClusteringConfig
from experiments.plotting.plot_clustering import ClusteringPlotter

from experiments.experiment2_fusion.main import FusionExperiment
from experiments.experiment2_fusion.config import FusionConfigFactory
from experiments.plotting.plot_fusion import FusionPlotter

from experiments.experiment3_categories.main import CategoriesExperiment
from experiments.experiment3_categories.config import CategoriesConfigFactory
from experiments.plotting.plot_categories import CategoriesPlotter

from src.utils.corpus_manager import CorpusManager  
from src.setup_experiments import ExperimentSetup
from src.experiment_logger import ExperimentLogger
from src.llm import LLM

from experiments.checkpoint_utils import (
    save_checkpoint,
    load_checkpoints,
    get_last_checkpoint_batch,
    merge_checkpoint_results
)

class OutputController:
    """Controls and filters experiment output."""
    def __init__(self, logger: ExperimentLogger):
        self.logger = logger
        self.output_buffer = []
        self.buffer_size = 100
        self.min_progress_interval = 1.0
        
    def write(self, text: str):
        """Write filtered output to buffer."""
        if self._should_log(text):
            self.output_buffer.append(text)
            if len(self.output_buffer) >= self.buffer_size:
                self.flush()
    
    def _should_log(self, text: str) -> bool:
        """Determine if text should be logged."""
        if not text.strip():
            return False
            
        if len(text) > 300:
            return False
            
        skip_patterns = [
            'token indices sequence',
            'Setting `pad_token_id`',
            'Special tokens have been added',
            'Using device:',
            'corpus_idx',
            'Document [',
            'embedding',
            'processing batch', 
            'tokenizer'
        ]
        
        if any(pattern in text.lower() for pattern in skip_patterns):
            return False
            
        allow_patterns = [
            '%|',  # tqdm progress
            'Error:',
            'WARNING:',
            'Running experiment',
            'Completed experiment',
            'Progress:',
            'Accuracy:'
        ]
        
        return any(pattern in text for pattern in allow_patterns)

    def flush(self):
        """Flush buffer to logger."""
        if self.output_buffer:
            output = ''.join(self.output_buffer)
            if output.strip():
                if any(('Error:' in line or 'WARNING:' in line) for line in self.output_buffer):
                    self.logger.experiment_logger.warning(output)
                else:
                    self.logger.experiment_logger.debug(output)
            self.output_buffer.clear()
    
    def __enter__(self):
        """Set up output capturing."""
        warnings.filterwarnings('ignore')
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        for logger_name in ["transformers", "torch", "pytorch_lightning"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original output streams."""
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.flush()
        
        if exc_type is not None:
            self.logger.log_error(exc_val, "Error output:\n" + '\n'.join(self.output_buffer))

class ExperimentPipeline:
    def __init__(self, base_output_dir: str = "experiments"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(base_output_dir)
        self.results_dir = self.base_output_dir / "results" / self.timestamp
        self.plots_dir = self.results_dir / "plots"
        self.setup = ExperimentSetup()
        
        # Initialize corpus manager
        self.corpus_manager = CorpusManager(
            base_corpus_path="data/processed/corpus_with_contriever_at150.json"
        )
        
        # Initialize logger
        self.logger = ExperimentLogger(
            experiment_name=f"experiment_pipeline_{self.timestamp}",
            base_log_dir="logs"
        )
        
        # Initialize LLM evaluator
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        
        self.output_controller = OutputController(self.logger)
        
        self.enabled_experiments = {
            'baseline': True,
            'clustering': True,
            'fusion': True,
            'categories': True
        }
        
        self.plotting_enabled = True

    def run_pipeline(self):
        with self.output_controller:
            start_time = time.time()
            self.logger.log_step_start("Initializing experiment pipeline")
            
            # Initialize corpus manager 
            corpus_manager = CorpusManager(
                base_corpus_path="data/processed/corpus_with_contriever_at150.json"
            )
            
            # Perform setup
            if not self.setup.setup():
                self.logger.log_error(None, "Setup failed")
                return

            # Create output directories
            self.results_dir.mkdir(parents=True, exist_ok=True)
            if self.plotting_enabled:
                self.plots_dir.mkdir(parents=True, exist_ok=True)

            results = {}
            
            # Run enabled experiments with corpus manager and LLM evaluator
            for exp_type, enabled in self.enabled_experiments.items():
                if enabled:
                    exp_start = time.time()
                    self.logger.log_step_start(f"Running {exp_type} experiments")
                    
                    # Run experiment with LLM evaluator
                    if exp_type == 'baseline':
                        results[exp_type] = self.run_baseline_experiments(corpus_manager)
                    elif exp_type == 'clustering':
                        results[exp_type] = self.run_clustering_experiments(corpus_manager)
                    elif exp_type == 'fusion':
                        results[exp_type] = self.run_fusion_experiments()
                    elif exp_type == 'categories':
                        results[exp_type] = self.run_categories_experiments()
                    
                    # Load and merge checkpoints
                    checkpoint_dir = self.results_dir / exp_type / "checkpoints"
                    if checkpoint_dir.exists():
                        checkpoint_results = load_checkpoints(checkpoint_dir)
                        results[exp_type] = merge_checkpoint_results(checkpoint_results)
                    
                    # Evaluate results using LLM
                    results[exp_type] = self._evaluate_results_with_llm(results[exp_type])
                    
                    # Save checkpoints
                    save_checkpoint(
                        results[exp_type],
                        len(results[exp_type]),
                        self.results_dir / exp_type
                    )
                    
                    # Generate plots if enabled
                    if self.plotting_enabled:
                        plotter_mapping = {
                            'baseline': BaselinePlotter,
                            'clustering': ClusteringPlotter,
                            'fusion': FusionPlotter,
                            'categories': CategoriesPlotter
                        }
                        plotter_class = plotter_mapping.get(exp_type)
                        if plotter_class:
                            plotter = plotter_class(self.results_dir)
                            plotter.plot_results(results[exp_type])
                    
                    self.logger.log_step_end(f"Running {exp_type} experiments", exp_start)
            
            # Generate and save final report
            self._generate_evaluation_report(results)
            self.logger.log_step_end("Pipeline execution", start_time)

    def _evaluate_results_with_llm(self, results: List[Dict]) -> List[Dict]:
        evaluated_results = []
        for result in tqdm(results, desc="Evaluating with LLM"):
            eval_prompt = f"""
            Question: {result['query']}
            Generated Answer: {result['generated_answer']}
            Gold Answer: {result['gold_answer']}
            Context: {result.get('context', '')}

            Evaluate the answer's correctness considering:
            1. Factual accuracy compared to gold answer
            2. Completeness of information
            3. Context utilization
            4. Semantic equivalence

            Rate each aspect 0-100 and explain why.
            """
            eval_result = self.llm.evaluate_answer(
                prompt=eval_prompt,
                context=result.get('context'),
                gold_answer=result['gold_answer']
            )
            result['evaluation'] = eval_result
            evaluated_results.append(result)
        return evaluated_results

    def _generate_evaluation_report(self, all_results: Dict[str, List[Dict]]) -> None:
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': self.timestamp,
            'experiments': {}
        }
        
        for exp_type, results in all_results.items():
            exp_metrics = {
                'accuracy': np.mean([r['llm_evaluation']['correct'] for r in results]),
                'avg_score': np.mean([r['llm_evaluation']['score'] for r in results]),
                'position_impact': self._analyze_position_impact(results),
                'noise_impact': self._analyze_noise_impact(results)
            }
            report['experiments'][exp_type] = exp_metrics
            
        report_path = self.results_dir / f"evaluation_report_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def _analyze_position_impact(self, results: List[Dict]) -> Dict:
        position_metrics = defaultdict(list)
        for result in results:
            if 'gold_position' in result:
                position_metrics[result['gold_position']].append(
                    result['llm_evaluation']['score']
                )
        return {
            pos: np.mean(scores) 
            for pos, scores in position_metrics.items()
        }
        
    def _analyze_noise_impact(self, results: List[Dict]) -> Dict:
        noise_metrics = defaultdict(list)
        for result in results:
            if 'noise_ratio' in result:
                noise_metrics[result['noise_ratio']].append(
                    result['llm_evaluation']['score']
                )
        return {
            ratio: np.mean(scores)
            for ratio, scores in noise_metrics.items()
        }

    def run_baseline_experiments(self, corpus_manager: CorpusManager) -> List[Dict]:
        """Run baseline experiments."""
        configs = [
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment0_baseline/main.py',
                'args': {
                    'retriever': 'contriever',
                    'corpus_manager': corpus_manager,
                    'llm_evaluator': self.llm,
                    'gold_position': None,
                    'num_documents': 7,
                    'use_test': False
                },
                'description': 'Baseline-Contriever'
            },
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment0_baseline/main.py',
                'args': {
                    'retriever': 'bm25',
                    'corpus_manager': corpus_manager,
                    'llm_evaluator': self.llm,
                    'gold_position': None,
                    'num_documents': 7,
                    'use_test': True
                },
                'description': 'Baseline-BM25'
            },
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment0_baseline/main.py',
                'args': {
                    'retriever': 'contriever',
                    'corpus_manager': corpus_manager,
                    'llm_evaluator': self.llm,
                    'use_random': True,
                    'num_documents': 7,
                    'use_test': False
                },
                'description': 'Baseline-Random'
            }
        ]
        return self.run_experiment_batch(configs, 'baseline')

    def run_clustering_experiments(self, corpus_manager: CorpusManager) -> List[Dict]:
        """Run clustering experiments."""
        configs = [
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment1_clustering/main.py',
                'args': {
                    'num_clusters': k,
                    'corpus_manager': corpus_manager,
                    'llm_evaluator': self.llm,
                    'use_random': False,
                    'compute_new_embeddings': True
                },
                'description': f'Clustering-{k}Clusters'
            }
            for k in [3, 5, 7]
        ]
        configs.append({
            'command': f'{sys.executable} {project_root}/experiments/experiment1_clustering/main.py',
            'args': {
                'num_clusters': 5,
                'corpus_manager': corpus_manager,
                'llm_evaluator': self.llm,
                'use_random': True,
                'compute_new_embeddings': True
            },
            'description': 'Clustering-Random'
        })
        return self.run_experiment_batch(configs, 'clustering')

    def run_fusion_experiments(self) -> List[Dict]:
        """Run fusion experiments."""
        configs = [
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment2_fusion/main.py',
                'args': {
                    'strategy': strategy,
                    'llm_evaluator': self.llm,
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
                'llm_evaluator': self.llm,
                'use_random': True
            },
            'description': 'Fusion-Random'
        })
        return self.run_experiment_batch(configs, 'fusion')

    def run_categories_experiments(self) -> List[Dict]:
        """Run categorization experiments."""
        configs = [
            {
                'command': f'{sys.executable} {project_root}/experiments/experiment3_categories/main.py',
                'args': {
                    'config_type': config_type,
                    'llm_evaluator': self.llm
                },
                'description': f'Categories-{config_type.capitalize()}'
            }
            for config_type in ['confidence', 'fusion', 'random']
        ]
        return self.run_experiment_batch(configs, 'categories')

    def run_experiment_batch(self, configs: List[Dict], experiment_type: str) -> List[Dict]:
        time.sleep(1)
        """Run a batch of experiments of the same type."""
        results = []
        
        for config in tqdm(configs, desc=f"Running {experiment_type} experiments"):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Reduce batch size if previous experiment failed
                if results and not results[-1].get('success', False):
                    config['args']['batch_size'] = max(1, config['args'].get('batch_size', 8) // 2)
                    
                output_dir = self.results_dir / experiment_type / config['description']
                output_dir.mkdir(parents=True, exist_ok=True)
                
                experiment_args = config['args'].copy()
                experiment_args['output_dir'] = str(output_dir)
                
                experiment_module = __import__(
                    f"experiments.{self.experiment_dir_mapping[experiment_type]}.main",
                    fromlist=['main']
                )
                experiment_results = experiment_module.main(experiment_args)
                
                if experiment_results:
                    results_dict, metrics_dict = experiment_results
                    result = self._process_experiment_results(
                        config, metrics_dict, output_dir
                    )
                    results.append(result)
                    
                    self.logger.experiment_logger.info(
                        f"Completed {config['description']} with "
                        f"accuracy: {metrics_dict.get('accuracy', 0):.4f}"
                    )
                    
            except Exception as e:
                self.logger.log_error(e, f"Error in {config['description']}")
                results.append({
                    'config': config,
                    'error': str(e),
                    'success': False
                })
                
            # Save checkpoint after each experiment
            checkpoint_path = output_dir / "checkpoints" / f"checkpoint_{len(results)}.json"
            save_checkpoint(results, len(results), output_dir)
                
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
        
        results_file = output_dir / 'experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        self.logger.log_metric(
            f"{config['description']}_accuracy",
            metrics_dict.get('accuracy', 0)
        )
        
        return result

def parse_arguments():
    """Parse command line arguments."""
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
        type=str,
        default='true',
        help='Whether to generate plots (true/false)'
    )
   
    args = parser.parse_args()
    args.save_plots = args.save_plots.lower() == 'true'
    return args

def main():
    """Main entry point for running all experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        args = parse_arguments()
        
        # Load global config and set GPU memory settings
        try:
            with open("experiments/experiments_config.json") as f:
                global_config = json.load(f)['global']
        except Exception as e:
            logging.error(f"Error loading global config: {str(e)}")
            raise
            
        # Set GPU memory settings
        if torch.cuda.is_available():
            gpu_memory_threshold = global_config.get('gpu_memory_threshold', 0.9)
            torch.cuda.set_per_process_memory_fraction(gpu_memory_threshold)
            
        # Create and configure pipeline
        pipeline = ExperimentPipeline(args.output_dir)
        
        experiments_to_run = {
            'baseline': not args.skip_baseline,
            'clustering': not args.skip_clustering,
            'fusion': False, #not args.skip_fusion,
            'categories': False #not args.skip_categories
        }
        
        pipeline.enabled_experiments = experiments_to_run
        pipeline.plotting_enabled = args.save_plots
        
        # Run pipeline
        pipeline.run_pipeline()
            
    except Exception as e:
        logging.error(f"Error in experiment pipeline: {str(e)}", exc_info=True)
        raise
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()