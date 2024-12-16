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
import numpy as np
import torch
import gc
import argparse
from collections import defaultdict

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.extend([str(project_root)])

from src.utils.file_utils import clear_memory
from experiments.experiment0_baseline.main import main as baseline_main
from experiments.experiment1_clustering.main import main as clustering_main
from experiments.experiment2_fusion.main import main as fusion_main
from experiments.experiment3_categories.main import main as categories_main

# Import utils and evaluation
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.llm_evaluator import LLMEvaluator
from src.utils.corpus_manager import CorpusManager
from src.utils.file_utils import seed_everything, clear_memory

# Plotters if needed
from experiments.plotting.plot_baseline import BaselinePlotter  
from experiments.plotting.plot_clustering import ClusteringPlotter
from experiments.plotting.plot_fusion import FusionPlotter
from experiments.plotting.plot_categories import CategoriesPlotter

# Import experiment configs
from experiments.experiment0_baseline.config import BaselineConfigFactory, BaselineConfig
from experiments.experiment1_clustering.config import ClusteringConfig

# Import experiments
from experiments.experiment0_baseline.main import BaselineExperiment 
from experiments.experiment1_clustering.main import ClusteringExperiment

# Import setup config
from src.setup_experiments import ExperimentLogger, ExperimentSetup

from dotenv import load_dotenv
load_dotenv() 

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
        self.corpus_manager = CorpusManager("data/processed/corpus_with_contriever_at150.json")
        self.logger = ExperimentLogger(
            experiment_name=f"experiment_pipeline_{self.timestamp}",
            base_log_dir="logs"
        )
        
        # Initialize LLM with proper error handling
        api_key = os.getenv("GEMINI_TOKEN")
        if not api_key:
            raise ValueError("GEMINI_TOKEN environment variable not found")
        self.llm = LLM(api_key=api_key)
        
        self.output_controller = OutputController(self.logger)
        
        self.enabled_experiments = {
            'baseline': True,
            'clustering': True,
            'fusion': True,
            'categories': True
        }
        
        self.plotter_mapping = {
            'baseline': BaselinePlotter,
            'clustering': ClusteringPlotter,
            'fusion': FusionPlotter,
            'categories': CategoriesPlotter
        }
        self.plotting_enabled = True

    def _save_results(self, results: Dict[str, Any], output_dir: Path):
        """
        Save final experiment results to final_results.json.
        Results should be a dictionary possibly with multiple scenarios.
        """
        final_path = output_dir / "final_results.json"
        with open(final_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.experiment_logger.info(f"Saved final results to {final_path}")

    def _generate_plots(self, exp_type: str, results: Dict[str, Any]):
        """Generate plots for the given experiment results if plotting is enabled."""
        if self.plotting_enabled and exp_type in self.plotter_mapping:
            exp_plot_dir = self.plots_dir / exp_type
            exp_plot_dir.mkdir(parents=True, exist_ok=True)
            plotter = self.plotter_mapping[exp_type](exp_plot_dir)
            plotter.plot_results(results)

    def _flatten_results(self, results: Any) -> List[Dict]:
        """
        Flatten nested dictionaries of scenarios into a single list of result dictionaries.
        """
        if isinstance(results, list):
            # If already a list of records
            return results
        if isinstance(results, dict):
            # If a dict, recursively gather lists
            combined = []
            for v in results.values():
                combined.extend(self._flatten_results(v))
            return combined
        return []

    def _compute_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute basic metrics from final results if needed.
        This is a simplified version: flatten and compute accuracy.
        """
        all_records = self._flatten_results(results)
        correct = [r.get('llm_evaluation', {}).get('correct', False) for r in all_records]
        scores = [r.get('llm_evaluation', {}).get('score', 0) for r in all_records]
        total = len(all_records)
        if total == 0:
            return {'accuracy':0.0,'avg_score':0.0,'total_examples':0}
        acc = sum(correct)/total
        avg_score = float(np.mean(scores)) if scores else 0.0
        return {
            'accuracy': acc,
            'avg_score': avg_score,
            'total_examples': total
        }

    def run_pipeline(self):
        self.logger.log_step_start("Initializing experiment pipeline")
        
        if not self.setup.setup():
            self.logger.log_error(None, "Setup failed")
            return

        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.plotting_enabled:
            self.plots_dir.mkdir(parents=True, exist_ok=True)

        final_results = {}
        
        experiment_order = ['baseline', 'clustering', 'fusion', 'categories']
        
        for exp_type in experiment_order:
            if not self.enabled_experiments[exp_type]:
                continue
                
            try:
                self.logger.log_step_start(f"Running {exp_type} experiments")
                exp_results = self._run_experiment(exp_type)
                
                if exp_results is None:
                    self.logger.error(f"{exp_type} experiment failed, stopping pipeline")
                    break
                    
                exp_dir = self.results_dir / exp_type
                exp_dir.mkdir(parents=True, exist_ok=True)

                self._save_results(exp_results, exp_dir)

                if self.plotting_enabled:
                    self._generate_plots(exp_type, exp_results)

                metrics = self._compute_metrics(exp_results)
                self.logger.experiment_logger.info(
                    f"Completed {exp_type} with accuracy: {metrics['accuracy']:.4f}, "
                    f"avg_score: {metrics['avg_score']:.4f} over {metrics['total_examples']} examples"
                )

                final_results[exp_type] = exp_results
                self.logger.log_step_end(f"{exp_type} experiments completed")
                
            except Exception as e:
                self.logger.log_error(e, f"Error in {exp_type} experiment")
                break

            clear_memory()

        return final_results
        
    def _verify_gold_documents(self, results: List[Dict]) -> None:
        """Verify that gold documents are present and properly positioned in results."""
        gold_count = 0
        for result in results:
            docs = result.get('document_indices', [])
            if docs:
                gold_doc = next((doc for doc in docs if doc['category'] == 'gold'), None)
                if gold_doc:
                    gold_count += 1
                    position = docs.index(gold_doc)
                    if position != 0:  # Gold should be first
                        self.logger.experiment_logger.warning(f"Gold document not in first position: {position}")
        
        self.logger.experiment_logger.info(f"Found {gold_count} gold documents in {len(results)} results")

    def _run_experiment(self, exp_type: str) -> Optional[Dict[str, Any]]:
        try:
            if exp_type == 'baseline':
                config = BaselineConfigFactory.get_config_for_retriever('contriever')
                corpus_manager = CorpusManager(str(config.corpus_path))
                llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
                experiment = BaselineExperiment(config, corpus_manager, llm_evaluator)
                return experiment.run()
                
            elif exp_type == 'clustering':
                config = ClusteringConfig()
                corpus_manager = CorpusManager(str(config.corpus_path))
                llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
                experiment = ClusteringExperiment(config, corpus_manager, llm_evaluator)
                return experiment.run()

        except Exception as e:
            self.logger.log_error(str(e), f"Error in {exp_type} experiment")
            raise

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run complete experiment pipeline")
   
    parser.add_argument('--output_dir', type=str, default='experiments', help='Base output directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for experiment logs')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline experiments')
    parser.add_argument('--skip_clustering', action='store_true', help='Skip clustering experiments')
    parser.add_argument('--skip_fusion', action='store_true', help='Skip fusion experiments')
    parser.add_argument('--skip_categories', action='store_true', help='Skip categories experiments')
    parser.add_argument('--save_plots', type=str, default='true', help='Whether to generate plots (true/false)')
   
    return parser.parse_args()

def main():
    try:
        baseline_config = BaselineConfig()
        clustering_config = ClusteringConfig()
        corpus_manager = CorpusManager(str(baseline_config.corpus_path))
        llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
        experiment_logger = ExperimentLogger("complete_pipeline", "logs")

        with tqdm(total=2, desc="Running experiments") as pbar:
            # Baseline experiment
            experiment_logger.info("Starting baseline experiment")
            baseline_exp = BaselineExperiment(baseline_config, corpus_manager, llm_evaluator)
            baseline_results = baseline_exp.run()
            pbar.update(1)
            clear_memory()

            # Clustering experiment
            experiment_logger.info("Starting clustering experiment")
            clustering_exp = ClusteringExperiment(clustering_config, corpus_manager, llm_evaluator)  
            clustering_results = clustering_exp.run()
            pbar.update(1)
            clear_memory()

            # Save final results
            final_results = {
                'baseline': baseline_results,
                'clustering': clustering_results
            }
            
            output_dir = Path("experiments/results")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"final_results_{datetime.now():%Y%m%d_%H%M%S}.json", 'w') as f:
                json.dump(final_results, f, indent=2)

    except Exception as e:
        experiment_logger.experiment_logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
