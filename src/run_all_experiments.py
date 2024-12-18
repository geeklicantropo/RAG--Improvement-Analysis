import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
import time

GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")
if not GEMINI_TOKEN:
    raise ValueError("GEMINI_TOKEN environment variable not found")
genai.configure(api_key=GEMINI_TOKEN)

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pickle
from datetime import datetime
import torch
import logging
import gc
import argparse
import json
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.extend([str(project_root)])

from src.utils.file_utils import clear_memory, seed_everything
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.llm_evaluator import LLMEvaluator
from src.utils.corpus_manager import CorpusManager

from experiments.plotting.plot_baseline import BaselinePlotter
from experiments.plotting.plot_clustering import ClusteringPlotter
from experiments.plotting.plot_comparisons import ComparativePlotter
from experiments.experiment0_baseline.config import BaselineConfigFactory, BaselineConfig
from experiments.experiment1_clustering.config import ClusteringConfig

from experiments.experiment0_baseline.main import BaselineExperiment
from experiments.experiment1_clustering.main import ClusteringExperiment

import warnings
warnings.filterwarnings('ignore')

class OutputController:
    """Controls and filters experiment output."""
    def __init__(self, logger: ExperimentLogger):
        self.logger = logger
        self.output_buffer = []
        self.buffer_size = 100

    def write(self, text: str):
        if text.strip():
            self.output_buffer.append(text)
            if len(self.output_buffer) >= self.buffer_size:
                self.flush()

    def flush(self):
        if self.output_buffer:
            output = ''.join(self.output_buffer)
            self.logger.experiment_logger.info(output)
            self.output_buffer.clear()

    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.flush()

class ExperimentPipeline:
    def __init__(self, base_output_dir: str = "experiments"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(base_output_dir)
        self.results_dir = self.base_output_dir / "results" / self.timestamp
        self.plots_dir = self.results_dir / "plots"
        self.logger = ExperimentLogger(
            experiment_name=f"experiment_pipeline_{self.timestamp}",
            base_log_dir="logs"
        )

        self.corpus_manager = CorpusManager("data/processed/corpus_with_contriever_at150.json")
        self.llm = LLM(api_key=GEMINI_TOKEN)

        self.enabled_experiments = {
            'baseline': True,
            'clustering': True
        }

        self.config = {
            'base_corpus_size': 1000,
            'random_corpus_path': 'data/processed/corpus_with_random_50_words.pkl',
            'adversarial_corpus_path': 'data/processed/reddit_corpus.pkl'
        }

    def _compute_comparison_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        comparison_metrics = {}
        for exp_type, exp_results in results.items():
            all_scores = [r['llm_evaluation']['score'] for r in exp_results if 'llm_evaluation' in r]
            correct = sum(1 for r in exp_results if r['llm_evaluation'].get('correct', False))
            total = len(exp_results)

            comparison_metrics[exp_type] = {
                'accuracy': correct / total if total > 0 else 0,
                'avg_score': np.mean(all_scores) if all_scores else 0,
                'total_examples': total
            }
        return comparison_metrics

    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        correct = sum(1 for r in results if r['llm_evaluation'].get('correct', False))
        all_scores = [r['llm_evaluation']['score'] for r in results if 'llm_evaluation' in r]
        total = len(results)

        return {
            'accuracy': correct / total if total > 0 else 0,
            'avg_score': np.mean(all_scores) if all_scores else 0,
            'total_examples': total
        }

    def _generate_plots(self, results: Dict[str, Any]):
        plotter = ComparativePlotter(self.results_dir)
        baseline_results = results.get('baseline', [])
        clustering_results = results.get('clustering', [])
        plotter.plot_experiment_results(
            baseline_results=baseline_results,
            clustering_results=clustering_results
        )

    def _save_results(self, results: Dict[str, Any]):
        results_file = self.results_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {results_file}")

    def _load_results(self, exp_dir: Path) -> Dict[str, Any]:
        results_file = exp_dir / "final_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return {}

    def run_pipeline(self):
        self.logger.log_step_start("Running Experiment Pipeline")

        final_results = {}
        experiment_names = [name for name, enabled in self.enabled_experiments.items() if enabled]

        with tqdm(total=len(experiment_names), desc="Experiment Pipeline", unit="experiment") as pbar:
            for experiment_name in experiment_names:
                exp_dir = self.results_dir / experiment_name
                exp_dir.mkdir(parents=True, exist_ok=True)

                self.logger.info(f"Starting {experiment_name} experiment...")
                try:
                    if experiment_name == 'baseline':
                        config = BaselineConfigFactory.get_config_for_retriever('contriever')
                        experiment = BaselineExperiment(config, self.corpus_manager, LLMEvaluator(GEMINI_TOKEN))
                        results = experiment.run()
                    elif experiment_name == 'clustering':
                        config = ClusteringConfig()
                        experiment = ClusteringExperiment(config, self.corpus_manager, LLMEvaluator(GEMINI_TOKEN))
                        results = experiment.run()
                    else:
                        raise ValueError(f"Unknown experiment type: {experiment_name}")

                    final_results[experiment_name] = results
                    self._save_results(results, exp_dir)
                    self.logger.info(f"Completed {experiment_name} experiment.")

                except Exception as e:
                    self.logger.experiment_logger.error(f"Error in {experiment_name} experiment: {str(e)}")
                    raise
                finally:
                    pbar.update(1)

        self._generate_plots(final_results)
        self.logger.log_step_end("Experiment Pipeline Completed")

        return final_results
    
    def _load_corpus_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load and validate corpus files"""
        corpus_base = self.corpus_manager.get_random_subset(self.config.base_corpus_size)
        
        # Load random and adversarial corpora
        with open(self.config.random_corpus_path, 'rb') as f:
            random_docs = [{'text': doc, 'title': ''} for doc in pickle.load(f)]
            
        with open(self.config.adversarial_corpus_path, 'rb') as f:
            adversarial_docs = [{'text': doc, 'title': ''} for doc in pickle.load(f)]
            
        return corpus_base, random_docs, adversarial_docs

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run complete experiment pipeline")
    parser.add_argument('--output_dir', type=str, default='experiments', help='Base output directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline experiments')
    parser.add_argument('--skip_clustering', action='store_true', help='Skip clustering experiments')
    return parser.parse_args()

def main():
    args = parse_arguments()
    seed_everything(10)

    pipeline = ExperimentPipeline(base_output_dir=args.output_dir)

    if args.skip_baseline:
        pipeline.enabled_experiments['baseline'] = False
    if args.skip_clustering:
        pipeline.enabled_experiments['clustering'] = False

    results = pipeline.run_pipeline()
    print("Experiment pipeline completed.", results)

if __name__ == "__main__":
    main()
