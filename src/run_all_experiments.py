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
from typing import Dict, List, Any, Optional
import pickle
from datetime import datetime
import torch
import logging
import gc
import argparse
import json
import numpy as np
from tqdm import tqdm
from utils.rate_limit import rate_limit

project_root = Path(__file__).parent.parent
sys.path.extend([str(project_root)])

from src.utils.file_utils import clear_memory, seed_everything, str2bool
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
        self.llm_evaluator = LLMEvaluator(api_key=GEMINI_TOKEN)

        self.enabled_experiments = {
            'baseline': True,
            'clustering': True
        }

        self.config = {
            'base_corpus_size': 10000,
            'random_corpus_path': 'data/processed/corpus_with_random_50_words.pkl',
            'adversarial_corpus_path': 'data/processed/reddit_corpus.pkl'
        }

        # Create result directories 
        for mode in ['baseline', 'clustering']:
            mode_dir = self.results_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            for submode in ['gold_only', 'gold_random', 'gold_adversarial']:
                submode_dir = mode_dir / submode
                submode_dir.mkdir(parents=True, exist_ok=True)
                (submode_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _compute_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        correct = sum(1 for r in results if 'llm_evaluation' in r and r['llm_evaluation'].get('correct', False))
        all_scores = [r['llm_evaluation']['score'] for r in results if 'llm_evaluation' in r]
        total = len(results)

        return {
            'accuracy': correct / total if total > 0 else 0,
            'avg_score': float(np.mean(all_scores)) if all_scores else 0.0,
            'total_examples': total
        }

    def _generate_plots(self, final_results: Dict[str, Any]):
        baseline_results = []
        if 'baseline' in final_results:
            for mode_results in final_results['baseline'].values():
                baseline_results.extend(mode_results)

        clustering_results = []
        if 'clustering' in final_results:
            for mode_results in final_results['clustering'].values():
                clustering_results.extend(mode_results)

        plotter = ComparativePlotter(self.results_dir)
        plotter.plot_experiment_results(
            baseline_results=baseline_results,
            clustering_results=clustering_results
        )

    def _save_results(self, results: Dict[str, Any], exp_dir: Path):
        final_path = exp_dir / "final_results.json"
        try:
            with open(final_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.experiment_logger.info(f"Saved final results to {final_path}")
        except Exception as e:
            self.logger.experiment_logger.error(f"Error saving final results: {str(e)}")

    @rate_limit
    def run_pipeline(self):
        self.logger.log_step_start("Running Experiment Pipeline")

        final_results = {}
        experiment_names = [name for name, enabled in self.enabled_experiments.items() if enabled]

        for experiment_name in experiment_names:
            exp_dir = self.results_dir / experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Starting {experiment_name} experiment...")
            try:
                if experiment_name == 'baseline':
                    config = BaselineConfigFactory.get_config_for_retriever('contriever')
                    experiment = BaselineExperiment(config, self.corpus_manager, self.llm_evaluator)
                    
                    # Run the experiment
                    results = experiment.run()
                    
                elif experiment_name == 'clustering':
                    config = ClusteringConfig()
                    experiment = ClusteringExperiment(config, self.corpus_manager, self.llm_evaluator)
                    
                    # Run the experiment
                    results = experiment.run()
                else:
                    raise ValueError(f"Unknown experiment type: {experiment_name}")

                final_results[experiment_name] = results
                self._save_results(results, exp_dir)
                self.logger.info(f"Completed {experiment_name} experiment.")

            except Exception as e:
                self.logger.experiment_logger.error(f"Error in {experiment_name} experiment: {str(e)}")
                raise

        self._generate_plots(final_results)
        self.logger.log_step_end("Experiment Pipeline Completed")

        return final_results


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

    with pipeline.logger:
        results = pipeline.run_pipeline()
    print("Experiment pipeline completed.")


if __name__ == "__main__":
    main()
