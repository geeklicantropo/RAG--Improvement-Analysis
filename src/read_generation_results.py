import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import torch
from scipy import stats
import gc

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

from src.llm import LLM

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read and evaluate generation results")
    parser.add_argument(
        '--results_dir', 
        type=str, 
        required=True,
        help='Directory containing generation results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory for saving evaluation results'
    )
    parser.add_argument(
        '--experiment_type',
        type=str,
        choices=['baseline', 'clustering', 'fusion', 'categories'],
        required=True,
        help='Type of experiment to evaluate'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        required=True,
        help='Gemini API Key'
    )
    return parser.parse_args()

class ResultsReader:
    def __init__(
        self,
        results_dir: str,
        output_dir: str,
        llm_evaluator: LLM,
        logger: Optional[logging.Logger] = None
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_evaluator = llm_evaluator
        self.logger = logger or self._setup_logger()
        self.metrics = defaultdict(dict)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ResultsReader")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def evaluate_results(self, experiment_type: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Evaluating {experiment_type} results")
            results = self._load_results(experiment_type)
            
            metrics = {
                'overall': self._evaluate_overall_performance(results),
                'position_impact': self._analyze_position_impact(results),
                'document_combinations': self._analyze_document_combinations(results),
                'category_performance': self._analyze_category_performance(results)
            }
            
            self._save_metrics(metrics, experiment_type)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating {experiment_type}: {str(e)}")
            raise

    def _evaluate_overall_performance(self, results: List[Dict]) -> Dict[str, float]:
        self.logger.info("Evaluating overall performance")
        evaluations = []
        
        for result in tqdm(results, desc="Evaluating answers"):
            eval_result = self.llm_evaluator.evaluate_answer(
                question=result['query'],
                generated_answer=result['generated_answer'],
                gold_answer=result['gold_answer']
            )
            evaluations.append(eval_result)
            
        return {
            'accuracy': np.mean([e['correct'] for e in evaluations]),
            'avg_score': np.mean([e['score'] for e in evaluations]),
            'avg_semantic_similarity': np.mean([e['semantic_similarity'] for e in evaluations])
        }

    def _analyze_position_impact(self, results: List[Dict]) -> Dict[str, Any]:
        self.logger.info("Analyzing position impact")
        position_metrics = defaultdict(list)
        
        for result in results:
            positions = result['document_positions']
            categories = result['document_categories']
            eval_result = self.llm_evaluator.evaluate_answer(
                result['query'], 
                result['generated_answer'],
                result['gold_answer']
            )
            
            for pos, cat in zip(positions, categories):
                position_metrics[f"{pos}_{cat}"].append(eval_result['correct'])
                
        return {
            pos: {
                'accuracy': np.mean(scores),
                'count': len(scores),
                'std': np.std(scores)
            }
            for pos, scores in position_metrics.items()
        }

    def _analyze_document_combinations(self, results: List[Dict]) -> Dict[str, Any]:
        self.logger.info("Analyzing document combinations")
        combination_metrics = defaultdict(list)
        
        for result in results:
            combo_key = self._get_combination_key(result['document_categories'])
            eval_result = self.llm_evaluator.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            combination_metrics[combo_key].append({
                'correct': eval_result['correct'],
                'score': eval_result['score'],
                'semantic_similarity': eval_result['semantic_similarity']
            })
            
        return {
            combo: {
                'accuracy': np.mean([r['correct'] for r in results]),
                'avg_score': np.mean([r['score'] for r in results]),
                'avg_similarity': np.mean([r['semantic_similarity'] for r in results]),
                'count': len(results)
            }
            for combo, results in combination_metrics.items()
        }

    def _analyze_category_performance(self, results: List[Dict]) -> Dict[str, Any]:
        self.logger.info("Analyzing category performance")
        category_metrics = defaultdict(list)
        
        for result in results:
            categories = result['document_categories']
            eval_result = self.llm_evaluator.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            for cat in set(categories):
                category_metrics[cat].append({
                    'correct': eval_result['correct'],
                    'score': eval_result['score']
                })
                
        return {
            cat: {
                'accuracy': np.mean([r['correct'] for r in metrics]),
                'avg_score': np.mean([r['score'] for r in metrics]),
                'count': len(metrics)
            }
            for cat, metrics in category_metrics.items()
        }

    def _get_combination_key(self, categories: List[str]) -> str:
        counts = defaultdict(int)
        for cat in categories:
            counts[cat] += 1
        return "_".join(f"{k}{v}" for k, v in sorted(counts.items()))

    def _load_results(self, experiment_type: str) -> List[Dict]:
        results_path = self.results_dir / experiment_type / "results.json"
        if not results_path.exists():
            raise ValueError(f"Results not found: {results_path}")
            
        with open(results_path) as f:
            return json.load(f)

    def _save_metrics(self, metrics: Dict[str, Any], experiment_type: str):
        metrics_path = self.output_dir / f"{experiment_type}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def compare_experiments(self, experiment_types: List[str]):
        self.logger.info("Comparing experiments")
        all_metrics = {}
        
        for exp_type in experiment_types:
            try:
                all_metrics[exp_type] = self.evaluate_results(exp_type)
            except Exception as e:
                self.logger.error(f"Error evaluating {exp_type}: {str(e)}")
                continue
                
        # Perform statistical tests
        stats_results = self._run_statistical_tests(all_metrics)
        
        # Save comparison results
        comparison_path = self.output_dir / "experiment_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump({
                'metrics': all_metrics,
                'statistical_tests': stats_results
            }, f, indent=2)

    def _run_statistical_tests(self, all_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        test_results = {}
        
        # Perform pairwise tests
        exp_types = list(all_metrics.keys())
        for i in range(len(exp_types)):
            for j in range(i + 1, len(exp_types)):
                exp1, exp2 = exp_types[i], exp_types[j]
                
                # t-test for scores
                scores1 = [m['score'] for m in all_metrics[exp1]['overall']]
                scores2 = [m['score'] for m in all_metrics[exp2]['overall']]
                t_stat, p_val = stats.ttest_ind(scores1, scores2)
                
                test_results[f"{exp1}_vs_{exp2}"] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_val)
                }
                
        return test_results

def main():
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize LLM evaluator
        llm_evaluator = LLM(args.api_key)
        
        # Initialize results reader
        reader = ResultsReader(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            llm_evaluator=llm_evaluator
        )
        
        # Evaluate results
        metrics = reader.evaluate_results(args.experiment_type)
        
        # Run experiment comparisons if all types present
        if all(os.path.exists(reader.results_dir / exp_type) 
               for exp_type in ['baseline', 'clustering', 'fusion', 'categories']):
            reader.compare_experiments([
                'baseline', 'clustering', 'fusion', 'categories'
            ])
        
        logging.info("Evaluation completed successfully")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()