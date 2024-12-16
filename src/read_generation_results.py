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
import gc

from src.llm import LLM
from src.llm_evaluator import LLMEvaluator
from src.experiment_logger import ExperimentLogger

class ResultsEvaluator:
    def __init__(self, api_key: str, output_dir: Path, logger: ExperimentLogger):
        self.output_dir = output_dir
        self.logger = logger
        self.llm = LLM(api_key=api_key)
        self.evaluator = LLMEvaluator(self.llm)
        self.metrics = defaultdict(dict)

    def evaluate_results(self, results: List[Dict], batch_size: int = 32) -> Dict[str, Any]:
        evaluated_results = []
        
        for batch_start in tqdm(range(0, len(results), batch_size), desc="Evaluating results"):
            batch = results[batch_start:batch_start + batch_size]
            
            for result in batch:
                context = self._get_context(result['document_indices'], result.get('corpus', {}))
                eval_result = self.evaluator.evaluate_answer(
                    question=result['query'],
                    generated_answer=result['generated_answer'],
                    gold_answer=result['gold_answer'],
                    context=context
                )
                result['llm_evaluation'] = eval_result
                evaluated_results.append(result)

        return self._compute_metrics(evaluated_results)

    def _get_context(self, doc_ids: List[int], corpus: Dict[int, Dict]) -> str:
        context = []
        for idx, doc_id in enumerate(doc_ids):
            if doc_id in corpus:
                doc = corpus[doc_id]
                context.append(f"Document [{idx+1}]: {doc.get('text', '')}")
        return "\n\n".join(context)

    def _compute_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        metrics = {
            'overall': self._compute_overall_metrics(results),
            'position_impact': self._analyze_position_impact(results),
            'category_performance': self._analyze_category_performance(results),
            'noise_impact': self._analyze_noise_impact(results)
        }
        
        self._save_metrics(metrics)
        return metrics

    def _compute_overall_metrics(self, results: List[Dict]) -> Dict[str, float]:
        evals = [r['llm_evaluation'] for r in results]
        return {
            'accuracy': np.mean([e['correct'] for e in evals]),
            'avg_score': np.mean([e['score'] for e in evals]),
            'score_std': np.std([e['score'] for e in evals]),
            'total_examples': len(results)
        }

    def _analyze_position_impact(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        position_metrics = defaultdict(list)
        
        for result in results:
            if 'position' in result:
                position = result['position']
                score = result['llm_evaluation']['score']
                position_metrics[position].append(score)
                
        return {
            pos: {
                'avg_score': np.mean(scores),
                'count': len(scores)
            }
            for pos, scores in position_metrics.items()
        }

    def _analyze_category_performance(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        category_metrics = defaultdict(list)
        
        for result in results:
            if 'category' in result:
                category = result['category']
                score = result['llm_evaluation']['score']
                category_metrics[category].append(score)
                
        return {
            cat: {
                'avg_score': np.mean(scores),
                'count': len(scores)
            }
            for cat, scores in category_metrics.items()
        }

    def _analyze_noise_impact(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        noise_metrics = defaultdict(list)
        
        for result in results:
            if 'noise_ratio' in result:
                noise = result['noise_ratio']
                score = result['llm_evaluation']['score']
                noise_metrics[noise].append(score)
                
        return {
            f'noise_{noise}': {
                'avg_score': np.mean(scores),
                'count': len(scores)
            }
            for noise, scores in noise_metrics.items()
        }

    def _save_metrics(self, metrics: Dict[str, Any]):
        output_file = self.output_dir / 'evaluation_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generation results using LLM")
    parser.add_argument('--results_dir', type=str, default="experiments/results",
                      help='Directory containing generation results')
    parser.add_argument('--output_dir', type=str, default="experiments/evaluation_results",
                      help='Directory for saving evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for LLM evaluation')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    logger = ExperimentLogger("results_evaluation", "logs")
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    batch_size = args.batch_size
    
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with logger:
            api_key = os.getenv("GEMINI_TOKEN")
            if not api_key:
                logger.experiment_logger.error("GEMINI_TOKEN environment variable not found")
                return

            evaluator = ResultsEvaluator(api_key, output_dir, logger)
            results_files = list(results_dir.glob("*.json"))
            all_results = []
            
            for file in results_files:
                with open(file) as f:
                    results = json.load(f)
                    all_results.extend(results)
            
            metrics = evaluator.evaluate_results(all_results, batch_size)
            logger.experiment_logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")
            
    except Exception as e:
        logger.experiment_logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()