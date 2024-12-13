import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from scipy import stats

from src.llm_evaluator import LLMEvaluator

class EvaluationMetrics:
    def __init__(self, llm_evaluator: LLMEvaluator):
        self.evaluator = llm_evaluator
        self.logger = logging.getLogger("EvaluationMetrics")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

    def compute_all_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        metrics = {}
        with tqdm(total=5, desc="Computing metrics") as pbar:
            metrics.update(self._compute_baseline_metrics(results))
            pbar.update(1)
            
            metrics.update(self._compute_categorical_metrics(results))
            pbar.update(1)
            
            metrics.update(self._compute_position_metrics(results))
            pbar.update(1)
            
            metrics.update(self._compute_noise_metrics(results))
            pbar.update(1)
            
            metrics.update(self._compute_statistical_metrics(results))
            pbar.update(1)
        
        return metrics

    def _compute_baseline_metrics(self, results: List[Dict]) -> Dict[str, float]:
        total = len(results)
        correct = sum(1 for r in results if r.get('llm_evaluation', {}).get('correct', False))
        scores = [r.get('llm_evaluation', {}).get('score', 0) for r in results]
        
        return {
            'accuracy': correct / total,
            'avg_score': np.mean(scores),
            'score_std': np.std(scores),
            'sample_size': total
        }

    def _compute_categorical_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        categories = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            if 'category' in result:
                eval_result = result.get('llm_evaluation', {})
                cat = result['category']
                categories[cat]['scores'].append(eval_result.get('score', 0))
                categories[cat]['correct'].append(eval_result.get('correct', False))
        
        return {
            cat: {
                'accuracy': sum(metrics['correct']) / len(metrics['correct']),
                'avg_score': np.mean(metrics['scores']),
                'score_std': np.std(metrics['scores']),
                'count': len(metrics['correct'])
            }
            for cat, metrics in categories.items()
        }

    def _compute_position_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        positions = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            if 'position' in result:
                eval_result = result.get('llm_evaluation', {})
                pos = result['position']
                positions[pos]['scores'].append(eval_result.get('score', 0))
                positions[pos]['correct'].append(eval_result.get('correct', False))
                
        return {
            pos: {
                'accuracy': sum(metrics['correct']) / len(metrics['correct']),
                'avg_score': np.mean(metrics['scores']),
                'score_std': np.std(metrics['scores']),
                'count': len(metrics['correct'])
            }
            for pos, metrics in positions.items()
        }

    def _compute_noise_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        noise_levels = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            if 'noise_ratio' in result:
                eval_result = result.get('llm_evaluation', {})
                noise = result['noise_ratio']
                noise_levels[noise]['scores'].append(eval_result.get('score', 0))
                noise_levels[noise]['correct'].append(eval_result.get('correct', False))
                
        noise_metrics = {}
        baseline = noise_levels[0.0] if 0.0 in noise_levels else None
        
        for noise, metrics in noise_levels.items():
            stats = {
                'accuracy': sum(metrics['correct']) / len(metrics['correct']),
                'avg_score': np.mean(metrics['scores']),
                'score_std': np.std(metrics['scores']),
                'count': len(metrics['correct'])
            }
            
            if baseline and noise > 0:
                stats['degradation'] = {
                    'accuracy': stats['accuracy'] - (sum(baseline['correct']) / len(baseline['correct'])),
                    'score': stats['avg_score'] - np.mean(baseline['scores'])
                }
                
            noise_metrics[f'noise_{noise}'] = stats
            
        return noise_metrics

    def _compute_statistical_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        scores = [r.get('llm_evaluation', {}).get('score', 0) for r in results]
        
        _, normality_p = stats.normaltest(scores)
        
        return {
            'normality_test': {
                'normal_distribution': normality_p > 0.05,
                'p_value': normality_p
            },
            'confidence_intervals': {
                'score_95': stats.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=stats.sem(scores)),
                'score_99': stats.t.interval(0.99, len(scores)-1, loc=np.mean(scores), scale=stats.sem(scores))
            },
            'effect_size': {
                'cohens_d': np.mean(scores) / np.std(scores) if np.std(scores) > 0 else 0
            }
        }