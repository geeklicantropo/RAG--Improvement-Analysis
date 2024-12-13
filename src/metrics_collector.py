import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from pathlib import Path

from src.llm_evaluator import LLMEvaluator
from src.experiment_logger import ExperimentLogger

class MetricsCollector:
    def __init__(self, llm_evaluator: LLMEvaluator):
        self.evaluator = llm_evaluator
        self.logger = self._setup_logger()
        self.metrics = defaultdict(dict)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("MetricsCollector")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def collect_metrics(
        self,
        results: List[Dict],
        experiment_type: str,
        configurations: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Collect comprehensive metrics for experiment results."""
        self.logger.info(f"Collecting metrics for {experiment_type}")
        
        # Base metrics
        metrics = self._compute_base_metrics(results)
        
        # Position-specific metrics
        if self._has_position_data(results):
            metrics.update(self._compute_position_metrics(results))
            
        # Category-specific metrics
        if self._has_category_data(results):
            metrics.update(self._compute_category_metrics(results))
            
        # Noise impact metrics
        if self._has_noise_data(results):
            metrics.update(self._compute_noise_metrics(results))
            
        # Configuration-specific metrics
        if configurations:
            metrics.update(self._compute_config_metrics(results, configurations))
            
        return metrics

    def _compute_base_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute basic performance metrics."""
        base_metrics = defaultdict(list)
        
        for result in tqdm(results, desc="Computing base metrics"):
            eval_result = self.evaluator.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            base_metrics['scores'].append(eval_result['score'])
            base_metrics['correct'].append(eval_result['correct'])
            
            if 'response_time' in result:
                base_metrics['response_times'].append(result['response_time'])
                
            if 'context_length' in result:
                base_metrics['context_lengths'].append(result['context_length'])

        return {
            'accuracy': sum(base_metrics['correct']) / len(base_metrics['correct']),
            'avg_score': np.mean(base_metrics['scores']),
            'score_std': np.std(base_metrics['scores']),
            'avg_response_time': np.mean(base_metrics['response_times']) if 'response_times' in base_metrics else None,
            'avg_context_length': np.mean(base_metrics['context_lengths']) if 'context_lengths' in base_metrics else None
        }

    def _compute_position_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute position-specific metrics."""
        position_metrics = defaultdict(lambda: defaultdict(list))
        
        for result in tqdm(results, desc="Computing position metrics"):
            position = result.get('gold_position', 'unknown')
            eval_result = self.evaluator.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            position_metrics[position]['scores'].append(eval_result['score'])
            position_metrics[position]['correct'].append(eval_result['correct'])

        return {
            f'position_{pos}': {
                'accuracy': sum(metrics['correct']) / len(metrics['correct']),
                'avg_score': np.mean(metrics['scores']),
                'score_std': np.std(metrics['scores'])
            }
            for pos, metrics in position_metrics.items()
        }

    def _compute_category_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute category-specific metrics."""
        category_metrics = defaultdict(lambda: defaultdict(list))
        
        for result in tqdm(results, desc="Computing category metrics"):
            category = result.get('category', 'unknown')
            eval_result = self.evaluator.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            category_metrics[category]['scores'].append(eval_result['score'])
            category_metrics[category]['correct'].append(eval_result['correct'])

        return {
            f'category_{cat}': {
                'accuracy': sum(metrics['correct']) / len(metrics['correct']),
                'avg_score': np.mean(metrics['scores']),
                'score_std': np.std(metrics['scores'])
            }
            for cat, metrics in category_metrics.items()
        }

    def _compute_noise_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute noise-related metrics."""
        noise_metrics = defaultdict(lambda: defaultdict(list))
        
        for result in tqdm(results, desc="Computing noise metrics"):
            noise_ratio = result.get('noise_ratio', 0.0)
            eval_result = self.evaluator.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            noise_metrics[noise_ratio]['scores'].append(eval_result['score'])
            noise_metrics[noise_ratio]['correct'].append(eval_result['correct'])

        metrics = {}
        baseline_accuracy = None
        
        for ratio, data in noise_metrics.items():
            accuracy = sum(data['correct']) / len(data['correct'])
            if ratio == 0.0:
                baseline_accuracy = accuracy
                
            metrics[f'noise_{ratio}'] = {
                'accuracy': accuracy,
                'avg_score': np.mean(data['scores']),
                'score_std': np.std(data['scores'])
            }
            
            if baseline_accuracy and ratio > 0:
                metrics[f'noise_{ratio}']['degradation'] = baseline_accuracy - accuracy
                
        return metrics

    def _compute_config_metrics(
        self,
        results: List[Dict],
        configurations: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Compute configuration-specific metrics."""
        config_metrics = defaultdict(lambda: defaultdict(list))
        
        for result in tqdm(results, desc="Computing configuration metrics"):
            config_id = result.get('config_id')
            if not config_id or config_id not in configurations:
                continue
                
            eval_result = self.evaluator.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            config_metrics[config_id]['scores'].append(eval_result['score'])
            config_metrics[config_id]['correct'].append(eval_result['correct'])

        return {
            f'config_{cfg_id}': {
                'accuracy': sum(metrics['correct']) / len(metrics['correct']),
                'avg_score': np.mean(metrics['scores']),
                'score_std': np.std(metrics['scores'])
            }
            for cfg_id, metrics in config_metrics.items()
        }

    def _has_position_data(self, results: List[Dict]) -> bool:
        return any('gold_position' in r for r in results)

    def _has_category_data(self, results: List[Dict]) -> bool:
        return any('category' in r for r in results)

    def _has_noise_data(self, results: List[Dict]) -> bool:
        return any('noise_ratio' in r for r in results)