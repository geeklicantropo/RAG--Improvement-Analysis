import logging
from typing import Dict, List, Any
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from src.llm_evaluator import LLMEvaluator
from src.experiment_logger import ExperimentLogger

class NoiseImpactAnalyzer:
    def __init__(self, llm_evaluator: LLMEvaluator):
        self.evaluator = llm_evaluator
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("NoiseImpactAnalyzer")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def run_noise_analysis(
        self,
        results: List[Dict],
        noise_ratios: List[float]
    ) -> Dict[str, Any]:
        """Run comprehensive noise impact analysis."""
        self.logger.info("Starting noise impact analysis")
        
        metrics = defaultdict(lambda: defaultdict(list))
        
        for result in tqdm(results, desc="Analyzing noise impact"):
            noise_ratio = result.get('noise_ratio', 0.0)
            if noise_ratio not in noise_ratios:
                continue
                
            eval_result = self.evaluator.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            metrics[noise_ratio]['scores'].append(eval_result['score'])
            metrics[noise_ratio]['correct'].append(eval_result['correct'])
            
            if 'response_time' in result:
                metrics[noise_ratio]['response_times'].append(result['response_time'])
                
            if 'context_length' in result:
                metrics[noise_ratio]['context_lengths'].append(result['context_length'])

        return self._compute_noise_metrics(metrics)

    def _compute_noise_metrics(
        self,
        raw_metrics: Dict[float, Dict[str, List[Any]]]
    ) -> Dict[str, Any]:
        """Compute statistical metrics for noise analysis."""
        processed_metrics = {}
        
        for noise_ratio, metrics in raw_metrics.items():
            ratio_metrics = {
                'accuracy': sum(metrics['correct']) / len(metrics['correct']),
                'avg_score': sum(metrics['scores']) / len(metrics['scores']),
                'score_std': np.std(metrics['scores']),
                'num_samples': len(metrics['correct'])
            }
            
            if 'response_times' in metrics:
                ratio_metrics.update({
                    'avg_response_time': sum(metrics['response_times']) / len(metrics['response_times']),
                    'response_time_std': np.std(metrics['response_times'])
                })
                
            if 'context_lengths' in metrics:
                ratio_metrics.update({
                    'avg_context_length': sum(metrics['context_lengths']) / len(metrics['context_lengths']),
                    'context_length_std': np.std(metrics['context_lengths'])
                })
                
            processed_metrics[f'noise_{noise_ratio}'] = ratio_metrics

        # Compute relative degradation
        if 0.0 in raw_metrics:
            base_accuracy = sum(raw_metrics[0.0]['correct']) / len(raw_metrics[0.0]['correct'])
            base_score = sum(raw_metrics[0.0]['scores']) / len(raw_metrics[0.0]['scores'])
            
            for noise_ratio in raw_metrics.keys():
                if noise_ratio == 0.0:
                    continue
                    
                curr_accuracy = sum(raw_metrics[noise_ratio]['correct']) / len(raw_metrics[noise_ratio]['correct'])
                curr_score = sum(raw_metrics[noise_ratio]['scores']) / len(raw_metrics[noise_ratio]['scores'])
                
                degradation_metrics = {
                    'accuracy_degradation': base_accuracy - curr_accuracy,
                    'score_degradation': base_score - curr_score,
                    'relative_accuracy_change': (base_accuracy - curr_accuracy) / base_accuracy,
                    'relative_score_change': (base_score - curr_score) / base_score
                }
                
                processed_metrics[f'noise_{noise_ratio}'].update(degradation_metrics)

        return processed_metrics

    def analyze_noise_types(
        self,
        results: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze impact of different noise types."""
        self.logger.info("Analyzing noise types")
        
        type_metrics = defaultdict(lambda: defaultdict(list))
        
        for result in tqdm(results, desc="Analyzing noise types"):
            noise_type = result.get('noise_type', 'unknown')
            
            eval_result = self.evaluator.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            type_metrics[noise_type]['scores'].append(eval_result['score'])
            type_metrics[noise_type]['correct'].append(eval_result['correct'])

        return {
            noise_type: {
                'accuracy': sum(metrics['correct']) / len(metrics['correct']),
                'avg_score': sum(metrics['scores']) / len(metrics['scores']),
                'score_std': np.std(metrics['scores']),
                'num_samples': len(metrics['correct'])
            }
            for noise_type, metrics in type_metrics.items()
        }