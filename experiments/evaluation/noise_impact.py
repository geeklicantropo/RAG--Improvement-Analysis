import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time

from src.llm_evaluator import LLMEvaluator

class NoiseImpactAnalyzer:
    def __init__(self, llm_evaluator: LLMEvaluator):
        self.evaluator = llm_evaluator
        self.logger = logging.getLogger("NoiseImpactAnalyzer")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        
    def analyze_noise_impact(
        self,
        results: List[Dict],
        noise_levels: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        if not noise_levels:
            noise_levels = [0.0, 0.1, 0.2, 0.3]
            
        metrics = {}
        baseline = None
        
        for noise in tqdm(noise_levels, desc="Analyzing noise impact"):
            noise_results = [r for r in results if r.get('noise_ratio', 0.0) == noise]
            
            if not noise_results:
                continue
                
            metrics[f'noise_{noise}'] = self._compute_noise_metrics(noise_results)
            
            if noise == 0.0:
                baseline = metrics[f'noise_{noise}']
            elif baseline:
                metrics[f'noise_{noise}']['degradation'] = {
                    'accuracy': baseline['accuracy'] - metrics[f'noise_{noise}']['accuracy'],
                    'score': baseline['avg_score'] - metrics[f'noise_{noise}']['avg_score']
                }
                
        return metrics
        
    def _compute_noise_metrics(self, results: List[Dict]) -> Dict[str, float]:
        evals = [r['llm_evaluation'] for r in results if 'llm_evaluation' in r]
        
        return {
            'accuracy': np.mean([e['correct'] for e in evals]),
            'avg_score': np.mean([e['score'] for e in evals]),
            'score_std': np.std([e['score'] for e in evals]),
            'count': len(results)
        }

    def evaluate_noise_types(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        noise_types = defaultdict(list)
        
        for result in results:
            if 'noise_type' in result:
                noise_types[result['noise_type']].append(result)
                
        evaluations = {}
        for noise_type, type_results in noise_types.items():
            metrics = self._compute_noise_metrics(type_results)
            
            # Evaluate impact on answer quality
            scores = []
            for r in type_results:
                eval_result = self.evaluator.evaluate_answer(
                    r['query'],
                    r['generated_answer'],
                    r['gold_answer']
                )
                scores.append(eval_result['score'])
                time.sleep(0.1)  # Rate limiting
                
            evaluations[noise_type] = {
                **metrics,
                'llm_score': np.mean(scores),
                'llm_score_std': np.std(scores)
            }
            
        return evaluations

    def analyze_position_impact(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        position_metrics = defaultdict(list)
        
        for result in results:
            if 'position' in result and 'noise_ratio' in result:
                position_metrics[(result['position'], result['noise_ratio'])].append(result)
                
        analysis = {}
        for (pos, noise), pos_results in position_metrics.items():
            metrics = self._compute_noise_metrics(pos_results)
            analysis[f'pos_{pos}_noise_{noise}'] = metrics
            
        return analysis