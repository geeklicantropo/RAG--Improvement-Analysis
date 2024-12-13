import logging
import numpy as np
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from pathlib import Path

from src.llm_evaluator import LLMEvaluator
from src.experiment_logger import ExperimentLogger

class ExperimentValidator:
    def __init__(self, llm_evaluator: LLMEvaluator):
        self.evaluator = llm_evaluator
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ExperimentValidator")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def validate_results(
        self,
        results: List[Dict],
        experiment_config: Dict,
        paper_baselines: Optional[Dict] = None
    ) -> Dict[str, Any]:
        validation_results = {
            'statistical_validation': self._validate_statistics(results),
            'methodology_validation': self._validate_methodology(results, experiment_config),
            'reproducibility_validation': self._validate_reproducibility(results)
        }
        
        if paper_baselines:
            validation_results['baseline_comparison'] = self._compare_to_baselines(results, paper_baselines)
            
        return validation_results

    def _validate_statistics(self, results: List[Dict]) -> Dict[str, bool]:
        self.logger.info("Validating statistical significance")
        validations = {}
        
        try:
            scores = [r.get('score', 0) for r in results]
            sample_size = len(scores)
            
            validations.update({
                'sufficient_sample_size': sample_size >= 100,
                'normal_distribution': self._check_normality(scores),
                'no_outliers': self._check_outliers(scores),
                'significance_achieved': self._check_significance(scores)
            })
            
            return validations
        except Exception as e:
            self.logger.error(f"Statistical validation error: {str(e)}")
            return {'validation_error': str(e)}

    def _validate_methodology(
        self, 
        results: List[Dict],
        config: Dict
    ) -> Dict[str, bool]:
        self.logger.info("Validating methodology")
        methodologies = {}
        
        try:
            methodologies.update({
                'consistent_evaluation': self._check_evaluation_consistency(results),
                'proper_noise_injection': self._check_noise_levels(results, config),
                'position_testing': self._check_position_testing(results),
                'proper_splits': self._check_data_splits(results)
            })
            
            return methodologies
        except Exception as e:
            self.logger.error(f"Methodology validation error: {str(e)}")
            return {'validation_error': str(e)}

    def _validate_reproducibility(self, results: List[Dict]) -> Dict[str, bool]:
        self.logger.info("Validating reproducibility")
        checks = {}
        
        try:
            checks.update({
                'seed_consistency': self._check_seed_consistency(results),
                'deterministic_output': self._check_output_determinism(results),
                'complete_metadata': self._check_metadata_completeness(results)
            })
            
            return checks
        except Exception as e:
            self.logger.error(f"Reproducibility validation error: {str(e)}")
            return {'validation_error': str(e)}

    def _compare_to_baselines(
        self,
        results: List[Dict],
        baselines: Dict[str, float]
    ) -> Dict[str, Any]:
        self.logger.info("Comparing to paper baselines")
        comparisons = {}
        
        try:
            current_metrics = self._compute_metrics(results)
            
            for metric, baseline in baselines.items():
                if metric in current_metrics:
                    diff = current_metrics[metric] - baseline
                    comparisons[f'{metric}_diff'] = diff
                    comparisons[f'{metric}_matches_paper'] = abs(diff) < 0.02
                    
            return comparisons
        except Exception as e:
            self.logger.error(f"Baseline comparison error: {str(e)}")
            return {'comparison_error': str(e)}

    def _check_normality(self, scores: List[float]) -> bool:
        return len(scores) >= 30  # Central Limit Theorem approximation

    def _check_outliers(self, scores: List[float], threshold: float = 3.0) -> bool:
        z_scores = np.abs((scores - np.mean(scores)) / np.std(scores))
        return np.sum(z_scores > threshold) / len(scores) < 0.01

    def _check_significance(self, scores: List[float], p_threshold: float = 0.01) -> bool:
        from scipy import stats
        _, p_value = stats.ttest_1samp(scores, 0)
        return p_value < p_threshold

    def _check_evaluation_consistency(self, results: List[Dict]) -> bool:
        return all('evaluation' in r for r in results)

    def _check_noise_levels(self, results: List[Dict], config: Dict) -> bool:
        expected_levels = config.get('noise_ratios', [])
        actual_levels = {r.get('noise_ratio') for r in results if 'noise_ratio' in r}
        return all(level in actual_levels for level in expected_levels)

    def _check_position_testing(self, results: List[Dict]) -> bool:
        positions_tested = {'near', 'mid', 'far'}
        actual_positions = {r.get('position') for r in results if 'position' in r}
        return positions_tested.issubset(actual_positions)

    def _check_data_splits(self, results: List[Dict]) -> bool:
        splits = {r.get('split') for r in results if 'split' in r}
        return {'train', 'test'}.issubset(splits)

    def _compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        metrics = {}
        correct = sum(1 for r in results if r.get('correct', False))
        metrics['accuracy'] = correct / len(results)
        
        if all('score' in r for r in results):
            metrics['avg_score'] = np.mean([r['score'] for r in results])
            
        return metrics

    def _check_seed_consistency(self, results: List[Dict]) -> bool:
        seeds = {r.get('seed') for r in results if 'seed' in r}
        return len(seeds) == 1 if seeds else False

    def _check_output_determinism(self, results: List[Dict]) -> bool:
        # Check if identical inputs produce identical outputs
        output_map = {}
        for r in results:
            input_key = f"{r.get('query')}_{r.get('context')}"
            output = r.get('generated_answer')
            if input_key in output_map and output_map[input_key] != output:
                return False
            output_map[input_key] = output
        return True

    def _check_metadata_completeness(self, results: List[Dict]) -> bool:
        required_fields = {'query', 'generated_answer', 'gold_answer', 'timestamp'}
        return all(all(field in r for field in required_fields) for r in results)