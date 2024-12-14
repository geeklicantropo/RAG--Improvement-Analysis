import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from scipy import stats
from tqdm import tqdm
import json
import hashlib

class ExperimentValidator:
    def __init__(self, output_dir: str = "experiments/validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.validation_cache = {}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ExperimentValidator")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def validate_experiment(
        self,
        results: List[Dict],
        config: Dict,
        experiment_type: str
    ) -> Dict[str, Any]:
        validations = {
            'evaluation_consistency': self._validate_llm_evaluation(results),
            'document_handling': self._validate_document_handling(results, config),
            'statistical_validity': self._validate_statistics(results),
            'methodology': self._validate_methodology(results, config)
        }
        
        self._save_validation(validations, experiment_type)
        return validations

    def _validate_llm_evaluation(self, results: List[Dict]) -> Dict[str, bool]:
        llm_validations = {}
        
        # Check evaluation presence
        has_evaluations = all('llm_evaluation' in r for r in results)
        llm_validations['complete_evaluations'] = has_evaluations
        
        if not has_evaluations:
            return llm_validations
            
        # Check evaluation consistency
        evaluations = [r['llm_evaluation'] for r in results]
        required_fields = {'score', 'correct', 'reasoning'}
        fields_present = all(
            all(field in eval_dict for field in required_fields)
            for eval_dict in evaluations
        )
        llm_validations['consistent_format'] = fields_present
        
        # Check score ranges
        scores = [e['score'] for e in evaluations]
        llm_validations['valid_scores'] = all(0 <= s <= 1 for s in scores)
        
        # Check reasoning presence
        has_reasoning = all(len(str(e.get('reasoning', ''))) > 0 for e in evaluations)
        llm_validations['has_reasoning'] = has_reasoning
        
        return llm_validations

    def _validate_document_handling(self, results: List[Dict], config: Dict) -> Dict[str, bool]:
        doc_validations = {}
        
        # Check category distribution
        if 'category' in results[0]:
            categories = [r['category'] for r in results]
            unique_cats = set(categories)
            expected_cats = {'gold', 'distracting', 'random'}
            doc_validations['correct_categories'] = expected_cats.issubset(unique_cats)
            
            # Check category balance
            cat_counts = {cat: categories.count(cat) for cat in unique_cats}
            max_imbalance = config.get('max_category_imbalance', 2.0)
            doc_validations['balanced_categories'] = (
                max(cat_counts.values()) / min(cat_counts.values()) <= max_imbalance
            )
            
        # Check position tracking
        if 'position' in results[0]:
            positions = [r['position'] for r in results]
            doc_validations['position_tracked'] = len(set(positions)) > 1
            
        # Check noise injection
        if 'noise_ratio' in results[0]:
            noise_ratios = [r['noise_ratio'] for r in results]
            expected_ratios = config.get('noise_ratios', [0.0, 0.1, 0.2])
            doc_validations['correct_noise_levels'] = all(r in expected_ratios for r in noise_ratios)
            
        return doc_validations

    def _validate_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        scores = [r.get('llm_evaluation', {}).get('score', 0) for r in results]
        
        stats_validation = {
            'sample_size': len(results),
            'sufficient_samples': len(results) >= 100,
        }
        
        if len(scores) > 1:
            # Normality test
            _, p_value = stats.normaltest(scores)
            stats_validation['normal_distribution'] = p_value > 0.05
            
            # Effect size
            effect_size = np.mean(scores) / np.std(scores) if np.std(scores) > 0 else 0
            stats_validation['significant_effect'] = abs(effect_size) > 0.3
            
            # Score distribution
            stats_validation['score_distribution'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
            
        return stats_validation

    def _validate_methodology(self, results: List[Dict], config: Dict) -> Dict[str, bool]:
        methodology_checks = {}
        
        # Check experiment conditions
        methodology_checks['consistent_conditions'] = self._check_experimental_conditions(results, config)
        
        # Check data splits
        if 'split' in results[0]:
            splits = set(r['split'] for r in results)
            methodology_checks['proper_splits'] = {'train', 'test'}.issubset(splits)
            
        # Check randomization
        if 'random_seed' in config:
            methodology_checks['controlled_randomization'] = True
            
        # Check evaluation independence
        evaluation_hashes = [
            hashlib.md5(str(r.get('llm_evaluation')).encode()).hexdigest()
            for r in results
        ]
        methodology_checks['independent_evaluations'] = len(set(evaluation_hashes)) > len(results) * 0.9
        
        return methodology_checks

    def _check_experimental_conditions(self, results: List[Dict], config: Dict) -> bool:
        required_conditions = config.get('required_conditions', [])
        for condition in required_conditions:
            if condition not in results[0]:
                return False
        return True

    def _save_validation(self, validations: Dict[str, Any], experiment_type: str) -> None:
        output_file = self.output_dir / f"{experiment_type}_validation.json"
        with open(output_file, 'w') as f:
            json.dump(validations, f, indent=2)
        self.logger.info(f"Saved validation results to {output_file}")