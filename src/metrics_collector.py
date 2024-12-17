import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import json
from datetime import datetime

class MetricsCollector:
    def __init__(self, output_dir: str = "experiments/metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.metrics = defaultdict(dict)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("MetricsCollector")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def collect_metrics(
    self,
    results: List[Dict],
    experiment_type: str,
    include_confidence: bool = True
    ) -> Dict[str, Any]:
        metrics = {
            'experiment_type': experiment_type,
            'total_examples': len(results),
            'timestamp': datetime.now().isoformat()
        }

        # Basic metrics
        eval_results = [r['llm_evaluation'] for r in results if 'llm_evaluation' in r]
        if eval_results:
            metrics.update({
                'accuracy': np.mean([e.get('correct', False) for e in eval_results]),
                'avg_score': np.mean([e.get('score', 0) for e in eval_results]),
                'score_std': np.std([e.get('score', 0) for e in eval_results])
            })

        # Category metrics
        if any('category' in r for r in results):
            metrics['category_metrics'] = self._compute_category_metrics(results)

        # Position metrics
        if any('position' in r for r in results):
            metrics['position_metrics'] = self._compute_position_metrics(results)

        # Noise metrics
        if any('noise_ratio' in r for r in results):
            metrics['noise_metrics'] = self._compute_noise_metrics(results)

        # Statistical metrics
        metrics['statistical_metrics'] = self._compute_statistical_metrics(eval_results)

        # Confidence intervals
        if include_confidence and eval_results:
            metrics['confidence_intervals'] = self._compute_confidence_intervals(eval_results)

        self._save_metrics(metrics, experiment_type)
        return metrics

    def _compute_base_metrics(self, results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {'accuracy': 0.0, 'avg_score': 0.0, 'score_std': 0.0, 'sample_size': 0}
        
        llm_evals = [r.get('llm_evaluation', {}) for r in results]
        scores = [e.get('score', 0) for e in llm_evals]
        correct = [e.get('correct', False) for e in llm_evals]
        
        return {
            'accuracy': float(np.mean(correct)) if correct else 0.0,
            'avg_score': float(np.mean(scores)) if scores else 0.0,
            'score_std': float(np.std(scores)) if scores else 0.0,
            'sample_size': len(results)
        }

    def _compute_category_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        categories = defaultdict(list)
        
        for result in results:
            cat_list = result.get('document_categories')
            llm_eval = result.get('llm_evaluation', {})
            # If multiple docs per result have categories, we aggregate them.
            # If 'category' is not per-result but per-doc, we can handle it differently.
            # We assume per result we already have doc categories as a list.
            if cat_list:
                # Average correctness for this result
                c = llm_eval.get('correct', False)
                s = llm_eval.get('score', 0)
                for cat in cat_list:
                    categories[cat].append({'score': s, 'correct': c})
        
        cat_metrics = {}
        for cat, cat_results in categories.items():
            correct_vals = [r['correct'] for r in cat_results]
            scores = [r['score'] for r in cat_results]
            cat_metrics[cat] = {
                'accuracy': float(np.mean(correct_vals)) if correct_vals else 0.0,
                'avg_score': float(np.mean(scores)) if scores else 0.0,
                'sample_size': len(cat_results)
            }

        return cat_metrics

    def _compute_position_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        positions = defaultdict(list)
        
        for result in results:
            pos_list = result.get('document_positions')
            llm_eval = result.get('llm_evaluation', {})
            c = llm_eval.get('correct', False)
            s = llm_eval.get('score', 0)
            if pos_list:
                # Each doc has a position (near, mid, far).
                # We'll aggregate by all positions present in this result.
                for p in pos_list:
                    positions[p].append({'score': s, 'correct': c})
                
        pos_metrics = {}
        for p, pos_results in positions.items():
            correct_vals = [r['correct'] for r in pos_results]
            scores = [r['score'] for r in pos_results]
            pos_metrics[p] = {
                'accuracy': float(np.mean(correct_vals)) if correct_vals else 0.0,
                'avg_score': float(np.mean(scores)) if scores else 0.0,
                'sample_size': len(pos_results)
            }

        return pos_metrics

    def _compute_noise_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        noise_levels = defaultdict(list)
        
        for result in results:
            llm_eval = result.get('llm_evaluation', {})
            c = llm_eval.get('correct', False)
            s = llm_eval.get('score', 0)
            nl = result.get('noise_ratio', None)
            if nl is not None:
                noise_levels[nl].append({'score': s, 'correct': c})

        metrics = {}
        baseline = noise_levels.get(0.0, [])
        baseline_acc = float(np.mean([r['correct'] for r in baseline])) if baseline else 0
        
        for noise_ratio, noise_results in noise_levels.items():
            correct_vals = [r['correct'] for r in noise_results]
            scores = [r['score'] for r in noise_results]
            accuracy = float(np.mean(correct_vals)) if correct_vals else 0.0
            metrics[f'noise_{noise_ratio}'] = {
                'accuracy': accuracy,
                'avg_score': float(np.mean(scores)) if scores else 0.0,
                'degradation': baseline_acc - accuracy if noise_ratio > 0 else 0.0,
                'sample_size': len(noise_results)
            }
            
        return metrics

    def _compute_mode_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        mode_map = defaultdict(lambda: {'correct': 0, 'total': 0, 'scores': []})
        for r in results:
            llm_eval = r.get('llm_evaluation', {})
            c = llm_eval.get('correct', False)
            s = llm_eval.get('score', 0)
            m = r.get('mode')
            if m is not None:
                mode_map[m]['total'] += 1
                if c:
                    mode_map[m]['correct'] += 1
                mode_map[m]['scores'].append(s)

        mode_metrics = {}
        for m, stats in mode_map.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            avg_s = float(np.mean(stats['scores'])) if stats['scores'] else 0.0
            mode_metrics[m] = {
                'accuracy': acc,
                'avg_score': avg_s,
                'sample_size': stats['total']
            }
        return mode_metrics

    def _compute_combination_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        combo_map = defaultdict(lambda: {'correct': 0, 'total': 0, 'scores': []})
        for r in results:
            llm_eval = r.get('llm_evaluation', {})
            c = llm_eval.get('correct', False)
            s = llm_eval.get('score', 0)
            ct = r.get('combination_type')
            if ct is not None:
                combo_map[ct]['total'] += 1
                if c:
                    combo_map[ct]['correct'] += 1
                combo_map[ct]['scores'].append(s)

        combo_metrics = {}
        for ct, stats in combo_map.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            avg_s = float(np.mean(stats['scores'])) if stats['scores'] else 0.0
            combo_metrics[ct] = {
                'accuracy': acc,
                'avg_score': avg_s,
                'sample_size': stats['total']
            }
        return combo_metrics

    def _compute_statistical_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        scores = [r.get('llm_evaluation', {}).get('score', 0) for r in results]
        if len(scores) < 2:  # not enough data for normality test
            return {'normality_test_p': None, 'effect_size': 0.0, 'statistically_significant': False}

        _, normality_p = stats.normaltest(scores)
        effect_size = (np.mean(scores) / np.std(scores)) if np.std(scores) > 0 else 0.0
        
        return {
            'normality_test_p': float(normality_p),
            'effect_size': effect_size,
            'statistically_significant': normality_p < 0.05
        }

    def _compute_confidence_intervals(self, results: List[Dict]) -> Dict[str, List[float]]:
        scores = [r.get('llm_evaluation', {}).get('score', 0) for r in results]
        if len(scores) < 2:
            return {'ci_95': [0, 0], 'ci_99': [0, 0]}

        sem = stats.sem(scores)
        mean = np.mean(scores)
        ci_95 = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=sem)
        ci_99 = stats.t.interval(0.99, len(scores)-1, loc=mean, scale=sem)

        return {
            'ci_95': [float(ci_95[0]), float(ci_95[1])] if ci_95 else [0,0],
            'ci_99': [float(ci_99[0]), float(ci_99[1])] if ci_99 else [0,0]
        }

    def _save_metrics(self, metrics: Dict[str, Any], experiment_type: str) -> None:
        output_file = self.output_dir / f"{experiment_type}_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Saved metrics to {output_file}")
