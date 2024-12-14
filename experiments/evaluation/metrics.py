import logging
import json
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from scipy import stats

class EvaluationMetrics:
    def __init__(self, output_dir: str = "experiments/evaluation/results"):
        """
        Computes evaluation metrics from experiment results.
        Saves computed metrics as JSON in output_dir.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("EvaluationMetrics")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def compute_metrics_from_file(self, results_file: str, include_confidence: bool = True) -> Dict[str, Any]:
        """
        Load results from a JSON file and compute metrics.
        Results expected to be a list or nested dicts with final records containing 'llm_evaluation'.
        """
        self.logger.info(f"Loading results from {results_file}")
        with open(results_file) as f:
            results = json.load(f)

        self.logger.info("Computing evaluation metrics...")
        metrics = self._compute_all_metrics(results, include_confidence)
        self._save_metrics(metrics, results_file)
        return metrics

    def _compute_all_metrics(self, results: Any, include_confidence: bool) -> Dict[str, Any]:
        """
        Recursively compute metrics for all nested scenarios in the results.
        If results is a list of dicts (records), compute scenario metrics.
        If results is a dict, recurse.
        """
        if isinstance(results, list) and results and isinstance(results[0], dict) and 'llm_evaluation' in results[0]:
            return self._compute_scenario_metrics(results, include_confidence)

        if isinstance(results, dict):
            out = {}
            for k, v in results.items():
                out[k] = self._compute_all_metrics(v, include_confidence)
            return out

        return {}

    def _compute_scenario_metrics(self, records: List[Dict], include_confidence: bool) -> Dict[str, Any]:
        """
        Compute metrics for a single scenario given a list of records.
        Records have 'llm_evaluation' with 'correct' and 'score'.
        """
        metrics = self._compute_base_metrics(records)

        if any('document_categories' in r for r in records) or any('category' in r for r in records):
            cat_metrics = self._compute_category_metrics(records)
            if cat_metrics:
                metrics['category_metrics'] = cat_metrics

        if any('document_positions' in r for r in records) or any('position' in r for r in records):
            pos_metrics = self._compute_position_metrics(records)
            if pos_metrics:
                metrics['position_metrics'] = pos_metrics

        if any('noise_ratio' in r for r in records):
            noise_metrics = self._compute_noise_metrics(records)
            if noise_metrics:
                metrics['noise_metrics'] = noise_metrics

        if any('mode' in r for r in records):
            mode_metrics = self._compute_mode_metrics(records)
            if mode_metrics:
                metrics['mode_metrics'] = mode_metrics

        if any('combination_type' in r for r in records):
            combo_metrics = self._compute_combination_metrics(records)
            if combo_metrics:
                metrics['combination_metrics'] = combo_metrics

        stats_metrics = self._compute_statistical_metrics(records)
        metrics['statistical_metrics'] = stats_metrics

        if include_confidence:
            ci = self._compute_confidence_intervals(records)
            metrics['confidence_intervals'] = ci

        return metrics

    def _compute_base_metrics(self, records: List[Dict]) -> Dict[str, float]:
        if not records:
            return {'accuracy':0.0,'avg_score':0.0,'score_std':0.0,'sample_size':0}
        
        scores = [r.get('llm_evaluation',{}).get('score',0) for r in records]
        correct = [r.get('llm_evaluation',{}).get('correct',False) for r in records]
        return {
            'accuracy': float(np.mean(correct)) if correct else 0.0,
            'avg_score': float(np.mean(scores)) if scores else 0.0,
            'score_std': float(np.std(scores)) if scores else 0.0,
            'sample_size': len(records)
        }

    def _compute_category_metrics(self, records: List[Dict]) -> Dict[str, Dict[str,float]]:
        # Attempt to derive categories from 'document_categories' or 'category'
        categories = defaultdict(list)
        for r in records:
            llm_eval = r.get('llm_evaluation',{})
            c = llm_eval.get('correct',False)
            s = llm_eval.get('score',0)
            
            # If we have per-record categories in 'document_categories' (list)
            doc_cats = r.get('document_categories')
            if doc_cats:
                for cat in doc_cats:
                    categories[cat].append({'score':s,'correct':c})
            
            # If we have a single 'category' per result
            single_cat = r.get('category')
            if single_cat:
                categories[single_cat].append({'score':s,'correct':c})

        cat_metrics = {}
        for cat, vals in categories.items():
            if vals:
                correct_vals = [v['correct'] for v in vals]
                scores = [v['score'] for v in vals]
                cat_metrics[cat] = {
                    'accuracy': float(np.mean(correct_vals)),
                    'avg_score': float(np.mean(scores)),
                    'score_std': float(np.std(scores)),
                    'sample_size': len(vals)
                }
        return cat_metrics

    def _compute_position_metrics(self, records: List[Dict]) -> Dict[str,Dict[str,float]]:
        positions = defaultdict(list)
        for r in records:
            llm_eval = r.get('llm_evaluation',{})
            c = llm_eval.get('correct',False)
            s = llm_eval.get('score',0)

            # If we have per-record 'document_positions' (list)
            doc_positions = r.get('document_positions')
            if doc_positions:
                for p in doc_positions:
                    positions[p].append({'score':s,'correct':c})

            # If we have single 'position'
            single_pos = r.get('position')
            if single_pos:
                positions[single_pos].append({'score':s,'correct':c})

        pos_metrics = {}
        for p, vals in positions.items():
            correct_vals = [v['correct'] for v in vals]
            scores = [v['score'] for v in vals]
            pos_metrics[p] = {
                'accuracy': float(np.mean(correct_vals)) if correct_vals else 0.0,
                'avg_score': float(np.mean(scores)) if scores else 0.0,
                'score_std': float(np.std(scores)) if scores else 0.0,
                'sample_size': len(vals)
            }
        return pos_metrics

    def _compute_noise_metrics(self, records: List[Dict]) -> Dict[str, Dict[str,float]]:
        noise_map = defaultdict(list)
        for r in records:
            llm_eval = r.get('llm_evaluation',{})
            c = llm_eval.get('correct',False)
            s = llm_eval.get('score',0)
            nl = r.get('noise_ratio',None)
            if nl is not None:
                noise_map[nl].append({'score':s,'correct':c})

        noise_metrics = {}
        baseline_vals = noise_map.get(0.0,[])
        baseline_acc = float(np.mean([x['correct'] for x in baseline_vals])) if baseline_vals else 0.0

        for nl, vals in noise_map.items():
            correct_vals = [x['correct'] for x in vals]
            scores = [x['score'] for x in vals]
            acc = float(np.mean(correct_vals)) if correct_vals else 0.0
            degr = (acc - baseline_acc) if nl!=0.0 else 0.0
            noise_metrics[f'noise_{nl}'] = {
                'accuracy': acc,
                'avg_score': float(np.mean(scores)) if scores else 0.0,
                'score_std': float(np.std(scores)) if scores else 0.0,
                'degradation': degr,
                'sample_size': len(vals)
            }
        return noise_metrics

    def _compute_mode_metrics(self, records: List[Dict]) -> Dict[str, Dict[str,float]]:
        mode_map = defaultdict(list)
        for r in records:
            llm_eval = r.get('llm_evaluation',{})
            c = llm_eval.get('correct',False)
            s = llm_eval.get('score',0)
            m = r.get('mode',None)
            if m:
                mode_map[m].append({'score':s,'correct':c})

        mode_metrics = {}
        for m, vals in mode_map.items():
            correct_vals = [x['correct'] for x in vals]
            scores = [x['score'] for x in vals]
            mode_metrics[m] = {
                'accuracy':float(np.mean(correct_vals)) if correct_vals else 0.0,
                'avg_score':float(np.mean(scores)) if scores else 0.0,
                'score_std':float(np.std(scores)) if scores else 0.0,
                'sample_size':len(vals)
            }

        return mode_metrics

    def _compute_combination_metrics(self, records: List[Dict]) -> Dict[str,Dict[str,float]]:
        combo_map = defaultdict(list)
        for r in records:
            llm_eval = r.get('llm_evaluation',{})
            c = llm_eval.get('correct',False)
            s = llm_eval.get('score',0)
            ct = r.get('combination_type',None)
            if ct:
                combo_map[ct].append({'score':s,'correct':c})

        combo_metrics = {}
        for ct, vals in combo_map.items():
            correct_vals = [x['correct'] for x in vals]
            scores = [x['score'] for x in vals]
            combo_metrics[ct] = {
                'accuracy': float(np.mean(correct_vals)) if correct_vals else 0.0,
                'avg_score': float(np.mean(scores)) if scores else 0.0,
                'score_std': float(np.std(scores)) if scores else 0.0,
                'sample_size': len(vals)
            }

        return combo_metrics

    def _compute_statistical_metrics(self, records: List[Dict]) -> Dict[str,Any]:
        scores = [r.get('llm_evaluation',{}).get('score',0) for r in records]
        if len(scores)<2:
            return {
                'normality_test':{'normal_distribution':False,'p_value':None},
                'effect_size':{'cohens_d':0.0},
                'confidence_intervals':{
                    'score_95':[0.0,0.0],
                    'score_99':[0.0,0.0]
                }
            }

        _, normality_p = stats.normaltest(scores)
        effect_size = float(np.mean(scores)/np.std(scores)) if np.std(scores)>0 else 0.0

        ci_95 = stats.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=stats.sem(scores))
        ci_99 = stats.t.interval(0.99, len(scores)-1, loc=np.mean(scores), scale=stats.sem(scores))

        return {
            'normality_test':{
                'normal_distribution':normality_p>0.05,
                'p_value':float(normality_p)
            },
            'effect_size':{'cohens_d':effect_size},
            'confidence_intervals':{
                'score_95':[float(ci_95[0]),float(ci_95[1])] if ci_95 else [0.0,0.0],
                'score_99':[float(ci_99[0]),float(ci_99[1])] if ci_99 else [0.0,0.0]
            }
        }

    def _compute_confidence_intervals(self, records: List[Dict]) -> Dict[str,List[float]]:
        scores = [r.get('llm_evaluation',{}).get('score',0) for r in records]
        if len(scores)<2:
            return {'ci_95':[0.0,0.0],'ci_99':[0.0,0.0]}
        sem = stats.sem(scores)
        mean = np.mean(scores)
        ci_95 = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=sem)
        ci_99 = stats.t.interval(0.99, len(scores)-1, loc=mean, scale=sem)
        return {
            'ci_95':[float(ci_95[0]),float(ci_95[1])] if ci_95 else [0.0,0.0],
            'ci_99':[float(ci_99[0]),float(ci_99[1])] if ci_99 else [0.0,0.0]
        }

    def _save_metrics(self, metrics: Dict[str,Any], results_file:str):
        output_name = Path(results_file).stem + "_evaluation_metrics.json"
        output_path = self.output_dir / output_name
        with open(output_path,'w') as f:
            json.dump(metrics,f,indent=2)
        self.logger.info(f"Metrics saved to {output_path}")
