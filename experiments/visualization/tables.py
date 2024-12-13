import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path
from scipy import stats
from tqdm import tqdm

class ResultsTableGenerator:
    def __init__(self, output_dir: str = "experiments/visualization/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("TableGenerator")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

    def generate_all_tables(self, results: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
        tables = {}
        with tqdm(total=4, desc="Generating tables") as pbar:
            tables['accuracy'] = self.generate_accuracy_table(results)
            pbar.update(1)
            
            tables['noise_impact'] = self.generate_noise_table(results)
            pbar.update(1)
            
            tables['position_impact'] = self.generate_position_table(results)
            pbar.update(1)
            
            tables['statistical_significance'] = self.generate_significance_table(results)
            pbar.update(1)
            
        self._save_tables(tables)
        return tables

    def generate_accuracy_table(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        data = []
        for exp_name, exp_results in results.items():
            metrics = self._compute_metrics(exp_results)
            data.append({
                'Experiment': exp_name,
                'Accuracy': f"{metrics['accuracy']:.3f} ± {metrics['accuracy_std']:.3f}",
                'Avg Score': f"{metrics['avg_score']:.2f}",
                'N': metrics['count']
            })
        return pd.DataFrame(data)

    def generate_noise_table(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        data = []
        for exp_name, exp_results in results.items():
            noise_levels = sorted(set(r.get('noise_ratio', 0.0) for r in exp_results))
            for noise in noise_levels:
                noise_results = [r for r in exp_results if r.get('noise_ratio', 0.0) == noise]
                metrics = self._compute_metrics(noise_results)
                data.append({
                    'Experiment': exp_name,
                    'Noise Ratio': f"{noise:.1f}",
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Score Degradation': f"{metrics.get('degradation', 0):.3f}"
                })
        return pd.DataFrame(data)

    def generate_position_table(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        data = []
        for exp_name, exp_results in results.items():
            positions = sorted(set(r.get('position', 'unknown') for r in exp_results))
            for pos in positions:
                pos_results = [r for r in exp_results if r.get('position') == pos]
                metrics = self._compute_metrics(pos_results)
                data.append({
                    'Experiment': exp_name,
                    'Position': pos,
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Impact': f"{metrics.get('position_impact', 0):.3f}"
                })
        return pd.DataFrame(data)

    def generate_significance_table(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        experiments = list(results.keys())
        data = []
        
        for i, exp1 in enumerate(experiments):
            for exp2 in experiments[i+1:]:
                scores1 = [r['llm_evaluation']['score'] for r in results[exp1]]
                scores2 = [r['llm_evaluation']['score'] for r in results[exp2]]
                
                t_stat, p_value = stats.ttest_ind(scores1, scores2)
                effect_size = (np.mean(scores1) - np.mean(scores2)) / np.sqrt(
                    (np.std(scores1)**2 + np.std(scores2)**2) / 2)
                
                data.append({
                    'Comparison': f"{exp1} vs {exp2}",
                    'p-value': f"{p_value:.4f}",
                    'Effect Size': f"{effect_size:.3f}",
                    'Significant': "Yes" if p_value < 0.01 else "No"
                })
                
        return pd.DataFrame(data)

    def _compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        evals = [r['llm_evaluation'] for r in results]
        return {
            'accuracy': np.mean([e['correct'] for e in evals]),
            'accuracy_std': np.std([e['correct'] for e in evals]),
            'avg_score': np.mean([e['score'] for e in evals]),
            'score_std': np.std([e['score'] for e in evals]),
            'count': len(results)
        }

    def _save_tables(self, tables: Dict[str, pd.DataFrame]):
        for name, df in tables.items():
            path = self.output_dir / f"{name}_table.csv"
            df.to_csv(path, index=False)
            self.logger.info(f"Saved {name} table to {path}")