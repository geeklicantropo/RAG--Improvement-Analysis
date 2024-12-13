import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from collections import defaultdict

from src.llm import LLM
from src.experiment_logger import ExperimentLogger

class LLMEvaluator:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.logger = self._setup_logger()
        self.eval_cache = {}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("LLMEvaluator")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        gold_answer: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate answer correctness using LLM."""
        cache_key = f"{question}:{generated_answer}:{gold_answer}"
        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]

        eval_prompt = f"""
        Question: {question}
        Generated Answer: {generated_answer}
        Gold Answer: {gold_answer}
        {"Context: " + context if context else ""}

        Evaluate the generated answer, considering:
        1. Factual accuracy compared to gold answer
        2. Completeness of information
        3. Semantic equivalence

        Provide evaluation as:
        Score (0-100):
        Correct (Yes/No):
        Reasoning:"""

        response = self.llm.generate(eval_prompt)
        
        try:
            lines = response.strip().split('\n')
            score = float(lines[0].split(':')[1].strip())
            correct = 'yes' in lines[1].lower()
            reasoning = lines[2].split(':')[1].strip()
            
            result = {
                'score': score,
                'correct': correct,
                'reasoning': reasoning
            }
            
            self.eval_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing evaluation response: {str(e)}")
            return {'score': 0, 'correct': False, 'reasoning': 'Error in evaluation'}

    def evaluate_clusters(
        self,
        clusters: Dict[int, List[Dict]],
        query: str
    ) -> Dict[str, Any]:
        """Evaluate clustering quality using LLM."""
        self.logger.info("Evaluating clusters")
        metrics = defaultdict(float)
        
        for cluster_id, docs in tqdm(clusters.items(), desc="Evaluating clusters"):
            # Evaluate cluster coherence
            doc_texts = "\n".join(d['text'][:200] for d in docs)  # Truncate for prompt length
            coherence_prompt = f"""
            Query: {query}
            Documents in cluster:
            {doc_texts}

            Rate the semantic coherence and relevance of this cluster (0-100):"""
            
            try:
                coherence = float(self.llm.generate(coherence_prompt).strip())
                metrics[f'cluster_{cluster_id}_coherence'] = coherence
            except ValueError:
                self.logger.warning(f"Could not parse coherence score for cluster {cluster_id}")
                metrics[f'cluster_{cluster_id}_coherence'] = 0

        metrics['avg_coherence'] = sum(v for k, v in metrics.items() if 'coherence' in k) / len(clusters)
        return dict(metrics)

    def evaluate_noise_impact(
        self,
        clean_results: List[Dict],
        noisy_results: List[Dict]
    ) -> Dict[str, Any]:
        """Compare performance with and without noise."""
        self.logger.info("Evaluating noise impact")
        
        clean_scores = []
        noisy_scores = []
        
        for clean, noisy in tqdm(zip(clean_results, noisy_results), 
                                desc="Evaluating noise impact",
                                total=len(clean_results)):
            clean_eval = self.evaluate_answer(
                clean['query'],
                clean['generated_answer'],
                clean['gold_answer']
            )
            noisy_eval = self.evaluate_answer(
                noisy['query'],
                noisy['generated_answer'],
                noisy['gold_answer']
            )
            
            clean_scores.append(clean_eval['score'])
            noisy_scores.append(noisy_eval['score'])

        return {
            'clean_avg_score': sum(clean_scores) / len(clean_scores),
            'noisy_avg_score': sum(noisy_scores) / len(noisy_scores),
            'score_degradation': ((sum(clean_scores) / len(clean_scores)) - 
                                (sum(noisy_scores) / len(noisy_scores)))
        }

    def evaluate_position_impact(
        self,
        results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze impact of document positions."""
        self.logger.info("Evaluating position impact")
        
        position_metrics = defaultdict(lambda: defaultdict(list))
        
        for result in tqdm(results, desc="Evaluating positions"):
            eval_result = self.evaluate_answer(
                result['query'],
                result['generated_answer'],
                result['gold_answer']
            )
            
            position = result.get('gold_position', 'unknown')
            position_metrics[position]['scores'].append(eval_result['score'])
            position_metrics[position]['correct'].append(eval_result['correct'])

        return {
            pos: {
                'avg_score': sum(metrics['scores']) / len(metrics['scores']),
                'accuracy': sum(metrics['correct']) / len(metrics['correct'])
            }
            for pos, metrics in position_metrics.items()
        }