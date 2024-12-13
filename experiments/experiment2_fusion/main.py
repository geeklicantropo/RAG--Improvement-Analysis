import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import torch
from tqdm import tqdm
import logging
import gc
import numpy as np
import json

from src.utils import seed_everything, clear_memory
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.document_classifier import DocumentClassifier
from src.llm_evaluator import LLMEvaluator
from src.rag_fusion_utils import RAGFusionRanker
from .config import FusionConfig
from experiments.evaluation.noise_impact import NoiseImpactAnalyzer

class FusionExperiment:
    def __init__(
        self,
        config: FusionConfig,
        experiment_name: str,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.logger = logger or ExperimentLogger(
            experiment_name=experiment_name,
            base_log_dir=str(Path(__file__).parent.parent.parent / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        self.evaluator = LLMEvaluator(self.llm)
        self.doc_classifier = DocumentClassifier(self.llm)
        self.noise_evaluator = NoiseImpactAnalyzer(self.evaluator)
        self.fusion_ranker = RAGFusionRanker(
            strategy=config.fusion_strategy,
            normalize_scores=config.normalize_scores,
            score_weights=config.fusion_weights
        )

    def run_fusion_phase(self):
        """Run fusion with systematic testing of configurations."""
        self.logger.log_step_start("Fusion Phase")
        try:
            test_configs = self._generate_test_configs()
            results = []
            
            for config in tqdm(test_configs, desc="Testing configurations"):
                fusion_results = self._run_fusion_test(config)
                position_results = self._test_positions(fusion_results)
                noise_results = self._test_noise_injection(fusion_results)
                combination_results = self._test_document_combinations(fusion_results)
                
                # Evaluate using LLM
                evaluation = self.evaluator.evaluate_answer(
                    fusion_results, 
                    position_results,
                    noise_results,
                    combination_results
                )
                
                results.append({
                    "config": config,
                    "fusion": fusion_results,
                    "positions": position_results,
                    "noise": noise_results,
                    "combinations": combination_results,
                    "evaluation": evaluation
                })
                
                clear_memory()
                
            self._save_results(results)
            self.logger.log_step_end("Fusion Phase")
            return results
            
        except Exception as e:
            self.logger.log_error(e, "Error in fusion phase")
            raise

    def _generate_test_configs(self) -> List[Dict]:
        """Generate systematic test configurations."""
        configs = []
        
        # Test different fusion strategies
        for strategy in ['rrf', 'linear']:
            # Test different document positions
            for doc_positions in ['near', 'mid', 'far']:
                # Test different noise ratios
                for noise_ratio in [0.0, 0.1, 0.2]:
                    configs.append({
                        "fusion_strategy": strategy,
                        "doc_positions": doc_positions,
                        "noise_ratio": noise_ratio,
                        "weights": self.config.fusion_weights
                    })
                    
        return configs

    def _run_fusion_test(self, config: Dict) -> Dict:
        """Run fusion test with given configuration."""
        self.logger.log_step_start(f"Testing fusion strategy={config['fusion_strategy']}")
        
        # Update fusion ranker with config
        self.fusion_ranker.strategy = config['fusion_strategy']
        
        # Get retrieval results
        contriever_results = self._get_contriever_results()
        bm25_results = self._get_bm25_results()
        
        # Perform fusion
        fused_results = self.fusion_ranker.fuse_search_results({
            'contriever': contriever_results,
            'bm25': bm25_results
        })
        
        # Categorize documents
        categorized = self.doc_classifier.classify_documents(
            fused_results,
            self.config.score_thresholds
        )
        
        self.logger.log_step_end(f"Fusion={config['fusion_strategy']}")
        return {
            "fusion_results": fused_results,
            "categories": categorized
        }

    def _test_positions(self, fusion_results: Dict) -> Dict:
        """Test impact of document positions."""
        self.logger.log_step_start("Testing positions")
        
        positions = ['near', 'mid', 'far']
        position_results = {}
        
        for pos in positions:
            # Rearrange documents based on position
            docs = self._position_documents(
                fusion_results['fusion_results'],
                position=pos
            )
            
            # Evaluate with LLM
            eval_results = self.evaluator.evaluate_answer(docs)
            position_results[pos] = eval_results
            
        self.logger.log_step_end("Position testing")
        return position_results

    def _test_noise_injection(self, fusion_results: Dict) -> Dict:
        """Test impact of noise injection."""
        self.logger.log_step_start("Testing noise")
        
        noise_ratios = [0.1, 0.2, 0.3]
        noise_results = {}
        
        for ratio in noise_ratios:
            # Inject noise
            noisy_results = self._inject_noise(
                fusion_results['fusion_results'],
                ratio=ratio
            )
            
            # Evaluate with LLM
            eval_results = self.evaluator.evaluate_answer(noisy_results)
            noise_results[ratio] = eval_results
            
        self.logger.log_step_end("Noise testing")
        return noise_results

    def _test_document_combinations(self, fusion_results: Dict) -> Dict:
        """Test different document combinations."""
        self.logger.log_step_start("Testing document combinations")
        
        # Gold + BM25
        gold_bm25 = self._combine_results(
            fusion_results['fusion_results'], 
            'gold_bm25'
        )
        gold_bm25_eval = self.evaluator.evaluate_answer(gold_bm25)
        
        # Gold + Contriever
        gold_contriever = self._combine_results(
            fusion_results['fusion_results'],
            'gold_contriever'
        )
        gold_contriever_eval = self.evaluator.evaluate_answer(gold_contriever)
        
        # BM25 + Contriever
        bm25_contriever = self._combine_results(
            fusion_results['fusion_results'],
            'bm25_contriever'
        )
        bm25_contriever_eval = self.evaluator.evaluate_answer(bm25_contriever)
        
        combination_results = {
            'gold_bm25': gold_bm25_eval,
            'gold_contriever': gold_contriever_eval,
            'bm25_contriever': bm25_contriever_eval
        }
        
        self.logger.log_step_end("Document combination testing")
        return combination_results

    def _position_documents(self, results: List[Dict], position: str) -> List[Dict]:
        """Position documents according to strategy."""
        positioned_results = []
        for result in results:
            doc_scores = list(zip(result['documents'], result['scores']))
            
            if position == 'near':
                doc_scores.sort(key=lambda x: x[1], reverse=True)
            elif position == 'mid':
                mid = len(doc_scores) // 2
                doc_scores = (
                    doc_scores[mid:] + 
                    doc_scores[:mid]
                )
            else:  # 'far'
                doc_scores.sort(key=lambda x: x[1])
                
            docs, scores = zip(*doc_scores)
            positioned_results.append({
                'documents': list(docs),
                'scores': list(scores)
            })
            
        return positioned_results

    def _inject_noise(self, results: List[Dict], ratio: float) -> List[Dict]:
        """Inject noise into fusion results."""
        noisy_results = []
        for result in results:
            num_noise = max(1, int(len(result['documents']) * ratio))
            noise_docs = self._generate_noise_docs(num_noise)
            
            noisy_docs = (
                result['documents'][:-num_noise] + 
                noise_docs
            )
            noisy_scores = (
                result['scores'][:-num_noise] + 
                [0.0] * num_noise
            )
            
            noisy_results.append({
                'documents': noisy_docs,
                'scores': noisy_scores
            })
            
        return noisy_results

    def _save_results(self, results: List[Dict]):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for phase in ['fusion', 'positions', 'noise', 'combinations']:
            phase_results = {
                config['fusion_strategy']: result[phase]
                for result in results
                for config in [result['config']]
            }
            
            output_path = output_dir / f"{phase}_results.json"
            with open(output_path, 'w') as f:
                json.dump(phase_results, f, indent=2)

    def run(self):
        """Run complete fusion experiment."""
        try:
            self.logger.log_step_start("Running fusion experiment")
            
            # Run fusion phase
            fusion_results = self.run_fusion_phase()
            
            # Save and return results
            self._save_results(fusion_results)
            
            self.logger.log_step_end("Fusion experiment completed")
            return fusion_results
            
        except Exception as e:
            self.logger.log_error(e, "Error in fusion experiment")
            raise
        finally:
            clear_memory()

def main():
    config = FusionConfig()
    experiment = FusionExperiment(config, "fusion_experiment")
    results = experiment.run()
    return results

if __name__ == "__main__":
    seed_everything(10)
    main()