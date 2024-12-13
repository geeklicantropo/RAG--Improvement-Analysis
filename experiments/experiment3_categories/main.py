import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from tqdm import tqdm
import gc

from src.utils import seed_everything, clear_memory
from src.experiment_logger import ExperimentLogger
from src.llm import LLM 
from src.document_classifier import DocumentClassifier
from src.llm_evaluator import LLMEvaluator
from src.rag_fusion_utils import RAGFusionRanker
from .config import CategoriesConfig
from experiments.evaluation.noise_impact import NoiseImpactAnalyzer

class CategoriesExperiment:
    def __init__(
        self,
        config: CategoriesConfig,
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

    def run_categories_phase(self):
        self.logger.log_step_start("Categories Phase")
        try:
            test_configs = self._generate_test_configs()
            results = []
            
            for config in tqdm(test_configs, desc="Testing configurations"):
                category_results = self._run_category_test(config)
                position_results = self._test_positions(category_results)
                noise_results = self._test_noise_injection(category_results)
                combination_results = self._test_document_combinations(category_results)
                
                # Evaluate each category result
                category_evaluations = {}
                for category, docs in category_results['documents'].items():
                    for doc in docs:
                        eval_result = self.evaluator.evaluate_answer(
                            question=doc['query'],
                            generated_answer=doc['generated_answer'],
                            gold_answer=doc['gold_answer']
                        )
                        category_evaluations[category] = eval_result
                
                results.append({
                    "config": config,
                    "categories": category_results,
                    "positions": position_results,
                    "noise": noise_results,
                    "combinations": combination_results,
                    "evaluations": category_evaluations
                })
                            
                clear_memory()
                
            self._save_results(results)
            self.logger.log_step_end("Categories Phase")
            return results
            
        except Exception as e:
            self.logger.log_error(e, "Error in categories phase")
            raise

    def _generate_test_configs(self) -> List[Dict]:
        configs = []
        confidence_levels = [0.8, 0.6, 0.4]
        doc_positions = ['near', 'mid', 'far']
        noise_ratios = [0.0, 0.1, 0.2]
        
        for confidence in confidence_levels:
            for position in doc_positions:
                for noise in noise_ratios:
                    configs.append({
                        "confidence_threshold": confidence,
                        "doc_position": position,
                        "noise_ratio": noise,
                        "max_docs_per_category": self.config.max_docs_per_category
                    })
        return configs

    def _run_category_test(self, config: Dict) -> Dict:
        self.logger.log_step_start(f"Testing configuration threshold={config['confidence_threshold']}")
        
        docs = self._get_documents()
        categorized = self.doc_classifier.classify_documents(
            docs,
            self.config.score_thresholds
        )
        
        pruned = self._prune_categories(
            categorized,
            max_docs=config['max_docs_per_category']
        )
        
        self.logger.log_step_end("Category test completed")
        return {"documents": pruned}

    def _test_positions(self, results: Dict) -> Dict:
        self.logger.log_step_start("Testing positions")
        position_results = {}
        
        for position in ['near', 'mid', 'far']:
            positioned_docs = self._position_documents(
                results['documents'],
                position=position
            )
            eval_results = self.evaluator.evaluate_answer(positioned_docs)
            position_results[position] = eval_results
            
        self.logger.log_step_end("Position testing")
        return position_results

    def _test_noise_injection(self, results: Dict) -> Dict:
        self.logger.log_step_start("Testing noise")
        noise_results = {}
        
        for ratio in [0.1, 0.2, 0.3]:
            noisy_docs = self._inject_noise(
                results['documents'],
                ratio=ratio
            )
            eval_results = self.evaluator.evaluate_answer(noisy_docs)
            noise_results[ratio] = eval_results
            
        self.logger.log_step_end("Noise testing")
        return noise_results

    def _test_document_combinations(self, results: Dict) -> Dict:
        self.logger.log_step_start("Testing combinations")
        combinations = {}
        
        # Gold + High confidence
        gold_high = self._combine_categories(
            results['documents'],
            ['gold', 'high_confidence']
        )
        combinations['gold_high'] = self.evaluator.evaluate_answer(gold_high)
        
        # Gold + Medium confidence
        gold_med = self._combine_categories(
            results['documents'], 
            ['gold', 'medium_confidence']
        )
        combinations['gold_medium'] = self.evaluator.evaluate_answer(gold_med)
        
        # High + Medium confidence
        high_med = self._combine_categories(
            results['documents'],
            ['high_confidence', 'medium_confidence']
        )
        combinations['high_medium'] = self.evaluator.evaluate_answer(high_med)
        
        self.logger.log_step_end("Combination testing")
        return combinations

    def _position_documents(self, docs: Dict[str, List], position: str) -> Dict[str, List]:
        positioned = docs.copy()
        if 'gold' in positioned:
            gold_docs = positioned.pop('gold')
            if position == 'near':
                positioned = {'gold': gold_docs, **positioned}
            elif position == 'mid':
                keys = list(positioned.keys())
                mid = len(keys) // 2
                for i, k in enumerate(keys):
                    if i == mid:
                        positioned['gold'] = gold_docs
                    positioned[k] = positioned[k]
            else:  # far
                positioned['gold'] = gold_docs
        return positioned

    def _inject_noise(self, docs: Dict[str, List], ratio: float) -> Dict[str, List]:
        total_docs = sum(len(d) for d in docs.values())
        num_noise = max(1, int(total_docs * ratio))
        noise_docs = self._generate_noise_docs(num_noise)
        
        noisy_docs = docs.copy()
        noisy_docs['noise'] = noise_docs
        return noisy_docs

    def _combine_categories(self, docs: Dict[str, List], categories: List[str]) -> Dict[str, List]:
        return {
            cat: docs[cat]
            for cat in categories
            if cat in docs
        }

    def _save_results(self, results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for phase in ['categories', 'positions', 'noise', 'combinations']:
            phase_results = {
                f"threshold_{result['config']['confidence_threshold']}": result[phase]
                for result in results
            }
            
            output_path = output_dir / f"{phase}_results.json"
            with open(output_path, 'w') as f:
                json.dump(phase_results, f, indent=2)

    def run(self):
        try:
            self.logger.log_step_start("Running categories experiment")
            results = self.run_categories_phase()
            self._save_results(results)
            self.logger.log_step_end("Categories experiment completed")
            return results
            
        except Exception as e:
            self.logger.log_error(e, "Error in categories experiment")
            raise
        finally:
            clear_memory()

def main():
    config = CategoriesConfig()
    experiment = CategoriesExperiment(config, "categories_experiment")
    results = experiment.run()
    return results

if __name__ == "__main__":
    seed_everything(10)
    main()