import os
import torch
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import logging
from tqdm import tqdm
from .config import BaselineConfig

from src.utils.corpus_manager import CorpusManager
from src.utils.file_utils import seed_everything, clear_memory
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.llm_evaluator import LLMEvaluator
from src.generate_answers_llm import _format_prompt, _contains_answer
from src.utils.rate_limit import rate_limit

class BaselineExperiment:
    def __init__(
        self,
        config: BaselineConfig,
        corpus_manager: CorpusManager,
        llm_evaluator: LLMEvaluator,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.corpus_manager = corpus_manager
        self.llm_evaluator = llm_evaluator
        self.logger = logger or ExperimentLogger(
            experiment_name="baseline_experiment",
            base_log_dir=str(Path(__file__).parent.parent.parent / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        self.output_dir = Path("experiments/experiment0_baseline/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        with open(config.train_dataset_path) as f:
            self.train_data = json.load(f)
        with open(config.test_dataset_path) as f:
            self.test_data = json.load(f)
        
        # Fix: Add corpus_type='base' when calling get_random_subset
        self.base_corpus = self.corpus_manager.get_random_subset(
            corpus_type='base', 
            num_docs=config.base_corpus_size
        )

        # Random and adversarial docs
        self.random_corpus = self.corpus_manager.get_random_subset(
            corpus_type='random',
            num_docs=config.num_random_docs,
            seed=config.random_seed
        )
        self.adversarial_corpus = self.corpus_manager.get_random_subset(
            corpus_type='adversarial',
            num_docs=config.num_adversarial_docs,
            seed=config.adversarial_seed
        )


    @rate_limit
    def run(self) -> Dict[str, List[Dict]]:
        modes = ["gold_only", "gold_random", "gold_adversarial"]
        results = {}

        for mode in modes:
            if self._check_completion(mode):
                self.logger.info(f"Mode {mode} already completed, skipping...")
                results_files = list(Path(f"experiments/experiment0_baseline/results/{mode}").glob("results_*.json"))
                with open(results_files[0]) as f:
                    results[mode] = json.load(f)
                continue

            self.logger.info(f"Running {mode} experiment")
            augment_docs = None
            if mode == "gold_random":
                augment_docs = self.random_corpus[:1000]
            elif mode == "gold_adversarial":
                augment_docs = self.adversarial_corpus[:1000]
                
            results[mode] = self._evaluate(mode=mode, augment_docs=augment_docs)
            
        return results
    

    def _check_completion(self, mode: str) -> bool:
        results_dir = Path(f"experiments/experiment0_baseline/results/{mode}")
        if results_dir.exists():
            results_files = list(results_dir.glob("results_*.json"))
            return len(results_files) > 0
        return False

    @rate_limit
    def _evaluate(self, mode: str, augment_docs: Optional[List[Dict]] = None) -> List[Dict]:
        mode_dir = self.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = mode_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing final results
        final_results = list(mode_dir.glob("results_*.json"))
        if final_results:
            self.logger.info(f"Found completed results for {mode}, skipping evaluation")
            with open(final_results[0]) as f:
                return json.load(f)

        # Resume from checkpoint if exists
        results = []
        processed_ids = set()
        
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))
            if checkpoints:
                self.logger.info(f"Resuming from checkpoint: {checkpoints[-1]}")
                with open(checkpoints[-1]) as f:
                    results = json.load(f)
                    processed_ids = {r["example_id"] for r in results}

        # Process remaining examples
        remaining_examples = [
            ex for ex in self.test_data 
            if ex["example_id"] not in processed_ids
        ]

        for example in tqdm(remaining_examples):
            question = example["question"]
            gold_answer = example["answers"][0]
            gold_doc = {"text": example["text"], "title": "", "is_gold": True}

            try:
                # Build context docs: always have gold doc
                context_docs = [gold_doc]

                # If augment_docs is provided (for random or adversarial), add them
                if augment_docs and len(augment_docs) > 0:
                    num_augment = min(len(augment_docs), self.config.num_documents_in_context - 1)
                    selected_augment = random.sample(augment_docs, num_augment)
                    context_docs.extend(selected_augment)

                prompt = _format_prompt(question, context_docs, gold_answer)
                generated_responses = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)
                if not generated_responses:
                    raise ValueError("No response generated by LLM")
                    
                generated_answer = generated_responses[0]
                evaluation = self.llm_evaluator.evaluate_answer(
                    question=question,
                    generated_answer=generated_answer,
                    gold_answer=gold_answer
                )

                result = {
                    "query": question,
                    "example_id": example["id"],
                    "generated_answer": generated_answer,
                    "gold_answer": gold_answer,
                    "llm_evaluation": evaluation,
                    "context": [doc.get("text", "") for doc in context_docs],
                    "experiment_type": 'random' if mode == 'gold_random' else ('adore' if mode == 'gold_adversarial' else 'baseline')
                }
                results.append(result)
                
                # Save checkpoint every N examples
                if len(results) % self.config.save_every == 0:
                    self._save_checkpoint(results, checkpoint_dir, len(results) // self.config.save_every)

            except Exception as e:
                self.logger.experiment_logger.error(f"Error evaluating example {example['id']}: {str(e)}")
                continue

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_results_path = mode_dir / f"results_{mode}_{timestamp}.json"
        with open(final_results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _save_checkpoint(self, results: List[Dict], checkpoint_dir: Path, batch_idx: int):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"checkpoint_batch_{batch_idx}_{timestamp}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--train_path", type=str, default="data/10k_train_dataset.json", help="Path to train dataset.")
    parser.add_argument("--test_path", type=str, default="data/test_dataset.json", help="Path to test dataset.")
    parser.add_argument("--results_dir", type=str, default="experiments/experiment0_baseline/results", help="Results directory.")
    parser.add_argument("--base_corpus_size", type=int, default=1000, help="Number of documents in base corpus.")
    parser.add_argument("--num_random_docs", type=int, default=1000, help="Number of random documents.")
    parser.add_argument("--num_adversarial_docs", type=int, default=1000, help="Number of adversarial documents.")
    parser.add_argument("--max_new_tokens", type=int, default=15, help="Maximum tokens for generation.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed for random sampling.")
    parser.add_argument("--adversarial_seed", type=int, default=42, help="Seed for adversarial sampling.")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N examples.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    seed_everything(42)
    config = BaselineConfig(**vars(args))
    corpus_manager = CorpusManager(str(config.corpus_path))
    llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))

    experiment = BaselineExperiment(config, corpus_manager, llm_evaluator)
    experiment.run()

if __name__ == "__main__":
    main()