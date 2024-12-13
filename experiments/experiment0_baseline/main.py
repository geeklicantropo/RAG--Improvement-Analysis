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
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.train_data = self._load_json(config.train_dataset_path)
        self.test_data = self._load_json(config.test_dataset_path)
        self.base_corpus = self.corpus_manager.get_random_subset(num_docs=config.base_corpus_size)

        # Random and adversarial docs
        self.random_corpus = self.corpus_manager.get_random_subset(
            num_docs=config.num_random_docs,
            seed=config.random_seed
        )
        self.adversarial_corpus = self.corpus_manager.get_random_subset(
            num_docs=config.num_adversarial_docs,
            seed=config.adversarial_seed
        )

    def _load_json(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.logger.info(f"Loaded {len(data)} examples from {path}.")
        return data

    def _load_checkpoint(self, checkpoint_dir: Path) -> List[Dict]:
        if checkpoint_dir.exists():
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.json"))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]
                self.logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                with open(latest_checkpoint, "r") as f:
                    return json.load(f)
        return []

    def _save_checkpoint(self, results: List[Dict], checkpoint_dir: Path, batch_idx: int):
        """Save checkpoint with proper error handling and validation."""
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #checkpoint_file = checkpoint_dir / f"checkpoint_batch_{batch_idx}_{timestamp}.json"
            checkpoint_file = checkpoint_dir / f"checkpoint_batch_{batch_idx}_{timestamp}.json"


            
            # Validate results before saving
            if not results:
                self.logger.experiment_logger.error("Attempting to save empty results")
                return
                
            # Clean results for serialization
            clean_results = []
            for result in results:
                try:
                    clean_result = {
                        "query": result.get("query", ""),
                        "gold_answer": result.get("gold_answer", ""),
                        "generated_answer": result.get("generated_answer", ""),
                        "evaluation": result.get("evaluation", {}),
                        "context": result.get("context", []),
                        "example_id": result.get("example_id", "")
                    }
                    clean_results.append(clean_result)
                except Exception as e:
                    self.logger.experiment_logger.error(f"Error cleaning result: {str(e)}")
                    continue

            with open(checkpoint_file, 'w') as f:
                json.dump(clean_results, f, indent=2)
                
            self.logger.info(f"Saved checkpoint to {checkpoint_file}")

        except Exception as e:
            self.logger.experiment_logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def run(self) -> Dict[str, Any]:
        modes = ["gold_only", "gold_random", "gold_adversarial"]
        results = {}
        
        for mode in modes:
            self.logger.info(f"Running {mode} experiment")
            augment_docs = None
            if mode == "gold_random":
                augment_docs = self.random_corpus[:1000]
            elif mode == "gold_adversarial":
                augment_docs = self.adversarial_corpus[:1000]
                
            results[mode] = self._evaluate(mode=mode, augment_docs=augment_docs)
            
            # Save mode results
            #mode_dir = self.output_dir / "naive_rag" / mode
            mode_dir = self.output_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            results_file = mode_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(results_file, 'w') as f:
                json.dump(results[mode], f, indent=2)
        
        return results

    def _evaluate(self, mode: str, augment_docs: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Evaluate the model on the test_data for the given mode.
        mode: one of "gold_only", "gold_random", or "gold_adversarial".
        augment_docs: If provided, these documents will be added to the gold doc context.
        """
        mode_dir = self.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = mode_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoint if available
        results = self._load_checkpoint(checkpoint_dir)
        processed_ids = {res["example_id"] for res in results}

        # Determine remaining examples
        remaining_examples = [example for example in self.test_data if example["example_id"] not in processed_ids]

        batch_idx = len(results) // self.config.save_every

        for idx, example in enumerate(tqdm(remaining_examples, desc=f"Evaluating {mode}")):
            question = example["question"]
            gold_answer = example["answers"][0]
            gold_doc = {"text": example["text"], "title": "", "is_gold": True}

            # Build context docs: always have gold doc
            context_docs = [gold_doc]

            # If augment_docs is provided (for random or adversarial), add them
            if augment_docs and len(augment_docs) > 0:
                num_augment = min(len(augment_docs), self.config.num_documents_in_context - 1)
                selected_augment = random.sample(augment_docs, num_augment)
                context_docs.extend(selected_augment)

            try:
                prompt = _format_prompt(question, context_docs, gold_answer)
                generated_answers = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)
                if not generated_answers:
                    raise ValueError("LLM returned empty response.")
                generated_answer = generated_answers[0]

                evaluation = self.llm_evaluator.evaluate_answer(
                    question=question,
                    generated_answer=generated_answer,
                    gold_answer=gold_answer
                )

                results.append({
                    "query": question,
                    "gold_answer": gold_answer,
                    "generated_answer": generated_answer,
                    "llm_evaluation": evaluation,
                    "context": [d.get("text", "") for d in context_docs],
                    "example_id": example["example_id"]
                })

                # Save checkpoint periodically
                if (len(results) % self.config.save_every) == 0:
                    batch_idx += 1
                    self._save_checkpoint(results, checkpoint_dir, batch_idx)

            except Exception as e:
                self.logger.experiment_logger.error(
                    f"Error evaluating example {example.get('example_id', 'unknown')}: {str(e)}"
                )
                raise e

        # After processing all, save final checkpoint
        self._save_checkpoint(results, checkpoint_dir, "final")

        # Also save final results file with mode and timestamp
        final_results_file = mode_dir / f"results_{mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Final results saved to {final_results_file}")

        return results

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
    config = vars(args)

    corpus_manager = CorpusManager(base_corpus_path="data/processed/corpus_with_contriever_at150.json")
    llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))

    experiment = BaselineExperiment(config, corpus_manager, llm_evaluator)
    experiment.run()


if __name__ == "__main__":
    main()
