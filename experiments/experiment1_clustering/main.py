import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import logging
import gc
from tqdm import tqdm
from datetime import datetime
import json
import random
import argparse
import numpy as np

from experiments.checkpoint_utils import load_checkpoints, get_last_checkpoint_batch, save_checkpoint
from src.utils.file_utils import seed_everything, clear_memory
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.llm_evaluator import LLMEvaluator
from src.utils.corpus_manager import CorpusManager
from src.generate_answers_llm import _format_prompt, _contains_answer
from .config import ClusteringConfig
from src.document_classifier import DocumentClassifier
from src.cluster_utils import DocumentClusterer


class ClusteringExperiment:
    def __init__(
        self,
        config: ClusteringConfig,
        corpus_manager: CorpusManager,
        llm_evaluator: LLMEvaluator,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.corpus_manager = corpus_manager
        self.llm_evaluator = llm_evaluator
        self.logger = logger or ExperimentLogger(
            experiment_name="clustering_experiment",
            base_log_dir=str(Path(__file__).parent.parent.parent / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        self.output_dir = Path("experiments/experiment1_clustering/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.train_data = self._load_json_data(config.train_dataset_path)
        self.test_data = self._load_json_data(config.test_dataset_path)
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
        self.doc_classifier = DocumentClassifier(self.llm_evaluator)

    def _load_json_data(self, path: str) -> List[Dict]:
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
        checkpoint_file = checkpoint_dir / f"checkpoint_{batch_idx}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved checkpoint to {checkpoint_file}")

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
            mode_dir = self.output_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            results_file = mode_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(results_file, 'w') as f:
                json.dump(results[mode], f, indent=2)
        
        return results
    
    def _get_document_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """Get embeddings for documents using Contriever"""
        texts = [doc["text"] for doc in documents]
        embeddings = []
        for text in texts:
            embedding = self.llm.model.generate_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)

    def _evaluate(self, mode: str, augment_docs: Optional[List[Dict]] = None) -> List[Dict]:
        mode_dir = self.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = mode_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoints
        results = self._load_checkpoint(checkpoint_dir)
        processed_queries = {res["query"] for res in results}

        # Determine remaining examples
        remaining_examples = [example for example in self.test_data if example["question"] not in processed_queries]

        # Select augmentation docs if needed
        # For gold_only mode, no augmentation. For gold_random or gold_adversarial, set augment_docs accordingly.
        if mode == "gold_random":
            augment_docs = self.random_corpus
        elif mode == "gold_adversarial":
            augment_docs = self.adversarial_corpus
        else:
            augment_docs = []  # gold_only mode

        batch_idx = len(results) // self.config.save_every

        # Process each example
        for idx, example in enumerate(tqdm(remaining_examples, desc=f"Evaluating {mode}")):
            question = example["question"]
            gold_answer = example["answers"][0]

            # Build context docs
            gold_doc = {"text": example["text"], "title": "", "is_gold": True}
            context_docs = [gold_doc]

            try:
                # If augment_docs are available, sample them
                if augment_docs:
                    num_augment = min(len(augment_docs), self.config.num_documents_in_context - 1)
                    selected_docs = random.sample(augment_docs, num_augment)
                    all_docs = context_docs + selected_docs

                    # Here we assume no clustering embedding generation errors. If errors occur, log and break.
                    # Generate embeddings (Replace with actual embedding generation if available)
                    doc_embeddings = []
                    for doc in all_docs:
                        # Create a prompt for embeddings (This is pseudo. Adjust as needed for real embedding generation.)
                        prompt = f"Generate a 768-dimensional dense vector for: {doc['text'][:500]}"
                        response = self.llm.generate(prompt, max_new_tokens=768)[0]
                        try:
                            embedding = [float(x) for x in response.strip().split()]
                            # If not exactly 768 dims, handle gracefully by padding or truncating
                            if len(embedding) < 768:
                                embedding.extend([0.0] * (768 - len(embedding)))
                            elif len(embedding) > 768:
                                embedding = embedding[:768]
                            doc_embeddings.append(embedding)
                        except Exception as e:
                            self.logger.experiment_logger.error(f"Error parsing embedding: {str(e)}")
                            break

                    if len(doc_embeddings) != len(all_docs):
                        # If we failed to get embeddings for all docs, skip this example
                        self.logger.experiment_logger.error("Embedding generation failed for one or more documents. Skipping example.")

                        continue

                    doc_embeddings = np.array(doc_embeddings)

                    # Cluster the documents
                    # Adjust parameters if needed
                    clusterer = DocumentClusterer(
                        api_key=os.getenv("GEMINI_TOKEN"),
                        num_clusters=3,
                        random_seed=42
                    )

                    clusters = clusterer.fit_clusters(doc_embeddings)
                    # Rebuild context_docs from clusters
                    context_docs = []
                    for cluster_id in clusters:
                        cluster_docs = [all_docs[doc_id] for doc_id in clusters[cluster_id]]
                        context_docs.extend(cluster_docs)
                else:
                    clusters = None

                # Build prompt and generate answer
                prompt = _format_prompt(question, context_docs, gold_answer)
                generated_responses = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)
                if not generated_responses or len(generated_responses) == 0:
                    self.logger.experiment_logger.error("No response generated by LLM.")
                    # No continue on error, just skip this example
                    continue
                generated_answer = generated_responses[0]

                evaluation = self.llm_evaluator.evaluate_answer(
                    question=question,
                    generated_answer=generated_answer,
                    gold_answer=gold_answer
                )

                results.append({
                    "query": question,
                    "gold_answer": gold_answer,
                    "generated_answer": generated_answer,
                    "evaluation": evaluation,
                    "context": [doc.get("text", "") for doc in context_docs],
                    "clusters": clusters,
                    "example_id": example["id"]
                })

                # Save checkpoints periodically
                if (idx + 1) % self.config.save_every == 0:
                    batch_idx += 1
                    try:
                        self._save_checkpoint(results, checkpoint_dir, batch_idx)
                    except Exception as e:
                        self.logger.experiment_logger.error(f"Error saving checkpoint: {str(e)}")

                        break

            except Exception as e:
                # Log the error and do not 'continue' silently
                self.logger.experiment_logger.error(f"Error evaluating example {example.get('id', 'unknown')}: {str(e)}")

                continue

        # After all examples processed, save final results
        # Use a mode-specific timestamped results file
        final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_results_file = mode_dir / f"results_{mode}_{final_timestamp}.json"
        try:
            with open(final_results_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Final results saved to {final_results_file}")
        except Exception as e:
            self.logger.experiment_logger.error(f"Error saving final results: {str(e)}")
            # If there's an error saving final results, we log it. Since it's the end, no need to continue.

        return results

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run clustering-based retrieval experiments")
    parser.add_argument("--train_path", type=str, default="data/10k_train_dataset.json", help="Path to train dataset.")
    parser.add_argument("--test_path", type=str, default="data/test_dataset.json", help="Path to test dataset.")
    parser.add_argument("--results_dir", type=str, default="experiments/experiment1_clustering/results", help="Results directory.")
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
    config = ClusteringConfig(**vars(args))
    corpus_manager = CorpusManager(config.corpus_path)
    llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))

    experiment = ClusteringExperiment(config, corpus_manager, llm_evaluator)
    experiment.run()


if __name__ == "__main__":
    main()
