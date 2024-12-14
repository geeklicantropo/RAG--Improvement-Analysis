import os
import sys
import torch
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import logging
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.corpus_manager import CorpusManager
from src.utils.file_utils import seed_everything, clear_memory
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.llm_evaluator import LLMEvaluator
from src.document_classifier import DocumentClassifier
from src.prompt_dataset import PromptDataset
from experiments.checkpoint_utils import save_checkpoint, load_checkpoints, get_last_checkpoint_batch
from .config import ClusteringConfig, ClusteringConfigFactory
from src.cluster_utils import fit_clusters, get_top_k_docs_from_cluster
import numpy as np

class ClusteringExperiment:
    def __init__(self, config: ClusteringConfig, corpus_manager: CorpusManager, llm_evaluator: LLMEvaluator):
        self.config = config
        self.corpus_manager = corpus_manager
        self.logger = ExperimentLogger("clustering_experiment", str(Path(__file__).parent.parent.parent / "logs"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_evaluator = llm_evaluator
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        self.document_classifier = DocumentClassifier(self.llm_evaluator)
        self._setup_environment()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.train_data = self._load_json_data("data/10k_train_dataset.json")
        self.test_data = self._load_json_data("data/test_dataset.json")
        self.base_corpus = self._load_json_data("data/processed/corpus_with_contriever_at150.json")
        self.random_corpus = self._load_pickle_data("data/processed/corpus_with_random_50_words.pkl")
        self.reddit_corpus = self._load_pickle_data("data/processed/reddit_corpus.pkl")
        self.distractor_results = self._load_pickle_data("data/contriever_search_results_at150.pkl")

        # Convert docs to embeddings
        try:
            embeddings = np.array([doc['embedding'] for doc in self.base_corpus])
        except KeyError:
            # If no embeddings, use text features
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=768)
            texts = [doc['text'] for doc in self.base_corpus]
            embeddings = vectorizer.fit_transform(texts).toarray()

        # Fit clusters
        self.cluster_model = fit_clusters(embeddings, config.num_clusters)

    def _setup_environment(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_utilization)  
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def _load_json_data(self, path: str) -> List[Dict]:
        with open(path) as f:
            return json.load(f)

    def _load_pickle_data(self, path: str) -> Any:
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def run(self):
        self.logger.log_step_start("Clustering Experiment")
        # We'll produce results in experiments/experiment1_clustering/results/clustered
        final_results = {
            'clustered': {
                'gold_only': self._generate_and_evaluate('gold_only'),
                'gold_random': self._generate_and_evaluate('gold_random'),
                'gold_distractor': self._generate_and_evaluate('gold_distractor')
            }
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self.output_dir / "clustered" / f"final_results_{timestamp}.json"
        (self.output_dir / "clustered").mkdir(parents=True, exist_ok=True)
        with open(final_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        self.logger.log_step_end("Clustering Experiment")
        return final_results

    def _generate_and_evaluate(self, mode: str) -> List[Dict]:
        # mode: gold_only, gold_random, gold_distractor
        mode_dir = self.output_dir / "clustered" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Similar approach to baseline:
        # For each query in test_data, get top_k docs from cluster,
        # classify them, add random or distractor as per mode, build prompt, evaluate.
        results = []
        batch_size = self.config.batch_size
        test_queries = self.test_data
        all_items = []

        top_k = self.config.num_clusters  # or some top_k defined; we can use config.num_clusters as placeholder

        for example in test_queries:
            question = example['question']
            gold_answer = example['answers'][0]

            # Get top_k docs from clusters using cluster_utils
            top_docs = get_top_k_docs_from_cluster(question, gold_answer, self.cluster_model, top_k)
            categories = self.document_classifier.classify_documents(top_docs, question, gold_answer)
            gold_docs = categories['gold']

            if mode == 'gold_random':
                random_docs = self._select_random_docs(self.random_corpus, top_k)
                gold_docs.extend(random_docs)

            if mode == 'gold_distractor':
                distractor_indices, _ = self.distractor_results[example['example_id']]
                distractor_candidates = [self.base_corpus[idx] for idx in distractor_indices[:top_k]]
                d_cat = self.document_classifier.classify_documents(distractor_candidates, question, gold_answer)
                distractors = [d for d in d_cat['distracting']+d_cat['random'] if d['category'] != 'gold']
                gold_docs.extend(distractors)

            prompt = self._format_prompt(question, gold_docs)
            if self._validate_prompt(prompt):
                item = {
                    'prompt': prompt,
                    'query': question,
                    'document_indices': [d['example_id'] for d in gold_docs],
                    'document_categories': [d['category'] for d in gold_docs],
                    'document_positions': [d['position'] for d in gold_docs],
                    'example_id': example['example_id'],
                    'gold_answer': gold_answer
                }
                all_items.append(item)

        for i in tqdm(range(0, len(all_items), batch_size), desc=f"Evaluating {mode}"):
            batch = all_items[i:i+batch_size]
            for item in batch:
                generated_answer = self.llm.generate(item['prompt'], max_new_tokens=self.config.max_new_tokens)[0]
                eval_result = self.llm_evaluator.evaluate_answer(
                    question=item['query'],
                    generated_answer=generated_answer,
                    gold_answer=item['gold_answer'],
                    context=self._build_context(item['document_indices'], item['document_categories'], item['document_positions'], item['query'], item['gold_answer'], item['document_indices'], gold_docs)
                )
                res = {
                    'query': item['query'],
                    'generated_answer': generated_answer,
                    'llm_evaluation': eval_result,
                    'document_indices': item['document_indices'],
                    'document_categories': item['document_categories'],
                    'document_positions': item['document_positions'],
                    'example_id': item['example_id'],
                    'gold_answer': item['gold_answer'],
                    'mode': mode
                }
                results.append(res)

        save_checkpoint(results, 'final', mode_dir)
        return results

    def _format_prompt(self, question: str, documents: List[Dict]) -> str:
        context_parts = [f"Question: {question}\n\nContext:"]
        for idx, doc in enumerate(documents, 1):
            context_parts.append(
                f"\nDocument [{idx}] (Category: {doc['category']}, Position: {doc['position']}):\n{doc['text']}"
            )
        context_parts.append("\nAnswer:")
        return "\n".join(context_parts)

    def _validate_prompt(self, prompt: str) -> bool:
        tokens = self.llm.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_length_encoder,
            return_tensors="pt"
        )
        return len(tokens['input_ids'][0]) <= self.config.max_length_encoder

    def _select_random_docs(self, corpus: List[Dict], count: int) -> List[Dict]:
        selected = random.sample(corpus, min(count, len(corpus)))
        cat = self.document_classifier.classify_documents(selected, "dummy_q?", "dummy_a?")
        all_docs = cat['gold'] + cat['distracting'] + cat['random']
        return all_docs

    def _build_context(self, indices, categories, positions, question, gold_answer, doc_indices, doc_set):
        # doc_set here is actually the docs we ended up using (gold_docs and random/distractor).
        # in _generate_and_evaluate we have 'gold_docs' final set as doc_set
        # build a mapping id->doc
        # doc_set is the final chosen docs for that query
        id_to_doc = {d['id']: d for d in doc_set}
        docs = [id_to_doc[i] for i in indices if i in id_to_doc]
        context = "\n\n".join([f"Document [{idx+1}]:\n{doc['text']}" for idx, doc in enumerate(docs)])
        return context

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run clustering-based retrieval experiments")
    parser.add_argument('--output_dir', type=str, default="experiments/experiment1_clustering/results", help="Output directory")
    return parser.parse_args()

def main():
    seed_everything(10)
    args = parse_arguments()
    config = ClusteringConfigFactory.get_base_config() 
    config.output_dir = args.output_dir
    corpus_manager = CorpusManager("data/processed/corpus_with_contriever_at150.json")
    llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
    experiment = ClusteringExperiment(config, corpus_manager, llm_evaluator)
    experiment.run()

if __name__ == "__main__":
    main()
