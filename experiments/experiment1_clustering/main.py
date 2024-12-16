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
import numpy as np
import random
import argparse


from experiments.checkpoint_utils import save_checkpoint
from src.utils.file_utils import seed_everything, clear_memory
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.document_classifier import DocumentClassifier
from src.llm_evaluator import LLMEvaluator
from src.utils.corpus_manager import CorpusManager
from src.generate_answers_llm import _format_prompt, _contains_answer
from .config import ClusteringConfig, ClusteringConfigFactory


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
        self.doc_classifier = DocumentClassifier(self.llm_evaluator)
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        self.output_dir = Path(config.output_dir)

        # Load data
        self.train_data = self._load_json_data("data/10k_train_dataset.json")
        self.test_data = self._load_json_data("data/test_dataset.json")
        self.base_corpus = self._load_json_data("data/processed/corpus_with_contriever_at150.json")
        self.cluster_model = self._initialize_clusters()

        self.random_corpus = self._load_pickle_data("data/processed/corpus_with_random_50_words.pkl")
        self.distractor_results = self._load_pickle_data("data/contriever_search_results_at150.pkl")
        self.reddit_corpus = self._load_pickle_data("data/processed/reddit_corpus.pkl")



    def _load_pickle_data(self, path: str) -> Any:
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_json_data(self, path: str):
        with open(path) as f:
            return json.load(f)

    def _initialize_clusters(self):
        try:
            embeddings = np.array([doc['embedding'] for doc in self.base_corpus])
        except KeyError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=768)
            texts = [doc['text'] for doc in self.base_corpus]
            embeddings = vectorizer.fit_transform(texts).toarray()

        from src.cluster_utils import fit_clusters
        return fit_clusters(embeddings, self.config.num_clusters)

    def run(self):
        self.logger.log_step_start("Clustering Experiment")
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

    def _select_random_docs(self, corpus: List[Dict], count: int) -> List[Dict]:
        selected = random.sample(corpus, min(count, len(corpus)))
        cat = self.doc_classifier.classify_documents(selected, "dummy_q?", "dummy_a?")
        return cat['gold'] + cat['distracting'] + cat['random']

    def _generate_and_evaluate(self, mode: str) -> List[Dict]:
        mode_dir = self.output_dir / "clustered" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        results = []
        batch_size = self.config.batch_size
        test_queries = self.test_data
        gold_docs = self.corpus_manager.get_gold_documents()

        with tqdm(total=len(test_queries), desc=f"Processing {mode}") as pbar:
            for example in test_queries:
                try:
                    question = example['question']
                    gold_answer = example['answers'][0]

                    # Find gold document
                    gold_doc = None
                    for doc in gold_docs:
                        if doc.get('id') == example.get('id') or self._contains_answer(doc['text'], gold_answer):
                            gold_doc = doc
                            break

                    if not gold_doc:
                        self.logger.experiment_logger.warning(f"No gold document found for question: {question}")

                        pbar.update(1)
                        continue

                    from src.cluster_utils import get_top_k_docs_from_cluster
                    top_k = self.config.num_clusters

                    if mode == 'gold_only':
                        top_docs = [gold_doc] + get_top_k_docs_from_cluster(
                            question, gold_answer, self.cluster_model, top_k - 1
                        )
                    elif mode == 'gold_random':
                        random_docs = self._select_random_docs(self.random_corpus, top_k - 1)
                        top_docs = [gold_doc] + random_docs
                    else:  # gold_distractor
                        distractor_indices, _ = self.distractor_results[example['id']]
                        distractor_docs = [self.base_corpus[idx] for idx in distractor_indices[:top_k-1]]
                        d_cat = self.doc_classifier.classify_documents(distractor_docs, question, gold_answer)
                        distractors = [d for d in d_cat['random'] + d_cat['distracting'] if d['category'] != 'gold']
                        top_docs = [gold_doc] + distractors[:top_k-1]

                    prompt = _format_prompt(question, top_docs, gold_answer)
                    if self._validate_prompt(prompt):
                        generated_answer = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)[0]

                        eval_result = self.llm_evaluator.evaluate_answer(
                            question=question,
                            generated_answer=generated_answer,
                            gold_answer=gold_answer,
                            context=self._build_context(
                                [d.get('id') for d in top_docs],
                                [d.get('category', 'gold' if d == gold_doc else 'other') for d in top_docs],
                                list(range(len(top_docs))),
                                question,
                                gold_answer
                            )
                        )

                        results.append({
                            'query': question,
                            'generated_answer': generated_answer,
                            'llm_evaluation': eval_result,
                            'document_indices': [d.get('id') for d in top_docs],
                            'document_categories': ['gold' if d == gold_doc else d.get('category', 'other') for d in top_docs],
                            'example_id': example['id'],
                            'gold_answer': gold_answer,
                            'mode': mode
                        })

                    pbar.update(1)

                    if len(results) % self.config.save_every == 0:
                        save_checkpoint(results, len(results), mode_dir)

                except Exception as e:
                    self.logger.experiment_logger.error(f"Error processing example {example.get('id')}: {str(e)}")
                    pbar.update(1)
                    continue

        save_checkpoint(results, 'final', mode_dir)
        return results

    def _validate_prompt(self, prompt: str) -> bool:
        try:
            _ = self.llm.generate(prompt, max_new_tokens=1)
            return True
        except Exception as e:
            self.logger.experiment_logger.warning(f"Invalid prompt: {prompt}")
            return False

    def _build_context(self, indices, categories, positions, question, gold_answer) -> str:
        docs = []
        for idx, cat, pos in zip(indices, categories, positions):
            doc = self.base_corpus[idx]
            doc_text = f"Document [{pos}]: {doc['text']}"
            if _contains_answer(doc['text'], gold_answer):
                doc_text = f"[GOLD] {doc_text}"
            docs.append(doc_text)
        return "\n\n".join(docs)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run clustering-based retrieval experiments")
    parser.add_argument('--output_dir', type=str, default="experiments/experiment1_clustering/results", help="Output directory")
    return parser.parse_known_args()[0]

def main():
    seed_everything(10)
    try:
        config = ClusteringConfig()
        corpus_manager = CorpusManager(str(config.corpus_path))
        llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
        experiment = ClusteringExperiment(config, corpus_manager, llm_evaluator)
        return experiment.run()
    except Exception as e:
        logging.error(f"Error in clustering experiment: {str(e)}")
        return None

if __name__ == "__main__":  
    seed_everything(10)
    main()