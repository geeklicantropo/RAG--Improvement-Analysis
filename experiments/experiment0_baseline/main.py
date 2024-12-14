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
import pickle

from src.utils.corpus_manager import CorpusManager
from src.utils.file_utils import seed_everything, clear_memory, write_pickle
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.llm_evaluator import LLMEvaluator
from src.document_classifier import DocumentClassifier
from src.prompt_dataset import PromptDataset
from experiments.checkpoint_utils import save_checkpoint, load_checkpoints, get_last_checkpoint_batch
from .config import BaselineConfig, BaselineConfigFactory

class BaselineExperiment:
    def __init__(self, config: BaselineConfig, corpus_manager: CorpusManager, llm_evaluator: LLMEvaluator):
        self.config = config
        self.corpus_manager = corpus_manager
        self.logger = ExperimentLogger("baseline", str(Path(__file__).parent.parent.parent / "logs"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_evaluator = llm_evaluator
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        self.document_classifier = DocumentClassifier(self.llm_evaluator)
        self.output_dir = Path(config.output_dir)
        self._setup_environment()

        # Load data with proper chunking handling
        self.train_data = self._load_json_data_chunked("data/10k_train_dataset.json", chunk_size=1000)
        self.test_data = self._load_json_data_chunked("data/test_dataset.json", chunk_size=1000)
        self.base_corpus = self._load_json_data_chunked("data/processed/corpus_with_contriever_at150.json", chunk_size=1000)
        self.random_corpus = self._load_pickle_data_chunked("data/processed/corpus_with_random_50_words.pkl", chunk_size=1000)
        self.reddit_corpus = self._load_pickle_data_chunked("data/processed/reddit_corpus.pkl", chunk_size=1000)
        self.distractor_results = self._load_pickle_data_chunked("data/contriever_search_results_at150.pkl", chunk_size=1000)


    def _load_json_data_chunked(self, path: str, chunk_size: int = 1000) -> List[Dict]:
        """Load and parse JSON data file in chunks."""
        try:
            with open(path) as f:
                data = []
                chunk = []
                for line in json.load(f):  
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        data.extend(chunk)
                        chunk = []
                        gc.collect()
                if chunk:
                    data.extend(chunk)
                return data
        except Exception as e:
            self.logger.error(f"Error loading JSON from {path}: {str(e)}")
            raise

    def _load_pickle_data_chunked(self, path: str, chunk_size: int = 1000) -> Any:
        """Load pickle data in chunks."""
        try:
            with open(path, 'rb') as f:
                data = []
                while True:
                    try:
                        chunk = []
                        for _ in range(chunk_size):
                            chunk.append(pickle.load(f))
                            if len(chunk) >= chunk_size:
                                data.extend(chunk)
                                chunk = []
                                gc.collect()
                    except EOFError:
                        if chunk:
                            data.extend(chunk)
                        break
                return data
        except Exception as e:
            self.logger.error(f"Error loading pickle from {path}: {str(e)}")
            raise

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
        # utility function for loading pickle data
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def run_generation_phase(self):
        self.logger.log_step_start("Generation Phase")
        
        # We will create three sets of results: gold_only, gold_random, gold_distractor as per your instructions.
        # gold_only: Use only gold docs from base_corpus.
        # gold_random: Mix gold docs from base_corpus and random docs from random_corpus.
        # gold_distractor: Mix gold docs from base_corpus and distractor docs from contriever results.

        final_results = {
            'naive_rag': {
                'gold_only': self._generate_and_evaluate('gold_only'),
                'gold_random': self._generate_and_evaluate('gold_random'),
                'gold_distractor': self._generate_and_evaluate('gold_distractor')
            }
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self.output_dir / f"final_results_{timestamp}.json"
        with open(final_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return final_results

    def _generate_and_evaluate(self, mode: str) -> List[Dict]:
        mode_dir = self.output_dir / "naive_rag" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        results = []
        batch_size = self.config.batch_size
        test_queries = self.test_data
        all_items = []

        for example in test_queries:
            question = example['question']
            gold_answer = example['answers'][0]

            top_docs = self.base_corpus[:self.config.num_documents_in_context]  
            categories = self.document_classifier.classify_documents(top_docs, question, gold_answer)
            gold_docs = categories['gold']

            if mode == 'gold_random':
                random_docs = self._select_random_docs(self.random_corpus, self.config.num_documents_in_context)
                gold_docs.extend(random_docs)
            
            if mode == 'gold_distractor':
                distractor_indices, _ = self.distractor_results[example['example_id']]
                distractor_docs = [self.base_corpus[idx] for idx in distractor_indices[:self.config.num_documents_in_context]]
                d_categories = self.document_classifier.classify_documents(distractor_docs, question, gold_answer)
                distractors = [d for d in d_categories['random'] + d_categories['distracting'] if d['category'] != 'gold']
                gold_docs.extend(distractors)

            prompt = self._format_prompt(question, gold_docs)
            item = {
                'prompt': prompt,
                'query': question,
                'document_indices': [d.get('example_id', d.get('id')) for d in gold_docs], # Handle both id types
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
                    context=self._build_context(item['document_indices'], 
                                            item['document_categories'], 
                                            item['document_positions'], 
                                            item['query'],
                                            item['gold_answer'],
                                            gold_docs)
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
        # Builds the final prompt
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
            max_length=self.config.model_max_length,
            return_tensors="pt"
        )
        return len(tokens['input_ids'][0]) <= self.config.model_max_length

    def _select_random_docs(self, corpus: List[Dict], count: int) -> List[Dict]:
        # selects 'count' random docs from given corpus and classifies them
        # after classification, returns them
        selected = random.sample(corpus, min(count, len(corpus)))
        # classify to assign category and position
        # we can treat them as doc set for classification
        cat = self.document_classifier.classify_documents(selected, "dummy?", "dummy?")
        # flatten categories
        all_docs = cat['gold'] + cat['distracting'] + cat['random']
        return all_docs

    def _build_context_from_docs(self, indices, categories, positions, question, gold_answer, doc_set):
        # rebuild doc texts from doc_set by matching ids
        # doc_set are the docs we used to form the prompt
        # we know doc_set might not be indexed the same as original corpus, so we rely on 'id' field
        id_to_doc = {d['id']: d for d in doc_set}
        docs = [id_to_doc[idx] for idx in indices if idx in id_to_doc]
        context = "\n\n".join([f"Document [{i+1}]:\n{doc['text']}" for i, doc in enumerate(docs)])
        return context

    def _run_noise_tests(self, noise_levels: List[float]) -> Dict:
        # Implement noise tests by injecting random docs from random_corpus and reddit_corpus
        noise_results = {}
        for nl in noise_levels:
            # For each noise level, we inject 'nl * num_documents_in_context' random docs
            # and measure performance. We'll do a simple approach: 
            # take each query from test_data, mix in random nonsense docs.
            nr_dir = self.output_dir / "naive_rag" / f"noise_injection_{int(nl*100)}"
            nr_dir.mkdir(parents=True, exist_ok=True)
            run_results = []
            for example in tqdm(self.test_data, desc=f"Noise Tests {nl}"):
                question = example['question']
                gold_answer = example['answers'][0]

                # get some gold docs from base corpus
                top_docs = self.base_corpus[:self.config.num_documents_in_context]
                categories = self.document_classifier.classify_documents(top_docs, question, gold_answer)
                gold_docs = categories['gold']

                # inject noise docs
                noise_count = int(nl * self.config.num_documents_in_context)
                # pick noise_count docs from reddit_corpus or random_corpus:
                noise_docs = self._select_random_docs(self.reddit_corpus, noise_count)
                # combine gold and noise
                final_docs = gold_docs + noise_docs

                prompt = self._format_prompt(question, final_docs)
                if self._validate_prompt(prompt):
                    generated_answer = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)[0]
                    eval_res = self.llm_evaluator.evaluate_answer(question, generated_answer, gold_answer,
                        context=self._build_context_from_docs([d['id'] for d in final_docs],
                        [d['category'] for d in final_docs],
                        [d['position'] for d in final_docs], question, gold_answer, final_docs)
                    )
                    run_results.append({
                        'query': question,
                        'generated_answer': generated_answer,
                        'llm_evaluation': eval_res,
                        'noise_level': nl,
                        'docs_used': [d['id'] for d in final_docs]
                    })
            
            save_checkpoint(run_results, 'final', nr_dir)
            noise_results[nl] = run_results

        return noise_results

    def _run_combination_tests(self) -> Dict:
        # Combination tests could mean mixing various doc types (gold + random + distractor + reddit)
        # We'll do a combination test: gold + random + distractor docs together.
        combo_dir = self.output_dir / "naive_rag" / "combination_tests"
        combo_dir.mkdir(parents=True, exist_ok=True)
        combo_results = []

        for example in tqdm(self.test_data, desc="Combination Tests"):
            question = example['question']
            gold_answer = example['answers'][0]

            top_docs = self.base_corpus[:self.config.num_documents_in_context]
            cat_gold = self.document_classifier.classify_documents(top_docs, question, gold_answer)
            gold_docs = cat_gold['gold']

            random_docs = self._select_random_docs(self.random_corpus, self.config.num_documents_in_context // 2)
            distractor_indices, _ = self.distractor_results[example['id']]
            distractor_docs = [self.base_corpus[idx] for idx in distractor_indices[:self.config.num_documents_in_context // 2]]
            cat_dist = self.document_classifier.classify_documents(distractor_docs, question, gold_answer)
            distractor_final = [d for d in cat_dist['distracting']+cat_dist['random'] if d['category'] != 'gold']

            final_docs = gold_docs + random_docs + distractor_final
            prompt = self._format_prompt(question, final_docs)
            if self._validate_prompt(prompt):
                generated_answer = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)[0]
                eval_res = self.llm_evaluator.evaluate_answer(
                    question=question,
                    generated_answer=generated_answer,
                    gold_answer=gold_answer,
                    context=self._build_context_from_docs([d['id'] for d in final_docs],
                        [d['category'] for d in final_docs],
                        [d['position'] for d in final_docs],
                        question, gold_answer, final_docs)
                )
                combo_results.append({
                    'query': question,
                    'generated_answer': generated_answer,
                    'llm_evaluation': eval_res,
                    'docs_used': [d['id'] for d in final_docs],
                    'combination_type': 'gold+random+distractor'
                })

        save_checkpoint(combo_results, 'final', combo_dir)
        return {'gold_random_distractor': combo_results}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument('--retriever', type=str, choices=['contriever', 'bm25', 'random'], 
                       default='contriever', help="Type of retriever to use") # Made optional with default
    parser.add_argument('--output_dir', type=str, 
                       default="experiments/experiment0_baseline/results", help="Output directory")
    return parser.parse_args()

def main():
    seed_everything(10)
    experiment_args = parse_arguments()
    config = BaselineConfigFactory.get_config_for_retriever(experiment_args.retriever)
    corpus_manager = CorpusManager("data/processed/corpus_with_contriever_at150.json")
    llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
    experiment = BaselineExperiment(config, corpus_manager, llm_evaluator)
    experiment.run_generation_phase()

if __name__ == "__main__":
    main()
