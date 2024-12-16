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
from experiments.checkpoint_utils import save_checkpoint
from .config import BaselineConfig, BaselineConfigFactory
from src.generate_answers_llm import _format_prompt, _contains_answer

from src.generate_answers_llm import _contains_answer
from src.retriever import Retriever
from transformers import AutoTokenizer, AutoConfig
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaselineExperiment:
    def __init__(self, config: BaselineConfig, corpus_manager: CorpusManager, llm_evaluator: LLMEvaluator):
        self.config = config
        self.corpus_manager = corpus_manager
        self.llm_evaluator = llm_evaluator
        self.logger = ExperimentLogger(
            experiment_name="baseline_experiment",
            base_log_dir=str(Path(__file__).parent.parent.parent / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.doc_classifier = DocumentClassifier(self.llm_evaluator)
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        self.output_dir = Path(config.output_dir)
        #self._setup_environment()

        '''
        # Load data with chunking
        self.train_data = self._load_json_data_chunked("data/10k_train_dataset.json", chunk_size=1000)
        self.test_data = self._load_json_data_chunked("data/test_dataset.json", chunk_size=1000)
        self.base_corpus = self._load_json_data_chunked("data/processed/corpus_with_contriever_at150.json", chunk_size=1000)
        self.random_corpus = self._load_pickle_data("data/processed/corpus_with_random_50_words.pkl")
        self.reddit_corpus = self._load_pickle_data("data/processed/reddit_corpus.pkl")
        self.distractor_results = self._load_pickle_data("data/contriever_search_results_at150.pkl")
        '''
        # Load data
        self.train_data = self._load_json_data("data/10k_train_dataset.json")
        self.test_data = self._load_json_data("data/test_dataset.json")
        self.base_corpus = self._load_json_data("data/processed/corpus_with_contriever_at150.json")
        self.random_corpus = self._load_pickle_data("data/processed/corpus_with_random_50_words.pkl")
        self.reddit_corpus = self._load_pickle_data("data/processed/reddit_corpus.pkl")
        self.distractor_results = self._load_pickle_data("data/contriever_search_results_at150.pkl")

        '''
        # Initialize retriever
        tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_id)
        self.retriever = Retriever(
            device=self.device,
            tokenizer=tokenizer,
            query_encoder=self.llm,
            max_length=self.config.max_length_encoder,
            norm_doc_emb=self.config.normalize_embeddings
        )
        '''
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
        
    def run(self):
        self.logger.log_step_start("Running Baseline Experiment")
        final_results = {
            'naive_rag': {
                'gold_only': self._generate_and_evaluate('gold_only'),
                'gold_random': self._generate_and_evaluate('gold_random'),
                'gold_distractor': self._generate_and_evaluate('gold_distractor')
            }
        }

        # Noise tests
        noise_results = self._run_noise_tests([0.1, 0.2, 0.3])
        final_results['noise_tests'] = noise_results

        # Combination tests
        combo_results = self._run_combination_tests()
        final_results['combination_tests'] = combo_results

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self.output_dir / "naive_rag" / f"final_results_{timestamp}.json"
        (self.output_dir / "naive_rag").mkdir(parents=True, exist_ok=True)
        
        with open(final_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        self.logger.log_step_end("Baseline Experiment")
        return final_results

    def run_generation_phase(self):
        self.logger.log_step_start("Generation Phase")
        try:
            
            test_queries = []
            gold_answers = {}
            results = []
            
            for item in self.test_data:
                test_queries.append(item['question'])
                gold_answers[item['question']] = item['answers'][0]

            for query in test_queries:
                sample_docs = random.sample(self.base_corpus, self.config.num_documents_in_context)
                classified_docs = self.doc_classifier.classify_documents(
                    documents=sample_docs,
                    query=query,
                    gold_answer=gold_answers[query]
                )
                
                # Flatten categories with gold tracking
                flattened_docs = []
                for cat in ['gold', 'distracting', 'random']:
                    for doc in classified_docs[cat]:
                        flattened_docs.append(doc)
                        if _contains_answer(doc['text'], gold_answers[query]):
                            self.logger.info(f"Found gold document {doc.get('id', '')} for query: {query}")

                prompt = _format_prompt(query, flattened_docs, gold_answers[query])
                if self._validate_prompt(prompt):
                    response = self.llm.generate(prompt)
                    results.append({
                        'query': query,
                        'docs': classified_docs,
                        'response': response,
                        'gold_answer': gold_answers[query]
                    })
                
                clear_memory()
            return results
                
        except Exception as e:
            self.logger.log_error(str(e), "Error in generation phase")
            raise
        finally:
            self.logger.log_step_end("Generation Phase")

    def _generate_and_evaluate(self, mode: str) -> List[Dict]:
        mode_dir = self.output_dir / "naive_rag" / mode
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

                    # Find gold document for this query
                    gold_doc = None
                    for doc in gold_docs:
                        if doc.get('id') == example.get('id') or self._contains_answer(doc['text'], gold_answer):
                            gold_doc = doc
                            break

                    if not gold_doc:
                        self.logger.experiment_logger.warning(f"No gold document found for question: {question}")

                        pbar.update(1)
                        continue

                    # Prepare documents based on mode
                    if mode == 'gold_only':
                        docs = [gold_doc]
                    elif mode == 'gold_random':
                        random_docs = self._select_random_docs(
                            self.random_corpus,
                            self.config.num_documents_in_context - 1
                        )
                        docs = [gold_doc] + random_docs
                    else:  # gold_distractor
                        distractor_indices = self.distractor_results[example['id']][0]
                        distractor_docs = [
                            self.base_corpus[idx] 
                            for idx in distractor_indices[:self.config.num_documents_in_context-1]
                        ]
                        docs = [gold_doc] + distractor_docs

                    # Generate prompt and get response
                    prompt = _format_prompt(question, docs, gold_answer)
                    if self._validate_prompt(prompt):
                        generated_answer = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)[0]
                        
                        # Evaluate response
                        eval_result = self.llm_evaluator.evaluate_answer(
                            question=question,
                            generated_answer=generated_answer,
                            gold_answer=gold_answer,
                            context=self._build_context(
                                [d.get('id') for d in docs],
                                [d.get('category', 'gold' if d == gold_doc else 'other') for d in docs],
                                list(range(len(docs))),
                                question,
                                gold_answer
                            )
                        )

                        results.append({
                            'query': question,
                            'generated_answer': generated_answer,
                            'llm_evaluation': eval_result,
                            'document_indices': [d.get('id') for d in docs],
                            'document_categories': [
                                'gold' if d == gold_doc else d.get('category', 'other') 
                                for d in docs
                            ],
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

    def _build_context(self, indices, categories, positions, question, gold_answer, doc_set):
        
        id_to_doc = {d['id']: d for d in doc_set}
        docs = [id_to_doc[idx] for idx in indices if idx in id_to_doc]
        context_parts = []
        for i, doc in enumerate(docs):
            prefix = "[GOLD] " if _contains_answer(doc['text'], gold_answer) else ""
            context_parts.append(f"Document [{i+1}] {prefix}:\n{doc['text']}")
        return "\n\n".join(context_parts)

    def validate_prompt(self, prompt: str) -> bool:
        try:
            # Use the LLM's generate method to check if the prompt is valid
            _ = self.llm.generate(prompt, max_new_tokens=1)
            return True
        except Exception as e:
            self.logger.experiment_logger.warning(f"Invalid prompt: {prompt}")
            self.logger.experiment_logger.warning(f"Error: {str(e)}")
            return False

    def _select_random_docs(self, corpus: List[Dict], count: int) -> List[Dict]:
        # selects 'count' random docs from given corpus and classifies them
        # after classification, returns them
        selected = random.sample(corpus, min(count, len(corpus)))
        # classify to assign category and position
        # we can treat them as doc set for classification
        cat = self.doc_classifier.classify_documents(selected, "dummy?", "dummy?")
        # flatten categories
        all_docs = cat['gold'] + cat['distracting'] + cat['random']
        return all_docs

    def _build_context(self, indices, categories, positions, question, gold_answer, doc_set):
        # rebuild doc texts from doc_set by matching ids
        # doc_set are the docs we used to form the prompt
        # we know doc_set might not be indexed the same as original corpus, so we rely on 'id' field
        id_to_doc = {d['id']: d for d in doc_set}
        docs = [id_to_doc[idx] for idx in indices if idx in id_to_doc]
        context = "\n\n".join([f"Document [{i+1}]:\n{doc['text']}" for i, doc in enumerate(docs)])
        return context
    
    def _validate_prompt(self, prompt: str) -> bool:
        """Validate prompt using LLM."""
        try:
            validation_prompt = f"""Evaluate if this prompt is well-formed and appropriate:
            {prompt}
            
            Valid prompts should:
            1. Have clear question
            2. Include relevant context
            3. Be under reasonable length
            
            Answer only 'yes' or 'no':"""
            
            response = self.llm.generate(validation_prompt)[0]  # Get first element since generate returns list
            return 'yes' in response.lower()
        except Exception as e:
            self.logger.experiment_logger.error(f"Prompt validation failed: {str(e)}") 
            return False

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
                categories = self.doc_classifier.classify_documents(top_docs, question, gold_answer)
                gold_docs = categories['gold']

                # inject noise docs
                noise_count = int(nl * self.config.num_documents_in_context)
                # pick noise_count docs from reddit_corpus or random_corpus:
                noise_docs = self._select_random_docs(self.reddit_corpus, noise_count)
                # combine gold and noise
                final_docs = gold_docs + noise_docs

                prompt = _format_prompt(question, final_docs)
                if self._validate_prompt(prompt):
                    generated_answer = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)[0]
                    eval_res = self.llm_evaluator.evaluate_answer(question, generated_answer, gold_answer,
                        context=self._build_context([d['id'] for d in final_docs],
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
            cat_gold = self.doc_classifier.classify_documents(top_docs, question, gold_answer)
            gold_docs = cat_gold['gold']

            random_docs = self._select_random_docs(self.random_corpus, self.config.num_documents_in_context // 2)
            distractor_indices, _ = self.distractor_results[example['id']]
            distractor_docs = [self.base_corpus[idx] for idx in distractor_indices[:self.config.num_documents_in_context // 2]]
            cat_dist = self.doc_classifier.classify_documents(distractor_docs, question, gold_answer)
            distractor_final = [d for d in cat_dist['distracting']+cat_dist['random'] if d['category'] != 'gold']

            final_docs = gold_docs + random_docs + distractor_final
            prompt = _format_prompt(question, final_docs)
            if self._validate_prompt(prompt):
                generated_answer = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)[0]
                eval_res = self.llm_evaluator.evaluate_answer(
                    question=question,
                    generated_answer=generated_answer,
                    gold_answer=gold_answer,
                    context=self._build_context([d['id'] for d in final_docs],
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
    try:
        config = BaselineConfig()
        corpus_manager = CorpusManager(str(config.corpus_path))
        llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
        experiment = BaselineExperiment(config, corpus_manager, llm_evaluator)
        return experiment.run()
    except Exception as e:
        logging.error(f"Error in baseline experiment: {str(e)}")
        return None

if __name__ == "__main__":
    main()