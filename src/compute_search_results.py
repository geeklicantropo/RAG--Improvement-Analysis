import os
import argparse
import warnings
import torch
from transformers import AutoTokenizer, AutoConfig
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import logging
import gc
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)

from retriever import Retriever, Encoder
from src.utils.file_utils import seed_everything, read_pickle, write_pickle
from experiment_logger import ExperimentLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED = 10

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, required=True)
    parser.add_argument('--encoder_id', type=str, default='facebook/contriever')
    parser.add_argument('--use_hybrid', action='store_true', help='Use hybrid retrieval (Contriever + BM25)')
    parser.add_argument('--inject_noise', action='store_true', help='Inject random/distractor documents')
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    parser.add_argument('--max_length_encoder', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='data/search_results/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_memory_usage', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=100)
    return parser.parse_args()

class BM25Retriever:
    def __init__(self, corpus: List[Dict], logger: ExperimentLogger):
        self.logger = logger
        try:
            self.logger.log_step_start("Initializing BM25")
            corpus_texts = [doc.get('text', '') for doc in corpus]
            tokenized_corpus = [word_tokenize(text.lower()) for text in corpus_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.logger.log_step_end("BM25 initialization")
        except Exception as e:
            self.logger.log_error(e, "BM25 initialization failed")
            raise

    def search(self, query: str, top_k: int = 100) -> Tuple[List[int], List[float]]:
        try:
            tokenized_query = word_tokenize(query.lower())
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[-top_k:][::-1]
            top_scores = doc_scores[top_indices]
            return list(top_indices), list(top_scores)
        except Exception as e:
            self.logger.log_error(e, f"BM25 search failed for query: {query}")
            raise

def compute_search_results(
    retriever: Retriever,
    bm25_retriever: Optional[BM25Retriever],
    queries: List[str],
    corpus: List[Dict],
    args: argparse.Namespace,
    logger: ExperimentLogger
) -> List[Tuple[List[int], List[float]]]:
    try:
        logger.log_step_start("Computing search results")
        batch_size = args.batch_size
        results = []

        for batch_start in tqdm(range(0, len(queries), batch_size), desc="Processing queries"):
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if current_memory > args.max_memory_usage:
                    batch_size = max(1, batch_size // 2)
                    logger.log_metric("batch_size_adjusted", batch_size)

            batch_end = min(batch_start + batch_size, len(queries))
            batch_queries = queries[batch_start:batch_end]

            # Get dense retrieval results
            dense_results = retriever.search(batch_queries, corpus)

            if args.use_hybrid and bm25_retriever:
                # Get BM25 results for each query in batch
                sparse_results = []
                for query in batch_queries:
                    doc_ids, scores = bm25_retriever.search(query, args.top_k)
                    sparse_results.append((doc_ids, scores))
                # Combine results
                batch_results = combine_results(dense_results, sparse_results)
            else:
                batch_results = dense_results

            if args.inject_noise:
                batch_results = inject_noise(batch_results, corpus, args.noise_ratio)

            results.extend(batch_results)

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        logger.log_step_end("Search results computation")
        return results

    except Exception as e:
        logger.log_error(e, "Error computing search results")
        raise

def combine_results(
    dense_results: List[Tuple[List[int], List[float]]],
    sparse_results: List[Tuple[List[int], List[float]]]
) -> List[Tuple[List[int], List[float]]]:
    combined_results = []
    k1 = 60.0  # RRF constant

    for dense, sparse in zip(dense_results, sparse_results):
        # Compute reciprocal rank fusion scores
        scores = {}
        
        # Add dense retrieval scores
        for rank, (doc_id, score) in enumerate(zip(dense[0], dense[1])):
            scores[doc_id] = 1 / (rank + k1)
            
        # Add BM25 scores
        for rank, (doc_id, score) in enumerate(zip(sparse[0], sparse[1])):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k1)
        
        # Sort by combined scores
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        doc_ids, doc_scores = zip(*sorted_docs)
        
        # Normalize scores
        max_score = max(doc_scores)
        normalized_scores = [score / max_score for score in doc_scores]
        
        combined_results.append((list(doc_ids), normalized_scores))
    
    return combined_results

def inject_noise(
    results: List[Tuple[List[int], List[float]]],
    corpus: List[Dict],
    noise_ratio: float
) -> List[Tuple[List[int], List[float]]]:
    noisy_results = []
    for doc_ids, scores in results:
        num_docs = len(doc_ids)
        num_noise = int(num_docs * noise_ratio)
        
        # Select random distractor documents
        available_ids = set(range(len(corpus))) - set(doc_ids)
        noise_ids = list(np.random.choice(list(available_ids), num_noise, replace=False))
        
        # Combine original and noise documents
        new_doc_ids = doc_ids[:-num_noise] + noise_ids
        new_scores = scores[:-num_noise] + [0.0] * num_noise
        
        noisy_results.append((new_doc_ids, new_scores))
    return noisy_results

def initialize_retriever(args: argparse.Namespace, logger: ExperimentLogger) -> Retriever:
    try:
        logger.log_step_start("Initializing retriever")
        
        retriever = Retriever(
            device=device,
            api_key=os.getenv("GEMINI_TOKEN"),
            batch_size=args.batch_size,
            max_length=args.max_length_encoder
        )
        
        logger.log_step_end("Retriever initialization")
        return retriever
        
    except Exception as e:
        logger.log_error(e, "Retriever initialization failed")
        raise

def main():
    args = parse_arguments()
    logger = ExperimentLogger("compute_search_results", "logs")

    try:
        with logger:
            logger.log_experiment_params(vars(args))
            logger.log_system_info()

            # Initialize retrievers
            retriever = initialize_retriever(args, logger)
            
            # Load queries and corpus
            logger.log_step_start("Loading data")
            queries = read_pickle(args.queries_path)
            corpus = read_pickle(args.corpus_path)
            logger.log_step_end("Data loading")

            # Initialize BM25 if using hybrid retrieval
            bm25_retriever = None
            if args.use_hybrid:
                bm25_retriever = BM25Retriever(corpus, logger)

            # Compute search results
            results = compute_search_results(
                retriever, 
                bm25_retriever,
                queries, 
                corpus, 
                args, 
                logger
            )

            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, 'search_results.pkl')
            write_pickle(results, output_path)
            logger.log_metric("results_saved", output_path)

    except Exception as e:
        logger.log_error(e, "Main execution failed")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    seed_everything(SEED)
    main()