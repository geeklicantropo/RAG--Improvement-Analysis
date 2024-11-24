import os 
import pickle
import argparse
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig

from index import Indexer, merge_ip_search_results
from retriever import Encoder, Retriever
from utils import *
from experiment_logger import ExperimentLogger
from rag_fusion_utils import RAGFusionRanker
from pyserini.search.lucene import LuceneSearcher

import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED = 10

info = {
    "train": {
        "data_path": 'data/10k_train_dataset.json',
    },
    "test": {
        "data_path": 'data/test_dataset.json',
    }
}

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search and Index Utility with RAG-Fusion support.")
    
    # Index configuration
    parser.add_argument('--faiss_dir', type=str, help='Directory containing FAISS index data')
    parser.add_argument('--use_index_on_gpu', type=str2bool, help='Flag to use index on GPU')
    parser.add_argument('--gpu_ids', nargs='+', type=int, help='GPU IDs for indexing')
    parser.add_argument('--vector_sz', type=int, default=768, help='Size of the vectors to be indexed')
    parser.add_argument('--idx_type', type=str, default='IP', help='Index type (IP for Inner Product)')
    
    # Retriever configuration
    parser.add_argument('--encoder_id', type=str, default='facebook/contriever', help='Model identifier for the encoder')
    parser.add_argument('--max_length_encoder', type=int, default=512, help='Maximum sequence length for the encoder')
    parser.add_argument('--normalize_embeddings', type=str2bool, default=False, help='Whether to normalize embeddings')
    parser.add_argument('--lower_case', type=str2bool, default=False, help='Whether to lower case the corpus text')
    parser.add_argument('--do_normalize_text', type=str2bool, default=True, help='Whether to normalize the corpus text')
    
    # Search configuration
    parser.add_argument('--top_docs', type=int, default=150, help='Number of documents to retrieve')
    parser.add_argument('--use_test', type=str2bool, default=False, help='Use the test set')
    parser.add_argument('--output_dir', type=str, default='data/faiss', help='Output directory for results')
    parser.add_argument('--prefix_name', type=str, default='contriever', help='Prefix for saved results')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for encoding')
    parser.add_argument('--index_batch_size', type=int, default=64, help='Batch size for FAISS search')
    
    # RAG-Fusion configuration
    parser.add_argument('--use_fusion', type=str2bool, default=False, help='Whether to use RAG-Fusion')
    parser.add_argument('--fusion_strategy', type=str, choices=['rrf', 'linear'], default='rrf', 
                        help='Fusion strategy to use')
    parser.add_argument('--fusion_k', type=float, default=60.0, help='k parameter for RRF fusion')
    parser.add_argument('--normalize_scores', type=str2bool, default=True, 
                        help='Normalize scores before fusion')
    parser.add_argument('--contriever_weight', type=float, default=0.7, 
                        help='Weight for Contriever scores in fusion')
    parser.add_argument('--bm25_weight', type=float, default=0.3, 
                        help='Weight for BM25 scores in fusion')
    
    args = parser.parse_args()
    args.split = "test" if args.use_test else "train"

    if args.use_index_on_gpu and (args.gpu_ids is None or len(args.gpu_ids) == 0):
        parser.error('--gpu_ids must be set when --use_index_on_gpu is used.')

    return args

def initialize_logger(args: argparse.Namespace) -> ExperimentLogger:
    """Initialize the experiment logger."""
    logger = ExperimentLogger(
        experiment_name="search_results_computation",
        base_log_dir="logs"
    )
    logger.log_experiment_params(vars(args))
    return logger

def load_queries(args: argparse.Namespace, logger: ExperimentLogger) -> List[str]:
    """Load queries from dataset with logging."""
    logger.log_step_start("Loading queries")
    df = pd.read_json(info[args.split]['data_path'])
    queries = df['query'].tolist() if 'query' in df.columns else df['question'].tolist()
    logger.log_step_end("Loading queries", time.time())
    logger.log_metric("num_queries", len(queries))
    return queries

def initialize_retrievers(args: argparse.Namespace, logger: ExperimentLogger) -> Dict[str, Retriever]:
    """Initialize multiple retrievers based on configuration."""
    retrievers = {}
    
    # Initialize Contriever
    logger.log_step_start("Initializing Contriever")
    config = AutoConfig.from_pretrained(args.encoder_id)
    encoder = Encoder(config).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_id)
    retrievers['contriever'] = Retriever(
        device=device,
        tokenizer=tokenizer,
        query_encoder=encoder,
        max_length=args.max_length_encoder,
        norm_query_emb=args.normalize_embeddings,
        lower_case=args.lower_case,
        do_normalize_text=args.do_normalize_text
    )
    logger.log_step_end("Initializing Contriever", time.time())
    
    if args.use_fusion:
        # Initialize BM25 (assuming pyserini is available)
        logger.log_step_start("Initializing BM25")
        
        retrievers['bm25'] = LuceneSearcher.from_prebuilt_index('wikipedia-dpr')
        logger.log_step_end("Initializing BM25", time.time())
    
    return retrievers

def process_queries(
    retriever: Retriever,
    queries: List[str],
    batch_size: int,
    logger: ExperimentLogger
) -> np.ndarray:
    """Process queries through the retriever with logging."""
    logger.log_step_start("Processing queries")
    embeddings = retriever.encode_queries(queries, batch_size=batch_size).numpy()
    logger.log_step_end("Processing queries", time.time())
    return embeddings

def perform_search(
    args: argparse.Namespace,
    retrievers: Dict[str, Retriever],
    queries: List[str],
    query_embeddings: Optional[np.ndarray],
    logger: ExperimentLogger
) -> Dict[str, List[Tuple[List[str], List[float]]]]:
    """
    Perform search using multiple retrievers and optionally fuse results.
    
    Args:
        args: Command line arguments
        retrievers: Dictionary of retriever instances
        queries: List of query strings
        query_embeddings: Pre-computed query embeddings for dense retrieval
        logger: Experiment logger instance
        
    Returns:
        Dictionary mapping retriever names to their search results
    """
    search_results = {}
    
    # Dense retrieval with Contriever
    if 'contriever' in retrievers:
        logger.log_step_start("Performing Contriever search")
        contriever_results = []
        
        # Initialize indexes
        indexes = []
        if args.use_index_on_gpu:
            for i, gpu_id in enumerate(args.gpu_ids):
                index = Indexer(args.vector_sz, idx_type=args.idx_type)
                index.deserialize_from(
                    args.faiss_dir,
                    f'{args.idx_type}_index{i+1}.faiss',
                    f'{args.idx_type}_index{i+1}_meta.faiss',
                    gpu_id=gpu_id
                )
                indexes.append(index)
        else:  # CPU
            index = Indexer(args.vector_sz, idx_type=args.idx_type)
            index.deserialize_from(args.faiss_dir)
            indexes.append(index)
            
        # Perform search
        if args.use_index_on_gpu:
            results = merge_ip_search_results(
                indexes[0], indexes[1], query_embeddings,
                top_docs=args.top_docs,
                index_batch_size=args.index_batch_size
            )
        else:
            results = indexes[0].search_knn(
                query_embeddings,
                top_docs=args.top_docs,
                index_batch_size=args.index_batch_size
            )
            
        search_results['contriever'] = results
        logger.log_step_end("Performing Contriever search", time.time())
        
    # BM25 retrieval
    if 'bm25' in retrievers and args.use_fusion:
        logger.log_step_start("Performing BM25 search")
        bm25_results = []
        
        for query in tqdm(queries, desc="BM25 search"):
            hits = retrievers['bm25'].search(query, k=args.top_docs)
            doc_ids = [hit.docid for hit in hits]
            scores = [hit.score for hit in hits]
            bm25_results.append((doc_ids, scores))
            
        search_results['bm25'] = bm25_results
        logger.log_step_end("Performing BM25 search", time.time())
    
    return search_results

def save_search_results(
    args: argparse.Namespace,
    search_results: Dict[str, List[Tuple[List[str], List[float]]]],
    logger: ExperimentLogger
) -> None:
    """Save search results to disk with logging."""
    logger.log_step_start("Saving search results")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save individual retriever results
    for retriever_name, results in search_results.items():
        file_path = os.path.join(
            args.output_dir,
            f'{retriever_name}_{args.idx_type}_{args.split}_search_results_at{args.top_docs}.pkl'
        )
        write_pickle(results, file_path)
        logger.log_metric(f"{retriever_name}_results_saved", file_path)
    
    # If using fusion, save fused results
    if args.use_fusion and len(search_results) > 1:
        # Initialize RAG-Fusion
        fusion_ranker = RAGFusionRanker(
            k=args.fusion_k,
            strategy=args.fusion_strategy,
            normalize_scores=args.normalize_scores,
            score_weights={
                'contriever': args.contriever_weight,
                'bm25': args.bm25_weight
            }
        )
        
        # Perform fusion
        fused_results = fusion_ranker.fuse_search_results(search_results)
        
        # Save fused results
        fusion_file_path = os.path.join(
            args.output_dir,
            f'fusion_{args.fusion_strategy}_{args.split}_search_results_at{args.top_docs}.pkl'
        )
        write_pickle(fused_results, fusion_file_path)
        logger.log_metric("fusion_results_saved", fusion_file_path)
    
    logger.log_step_end("Saving search results", time.time())

def main():
    args = parse_arguments()
    
    # Initialize logger
    logger = initialize_logger(args)
    
    try:
        with logger:
            # Load queries
            queries = load_queries(args, logger)
            
            # Initialize retrievers
            retrievers = initialize_retrievers(args, logger)
            
            # Process queries for dense retrieval
            if 'contriever' in retrievers:
                query_embeddings = process_queries(
                    retrievers['contriever'],
                    queries,
                    args.batch_size,
                    logger
                )
            else:
                query_embeddings = None
            
            # Perform search
            search_results = perform_search(
                args, retrievers, queries, query_embeddings, logger
            )
            
            # Save results
            save_search_results(args, search_results, logger)
            
    except Exception as e:
        logger.log_error(e, "Error in main execution")
        raise

if __name__ == '__main__':
    seed_everything(SEED)
    main()