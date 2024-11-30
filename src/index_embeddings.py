import os
import argparse
import warnings
import numpy as np
import faiss
import torch
from pathlib import Path
from typing import List, Optional
from experiment_logger import ExperimentLogger
from utils import *
import gc

warnings.filterwarnings('ignore')
SEED = 10

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for indexing embeddings.")
    parser.add_argument('--corpus_size', type=int, required=True)
    parser.add_argument('--vector_sz', type=int, default=768)
    parser.add_argument('--idx_type', type=str, default='IP')
    parser.add_argument('--faiss_dir', type=str, required=True)
    parser.add_argument('--embeddings_dir', type=str, required=True)
    parser.add_argument('--prefix_name', type=str, default='contriever')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--max_memory_usage', type=float, default=0.8)
    return parser.parse_args()

def load_embeddings_batch(
    start_idx: int,
    batch_size: int,
    embeddings_dir: str,
    prefix_name: str,
    logger: ExperimentLogger
) -> np.ndarray:
    try:
        batch_path = f'{embeddings_dir}/{prefix_name}_{start_idx + batch_size}_embeddings.npy'
        embeddings = np.load(batch_path, mmap_mode='r')
        return embeddings
    except Exception as e:
        logger.log_error(e, f"Error loading embeddings batch at {start_idx}")
        raise

def create_index(
    args: argparse.Namespace,
    logger: ExperimentLogger
) -> faiss.Index:
    try:
        if args.idx_type == 'IP':
            index = faiss.IndexFlatIP(args.vector_sz)
        elif args.idx_type == 'L2':
            index = faiss.IndexFlatL2(args.vector_sz)
        else:
            raise ValueError("Only IP and L2 indices supported")
        
        logger.log_metric("index_type", args.idx_type)
        return index
    except Exception as e:
        logger.log_error(e, "Error creating index")
        raise

def index_embeddings_with_memory_tracking(
    args: argparse.Namespace,
    index: faiss.Index,
    logger: ExperimentLogger
) -> None:
    try:
        batch_size = args.batch_size
        total_batches = (args.corpus_size + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            # Check memory usage and adjust batch size
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if current_memory > args.max_memory_usage:
                    batch_size = max(1, batch_size // 2)
                    logger.log_metric("batch_size_adjusted", batch_size)

            start_idx = batch_idx * batch_size
            embeddings_batch = load_embeddings_batch(
                start_idx, batch_size, args.embeddings_dir, args.prefix_name, logger
            )
            
            # Add embeddings to index
            index.add(embeddings_batch.astype('float32'))

            # Memory cleanup
            del embeddings_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        logger.log_error(e, "Error during indexing")
        raise

def save_index(
    index: faiss.Index,
    args: argparse.Namespace,
    logger: ExperimentLogger
) -> None:
    try:
        os.makedirs(args.faiss_dir, exist_ok=True)
        index_path = f'{args.faiss_dir}/{args.idx_type}_index.faiss'
        faiss.write_index(index, index_path)
        logger.log_metric("index_saved", index_path)
    except Exception as e:
        logger.log_error(e, "Error saving index")
        raise

def main():
    args = parse_arguments()
    logger = ExperimentLogger("indexing_embeddings", "logs")

    try:
        with logger:
            logger.log_experiment_params(vars(args))
            logger.log_system_info()

            # Create index
            index = create_index(args, logger)

            # Index embeddings with memory tracking
            index_embeddings_with_memory_tracking(args, index, logger)

            # Save index
            save_index(index, args, logger)

    except Exception as e:
        logger.log_error(e, "Main execution failed")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()