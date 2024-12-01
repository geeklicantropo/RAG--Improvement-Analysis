import os
import argparse
import warnings
import numpy as np
import faiss
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from experiment_logger import ExperimentLogger
from utils import *
import gc
import json
from tqdm import tqdm

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
    parser.add_argument('--subset_type', type=str, choices=['cluster', 'noise', 'full'], default='full')
    parser.add_argument('--subset_id', type=str, help='Identifier for the subset index')
    parser.add_argument('--hybrid_retrieval', type=str2bool, default=False, help='Enable hybrid retrieval (dense + sparse)')
    parser.add_argument('--bm25_index_path', type=str, help='Path to prebuilt BM25 index for hybrid retrieval')
    return parser.parse_args()


class IndexManager:
    def __init__(self, args: argparse.Namespace, logger: ExperimentLogger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_memory_usage = 0.0

    def load_embeddings_batch(self, start_idx: int, batch_size: int) -> np.ndarray:
        try:
            batch_path = f'{self.args.embeddings_dir}/{self.args.prefix_name}_{start_idx + batch_size}_embeddings.npy'
            embeddings = np.load(batch_path, mmap_mode='r')
            return embeddings
        except Exception as e:
            self.logger.log_error(e, f"Error loading embeddings batch at {start_idx}")
            raise

    def create_index(self) -> faiss.Index:
        try:
            if self.args.idx_type == 'IP':
                index = faiss.IndexFlatIP(self.args.vector_sz)
            elif self.args.idx_type == 'L2':
                index = faiss.IndexFlatL2(self.args.vector_sz)
            else:
                raise ValueError("Only IP and L2 indices supported")

            self.logger.log_metric("index_type", self.args.idx_type)
            return index
        except Exception as e:
            self.logger.log_error(e, "Error creating index")
            raise

    def adjust_batch_size(self) -> int:
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if current_memory > self.args.max_memory_usage:
                self.args.batch_size = max(1000, int(self.args.batch_size * 0.8))
                self.logger.log_metric("batch_size_adjusted", self.args.batch_size)
        return self.args.batch_size

    def index_embeddings(self, index: faiss.Index) -> None:
        try:
            total_batches = (self.args.corpus_size + self.args.batch_size - 1) // self.args.batch_size

            for batch_idx in tqdm(range(total_batches), desc="Indexing embeddings"):
                self.args.batch_size = self.adjust_batch_size()
                start_idx = batch_idx * self.args.batch_size

                embeddings_batch = self.load_embeddings_batch(start_idx, self.args.batch_size)

                # Add embeddings to index
                index.add(embeddings_batch.astype('float32'))

                # Memory cleanup
                del embeddings_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    self.logger.log_metric("processed_batches", batch_idx + 1)

        except Exception as e:
            self.logger.log_error(e, "Error during indexing")
            raise

    def get_index_path(self) -> Path:
        subset_dir = self.args.faiss_dir
        if self.args.subset_type != 'full':
            subset_dir = os.path.join(subset_dir, self.args.subset_type)
            if self.args.subset_id:
                subset_dir = os.path.join(subset_dir, self.args.subset_id)

        os.makedirs(subset_dir, exist_ok=True)
        return Path(subset_dir) / f'{self.args.idx_type}_index.faiss'

    def save_index(self, index: faiss.Index) -> None:
        try:
            index_path = self.get_index_path()
            faiss.write_index(index, str(index_path))
            self.logger.log_metric("index_saved", str(index_path))

            # Save metadata
            metadata = {
                'corpus_size': self.args.corpus_size,
                'vector_size': self.args.vector_sz,
                'index_type': self.args.idx_type,
                'subset_type': self.args.subset_type,
                'subset_id': self.args.subset_id
            }

            metadata_path = index_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.logger.log_error(e, "Error saving index")
            raise

    def enable_hybrid_retrieval(self) -> None:
        if self.args.hybrid_retrieval and self.args.bm25_index_path:
            self.logger.experiment_logger.info("Enabling hybrid retrieval (dense + sparse).")
            # Integration logic for hybrid retrieval would go here.

def main():
    args = parse_arguments()
    logger = ExperimentLogger("indexing_embeddings", "logs")

    try:
        with logger:
            logger.log_experiment_params(vars(args))
            logger.log_system_info()

            index_manager = IndexManager(args, logger)

            # Create index
            index = index_manager.create_index()

            # Index embeddings
            index_manager.index_embeddings(index)

            # Save index and metadata
            index_manager.save_index(index)

            # Enable hybrid retrieval if specified
            if args.hybrid_retrieval:
                index_manager.enable_hybrid_retrieval()

    except Exception as e:
        logger.log_error(e, "Main execution failed")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()
