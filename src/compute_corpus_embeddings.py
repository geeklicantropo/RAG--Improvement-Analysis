import os
import argparse
import warnings
import torch
from transformers import AutoTokenizer, AutoConfig
from retriever import Retriever
#from utils import read_corpus_json, str2bool, read_json
from src.utils.file_utils import read_corpus_json, str2bool, read_json
from pathlib import Path
import numpy as np

from experiment_logger import ExperimentLogger
import time
import gc
from typing import Optional, List, Dict
from tqdm import tqdm

#os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
#SEED = 10

'''
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for computing corpus embeddings.")
    parser.add_argument('--corpus_path', type=str, required=True)
    parser.add_argument('--encoder_id', type=str, default='facebook/contriever')
    parser.add_argument('--max_length_encoder', type=int, default=512)
    parser.add_argument('--normalize_embeddings', type=str2bool, default=False)
    parser.add_argument('--output_dir', type=str, default='data/corpus/embeddings/')
    parser.add_argument('--prefix_name', type=str, default='contriever')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--max_memory_usage', type=float, default=0.8)
    parser.add_argument('--subset_indices', type=str, default=None, help="Path to subset indices file")
    return parser.parse_args()


def initialize_retriever(args: argparse.Namespace, logger: ExperimentLogger) -> Retriever:
    try:
        logger.log_step_start("Initializing retriever")
        config = AutoConfig.from_pretrained(args.encoder_id)
        encoder = AutoTokenizer.from_pretrained(args.encoder_id)
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_id)

        retriever = Retriever(
            device=device,
            tokenizer=tokenizer,
            query_encoder=encoder,
            max_length=args.max_length_encoder,
            norm_doc_emb=args.normalize_embeddings
        )

        logger.log_step_end("Retriever initialization")
        return retriever

    except Exception as e:
        logger.log_error(e, "Retriever initialization failed")
        raise


def load_subset_indices(subset_indices_path: Optional[str], corpus_size: int, logger: ExperimentLogger) -> List[int]:
    """
    Load subset indices for embedding a specific subset of the corpus.

    Args:
        subset_indices_path (str): Path to subset indices file.
        corpus_size (int): Total size of the corpus.
        logger (ExperimentLogger): Logger instance.

    Returns:
        List[int]: Subset indices.
    """
    if not subset_indices_path:
        return list(range(corpus_size))

    try:
        logger.log_step_start("Loading subset indices")
        subset_indices = read_json(subset_indices_path)
        if not isinstance(subset_indices, list) or not all(isinstance(idx, int) for idx in subset_indices):
            raise ValueError("Invalid format for subset indices file. Expected a list of integers.")
        logger.log_step_end("Subset indices loaded")
        return subset_indices
    except Exception as e:
        logger.log_error(e, "Failed to load subset indices")
        raise


def compute_embeddings_with_memory_tracking(
    retriever: Retriever,
    corpus: List[Dict],
    subset_indices: List[int],
    args: argparse.Namespace,
    logger: ExperimentLogger
) -> None:
    try:
        batch_size = args.batch_size
        total_batches = (len(subset_indices) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="Computing embeddings", unit="batch"):
            # Check memory usage and adjust batch size
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if current_memory > args.max_memory_usage:
                    batch_size = max(1, batch_size // 2)
                    logger.log_metric("batch_size_adjusted", batch_size)

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(subset_indices))
            batch_indices = subset_indices[start_idx:end_idx]
            batch_corpus = [corpus[idx] for idx in batch_indices]

            # Compute embeddings for batch
            retriever.encode_corpus(
                batch_corpus,
                batch_size=batch_size,
                output_dir=args.output_dir,
                prefix_name=f"{args.prefix_name}_batch_{batch_idx}",
                save_every=args.save_every
            )

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        logger.log_error(e, "Error computing embeddings")
        raise
    '''
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for computing corpus embeddings.")
    parser.add_argument('--corpus_path', type=str, required=True)
    parser.add_argument('--encoder_id', type=str, default='facebook/contriever')
    parser.add_argument('--max_length_encoder', type=int, default=512)
    parser.add_argument('--normalize_embeddings', type=str2bool, default=False)
    parser.add_argument('--output_dir', type=str, default='data/corpus/embeddings/')
    parser.add_argument('--prefix_name', type=str, default='contriever')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--max_memory_usage', type=float, default=0.8)
    parser.add_argument('--subset_indices', type=str, default=None, help="Path to subset indices file")
    return parser.parse_args()

def initialize_retriever(args: argparse.Namespace, logger: ExperimentLogger) -> Retriever:
    try:
        logger.log_step_start("Initializing retriever")
        config = AutoConfig.from_pretrained(args.encoder_id)
        encoder = AutoTokenizer.from_pretrained(args.encoder_id)
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_id)

        retriever = Retriever(
            device=device,
            tokenizer=tokenizer,
            query_encoder=encoder,
            max_length=args.max_length_encoder,
            norm_doc_emb=args.normalize_embeddings
        )

        logger.log_step_end("Retriever initialization")
        return retriever

    except Exception as e:
        logger.log_error(e, "Retriever initialization failed")
        raise

class EmbeddingsComputer:
    def __init__(self, args: argparse.Namespace, logger: ExperimentLogger):
        self.args = args
        self.logger = logger
        self.cache_dir = Path(args.output_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_tracker = MemoryTracker(args.max_memory_usage)

    def compute_embeddings(self, corpus: List[Dict], retriever) -> None:
        total_batches = (len(corpus) + self.args.batch_size - 1) // self.args.batch_size
        
        for batch_idx in tqdm(range(total_batches)):
            try:
                # Adjust batch size based on memory
                self.args.batch_size = self.memory_tracker.get_optimal_batch_size(
                    self.args.batch_size,
                    self.args.min_batch_size
                )

                # Process batch
                start_idx = batch_idx * self.args.batch_size
                end_idx = min((batch_idx + 1) * self.args.batch_size, len(corpus))
                batch_corpus = corpus[start_idx:end_idx]

                # Compute embeddings
                batch_embeddings = self._compute_batch_embeddings(
                    batch_corpus,
                    retriever
                )

                # Cache results
                self._cache_embeddings(
                    batch_embeddings,
                    start_idx,
                    end_idx
                )

                # Memory cleanup
                self.memory_tracker.cleanup()

            except Exception as e:
                self.logger.log_error(e, f"Error processing batch {batch_idx}")
                continue

    def _compute_batch_embeddings(
        self,
        batch_corpus: List[Dict],
        retriever
    ) -> np.ndarray:
        return retriever.encode_corpus(
            batch_corpus,
            batch_size=self.args.batch_size,
            show_progress=False
        )

    def _cache_embeddings(
        self,
        embeddings: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> None:
        cache_file = self.cache_dir / f"embeddings_{start_idx}_{end_idx}.npy"
        np.save(cache_file, embeddings)

class MemoryTracker:
    def __init__(self, max_memory_usage: float):
        self.max_memory_usage = max_memory_usage
        
    def get_optimal_batch_size(self, current_batch_size: int, min_batch_size: int) -> int:
        if torch.cuda.is_available():
            current_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if current_usage > self.max_memory_usage:
                return max(min_batch_size, current_batch_size // 2)
        return current_batch_size

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def main():
    args = parse_arguments()
    logger = ExperimentLogger("corpus_embeddings", "logs")

    try:
        with logger:
            corpus = read_corpus_json(args.corpus_path)
            retriever = initialize_retriever(args, logger)
            embeddings_computer = EmbeddingsComputer(args, logger)
            embeddings_computer.compute_embeddings(corpus, retriever)

    except Exception as e:
        logger.log_error(e, "Main execution failed")
        raise

if __name__ == '__main__':
    main()