import os 
import argparse
import warnings
import torch
from transformers import AutoTokenizer, AutoConfig
from retriever import *
from utils import *
from experiment_logger import ExperimentLogger
import time
import gc
from typing import Optional
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED = 10

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
    return parser.parse_args()

def initialize_retriever(args: argparse.Namespace, logger: ExperimentLogger) -> Retriever:
    try:
        logger.log_step_start("Initializing retriever")
        config = AutoConfig.from_pretrained(args.encoder_id)
        encoder = Encoder(config).eval()
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

def compute_embeddings_with_memory_tracking(
    retriever: Retriever,
    corpus: List[Dict],
    args: argparse.Namespace,
    logger: ExperimentLogger
) -> None:
    try:
        batch_size = args.batch_size
        total_batches = (len(corpus) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches)):
            # Check memory usage and adjust batch size
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if current_memory > args.max_memory_usage:
                    batch_size = max(1, batch_size // 2)
                    logger.log_metric("batch_size_adjusted", batch_size)

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(corpus))
            batch_corpus = corpus[start_idx:end_idx]

            # Compute embeddings for batch
            retriever.encode_corpus(
                batch_corpus,
                batch_size=batch_size,
                output_dir=args.output_dir,
                prefix_name=f"{args.prefix_name}_{end_idx}",
                save_every=1  # Save each batch
            )

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        logger.log_error(e, "Error computing embeddings")
        raise

def main():
    args = parse_arguments()
    logger = ExperimentLogger("corpus_embeddings", "logs")

    try:
        with logger:
            logger.log_experiment_params(vars(args))
            logger.log_system_info()

            # Load corpus
            logger.log_step_start("Loading corpus")
            corpus = read_json(args.corpus_path)
            logger.log_step_end("Corpus loading")

            # Initialize retriever
            retriever = initialize_retriever(args, logger)

            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)

            # Compute embeddings with memory tracking
            compute_embeddings_with_memory_tracking(retriever, corpus, args, logger)

    except Exception as e:
        logger.log_error(e, "Main execution failed")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()