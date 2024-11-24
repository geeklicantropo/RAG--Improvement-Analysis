import os 
import argparse
import warnings
import torch
from transformers import AutoTokenizer, AutoConfig
from retriever import *
from utils import *
from experiment_logger import ExperimentLogger
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED = 10

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for computing the embeddings of a corpus.")
    parser.add_argument('--corpus_path', type=str, required=True, help='Path to the JSON corpus data')
    parser.add_argument('--encoder_id', type=str, default='facebook/contriever', help='Model identifier for the encoder')
    parser.add_argument('--max_length_encoder', type=int, default=512, help='Maximum sequence length for the encoder')
    parser.add_argument('--normalize_embeddings', type=str2bool, default=False, help='Whether to normalize embeddings')
    parser.add_argument('--lower_case', type=str2bool, default=False, help='Whether to lower case the corpus text')
    parser.add_argument('--do_normalize_text', type=str2bool, default=True, help='Whether to normalize the corpus text')
    parser.add_argument('--output_dir', type=str, default='data/corpus/embeddings/', help='Output directory for saving embeddings')
    parser.add_argument('--prefix_name', type=str, default='contriever', help='Initial part of the name of the saved embeddings')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for embedding documents')
    parser.add_argument('--save_every', type=int, default=500, help='Save embeddings every N batches')
    
    args = parser.parse_args()
    return args

def initialize_retriever(args: argparse.Namespace, logger: ExperimentLogger) -> Retriever:
    """Initialize the encoder and retriever with logging."""
    try:
        logger.log_step_start("Initializing retriever")
        start_time = time.time()

        config = AutoConfig.from_pretrained(args.encoder_id)
        encoder = Encoder(config).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_id)
        retriever = Retriever(
            device=device, 
            tokenizer=tokenizer, 
            query_encoder=encoder, 
            max_length=args.max_length_encoder,
            norm_doc_emb=args.normalize_embeddings,
            lower_case=args.lower_case,
            do_normalize_text=args.do_normalize_text
        )

        logger.log_step_end("Initializing retriever", start_time)
        return retriever

    except Exception as e:
        logger.log_error(e, "Error initializing retriever")
        raise

def main():
    args = parse_arguments()

    # Initialize logger
    logger = ExperimentLogger(
        experiment_name="corpus_embeddings",
        base_log_dir="logs"
    )

    try:
        with logger:
            # Log experiment parameters
            logger.log_experiment_params(vars(args))
            
            # Log system information
            logger.log_system_info()

            # Load corpus with progress bar
            logger.log_step_start("Loading corpus")
            corpus = read_json(args.corpus_path)
            logger.log_step_end("Loading corpus", time.time())
            logger.log_metric("corpus_size", len(corpus))
            logger.experiment_logger.info(f"Corpus loaded with {len(corpus)} documents")

            # Initialize retriever
            retriever = initialize_retriever(args, logger)
            logger.experiment_logger.info("Retriever initialized successfully")

            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)

            # Compute embeddings with progress tracking
            logger.log_step_start("Computing embeddings")
            retriever.encode_corpus(
                corpus, 
                batch_size=args.batch_size, 
                output_dir=args.output_dir,
                prefix_name=args.prefix_name,
                save_every=args.save_every,
                logger=logger  # Pass logger to encode_corpus
            )
            logger.log_step_end("Computing embeddings", time.time())

            # Log final metrics
            logger.log_metric("total_batches_processed", len(corpus) // args.batch_size)
            logger.log_metric("final_save_checkpoint", len(corpus) - 1)

    except Exception as e:
        logger.log_error(e, "Error in main execution")
        raise

if __name__ == '__main__':
    seed_everything(SEED)
    main()