import logging
import logging.config
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List
from datetime import datetime

import torch
import argparse
import warnings

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from rich.console import Console
from rich.progress import track
import yaml

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import modules from your project
from src.utils.llm import LLM
from src.utils import *
from src.prompt_dataset import MultiCorpusDataset
from src.core.env.env_loader import load_environment
from src.core.utils.system_check import perform_system_checks
from src.config import paths, settings

# Initialize Console for Rich
console = Console()

# Initialize Logging
def configure_logging():
    with open(paths.LOGGING_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

def seed_everything(seed: int):
    """Set seed for reproducibility."""
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiCorpusExperiment:
    """Track and manage multi-corpus generation experiments."""
    def __init__(self, config: dict):
        self.start_time = datetime.now()
        self.config = config
        self.metrics = {}
        self.batch_metrics = []
        logging.getLogger(__name__).info(f"Initialized multi-corpus experiment with config: {config}")
    
    def log_batch_metric(self, batch_idx: int, metrics: Dict):
        """Log metrics for a batch."""
        self.batch_metrics.append({
            'batch_idx': batch_idx,
            'metrics': metrics,
            'timestamp': datetime.now()
        })
        logging.getLogger(__name__).info(f"Batch {batch_idx} metrics: {metrics}")
    
    def end_experiment(self):
        """End experiment and compute final metrics."""
        duration = datetime.now() - self.start_time
        self.metrics['duration'] = duration
        self.metrics['total_batches'] = len(self.batch_metrics)
        
        if self.batch_metrics:
            avg_generation_time = sum(
                m['metrics'].get('generation_time', 0) 
                for m in self.batch_metrics
            ) / len(self.batch_metrics)
            self.metrics['avg_generation_time'] = avg_generation_time
            
        logging.getLogger(__name__).info(f"Experiment completed in {duration}")
        logging.getLogger(__name__).info(f"Final metrics: {self.metrics}")

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def timing_decorator(func):
    """Decorator to time functions."""
    import functools, time
    @functools.wraps(func)
    def wrapper_timings(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.getLogger(__name__).info(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds.")
        return result
    return wrapper_timings

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM Generation with multi-corpus approach."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/gen_res',
        help='Output directory'
    )
    parser.add_argument(
        '--llm_id',
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
        help='LLM model identifier'
    )
    parser.add_argument(
        '--model_max_length',
        type=int,
        default=4096,
        help='Maximum input length for the LLM model'
    )
    parser.add_argument(
        '--load_full_corpus',
        type=str2bool,
        default=True,
        help='Load the full corpus'
    )
    parser.add_argument(
        '--use_bm25',
        type=str2bool,
        default=False,
        help='Use BM25 retrieved documents'
    )
    parser.add_argument(
        '--use_corpus_nonsense',
        type=str2bool,
        default=False,
        help='Use documents composed of random words'
    )
    parser.add_argument(
        '--num_main_documents',
        type=int,
        required=True,
        help='Number of main corpus documents'
    )
    parser.add_argument(
        '--num_other_documents',
        type=int,
        required=True,
        help='Number of other corpus documents'
    )
    parser.add_argument(
        '--put_main_first',
        type=str2bool,
        default=False,
        help='Put main corpus documents first'
    )
    parser.add_argument(
        '--use_test',
        type=str2bool,
        default=True,
        help='Use test set'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=15,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=True,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=250,
        help='Save results every N batches'
    )
    parser.add_argument(
        '--experiment_mode',
        type=str,
        choices=['baseline', 'kmeans', 'fusion', 'categories'],
        default='baseline',
        help='Experiment mode for document selection'
    )
    
    args = parser.parse_args()
    args.split = "test" if args.use_test else "train"
    args.num_documents_in_context = args.num_main_documents + args.num_other_documents

    if args.num_main_documents <= 0 and args.num_other_documents <= 0:
        parser.error("At least one of num_main_documents or num_other_documents must be positive")

    return args

@timing_decorator
def load_corpus(args: argparse.Namespace) -> Tuple[List[Dict], Optional[Dict[int, int]], List[Dict]]:
    """Load both main and other corpus with experiment tracking."""
    logger = logging.getLogger(__name__)
    logger.info("Loading corpora")
    try:
        # Load other corpus based on configuration
        if args.use_corpus_nonsense:
            other_corpus = read_pickle('data/processed/corpus_with_random_50_words.pkl')
            logger.info("Loaded nonsense corpus")
        else:
            other_corpus = read_pickle('data/processed/reddit_corpus.pkl')
            logger.info("Loaded Reddit corpus")

        # Load main corpus
        if args.load_full_corpus:
            corpus = read_corpus_json('data/corpus.json')
            logger.info("Loaded full main corpus")
            return corpus, None, other_corpus

        if args.use_test:
            if args.use_bm25:
                corpus_loader = read_test_corpus_with_random_and_bm25
                logger.info("Loading test corpus with BM25")
            else:
                corpus_loader = read_test_corpus_with_random_and_contriever
                logger.info("Loading test corpus with Contriever")
        else:
            corpus_loader = read_corpus_with_random_and_contriever
            logger.info("Loading training corpus")
        
        corpus, full_to_subset_idx_map = corpus_loader()
        return corpus, full_to_subset_idx_map, other_corpus
            
    except Exception as e:
        logger.error(f"Error loading corpus: {e}")
        raise

@timing_decorator
def load_search_results(
    args: argparse.Namespace
) -> Tuple[List[Tuple[List[int], List[float]]], List[Tuple[List[int], List[float]]]]:
    """Load search results for both corpora."""
    logger = logging.getLogger(__name__)
    logger.info("Loading search results for both corpora")
    try:
        # Load main corpus search results
        if args.use_bm25:
            search_results_path = info[args.split]['bm25_search_results_path']
            logger.info("Loading BM25 results for main corpus")
        else:
            search_results_path = info[args.split]['contriever_search_results_path']
            logger.info("Loading Contriever results for main corpus")
        search_results = read_pickle(search_results_path)

        # Load other corpus search results
        if args.use_corpus_nonsense:
            other_results_path = info[args.split]['nonsense_results_path']
            logger.info("Loading nonsense corpus results")
        else:
            other_results_path = info[args.split]['reddit_results_path']
            logger.info("Loading Reddit corpus results")
        search_results_other_corpus = read_pickle(other_results_path)

        logger.info(f"Loaded {len(search_results)} main corpus results and {len(search_results_other_corpus)} other corpus results")
        return search_results, search_results_other_corpus
        
    except Exception as e:
        logger.error(f"Error loading search results: {e}")
        raise

@timing_decorator
def initialize_dataset_and_loader(
    args: argparse.Namespace,
    corpus: List[Dict],
    other_corpus: List[Dict],
    full_to_subset_idx_map: Optional[Dict[int, int]],
    search_results: List[Tuple[List[int], List[float]]],
    search_results_other_corpus: List[Tuple[List[int], List[float]]],
    tokenizer: PreTrainedTokenizer
) -> DataLoader:
    """Initialize multi-corpus dataset and dataloader."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing multi-corpus dataset")
    try:
        documents_disposition_info = {
            "num_main_documents": args.num_main_documents,
            "num_other_documents": args.num_other_documents,
            "put_main_first": args.put_main_first,
        }
        
        prompt_ds = MultiCorpusDataset(
            corpus=corpus,
            data_path=info[args.split]['data_path'],
            tokenizer=tokenizer,
            max_tokenized_length=args.model_max_length - 2,
            search_results=search_results,
            documents_other_corpus=other_corpus,
            search_results_other_corpus=search_results_other_corpus,
            documents_disposition_info=documents_disposition_info,
            full_to_subset_idx_map=full_to_subset_idx_map,
            do_normalize_query=True,
            experiment_mode=args.experiment_mode
        )
            
        prompt_dataloader = DataLoader(
            prompt_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        
        logger.info(f"Initialized dataloader with {len(prompt_ds)} examples")
        return prompt_dataloader
        
    except Exception as e:
        logger.error(f"Error initializing dataset: {e}")
        raise

def print_info(args: argparse.Namespace):
    """Print experiment configuration."""
    console = Console()
    console.print("\n[bold cyan]Multi-Corpus Experiment Configuration:[/bold cyan]")
    console.print(f"Data: {info[args.split]['data_path']}")
    console.print(f"Use Test: {args.use_test}")
    console.print(f"Model: {args.llm_id}")
    console.print(f"Use BM25: {args.use_bm25}")
    console.print(f"Use Nonsense Corpus: {args.use_corpus_nonsense}")
    console.print(f"Main Documents: {args.num_main_documents}")
    console.print(f"Other Documents: {args.num_other_documents}")
    console.print(f"Main First: {args.put_main_first}")
    console.print(f"Total Documents: {args.num_documents_in_context}")
    console.print(f"Experiment Mode: {args.experiment_mode}")
    console.print(f"Batch Size: {args.batch_size}")
    console.print(f"Save Every: {args.save_every}\n")

@timing_decorator
def generate_and_save(
    args: argparse.Namespace,
    llm: LLM,
    prompt_dataloader: DataLoader,
    experiment: MultiCorpusExperiment
):
    """Generate answers and save results with experiment tracking."""
    logger = logging.getLogger(__name__)
    try:
        # Setup paths and configurations
        llm_folder = args.llm_id.split("/")[1] if '/' in args.llm_id else args.llm_id
        num_doc = args.num_documents_in_context
        retriever_str = "bm25" if args.use_bm25 else "contriever"
        other_corpus_str = "_nonsense" if args.use_corpus_nonsense else "_reddit"

        if args.put_main_first:
            first_type_str = f"_main{args.num_main_documents}"
            second_type_str = f"_other{args.num_other_documents}"
        else:
            first_type_str = f"_other{args.num_other_documents}"
            second_type_str = f"_main{args.num_main_documents}"

        # Create save directory
        saving_dir = Path(args.output_dir) / llm_folder / args.split / "multi_corpus" / retriever_str / f"{num_doc}_doc" / args.experiment_mode
        saving_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created save directory: {saving_dir}")
        
        # Determine answer string based on model
        answer_string = "### Response:" if 'mpt' in args.llm_id else "Answer:"
        
        all_info = []
        total_generation_time = 0
        
        for idx, prompt_batch in enumerate(track(prompt_dataloader, description="Generating answers")):
            batch_start_time = datetime.now()
            
            # Generate answers
            prompts = prompt_batch['prompt']
            generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
            
            # Extract answers
            generated_answers = []
            for output in generated_output:
                start = output.find(answer_string) + len(answer_string)
                response = output[start:].strip()
                generated_answers.append(response)

            # Store results
            prompt_batch['generated_answer'] = generated_answers
            all_info.append(prompt_batch)
            
            # Calculate metrics
            batch_time = (datetime.now() - batch_start_time).total_seconds()
            total_generation_time += batch_time
            
            # Log batch metrics
            experiment.log_batch_metric(idx, {
                'batch_size': len(prompts),
                'generation_time': batch_time,
                'avg_answer_length': sum(len(a) for a in generated_answers) / len(generated_answers)
            })
            
            # Save checkpoint if needed
            if (idx + 1) % args.save_every == 0 or (idx + 1) == len(prompt_dataloader):
                logger.info(f"Saving checkpoint at batch {idx + 1}")
                file_name = saving_dir / f"numdoc{num_doc}{first_type_str}{second_type_str}{other_corpus_str}_info_{idx+1}.pkl"
                write_pickle(all_info, file_name)
                all_info = []
                
        # Log final metrics
        experiment.metrics.update({
            'total_generation_time': total_generation_time,
            'total_batches': len(prompt_dataloader)
        })
        
    except Exception as e:
        logger.error(f"Error in generation: {e}")
        raise

def main():
    """Main execution with comprehensive experiment tracking."""
    start_time = datetime.now()
    try:
        # Configure logging
        configure_logging()

        # Load environment variables
        load_environment()

        # Perform system checks
        perform_system_checks()

        # Parse arguments and initialize experiment
        args = parse_arguments()
        experiment = MultiCorpusExperiment(vars(args))
        
        # Create necessary directories
        Path("logs/generation").mkdir(parents=True, exist_ok=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize LLM
        console.print("\n[bold cyan]Initializing LLM...[/bold cyan]")
        llm = LLM(
            model_id=args.llm_id,
            device=device,
            quantization_bits=4, 
            model_max_length=args.model_max_length
        )
        tokenizer = llm.tokenizer
        logging.getLogger(__name__).info("LLM initialized successfully")

        # Load data
        console.print("\n[bold cyan]Loading corpora and search results...[/bold cyan]")
        corpus, full_to_subset_idx_map, other_corpus = load_corpus(args)
        search_results, search_results_other_corpus = load_search_results(args)

        # Initialize dataset
        console.print("\n[bold cyan]Initializing dataset...[/bold cyan]")
        prompt_dataloader = initialize_dataset_and_loader(
            args, corpus, other_corpus, full_to_subset_idx_map,
            search_results, search_results_other_corpus,
            tokenizer
        )

        # Print configuration
        print_info(args)

        # Generate answers
        console.print("\n[bold cyan]Generating answers...[/bold cyan]")
        generate_and_save(args, llm, prompt_dataloader, experiment)

        # Log completion
        experiment.end_experiment()
        duration = datetime.now() - start_time
        console.print(f"\n[bold green]✓ Multi-corpus generation completed in {duration}[/bold green]")

    except Exception as e:
        logging.getLogger(__name__).error(f"Error in main execution: {e}")
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise

if __name__ == "__main__":
    seed_everything(settings.SEED)
    main()
