import os 
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rich.console import Console
from rich.progress import track
import logging
from rich.logging import RichHandler

from index import *
from utils import *

# Configure logging
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(
            f'logs/indexing/index_embeddings_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# Environment setup
warnings.filterwarnings('ignore')
SEED = 10

class IndexingExperiment:
    """Track and manage indexing experiments."""
    def __init__(self, config: Dict[str, Any]):
        self.start_time = datetime.now()
        self.config = config
        self.metrics = {}
        self.splits = []
        logger.info(f"Initialized indexing experiment with config: {config}")
    
    def log_split_metric(self, split_idx: int, metrics: Dict[str, Any]):
        """Log metrics for an index split."""
        self.splits.append({
            'split_idx': split_idx,
            'metrics': metrics,
            'timestamp': datetime.now()
        })
        logger.info(f"Split {split_idx} metrics: {metrics}")
    
    def end_experiment(self):
        """End experiment and compute final metrics."""
        duration = datetime.now() - self.start_time
        self.metrics['duration'] = duration
        self.metrics['total_splits'] = len(self.splits)
        
        # Compute aggregate metrics
        if self.splits:
            total_vectors = sum(s['metrics'].get('num_vectors', 0) for s in self.splits)
            self.metrics['total_vectors_indexed'] = total_vectors
            
        logger.info(f"Experiment completed in {duration}")
        logger.info(f"Final metrics: {self.metrics}")

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Script for indexing corpus embeddings.")
    parser.add_argument(
        '--corpus_size',
        type=int,
        required=True,
        help='Size of the embedded corpus'
    )
    parser.add_argument(
        '--vector_sz',
        type=int,
        default=768,
        help='Size of the vectors to be indexed (default: 768)'
    )
    parser.add_argument(
        '--idx_type',
        type=str,
        default='IP',
        help='Index type (IP for Inner Product) (default: IP)'
    )
    parser.add_argument(
        '--faiss_dir',
        type=str,
        required=True,
        help='Directory where to store the FAISS index data'
    )
    parser.add_argument(
        '--percentages_for_index_splitting',
        nargs='*',
        type=float,
        default=[],
        help='Percentages for splitting index (e.g., 40 means 40%%). Leave empty for single index.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/corpus/embeddings/',
        help='Output directory of the saved embeddings'
    )
    parser.add_argument(
        '--prefix_name',
        type=str,
        default='contriever',
        help='Initial part of the name of the saved embeddings'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Batch size used in compute_corpus_embeddings'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=500,
        help='Save every steps used in compute_corpus_embeddings'
    )
    
    args = parser.parse_args()

    if any(p <= 0 for p in args.percentages_for_index_splitting):
        raise ValueError("Percentages for index splitting must be positive")
        
    if sum(args.percentages_for_index_splitting) >= 100:
        raise ValueError("Sum of percentages must be less than 100")

    return args

@timing_decorator
def load_all_embeddings(args: argparse.Namespace) -> np.ndarray:
    """Load and merge all embeddings."""
    logger.info("Loading embeddings")
    
    try:
        all_embeddings_path = Path(args.output_dir) / f'{args.prefix_name}_all_embeddings.npy'

        # Check for pre-merged embeddings
        if all_embeddings_path.exists():
            logger.info("Loading pre-merged embeddings")
            embeddings = np.load(all_embeddings_path, mmap_mode='c')
            return embeddings

        # Load and merge individual embedding files
        all_embeddings = []
        num_embed = args.batch_size * args.save_every

        for i in track(
            range(num_embed - 1, args.corpus_size, num_embed),
            description="Loading embeddings"
        ):
            emb_path = Path(args.output_dir) / f'{args.prefix_name}_{i}_embeddings.npy'
            emb = np.load(emb_path, mmap_mode='c')
            all_embeddings.append(emb)

        # Load final embeddings
        last_idx = args.corpus_size - 1
        last_emb_path = Path(args.output_dir) / f'{args.prefix_name}_{last_idx}_embeddings.npy'
        last_emb = np.load(last_emb_path, mmap_mode='c')
        all_embeddings.append(last_emb)

        # Merge and save
        embeddings = np.concatenate(all_embeddings, axis=0)
        np.save(all_embeddings_path, embeddings)
        
        logger.info(f"Loaded and merged {len(all_embeddings)} embedding files")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise

@timing_decorator
def indexing_embeddings(args: argparse.Namespace, embeddings: np.ndarray, experiment: IndexingExperiment):
    """Create and save FAISS indexes."""
    try:
        os.makedirs(args.faiss_dir, exist_ok=True)
        logger.info(f"Created index directory: {args.faiss_dir}")

        # Single index case
        if not args.percentages_for_index_splitting:
            logger.info("Creating single index")
            index = Indexer(args.vector_sz, idx_type=args.idx_type)
            index.index_data(list(range(args.corpus_size)), embeddings)
            
            experiment.log_split_metric(0, {
                'type': 'single',
                'num_vectors': args.corpus_size
            })

            logger.info("Saving index")
            index.serialize(
                dir_path=args.faiss_dir,
                index_file_name=f'{args.idx_type}_index.faiss',
                meta_file_name=f'{args.idx_type}_index_meta.faiss'
            )
            return

        # Split index case
        logger.info("Creating split indexes")
        current_pos = 0
        for i, percentage in enumerate(args.percentages_for_index_splitting, 1):
            start_idx = current_pos
            end_idx = int((args.corpus_size * percentage) / 100)
            
            logger.info(f"Creating split {i}: {start_idx} to {end_idx}")
            
            index = Indexer(args.vector_sz, idx_type=args.idx_type)
            index.index_data(
                list(range(start_idx, end_idx)),
                embeddings[start_idx:end_idx]
            )
            
            experiment.log_split_metric(i, {
                'type': 'split',
                'start': start_idx,
                'end': end_idx,
                'num_vectors': end_idx - start_idx
            })

            logger.info(f"Saving split {i}")
            index.serialize(
                dir_path=args.faiss_dir,
                index_file_name=f'{args.idx_type}_index{i}.faiss',
                meta_file_name=f'{args.idx_type}_index{i}_meta.faiss'
            )
            
            current_pos = end_idx

        # Final split for remaining vectors
        logger.info("Creating final split")
        index = Indexer(args.vector_sz, idx_type=args.idx_type)
        index.index_data(
            list(range(current_pos, args.corpus_size)),
            embeddings[current_pos:]
        )
        
        experiment.log_split_metric(i + 1, {
            'type': 'final_split',
            'start': current_pos,
            'end': args.corpus_size,
            'num_vectors': args.corpus_size - current_pos
        })

        logger.info("Saving final split")
        index.serialize(
            dir_path=args.faiss_dir,
            index_file_name=f'{args.idx_type}_index{i + 1}.faiss',
            meta_file_name=f'{args.idx_type}_index{i + 1}_meta.faiss'
        )
        
    except Exception as e:
        logger.error(f"Error in indexing: {e}")
        raise

def main():
    """Main execution with comprehensive experiment tracking."""
    start_time = datetime.now()
    try:
        # Parse arguments and initialize experiment
        args = parse_arguments()
        experiment = IndexingExperiment(vars(args))
        
        # Create necessary directories
        os.makedirs("logs/indexing", exist_ok=True)
        os.makedirs(args.faiss_dir, exist_ok=True)

        # Load embeddings
        console.print("\n[bold cyan]Loading embeddings...[/bold cyan]")
        embeddings = load_all_embeddings(args)
        experiment.metrics['embeddings_shape'] = embeddings.shape
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Create indexes
        console.print("\n[bold cyan]Creating indexes...[/bold cyan]")
        indexing_embeddings(args, embeddings, experiment)

        # Log completion
        experiment.end_experiment()
        duration = datetime.now() - start_time
        console.print(f"\n[bold green]✓ Indexing completed in {duration}[/bold green]")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise

if __name__ == '__main__':
    seed_everything(SEED)
    main()