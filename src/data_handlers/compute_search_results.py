import os 
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import sys
import pickle
import json
import numpy as np
import torch
import logging
from rich.logging import RichHandler
from rich.progress import track
from datasets import load_from_disk, IterableDataset
import gc

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.retriever import Retriever
from src.core.env.env_loader import load_environment
from src.core.utils.system_check import perform_system_checks
from src.config import paths, settings

class SearchResultsComputer:
    """Computes and manages search results with memory-efficient processing."""
    
    def __init__(
        self,
        retriever_type: str = "contriever",
        device: Optional[str] = None,
        batch_size: int = 32,
        experiment_name: Optional[str] = None,
        chunk_size: int = 1000  # Size of chunks for saving results
    ):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.retriever_type = retriever_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.experiment_name = experiment_name or f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.retriever = None
        self.metrics = {
            'retriever_type': retriever_type,
            'device': self.device,
            'batch_size': batch_size,
            'start_time': datetime.now().isoformat()
        }
        
        # Load settings
        self.retriever_settings = settings.RETRIEVERS.get(retriever_type, {})
        if not self.retriever_settings:
            raise ValueError(f"No settings found for retriever type: {self.retriever_type}")
            
        self.logger.info(
            f"Initialized SearchResultsComputer with {retriever_type} on {self.device}"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = paths.LOGS_DIR / "retrieval"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                RichHandler(rich_tracebacks=True),
                logging.FileHandler(
                    log_dir / f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            ]
        )

    def _load_embeddings_iterator(self, embeddings_dir: Path) -> Iterator[np.ndarray]:
        """Memory-efficient embeddings iterator."""
        chunk_files = sorted(embeddings_dir.glob("embeddings_chunk_*.npy"))
        
        for chunk_file in chunk_files:
            # Memory map the numpy array
            embeddings = np.load(chunk_file, mmap_mode='r')
            yield embeddings
            
            # Clear memory
            del embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def initialize_components(self, corpus_path: str):
        """Initialize retriever with corpus path."""
        try:
            self.logger.info("Initializing components...")
            
            # Initialize retriever with corpus path
            self.retriever = Retriever(
                retriever_type=self.retriever_type,
                device=self.device,
                batch_size=self.batch_size,
                cache_embeddings=self.retriever_settings.get('cache_embeddings', True),
                dataset_path=corpus_path
            )
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise

    def compute_search_results(
        self,
        queries_dataset: str,
        top_k: int = 100,
        save_results: bool = True
    ) -> None:
        """
        Compute search results with memory-efficient processing.
        
        Args:
            queries_dataset: Path to queries dataset
            top_k: Number of top results to retrieve
            save_results: Whether to save results to disk
        """
        try:
            start_time = datetime.now()
            self.logger.info("Computing search results...")
            
            # Create output directory
            output_dir = paths.RETRIEVAL_DIR / self.retriever_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize results file
            results_file = output_dir / f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            # Process queries in chunks
            current_chunk = []
            chunk_count = 0
            total_queries = 0
            total_retrieval_time = 0
            
            # Load and process queries in batches
            for query_batch in self._load_queries_iterator(queries_dataset):
                current_chunk.extend(query_batch)
                total_queries += len(query_batch)
                
                # Process chunk when it reaches chunk_size
                if len(current_chunk) >= self.chunk_size:
                    self._process_query_chunk(
                        current_chunk,
                        chunk_count,
                        top_k,
                        results_file
                    )
                    chunk_count += 1
                    current_chunk = []
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # Process remaining queries
            if current_chunk:
                self._process_query_chunk(
                    current_chunk,
                    chunk_count,
                    top_k,
                    results_file
                )
            
            # Save metadata
            self._save_metadata(output_dir, total_queries, start_time)
            
            self.logger.info(
                f"Completed search results computation for {total_queries} queries "
                f"in {(datetime.now() - start_time).total_seconds():.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error computing search results: {e}")
            raise

    def _load_queries_iterator(self, queries_dataset: str) -> Iterator[List[str]]:
        """Memory-efficient query iterator."""
        dataset = load_from_disk(queries_dataset)
        
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:min(i + self.batch_size, len(dataset))]
            yield batch['question']

    def _process_query_chunk(
        self,
        queries: List[str],
        chunk_idx: int,
        top_k: int,
        results_file: Path
    ):
        """Process a chunk of queries."""
        try:
            batch_start = datetime.now()
            
            # Process queries in batches
            chunk_results = []
            for i in range(0, len(queries), self.batch_size):
                batch_queries = queries[i:i + self.batch_size]
                
                # Retrieve documents for batch
                batch_results = self._process_query_batch(batch_queries, top_k)
                chunk_results.extend(batch_results)
            
            # Save chunk results
            self._save_chunk_results(chunk_results, results_file)
            
            # Update metrics
            processing_time = (datetime.now() - batch_start).total_seconds()
            self.metrics[f'chunk_{chunk_idx}'] = {
                'size': len(queries),
                'processing_time': processing_time,
                'avg_results_per_query': np.mean([len(r[0]) for r in chunk_results])
            }
            
            self.logger.info(
                f"Processed chunk {chunk_idx}: {len(queries)} queries "
                f"in {processing_time:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_idx}: {e}")
            raise

    def _process_query_batch(
        self,
        queries: List[str],
        top_k: int
    ) -> List[Tuple[List[int], List[float]]]:
        """Process a batch of queries."""
        try:
            # Retrieve documents
            results = []
            for query in queries:
                retrieved_docs = self.retriever.retrieve(
                    query,
                    top_k=top_k,
                    include_metadata=True
                )
                
                # Extract corpus indices and scores
                corpus_indices = [doc.corpus_idx for doc in retrieved_docs]
                scores = [doc.score for doc in retrieved_docs]
                results.append((corpus_indices, scores))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing query batch: {e}")
            raise

    def _save_chunk_results(self, results: List, results_file: Path):
        """Save chunk results to file."""
        try:
            # Append results to file
            with open(results_file, 'ab') as f:
                pickle.dump(results, f)
            
        except Exception as e:
            self.logger.error(f"Error saving chunk results: {e}")
            raise

    def _save_metadata(self, output_dir: Path, total_queries: int, start_time: datetime):
        """Save metadata about the search results computation."""
        try:
            metadata = {
                'total_queries': total_queries,
                'chunks': list(self.metrics.keys()),
                'retriever_type': self.retriever_type,
                'device': str(self.device),
                'batch_size': self.batch_size,
                'chunk_size': self.chunk_size,
                'processing_metrics': self.metrics,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': (datetime.now() - start_time).total_seconds()
            }
            
            metadata_file = output_dir / "search_results_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved metadata to {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            raise

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute search results with memory optimization"
    )
    parser.add_argument(
        '--retriever_type',
        type=str,
        choices=['contriever', 'bm25', 'adore'],
        default='contriever',
        help='Type of retriever to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=1000,
        help='Chunk size for saving results'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=100,
        help='Number of top results to retrieve'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu)'
    )
    args = parser.parse_args()
    
    start_time = datetime.now()
    try:
        # Load environment
        load_environment()
        
        # Perform system checks
        perform_system_checks()
        
        # Initialize computer
        computer = SearchResultsComputer(
            retriever_type=args.retriever_type,
            device=args.device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size
        )
        
        # Initialize components with corpus path
        computer.initialize_components(str(paths.CORPUS_DIR))
        
        # Compute results
        computer.compute_search_results(
            queries_dataset=str(paths.DATASETS_DIR / "test"),
            top_k=args.top_k,
            save_results=True
        )
        
        # Log completion
        duration = datetime.now() - start_time
        logging.info(f"Search computation completed in {duration}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()