import os 
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Iterator
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from rich.console import Console
from rich.progress import track
import logging
from rich.logging import RichHandler
import yaml
from datasets import load_from_disk, Dataset, IterableDataset
import ijson
import gc

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.id_mapping import IDMapper
from src.utils.retriever import Retriever
from src.core.env.env_loader import load_environment
from src.core.utils.system_check import perform_system_checks
from src.config import paths, settings

class CorpusEmbeddingComputer:
    """Compute and manage corpus embeddings with memory-efficient processing."""
    
    def __init__(
        self,
        retriever_type: str = "contriever",
        device: Optional[str] = None,
        batch_size: int = 512,
        experiment_name: Optional[str] = None,
        chunk_size: int = 1000  # Size of chunks to save embeddings
    ):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.retriever_type = retriever_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.experiment_name = experiment_name or f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.retriever = None
        self.id_mapper = None
        self.metrics = {}
        
        # Load settings
        self.retriever_settings = settings.RETRIEVERS.get(retriever_type, {})
        if not self.retriever_settings:
            raise ValueError(f"No settings found for retriever type: {retriever_type}")
            
        self.logger.info(
            f"Initialized CorpusEmbeddingComputer with {retriever_type} on {self.device}"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = paths.LOGS_DIR / "embeddings"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                RichHandler(rich_tracebacks=True),
                logging.FileHandler(
                    log_dir / f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            ]
        )

    def initialize_components(self):
        """Initialize retriever and ID mapper."""
        try:
            # Initialize retriever
            self.retriever = Retriever(
                retriever_type=self.retriever_type,
                device=self.device,
                batch_size=self.batch_size,
                cache_embeddings=self.retriever_settings.get('cache_embeddings', True)
            )
            
            # Initialize ID mapper
            self.id_mapper = IDMapper(
                cache_size=settings.ID_MAPPING['cache_size'],
                experiment_name=self.experiment_name,
                batch_size=self.batch_size
            )
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise

    def _corpus_iterator(self) -> Iterator[Dict[str, Any]]:
        """Memory-efficient corpus iterator."""
        try:
            with open(paths.CORPUS_JSON, 'rb') as f:
                parser = ijson.parse(f)
                current_item = {}
                
                for prefix, event, value in parser:
                    if prefix.endswith('.text'):
                        current_item['text'] = value
                    elif prefix.endswith('.title'):
                        current_item['title'] = value
                        yield current_item
                        current_item = {}
                        
        except Exception as e:
            self.logger.error(f"Error in corpus iteration: {e}")
            raise

    def compute_embeddings(
        self,
        save_embeddings: bool = True,
        normalize: bool = True
    ) -> None:
        """
        Compute embeddings for corpus documents in chunks with memory optimization.
        
        Args:
            save_embeddings: Whether to save embeddings to disk
            normalize: Whether to normalize embeddings
        """
        try:
            start_time = datetime.now()
            self.logger.info("Starting embeddings computation")
            
            # Create output directory
            output_dir = paths.EMBEDDINGS_DIR / self.retriever_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process documents in chunks
            current_chunk = []
            chunk_count = 0
            total_docs = 0
            total_processing_time = 0
            
            # Process corpus in batches
            for doc in track(self._corpus_iterator(), description="Computing embeddings"):
                current_chunk.append(doc)
                total_docs += 1
                
                # Process chunk when it reaches chunk_size
                if len(current_chunk) >= self.chunk_size:
                    self._process_chunk(
                        current_chunk,
                        chunk_count,
                        output_dir,
                        normalize
                    )
                    chunk_count += 1
                    current_chunk = []
                    
                    # Clear GPU cache if using CUDA
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # Process remaining documents
            if current_chunk:
                self._process_chunk(
                    current_chunk,
                    chunk_count,
                    output_dir,
                    normalize
                )
            
            # Save metadata
            self._save_metadata(output_dir, total_docs, start_time)
            
            self.logger.info(
                f"Completed embeddings computation for {total_docs} documents "
                f"in {(datetime.now() - start_time).total_seconds():.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error computing embeddings: {e}")
            raise

    def _process_chunk(
        self,
        chunk: List[Dict],
        chunk_idx: int,
        output_dir: Path,
        normalize: bool
    ):
        """Process a chunk of documents."""
        try:
            batch_start = datetime.now()
            
            # Get texts from chunk
            texts = [doc['text'] for doc in chunk]
            
            # Compute embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Compute embeddings
                batch_embeddings = self.retriever.encode_texts(
                    texts=batch_texts,
                    normalize=normalize
                )
                all_embeddings.append(batch_embeddings)
                
                # Update index mapping
                for j, _ in enumerate(batch_texts):
                    doc_idx = chunk_idx * self.chunk_size + i + j
                    self.id_mapper.add_index_mapping(
                        doc_idx,
                        doc_idx,
                        'corpus_idx_to_embedding'
                    )
            
            # Combine embeddings for chunk
            chunk_embeddings = np.vstack(all_embeddings)
            
            # Save chunk embeddings
            chunk_file = output_dir / f"embeddings_chunk_{chunk_idx}.npy"
            np.save(chunk_file, chunk_embeddings)
            
            # Update metrics
            processing_time = (datetime.now() - batch_start).total_seconds()
            self.metrics[f'chunk_{chunk_idx}'] = {
                'size': len(chunk),
                'processing_time': processing_time
            }
            
            self.logger.info(
                f"Processed chunk {chunk_idx}: {len(chunk)} documents "
                f"in {processing_time:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_idx}: {e}")
            raise

    def _save_metadata(self, output_dir: Path, total_docs: int, start_time: datetime):
        """Save metadata about the embeddings computation."""
        try:
            metadata = {
                'total_documents': total_docs,
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
            
            metadata_file = output_dir / "embeddings_metadata.json"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, indent=2)
                
            # Save index mappings
            self.id_mapper.save_mappings(output_dir / "embeddings_index_mapping.pkl")
            
            self.logger.info(f"Saved metadata to {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            raise

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute corpus embeddings with memory optimization"
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
        default=512,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=1000,
        help='Chunk size for saving embeddings'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--normalize',
        type=bool,
        default=True,
        help='Whether to normalize embeddings'
    )
    args = parser.parse_args()
    
    start_time = datetime.now()
    try:
        # Load environment
        load_environment()
        
        # Perform system checks
        perform_system_checks()
        
        # Initialize computer
        computer = CorpusEmbeddingComputer(
            retriever_type=args.retriever_type,
            device=args.device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size
        )
        
        # Initialize components
        computer.initialize_components()
        
        # Compute embeddings
        computer.compute_embeddings(
            save_embeddings=True,
            normalize=args.normalize
        )
        
        # Log completion
        duration = datetime.now() - start_time
        logging.info(f"Embedding computation completed in {duration}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()