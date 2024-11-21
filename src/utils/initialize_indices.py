import os  
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Iterator
from tqdm import tqdm
from rich.console import Console
from rich.progress import track
import numpy as np
import gc
import ijson

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import paths
from src.utils.id_mapping import IDMapper

class IndexInitializer:
    """Initializes and manages index mappings across the RAG system with memory-efficient processing."""
    
    def __init__(
        self,
        experiment: str,
        retriever: str,
        batch_size: int = 1000,
        enable_gc: bool = True
    ):
        self.setup_logging()
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
        self.experiment = experiment
        self.retriever = retriever
        self.batch_size = batch_size
        self.enable_gc = enable_gc
        
        # Initialize ID mapper with batching support
        self.id_mapper = IDMapper(
            experiment_name=f"{experiment}_{retriever}",
            batch_size=batch_size,
            enable_gc=enable_gc
        )
        
        self.logger.info(
            f"Initializing indices for {experiment} experiment with "
            f"{retriever} retriever (batch_size={batch_size})"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = paths.LOGS_DIR / "indexing"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    log_dir / f"index_init_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            ]
        )

    def _iterate_corpus(self, file_path: str) -> Iterator[Dict]:
        """Memory-efficient corpus iterator using ijson."""
        with open(file_path, 'rb') as f:
            parser = ijson.parse(f)
            current_item = {}
            
            for prefix, event, value in parser:
                if prefix.endswith('.text'):
                    current_item['text'] = value
                elif prefix.endswith('.title'):
                    current_item['title'] = value
                    # Yield complete item and reset
                    yield current_item
                    current_item = {}

    def _write_batch_to_file(self, batch_data: Dict, output_file: Path, mode: str = 'a'):
        """Write a batch of mappings to file."""
        with open(output_file, mode) as f:
            if mode == 'w':  # New file, write opening bracket
                f.write('{\n')
            for idx, item in batch_data.items():
                f.write(f'"{idx}": {json.dumps(item)},\n')

    def initialize_corpus_mapping(self) -> None:
        """Initialize corpus index mapping with streaming processing."""
        self.logger.info("Initializing corpus index mapping...")
        
        try:
            output_path = paths.MAPPINGS_DIR / 'corpus_idx_mapping.json'
            
            # Write header
            with open(output_path, 'w') as f:
                f.write('{\n"metadata": {\n')
                f.write(f'"creation_time": "{datetime.now().isoformat()}",\n')
                f.write(f'"experiment": "{self.experiment}",\n')
                f.write(f'"retriever": "{self.retriever}",\n')
                f.write(f'"batch_size": {self.batch_size}\n')
                f.write('},\n"mappings": {\n')
            
            # Process corpus in batches
            batch_mappings = {}
            doc_count = 0
            
            for doc in track(self._iterate_corpus(paths.CORPUS_JSON), description="Processing corpus"):
                batch_mappings[str(doc_count)] = {
                    'corpus_idx': doc_count,
                    'title': doc.get('title', ''),
                    'embedding_idx': None
                }
                doc_count += 1
                
                if len(batch_mappings) >= self.batch_size:
                    self._write_batch_to_file(batch_mappings, output_path)
                    batch_mappings.clear()
                    
                    if self.enable_gc and doc_count % (self.batch_size * 10) == 0:
                        gc.collect()
            
            # Write remaining batch
            if batch_mappings:
                self._write_batch_to_file(batch_mappings, output_path)
            
            # Write file footer
            with open(output_path, 'a') as f:
                f.write('}}\n')
            
            self.logger.info(f"Created corpus mapping with {doc_count} documents")
            
        except Exception as e:
            self.logger.error(f"Error initializing corpus mapping: {e}")
            raise

    def _process_dataset_file(self, file_path: str, split: str, output_file: Path) -> int:
        """Process a dataset file in chunks."""
        doc_count = 0
        batch_mappings = {}
        
        # Use file streaming for JSON
        with open(file_path, 'r') as f:
            parser = ijson.parse(f, use_float=True)
            
            for prefix, event, value in parser:
                if event == 'map_key':
                    continue
                    
                if isinstance(value, dict):
                    batch_mappings[str(doc_count)] = {
                        'dataset_idx': doc_count,
                        'example_id': value.get('example_id', str(doc_count)),
                        'corpus_idx': value.get('idx_gold_in_corpus', -1),
                        'retrieval_idx': None
                    }
                    doc_count += 1
                    
                    if len(batch_mappings) >= self.batch_size:
                        self._write_batch_to_file(batch_mappings, output_file)
                        batch_mappings.clear()
                        
                        if self.enable_gc and doc_count % (self.batch_size * 10) == 0:
                            gc.collect()
        
        # Write remaining batch
        if batch_mappings:
            self._write_batch_to_file(batch_mappings, output_file)
            
        return doc_count

    def initialize_dataset_mapping(self) -> None:
        """Initialize dataset index mapping with streaming processing."""
        self.logger.info("Initializing dataset index mapping...")
        
        try:
            output_path = paths.MAPPINGS_DIR / 'dataset_idx_mapping.json'
            
            # Initialize file structure
            with open(output_path, 'w') as f:
                f.write('{\n"metadata": {\n')
                f.write(f'"creation_time": "{datetime.now().isoformat()}",\n')
                f.write(f'"experiment": "{self.experiment}",\n')
                f.write(f'"batch_size": {self.batch_size}\n')
                f.write('},\n')
            
            # Process train data
            train_count = self._process_dataset_file(
                paths.TRAIN_DATASET_PATH,
                'train',
                output_path
            )
            
            # Process test data
            test_count = self._process_dataset_file(
                paths.TEST_DATASET_PATH,
                'test',
                output_path
            )
            
            # Write file footer
            with open(output_path, 'a') as f:
                f.write('}\n')
            
            self.logger.info(
                f"Created dataset mapping for {train_count} train "
                f"and {test_count} test examples"
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing dataset mapping: {e}")
            raise

    def initialize_search_mapping(self) -> None:
        """Initialize search results index mapping with streaming processing."""
        self.logger.info("Initializing search results mapping...")
        
        try:
            results_path = {
                'contriever': paths.CONTRIEVER_RESULTS_PATH,
                'bm25': paths.BM25_RESULTS_PATH,
                'adore': paths.ADORE_RESULTS_PATH
            }.get(self.retriever)
            
            if not results_path:
                raise ValueError(f"Unknown retriever type: {self.retriever}")
            
            output_path = paths.MAPPINGS_DIR / 'search_idx_mapping.json'
            
            # Initialize file
            with open(output_path, 'w') as f:
                f.write('{\n"metadata": {\n')
                f.write(f'"creation_time": "{datetime.now().isoformat()}",\n')
                f.write(f'"experiment": "{self.experiment}",\n')
                f.write(f'"retriever": "{self.retriever}",\n')
                f.write(f'"batch_size": {self.batch_size}\n')
                f.write('},\n"mappings": {\n')
            
            # Process search results in batches
            doc_count = 0
            batch_mappings = {}
            
            # Read pickle file in chunks
            with open(results_path, 'rb') as f:
                while True:
                    try:
                        result = pickle.load(f)
                        
                        batch_mappings[str(doc_count)] = {
                            'query_idx': doc_count,
                            'corpus_indices': list(map(int, result[0])),
                            'scores': list(map(float, result[1])),
                            'retrieval_meta': {
                                'timestamp': datetime.now().isoformat(),
                                'top_k': len(result[0])
                            }
                        }
                        doc_count += 1
                        
                        if len(batch_mappings) >= self.batch_size:
                            self._write_batch_to_file(batch_mappings, output_path)
                            batch_mappings.clear()
                            
                            if self.enable_gc and doc_count % (self.batch_size * 10) == 0:
                                gc.collect()
                                
                    except EOFError:
                        break
            
            # Write remaining batch
            if batch_mappings:
                self._write_batch_to_file(batch_mappings, output_path)
            
            # Write file footer
            with open(output_path, 'a') as f:
                f.write('}}\n')
            
            self.logger.info(f"Created search mapping for {doc_count} queries")
            
        except Exception as e:
            self.logger.error(f"Error initializing search mapping: {e}")
            raise

    def validate_mappings(self) -> bool:
        """Validate index mappings with streaming checks."""
        self.logger.info("Validating index mappings...")
        
        try:
            mapping_files = {
                'corpus': 'corpus_idx_mapping.json',
                'dataset': 'dataset_idx_mapping.json',
                'search': 'search_idx_mapping.json'
            }
            
            valid = True
            
            for name, filename in mapping_files.items():
                file_path = paths.MAPPINGS_DIR / filename
                if not file_path.exists():
                    self.logger.error(f"Missing mapping file: {filename}")
                    valid = False
                    continue
                
                try:
                    # Validate file structure using ijson
                    with open(file_path, 'rb') as f:
                        parser = ijson.parse(f)
                        has_metadata = False
                        has_mappings = False
                        
                        for prefix, event, _ in parser:
                            if prefix == 'metadata':
                                has_metadata = True
                            elif prefix == 'mappings':
                                has_mappings = True
                            
                            if has_metadata and has_mappings:
                                break
                        
                        if not (has_metadata and has_mappings):
                            self.logger.error(f"Invalid structure in {filename}")
                            valid = False
                            
                except Exception as e:
                    self.logger.error(f"Error validating {filename}: {e}")
                    valid = False
            
            return valid
            
        except Exception as e:
            self.logger.error(f"Error validating mappings: {e}")
            return False

    def cleanup_old_mappings(self):
        """Clean up old mapping files."""
        try:
            for file in paths.MAPPINGS_DIR.glob('*_mapping*.json'):
                file.unlink()
            self.logger.info("Cleaned up old mapping files")
        except Exception as e:
            self.logger.warning(f"Error cleaning up old mappings: {e}")

def main():
    """Main execution function with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="Initialize index mappings for RAG experiments"
    )
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Name of the experiment'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        choices=['contriever', 'bm25', 'adore'],
        required=True,
        help='Type of retriever to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for processing (default: 1000)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Optional custom output directory for mappings'
    )
    parser.add_argument(
        '--enable_gc',
        type=bool,
        default=True,
        help='Enable garbage collection for memory optimization'
    )
    
    args = parser.parse_args()
    console = Console()
    
    try:
        # Override mappings directory if specified
        if args.output_dir:
            paths.MAPPINGS_DIR = Path(args.output_dir)
            paths.MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize  
        initializer = IndexInitializer(
            experiment=args.experiment,
            retriever=args.retriever,
            batch_size=args.batch_size,
            enable_gc=args.enable_gc
        )
        
        # Clean up old mappings
        initializer.cleanup_old_mappings()
        
        with console.status("[bold green]Initializing mappings...") as status:
            # Create corpus mapping
            status.update("[bold blue]Creating corpus mapping...")
            initializer.initialize_corpus_mapping()
            
            # Create dataset mapping
            status.update("[bold blue]Creating dataset mapping...")
            initializer.initialize_dataset_mapping()
            
            # Create search mapping
            status.update("[bold blue]Creating search mapping...")
            initializer.initialize_search_mapping()
            
            # Validate all mappings
            status.update("[bold yellow]Validating mappings...")
            if not initializer.validate_mappings():
                raise ValueError("Mapping validation failed")
        
        # Print success message
        console.print("\n[bold green]✓ Index initialization completed successfully")
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error in index initialization: {str(e)}")
        logging.error(f"Error in index initialization: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()