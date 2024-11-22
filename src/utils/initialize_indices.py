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
        enable_gc: bool = True,
        start_idx: int = 0
    ):
        self.setup_logging()
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
        self.experiment = experiment
        self.retriever = retriever
        self.batch_size = batch_size
        self.enable_gc = enable_gc
        self.start_idx = start_idx
        
        # Initialize ID mapper with batching support
        self.id_mapper = IDMapper(
            experiment_name=f"{experiment}_{retriever}",
            batch_size=batch_size,
            enable_gc=enable_gc
        )
        
        self.logger.info(
            f"Initializing indices for {experiment} experiment with "
            f"{retriever} retriever (batch_size={batch_size}, start_idx={start_idx})"
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
        """Write a batch of mappings to file with proper JSON formatting."""
        try:
            with open(output_file, mode) as f:
                for idx, item in batch_data.items():
                    line = f'"{idx}": {json.dumps(item)}'
                    if mode == 'w':  # First batch
                        f.write('{\n')
                        f.write('"metadata": {\n')
                        f.write(f'"creation_time": "{datetime.now().isoformat()}",\n')
                        f.write(f'"experiment": "{self.experiment}",\n')
                        f.write(f'"batch_size": {self.batch_size}\n')
                        f.write('},\n')
                        f.write('"mappings": {\n')
                    f.write(line + ',\n')
        except Exception as e:
            self.logger.error(f"Error writing batch to file: {e}")
            raise

    def _write_mappings_batch(self, batch_mappings: Dict, output_file: Path, is_first: bool):
        """Write a batch of mappings with proper JSON formatting."""
        try:
            with open(output_file, 'a') as f:
                for idx, (key, mapping) in enumerate(batch_mappings.items()):
                    # Add comma if not first item
                    if not is_first or idx > 0:
                        f.write(',\n')
                    # Write mapping entry
                    f.write(f'    "{key}": {json.dumps(mapping)}')
                    
        except Exception as e:
            self.logger.error(f"Error writing mappings batch: {e}")
            raise

    def initialize_corpus_mapping(self) -> None:
        """Initialize corpus index mapping with streaming processing."""
        self.logger.info("Initializing corpus index mapping...")
        
        try:
            output_path = paths.MAPPINGS_DIR / 'corpus_idx_mapping.json'
            temp_path = output_path.with_suffix('.tmp')
            
            # Track whether we're writing the first item
            is_first = True
            doc_count = 0
            
            # Write header to temp file
            with open(temp_path, 'w') as f:
                f.write('{\n')
                f.write('  "metadata": {\n')
                f.write(f'    "creation_time": "{datetime.now().isoformat()}",\n')
                f.write(f'    "experiment": "{self.experiment}",\n')
                f.write(f'    "retriever": "{self.retriever}",\n')
                f.write(f'    "batch_size": {self.batch_size}\n')
                f.write('  },\n')
                f.write('  "mappings": {\n')
            
            # Process corpus in batches
            batch_mappings = {}
            for doc in track(self._iterate_corpus(paths.CORPUS_JSON), description="Processing corpus"):
                # Create mapping entry
                batch_mappings[str(doc_count)] = {
                    'corpus_idx': doc_count,
                    'title': doc.get('title', ''),
                    'embedding_idx': None
                }
                doc_count += 1
                
                # Write batch when full
                if len(batch_mappings) >= self.batch_size:
                    self._write_mappings_batch(batch_mappings, temp_path, is_first)
                    is_first = False
                    batch_mappings.clear()
                    
                    # Memory cleanup
                    if self.enable_gc and doc_count % (self.batch_size * 10) == 0:
                        gc.collect()
            
            # Write remaining batch
            if batch_mappings:
                self._write_mappings_batch(batch_mappings, temp_path, is_first)
            
            # Write footer
            with open(temp_path, 'a') as f:
                f.write('\n  }\n}')
            
            # Validate JSON before finalizing
            if self._validate_json_file(temp_path):
                # Replace original file with temp file
                if output_path.exists():
                    output_path.unlink()
                temp_path.rename(output_path)
                self.logger.info(f"Created corpus mapping with {doc_count} documents")
            else:
                raise ValueError("Generated JSON validation failed")
                
        except Exception as e:
            self.logger.error(f"Error initializing corpus mapping: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _read_dataset_json(self, file_path: str) -> List[Dict]:
        """Read JSON dataset with memory efficient streaming."""
        try:
            items = []
            with open(file_path, 'r') as f:
                # Skip first character if it's a '['
                f.seek(1) if f.read(1) == '[' else f.seek(0)
                
                current_item = ""
                bracket_count = 0
                
                while True:
                    chunk = f.read(8192)  # Read in chunks
                    if not chunk:
                        break
                        
                    for char in chunk:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            
                        current_item += char
                        
                        if bracket_count == 0 and current_item.strip():
                            # Clean up the item string
                            clean_item = current_item.strip().rstrip(',')
                            if clean_item:
                                try:
                                    item_dict = json.loads(clean_item)
                                    items.append(item_dict)
                                except json.JSONDecodeError:
                                    pass
                            current_item = ""
                            
            return items
        except Exception as e:
            self.logger.error(f"Error reading dataset file {file_path}: {e}")
            return []

    def _process_dataset_file(
        self,
        file_path: str, 
        split: str,
        output_file: Path,
        file_mode: str = 'a'
    ) -> int:
        """Process a dataset file in chunks."""
        doc_count = 0
        batch_mappings = {}
        first_write = file_mode == 'w'
        
        try:
            if not Path(file_path).exists():
                self.logger.warning(f"Dataset file not found: {file_path}")
                return 0
                
            # Read dataset entries
            items = self._read_dataset_json(file_path)
            
            for item in items:
                batch_mappings[str(doc_count)] = {
                    'dataset_idx': doc_count,
                    'split': split,
                    'example_id': item.get('example_id', str(doc_count)),
                    'corpus_idx': item.get('idx_gold_in_corpus', -1),
                    'retrieval_idx': None
                }
                doc_count += 1
                
                if len(batch_mappings) >= self.batch_size:
                    if first_write:
                        # Write header for first batch
                        with open(output_file, 'w') as out_f:
                            out_f.write('{\n')
                            out_f.write('  "metadata": {\n')
                            out_f.write(f'    "creation_time": "{datetime.now().isoformat()}",\n')
                            out_f.write(f'    "experiment": "{self.experiment}",\n')
                            out_f.write(f'    "batch_size": {self.batch_size}\n')
                            out_f.write('  },\n')
                            out_f.write('  "mappings": {\n')
                        first_write = False
                    
                    # Write batch
                    with open(output_file, 'a') as out_f:
                        for map_idx, mapping in batch_mappings.items():
                            out_f.write(f'    "{map_idx}": {json.dumps(mapping)}')
                            if doc_count < len(items):
                                out_f.write(',\n')
                            
                    batch_mappings.clear()
                    
                    if self.enable_gc and doc_count % (self.batch_size * 10) == 0:
                        gc.collect()
            
            # Write remaining batch
            if batch_mappings:
                with open(output_file, 'a') as out_f:
                    if first_write:
                        out_f.write('{\n')
                        out_f.write('  "metadata": {\n')
                        out_f.write(f'    "creation_time": "{datetime.now().isoformat()}",\n')
                        out_f.write(f'    "experiment": "{self.experiment}",\n')
                        out_f.write(f'    "batch_size": {self.batch_size}\n')
                        out_f.write('  },\n')
                        out_f.write('  "mappings": {\n')
                    
                    for map_idx, mapping in batch_mappings.items():
                        out_f.write(f'    "{map_idx}": {json.dumps(mapping)}')
                        if map_idx != list(batch_mappings.keys())[-1]:
                            out_f.write(',\n')
                        else:
                            out_f.write('\n')
            
            return doc_count
                
        except Exception as e:
            self.logger.error(f"Error processing dataset file {file_path}: {e}")
            raise

    def _write_dataset_batch(
        self,
        batch_data: Dict,
        output_file: Path,
        split: str,
        is_first: bool
    ):
        """Write dataset batch with proper JSON formatting."""
        try:
            with open(output_file, 'a') as f:
                for idx, (key, mapping) in enumerate(batch_data.items()):
                    # Add comma if not first entry
                    if not is_first or idx > 0:
                        f.write(',\n')
                    # Write mapping entry with proper indentation
                    f.write(f'    "{key}": {json.dumps(mapping, separators=(",", ":"))}'.strip())
        except Exception as e:
            self.logger.error(f"Error writing dataset batch: {e}")
            raise

    def _process_dataset_line_by_line(
        self,
        file_path: str,
        split: str,
        output_file: Path,
        is_first: bool = True
    ) -> int:
        """Process dataset file line by line for malformed JSON."""
        doc_count = 0
        batch_data = {}
        first_in_file = is_first
        
        try:
            with open(file_path, 'r') as f:
                # Skip first line if it's an opening bracket
                first_line = f.readline().strip()
                if first_line != '[':
                    f.seek(0)
                
                for line in f:
                    line = line.strip()
                    if not line or line in ['[', ']']:
                        continue
                        
                    # Remove trailing comma if present
                    if line.endswith(','):
                        line = line[:-1]
                        
                    try:
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            continue
                            
                        example_id = item.get('example_id', str(doc_count))
                        corpus_idx = item.get('idx_gold_in_corpus', -1)
                        
                        batch_data[str(doc_count)] = {
                            'dataset_idx': doc_count,
                            'split': split,
                            'example_id': example_id,
                            'corpus_idx': corpus_idx,
                            'retrieval_idx': None
                        }
                        doc_count += 1
                        
                        if len(batch_data) >= self.batch_size:
                            self._write_dataset_batch(
                                batch_data,
                                output_file,
                                split,
                                first_in_file
                            )
                            first_in_file = False
                            batch_data.clear()
                            
                            if self.enable_gc and doc_count % (self.batch_size * 10) == 0:
                                gc.collect()
                                
                    except json.JSONDecodeError:
                        continue
                        
            # Write remaining batch
            if batch_data:
                self._write_dataset_batch(
                    batch_data,
                    output_file,
                    split,
                    first_in_file
                )
                
            return doc_count
            
        except Exception as e:
            self.logger.error(f"Error in line-by-line processing: {e}")
            raise

    def _process_dataset_batch(
        self,
        file_path: str,
        split: str,
        output_file: Path,
        is_first: bool = True
    ) -> int:
        """Process dataset file in batches with robust JSON handling."""
        doc_count = 0
        
        try:
            if not Path(file_path).exists():
                self.logger.warning(f"Dataset file not found: {file_path}")
                return 0
            
            # Initialize batch tracking
            batch_data = {}
            first_in_file = is_first
            current_item = {}
            
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
                
                try:
                    # Parse JSON data
                    data = json.loads(content)
                    if isinstance(data, list):
                        items = data
                    else:
                        items = [data]
                        
                    # Process items in batches
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                            
                        # Extract required fields
                        example_id = item.get('example_id', str(doc_count))
                        corpus_idx = item.get('idx_gold_in_corpus', -1)
                        
                        # Create mapping entry
                        batch_data[str(doc_count)] = {
                            'dataset_idx': doc_count,
                            'split': split,
                            'example_id': example_id,
                            'corpus_idx': corpus_idx,
                            'retrieval_idx': None
                        }
                        doc_count += 1
                        
                        # Write batch when full
                        if len(batch_data) >= self.batch_size:
                            self._write_dataset_batch(
                                batch_data,
                                output_file,
                                split,
                                first_in_file
                            )
                            first_in_file = False
                            batch_data.clear()
                            
                            # Memory cleanup
                            if self.enable_gc and doc_count % (self.batch_size * 10) == 0:
                                gc.collect()
                                
                    # Write remaining batch
                    if batch_data:
                        self._write_dataset_batch(
                            batch_data,
                            output_file,
                            split,
                            first_in_file
                        )
                        
                except json.JSONDecodeError as e:
                    # Fall back to line-by-line processing if JSON is malformed
                    self.logger.warning(f"Falling back to line-by-line processing for {file_path}")
                    return self._process_dataset_line_by_line(file_path, split, output_file, is_first)
                    
            return doc_count
            
        except Exception as e:
            self.logger.error(f"Error processing dataset file {file_path}: {e}")
            raise

    def _validate_dataset_file(self, file_path: str) -> bool:
        """Validate dataset file format."""
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                
                # Check if file is empty
                if not content:
                    return False
                    
                # Try to parse as JSON array
                try:
                    data = json.loads(content)
                    return isinstance(data, list)
                except json.JSONDecodeError:
                    # Try line-by-line validation
                    f.seek(0)
                    first_line = f.readline().strip()
                    if first_line != '[':
                        return False
                        
                    valid_lines = 0
                    for line in f:
                        line = line.strip()
                        if not line or line in ['[', ']']:
                            continue
                            
                        # Remove trailing comma if present
                        if line.endswith(','):
                            line = line[:-1]
                            
                        try:
                            item = json.loads(line)
                            if isinstance(item, dict):
                                valid_lines += 1
                        except json.JSONDecodeError:
                            continue
                            
                    return valid_lines > 0
                    
            return True
        except Exception as e:
            self.logger.error(f"Error validating dataset file {file_path}: {e}")
            return False
        

    def initialize_dataset_mapping(self) -> None:
        """Initialize dataset index mapping with streaming processing and proper JSON formatting."""
        self.logger.info("Initializing dataset index mapping...")
        
        try:
            # Validate dataset files first
            train_path = str(paths.TRAIN_DATASET_PATH)
            test_path = str(paths.TEST_DATASET_PATH)
            
            if not self._validate_dataset_file(train_path):
                raise ValueError(f"Invalid dataset file format: {train_path}")
            if not self._validate_dataset_file(test_path):
                raise ValueError(f"Invalid dataset file format: {test_path}")
            
            output_path = paths.MAPPINGS_DIR / 'dataset_idx_mapping.json'
            temp_path = output_path.with_suffix('.tmp')
            
            # Write header
            with open(temp_path, 'w') as f:
                f.write('{\n')
                f.write('  "metadata": {\n')
                f.write(f'    "creation_time": "{datetime.now().isoformat()}",\n')
                f.write(f'    "experiment": "{self.experiment}",\n')
                f.write(f'    "retriever": "{self.retriever}",\n')
                f.write(f'    "batch_size": {self.batch_size}\n')
                f.write('  },\n')
                f.write('  "mappings": {\n')
            
            # Process train data
            train_count = self._process_dataset_batch(
                str(paths.TRAIN_DATASET_PATH),
                'train',
                temp_path,
                is_first=True
            )
            
            # Process test data
            test_count = self._process_dataset_batch(
                str(paths.TEST_DATASET_PATH),
                'test',
                temp_path,
                is_first=False
            )
            
            # Write footer
            with open(temp_path, 'a') as f:
                f.write('\n  }\n}')
            
            # Validate JSON before finalizing
            if self._validate_json_file(temp_path):
                # Replace original file with temp file
                if output_path.exists():
                    output_path.unlink()
                temp_path.rename(output_path)
                self.logger.info(
                    f"Created dataset mapping for {train_count} train "
                    f"and {test_count} test examples"
                )
            else:
                raise ValueError("Generated JSON validation failed")
                
        except Exception as e:
            self.logger.error(f"Error initializing dataset mapping: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _write_search_batch(
        self,
        batch_data: Dict,
        output_file: Path,
        is_first: bool
    ):
        """Write search batch with proper JSON formatting."""
        try:
            with open(output_file, 'a') as f:
                for idx, (key, mapping) in enumerate(batch_data.items()):
                    # Add comma if not first entry
                    if not is_first or idx > 0:
                        f.write(',\n')
                    # Write mapping entry
                    f.write(f'    "{key}": {json.dumps(mapping)}')
        except Exception as e:
            self.logger.error(f"Error writing search batch: {e}")
            raise

    def initialize_search_mapping(self) -> None:
        """Initialize search results index mapping with streaming processing."""
        self.logger.info("Initializing search results mapping...")
        
        try:
            # Determine correct results path based on retriever
            results_path = {
                'contriever': paths.CONTRIEVER_RESULTS_PATH,
                'bm25': paths.BM25_RESULTS_PATH,
                'adore': paths.ADORE_RESULTS_PATH
            }.get(self.retriever)
            
            if not results_path or not results_path.exists():
                raise ValueError(f"No search results found for {self.retriever}")
            
            output_path = paths.MAPPINGS_DIR / 'search_idx_mapping.json'
            temp_path = output_path.with_suffix('.tmp')
            
            # Write header
            with open(temp_path, 'w') as f:
                f.write('{\n')
                f.write('  "metadata": {\n')
                f.write(f'    "creation_time": "{datetime.now().isoformat()}",\n')
                f.write(f'    "experiment": "{self.experiment}",\n')
                f.write(f'    "retriever": "{self.retriever}",\n')
                f.write(f'    "batch_size": {self.batch_size}\n')
                f.write('  },\n')
                f.write('  "mappings": {\n')
            
            # Process search results in batches
            doc_count = 0
            batch_data = {}
            first_batch = True
            
            with open(results_path, 'rb') as f:
                while True:
                    try:
                        # Read results in batches
                        batch = []
                        for _ in range(self.batch_size):
                            try:
                                result = pickle.load(f)
                                batch.append(result)
                            except EOFError:
                                break
                        
                        if not batch:
                            break
                        
                        # Process batch
                        for result in batch:
                            if isinstance(result, tuple) and len(result) == 2:
                                doc_indices, scores = result
                                
                                # Convert arrays to lists if needed
                                if isinstance(doc_indices, np.ndarray):
                                    doc_indices = doc_indices.tolist()
                                if isinstance(scores, np.ndarray):
                                    scores = scores.tolist()
                                
                                batch_data[str(doc_count)] = {
                                    'query_idx': doc_count,
                                    'corpus_indices': doc_indices,
                                    'scores': scores,
                                    'retrieval_meta': {
                                        'timestamp': datetime.now().isoformat(),
                                        'top_k': len(doc_indices)
                                    }
                                }
                                doc_count += 1
                        
                        # Write batch
                        if batch_data:
                            self._write_search_batch(batch_data, temp_path, first_batch)
                            first_batch = False
                            batch_data.clear()
                            
                            # Memory cleanup
                            if self.enable_gc and doc_count % (self.batch_size * 10) == 0:
                                gc.collect()
                                
                    except EOFError:
                        break
            
            # Write remaining batch
            if batch_data:
                self._write_search_batch(batch_data, temp_path, first_batch)
            
            # Write footer
            with open(temp_path, 'a') as f:
                f.write('\n  }\n}')
            
            # Validate and finalize
            if self._validate_json_file(temp_path):
                if output_path.exists():
                    output_path.unlink()
                temp_path.rename(output_path)
                self.logger.info(f"Created search mapping for {doc_count} queries")
            else:
                raise ValueError("Generated JSON validation failed")
                
        except Exception as e:
            self.logger.error(f"Error initializing search mapping: {e}")
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            raise

    def _validate_json_file(self, file_path: Path) -> bool:
        """Validate JSON file structure in chunks."""
        try:
            with open(file_path, 'r') as f:
                # Read and parse in chunks
                parser = ijson.parse(f)
                for prefix, event, value in parser:
                    pass  # Just iterate through to validate
            return True
        except Exception as e:
            self.logger.error(f"JSON validation failed for {file_path}: {e}")
            return False

    def check_index_mappings(self, experiment: str, retriever: str) -> bool:
        """Check the validity of index mapping files."""
        self.logger.info(f"Checking index mapping files for {experiment}/{retriever}...")
        
        mapping_dir = f"data/mappings/temp/{experiment}/{retriever}"
        required_mappings = [
            "corpus_idx_mapping.json",
            "dataset_idx_mapping.json",
            "search_idx_mapping.json"
        ]
        
        try:
            for mapping in required_mappings:
                file_path = Path(mapping_dir) / mapping
                if not file_path.exists():
                    self.logger.warning(f"Missing index mapping: {file_path}")
                    return False
                    
                # Read and validate the entire file
                with open(file_path) as f:
                    try:
                        # Load and validate the JSON structure
                        data = json.load(f)
                        
                        # Verify required sections
                        if "metadata" not in data:
                            self.logger.warning(f"Missing metadata section in {mapping}")
                            return False
                        if "mappings" not in data:
                            self.logger.warning(f"Missing mappings section in {mapping}")
                            return False
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON structure in {mapping}: {str(e)}")
                        return False
                        
            # Create final index file
            final_mappings = {
                'metadata': {
                    'creation_time': datetime.now().isoformat(),
                    'experiment': experiment,
                    'retriever': retriever
                },
                'mappings': {}
            }
            
            # Combine all mappings
            for mapping in required_mappings:
                file_path = Path(mapping_dir) / mapping
                with open(file_path) as f:
                    data = json.load(f)
                    mapping_type = mapping.replace('_mapping.json', '')
                    final_mappings['mappings'][mapping_type] = data['mappings']
                    
            # Save combined mappings
            output_file = Path(mapping_dir) / "final_indices.json"
            with open(output_file, 'w') as f:
                json.dump(final_mappings, f, indent=2)
                
            self.logger.info("Index mapping files verified successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during index mapping verification: {str(e)}")
            return False

    def initialize_indices(self) -> bool:
        """Initialize and validate all indices."""
        try:
            # Initialize corpus mapping
            self.initialize_corpus_mapping()
            
            # Initialize dataset mapping
            self.initialize_dataset_mapping()
            
            # Initialize search mapping
            self.initialize_search_mapping()
            
            # Create empty mappings if needed
            mapping_dir = Path(f"data/mappings/temp/{self.experiment}/{self.retriever}")
            mapping_dir.mkdir(parents=True, exist_ok=True)
            
            empty_structure = {
                "metadata": {
                    "creation_time": datetime.now().isoformat(),
                    "experiment": self.experiment,
                    "retriever": self.retriever,
                    "batch_size": self.batch_size
                },
                "mappings": {}
            }
            
            for mapping_type in ["corpus", "dataset", "search"]:
                file_path = mapping_dir / f"{mapping_type}_idx_mapping.json"
                if not file_path.exists():
                    with open(file_path, 'w') as f:
                        json.dump(empty_structure, f, indent=2)
                        
            # Validate mappings
            if not self.validate_mappings():
                return False
                
            # Create final indices file
            final_mappings = {
                "metadata": empty_structure["metadata"],
                "mappings": {}
            }
            
            for mapping_type in ["corpus", "dataset", "search"]:
                file_path = mapping_dir / f"{mapping_type}_idx_mapping.json"
                if file_path.exists():
                    with open(file_path) as f:
                        data = json.load(f)
                        final_mappings["mappings"][mapping_type] = data["mappings"]
                        
            with open(mapping_dir / "final_indices.json", 'w') as f:
                json.dump(final_mappings, f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing indices: {e}")
            return False

    def validate_mappings(self) -> bool:
        """Validate all mapping files."""
        self.logger.info("Validating index mappings...")
        
        try:
            mapping_dir = Path(f"data/mappings/temp/{self.experiment}/{self.retriever}")
            for mapping_type in ["corpus", "dataset", "search"]:
                file_path = mapping_dir / f"{mapping_type}_idx_mapping.json"
                
                if not file_path.exists():
                    self.logger.error(f"Missing mapping file: {file_path}")
                    return False
                    
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                        
                    # Verify structure
                    if "metadata" not in data:
                        self.logger.error(f"Missing metadata in {file_path}")
                        return False
                    if "mappings" not in data:
                        self.logger.error(f"Missing mappings in {file_path}")
                        return False
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in {file_path}: {str(e)}")
                    return False
                    
            return True
            
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
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Starting index for batch processing'
    )
    
    args = parser.parse_args()
    console = Console()
    
    try:
        # Override mappings directory if specified
        if args.output_dir:
            paths.MAPPINGS_DIR = Path(args.output_dir)
            paths.MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize with start_idx 
        initializer = IndexInitializer(
            experiment=args.experiment,
            retriever=args.retriever,
            batch_size=args.batch_size,
            enable_gc=args.enable_gc,
            start_idx=args.start_idx
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