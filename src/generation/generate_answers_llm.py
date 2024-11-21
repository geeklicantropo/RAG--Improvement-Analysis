import logging
import logging.config
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
from tqdm import tqdm
from rich.progress import track, Progress
from rich.console import Console
import gc
import argparse
import json
import tempfile
import os
import mmap
from contextlib import contextmanager
from queue import Queue
from threading import Thread

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.llm import LLM, GenerationResult
from src.utils.id_mapping import IDMapper
from src.core.env.env_loader import load_environment
from src.core.utils.system_check import perform_system_checks
from src.config import paths, settings

@dataclass
class GenerationTask:
    """Represents a generation task with memory-efficient tracking."""
    query: str
    context: str
    query_idx: int
    document_indices: List[int]
    prompt_tokens: int
    metadata: Optional[Dict[str, Any]] = None

class AnswerGenerator:
    """
    Memory-efficient implementation for generating answers using LLMs.
    """
    def __init__(
        self,
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        batch_size: int = 8,
        id_mapper: Optional[IDMapper] = None,
        enable_gc: bool = True,
        max_queue_size: int = 100
    ):
        self.setup_logging()
        self.console = Console()
        self.logger = logging.getLogger(f"{__name__}.{experiment_name}")
        
        # Initialize parameters
        self.model_id = model_id or settings.LLM_MODEL_ID
        self.experiment_name = experiment_name or f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.batch_size = batch_size
        self.enable_gc = enable_gc
        
        # Setup cache directory
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="generation_")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize components
        self.id_mapper = id_mapper or IDMapper(
            experiment_name=self.experiment_name,
            batch_size=batch_size,
            enable_gc=enable_gc
        )
        
        # Setup task queue
        self.task_queue = Queue(maxsize=max_queue_size)
        
        # Setup metrics tracking with memory mapping
        self.metrics_file = Path(self.cache_dir) / "metrics.mmap"
        self.metrics = self._init_metrics()
        
        # Create experiment directories
        self.setup_directories()
        
        self.logger.info(
            f"Initialized AnswerGenerator with model {self.model_id} "
            f"(batch_size={batch_size})"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        with open(paths.LOGGING_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)

    def setup_directories(self):
        """Create necessary directories."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.results_dir = Path('results/generation') / timestamp
            self.temp_dir = self.results_dir / "temp"
            self.cache_dir = self.results_dir / "cache"
            
            for dir_path in [self.results_dir, self.temp_dir, self.cache_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
                
            self.logger.info(f"Created results directory: {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            raise

    def _init_metrics(self) -> Dict[str, Any]:
        """Initialize metrics with memory mapping."""
        base_metrics = {
            'experiment_type': 'answer_generation',
            'model_id': self.model_id,
            'start_time': datetime.now().isoformat(),
            'batch_size': self.batch_size,
            'processing_stats': defaultdict(float),
            'generation_metrics': []
        }
        
        # Save initial metrics to memory-mapped file
        with open(self.metrics_file, 'wb') as f:
            f.write(json.dumps(base_metrics).encode())
            
        return base_metrics

    @contextmanager
    def _update_metrics(self):
        """Context manager for thread-safe metrics updates."""
        try:
            with open(self.metrics_file, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                metrics = json.loads(mm.read().decode())
                yield metrics
                
                # Write updated metrics
                mm.seek(0)
                mm.write(json.dumps(metrics).encode())
                mm.flush()
        finally:
            if 'mm' in locals():
                mm.close()

    def initialize_model(self):
        """Initialize LLM with memory optimization."""
        try:
            self.logger.info("Initializing model...")
            
            # Initialize device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize LLM
            self.model = LLM(
                model_id=self.model_id,
                device=device,
                quantization_bits=4,
                model_max_length=settings.MODEL_MAX_LENGTH,
                experiment_name=self.experiment_name
            )
            
            # Update metrics
            with self._update_metrics() as metrics:
                metrics.update({
                    'model_max_length': settings.MODEL_MAX_LENGTH,
                    'device': device
                })
            
            self.logger.info("Model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def _stream_dataset(self, dataset_path: str) -> Iterator[Dict]:
        """Stream dataset with memory efficiency."""
        try:
            with open(dataset_path, 'r') as f:
                # Skip header
                f.readline()
                
                for line in f:
                    if line.strip() == ']':  # End of array
                        break
                        
                    # Remove trailing comma if present
                    line = line.rstrip(',\n')
                    
                    try:
                        item = json.loads(line)
                        yield item
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing JSON: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error streaming dataset: {e}")
            raise

    def _process_batch(
        self,
        batch: List[Dict],
        process_fn: callable,
        desc: str = "Processing batch"
    ) -> List[Any]:
        """Process batch with memory optimization."""
        results = []
        
        with Progress() as progress:
            task = progress.add_task(desc, total=len(batch))
            
            for item in batch:
                try:
                    result = process_fn(item)
                    if result:
                        results.append(result)
                    progress.advance(task)
                    
                    # Clean memory periodically
                    if self.enable_gc and len(results) % (self.batch_size * 2) == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    self.logger.error(f"Error processing item: {e}")
                    continue
                    
        return results

    def _save_batch_results(
        self,
        results: List[Dict],
        batch_idx: int
    ):
        """Save batch results with memory efficiency."""
        try:
            # Save to temporary file first
            temp_file = self.temp_dir / f"batch_{batch_idx}_temp.json"
            results_file = self.results_dir / f"batch_{batch_idx}.json"
            
            # Write results in chunks
            with open(temp_file, 'w') as f:
                f.write('[\n')
                
                for idx, result in enumerate(results):
                    f.write(json.dumps(result))
                    if idx < len(results) - 1:
                        f.write(',\n')
                    
                f.write('\n]')
            
            # Rename temp file to final file
            os.replace(temp_file, results_file)
            
            # Update metrics
            with self._update_metrics() as metrics:
                batch_metrics = {
                    'batch_idx': batch_idx,
                    'batch_size': len(results),
                    'timestamp': datetime.now().isoformat()
                }
                metrics['generation_metrics'].append(batch_metrics)
                
            self.logger.info(f"Saved results for batch {batch_idx}")
            
        except Exception as e:
            self.logger.error(f"Error saving batch results: {e}")
            # Clean up temp file if it exists
            if 'temp_file' in locals():
                try:
                    temp_file.unlink()
                except:
                    pass
            raise

    def generate_answers(
        self,
        dataset_path: str,
        num_samples: int = -1,
        batch_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None
    ):
        """
        Generate answers with memory-efficient batch processing.
        """
        try:
            # Initialize model
            self.initialize_model()
            
            batch_size = batch_size or self.batch_size
            max_new_tokens = max_new_tokens or settings.MAX_NEW_TOKENS
            
            # Process dataset in streaming fashion
            total_processed = 0
            batch_data = []
            
            for item in self._stream_dataset(dataset_path):
                batch_data.append(item)
                
                if len(batch_data) >= batch_size:
                    # Process batch
                    results = self._process_batch(
                        batch_data,
                        lambda x: self._generate_answer(x, max_new_tokens),
                        f"Processing batch {total_processed // batch_size + 1}"
                    )
                    
                    if results:
                        # Save batch results
                        self._save_batch_results(
                            results,
                            total_processed // batch_size
                        )
                    
                    total_processed += len(batch_data)
                    batch_data = []
                    
                    # Check sample limit
                    if num_samples > 0 and total_processed >= num_samples:
                        break
                    
                    # Memory cleanup
                    if self.enable_gc:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            # Process remaining items
            if batch_data:
                results = self._process_batch(
                    batch_data,
                    lambda x: self._generate_answer(x, max_new_tokens),
                    "Processing final batch"
                )
                
                if results:
                    self._save_batch_results(
                        results,
                        total_processed // batch_size
                    )
            
            # Save final metrics
            self._save_final_results()
            
        except Exception as e:
            self.logger.error(f"Error generating answers: {e}")
            raise

    def _generate_answer(
        self,
        item: Dict,
        max_new_tokens: int
    ) -> Optional[Dict]:
        """Generate answer for a single item."""
        try:
            # Extract information
            query = item['question']
            query_idx = int(item.get('example_id', -1))
            
            # Start timing
            start_time = datetime.now()
            
            # Generate answer
            prompt = f"Question: {query}\nAnswer:"
            generation_result = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens
            )
            
            # Create result
            result = {
                'query': query,
                'query_idx': query_idx,
                'generated_answer': generation_result.text,
                'tokens_used': generation_result.tokens_used,
                'generation_time': (datetime.now() - start_time).total_seconds(),
                'prompt_tokens_len': len(prompt.split())
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return None

    def _save_final_results(self):
        """Save final results and metrics."""
        try:
            # Save metrics summary
            metrics_path = self.results_dir / "metrics.json"
            with open(self.metrics_file, 'rb') as f:
                metrics = json.loads(f.read().decode())
                
            metrics.update({
                'end_time': datetime.now().isoformat(),
                'total_duration': (
                    datetime.now() - datetime.fromisoformat(metrics['start_time'])
                ).total_seconds()
            })
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save experiment configuration
            config = {
                'model_id': self.model_id,
                'batch_size': self.batch_size,
                'model_config': {
                    'model_max_length': settings.MODEL_MAX_LENGTH,
                    'max_new_tokens': settings.MAX_NEW_TOKENS
                },
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat()
            }
            
            config_path = self.results_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            self.logger.info(f"Saved final results to {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving final results: {e}")
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            # Remove temp directory
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                for file in self.temp_dir.iterdir():
                    file.unlink()
                self.temp_dir.rmdir()
            
            # Remove cache directory
            if hasattr(self, 'cache_dir') and os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    os.unlink(os.path.join(self.cache_dir, file))
                os.rmdir(self.cache_dir)
            
            # Clean GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate answers with memory-efficient processing"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/gen_res',
        help='Output directory'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default=settings.LLM_MODEL_ID,
        help='Model identifier'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='Number of samples to process (-1 for all)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=None,
        help='Maximum number of new tokens'
    )
    parser.add_argument(
        '--enable_gc',
        type=bool,
        default=True,
        help='Enable garbage collection'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Custom experiment name'
    )
    
    args = parser.parse_args()
    
    try:
        # Load environment
        load_environment()
        
        # Perform system checks
        perform_system_checks()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize and run generator
        with AnswerGenerator(
            model_id=args.model_id,
            experiment_name=args.experiment_name,
            batch_size=args.batch_size,
            enable_gc=args.enable_gc
        ) as generator:
            generator.generate_answers(
                dataset_path=str(paths.TEST_DATASET_PATH),
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens
            )
            
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()