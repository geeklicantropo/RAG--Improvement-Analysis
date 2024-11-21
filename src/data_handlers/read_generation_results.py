import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from dataclasses import dataclass
import numpy as np
import json
from rich.console import Console
from rich.progress import track
from datasets import load_from_disk, Dataset
import gc
import mmap
import os
import tempfile
from contextlib import contextmanager

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.normalization import normalize_answers
from src.core.env.env_loader import load_environment
from src.core.utils.system_check import perform_system_checks
from src.config import paths

@dataclass
class GenerationEvalResult:
    """Stores evaluation results with index-based tracking."""
    query: str
    query_idx: int  # Dataset index
    corpus_indices: List[int]  # Used document indices
    generated_answer: str
    normalized_answer: str
    is_correct: bool
    metrics: Dict[str, Any]

class GenerationResultsReader:
    """Enhanced reader for generation results with memory-efficient processing."""
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        enable_gc: bool = True
    ):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        
        self.experiment_name = experiment_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.batch_size = batch_size
        self.enable_gc = enable_gc
        
        # Setup cache directory
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="gen_results_")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize metrics tracking with memory mapping
        self.metrics_file = Path(self.cache_dir) / "metrics.mmap"
        self._init_metrics()
        
        # Load test dataset with streaming
        self.test_dataset = self._load_test_dataset()
        
        self.logger.info(
            f"Initialized GenerationResultsReader with "
            f"{len(self.test_dataset) if self.test_dataset else 0} test examples"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = paths.LOGS_DIR / "evaluation"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            ]
        )

    def _init_metrics(self):
        """Initialize metrics with memory mapping."""
        base_metrics = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'dataset_size': 0,  # Will be updated later
            'batch_metrics': [],
            'evaluation_stats': defaultdict(float)
        }
        
        with open(self.metrics_file, 'wb') as f:
            f.write(json.dumps(base_metrics).encode())

    @contextmanager
    def _update_metrics(self):
        """Context manager for thread-safe metrics updates."""
        try:
            with open(self.metrics_file, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                metrics = json.loads(mm.read().decode())
                yield metrics
                
                mm.seek(0)
                mm.write(json.dumps(metrics).encode())
                mm.flush()
        finally:
            if 'mm' in locals():
                mm.close()

    def _load_test_dataset(self) -> Optional[Dataset]:
        """Load test dataset with streaming support."""
        try:
            self.logger.info("Loading test dataset...")
            dataset = load_from_disk(str(paths.DATASETS_DIR / "test"))
            
            # Create index mapping if needed
            if 'dataset_idx' not in dataset.features:
                dataset = dataset.map(
                    lambda x, idx: {'dataset_idx': idx},
                    with_indices=True
                )
            
            self.logger.info(f"Loaded test dataset with {len(dataset)} examples")
            
            # Update metrics
            with self._update_metrics() as metrics:
                metrics['dataset_size'] = len(dataset)
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading test dataset: {e}")
            return None

    def _result_file_iterator(
        self,
        results_dir: Path,
        batch_size: Optional[int] = None
    ) -> Iterator[Tuple[Dict, int]]:
        """Memory-efficient iterator for result files."""
        batch_size = batch_size or self.batch_size
        current_batch = []
        
        # Iterate through result files
        for file_path in sorted(results_dir.glob("*.json")):
            try:
                with open(file_path, 'r') as f:
                    # Process file in chunks
                    parser = json.load(f)
                    for item in parser:
                        current_batch.append((item, int(file_path.stem.split('_')[-1])))
                        
                        if len(current_batch) >= batch_size:
                            yield current_batch
                            current_batch = []
                            
                            # Clean memory
                            if self.enable_gc:
                                gc.collect()
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        # Yield remaining items
        if current_batch:
            yield current_batch

    def read_generation_results(
        self,
        results_dir: Union[str, Path],
        normalize_answers_flag: bool = True
    ) -> List[GenerationEvalResult]:
        """Read and evaluate generation results with memory efficiency."""
        try:
            results_dir = Path(results_dir)
            self.logger.info(f"Reading results from {results_dir}")
            
            # Validate directory
            if not results_dir.exists():
                raise FileNotFoundError(f"Results directory not found: {results_dir}")
            
            all_results = []
            total_correct = 0
            corpus_idx_usage = defaultdict(int)
            
            # Process results in batches
            for batch in track(
                self._result_file_iterator(results_dir),
                description="Processing results"
            ):
                for result, file_idx in batch:
                    eval_result = self._evaluate_generation(
                        result,
                        normalize_answers_flag
                    )
                    
                    if eval_result:
                        if eval_result.is_correct:
                            total_correct += 1
                            
                        # Track corpus index usage
                        for idx in eval_result.corpus_indices:
                            corpus_idx_usage[idx] += 1
                            
                        all_results.append(eval_result)
            
            # Update metrics
            with self._update_metrics() as metrics:
                metrics.update({
                    'total_evaluated': len(all_results),
                    'total_correct': total_correct,
                    'accuracy': total_correct / len(all_results) if all_results else 0,
                    'corpus_usage': {
                        'total_unique_docs': len(corpus_idx_usage),
                        'max_doc_reuse': max(corpus_idx_usage.values()),
                        'avg_doc_reuse': sum(corpus_idx_usage.values()) / len(corpus_idx_usage) if corpus_idx_usage else 0
                    }
                })
            
            # Save evaluation results
            self._save_evaluation_results(all_results, results_dir)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error reading generation results: {e}")
            raise

    def _evaluate_generation(
        self,
        result: Dict[str, Any],
        normalize_answers_flag: bool
    ) -> Optional[GenerationEvalResult]:
        """Evaluate a single generation result."""
        try:
            # Get test dataset entry
            dataset_idx = result.get('dataset_idx')
            if dataset_idx is None or dataset_idx >= len(self.test_dataset):
                self.logger.warning(f"Invalid dataset index: {dataset_idx}")
                return None
            
            test_example = self.test_dataset[dataset_idx]
            
            # Normalize answers if requested
            generated_answer = result['generated_answer']
            normalized_generation = (
                normalize_answers.normalize_answer(generated_answer)
                if normalize_answers_flag else generated_answer
            )
            
            # Check correctness
            correct_answers = test_example['answers']
            normalized_answers = [
                normalize_answers.normalize_answer(ans)
                if normalize_answers_flag else ans
                for ans in correct_answers
            ]
            
            is_correct = any(
                norm_gen == norm_truth
                for norm_gen in [normalized_generation]
                for norm_truth in normalized_answers
            )
            
            # Compile metrics
            metrics = {
                'dataset_idx': dataset_idx,
                'corpus_indices_used': result['document_indices'],
                'generation_time': result.get('generation_time', 0),
                'prompt_tokens': result.get('prompt_tokens_len', 0),
                'is_correct': is_correct,
                'attention_patterns': result.get('attention_patterns')
            }
            
            return GenerationEvalResult(
                query=test_example['question'],
                query_idx=dataset_idx,
                corpus_indices=result['document_indices'],
                generated_answer=generated_answer,
                normalized_answer=normalized_generation,
                is_correct=is_correct,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating generation result: {e}")
            return None

    def _save_evaluation_results(
        self,
        results: List[GenerationEvalResult],
        output_dir: Path
    ):
        """Save evaluation results with memory efficiency."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save detailed results in chunks
            results_file = output_dir / f"evaluation_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                # Write header
                f.write('[\n')
                
                # Write results in chunks
                for idx, result in enumerate(results):
                    result_dict = {
                        'query': result.query,
                        'query_idx': result.query_idx,
                        'corpus_indices': result.corpus_indices,
                        'generated_answer': result.generated_answer,
                        'normalized_answer': result.normalized_answer,
                        'is_correct': result.is_correct,
                        'metrics': result.metrics
                    }
                    
                    f.write(json.dumps(result_dict))
                    if idx < len(results) - 1:
                        f.write(',\n')
                        
                    # Clean memory periodically
                    if self.enable_gc and idx % self.batch_size == 0:
                        gc.collect()
                
                # Write footer
                f.write('\n]')
            
            # Save metrics summary
            metrics_file = output_dir / f"evaluation_metrics_{timestamp}.json"
            with open(self.metrics_file, 'rb') as f:
                metrics = json.loads(f.read().decode())
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Saved evaluation results to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            # Remove cache directory
            if hasattr(self, 'cache_dir') and os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    os.unlink(os.path.join(self.cache_dir, file))
                os.rmdir(self.cache_dir)
            
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
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="Read and analyze generation results")
        parser.add_argument(
            '--results_dir',
            type=str,
            required=True,
            help='Directory containing generation results'
        )
        parser.add_argument(
            '--normalize_answers',
            type=bool,
            default=True,
            help='Whether to normalize answers for comparison'
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            help='Batch size for processing'
        )
        parser.add_argument(
            '--enable_gc',
            type=bool,
            default=True,
            help='Enable garbage collection'
        )
        args = parser.parse_args()
        
        # Load environment variables
        load_environment()
        
        # Run result analysis
        with GenerationResultsReader(
            batch_size=args.batch_size,
            enable_gc=args.enable_gc
        ) as reader:
            results = reader.read_generation_results(
                results_dir=args.results_dir,
                normalize_answers_flag=args.normalize_answers
            )
            
            # Display summary
            total = len(results)
            correct = sum(1 for r in results if r.is_correct)
            accuracy = correct / total if total > 0 else 0
            
            print(f"\nResults Summary:")
            print(f"Total Examples: {total}")
            print(f"Correct Answers: {correct}")
            print(f"Accuracy: {accuracy:.4f}")
            
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()