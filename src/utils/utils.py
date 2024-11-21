import os
import json
import torch
import ijson
import pickle
import random
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime 
from typing import Dict, List, Optional, Tuple, Any, Callable
from functools import wraps
import logging
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def get_experiment_dir(experiment_name: str) -> Path:
    """Create and return experiment directory path."""
    exp_dir = PROJECT_ROOT / 'results' / experiment_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Log execution time
        duration = end_time - start_time
        logger.info(f"{func.__name__} took {duration:.2f} seconds to execute")
        
        # Save timing information
        timing_file = PROJECT_ROOT / 'logs' / 'timing' / 'performance_metrics.log'
        timing_file.parent.mkdir(parents=True, exist_ok=True)
        with timing_file.open('a') as f:
            f.write(f"{datetime.now()},{func.__name__},{duration:.2f}\n")
            
        return result
    return wrapper

def seed_everything(seed: int = 10):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logger.info(f"Random seed set to {seed}")

def str2bool(v: Any) -> bool:
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@timing_decorator
def read_pickle(file_path: str) -> Any:
    """Read data from pickle file with timing and logging."""
    logger.info(f"Reading pickle file: {file_path}")
    try:
        with open(file_path, "rb") as reader:
            data = pickle.load(reader)
        logger.debug(f"Successfully read pickle file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading pickle file {file_path}: {e}")
        raise

@timing_decorator
def write_pickle(data: Any, file_path: str):
    """Write data to pickle file with timing and logging."""
    logger.info(f"Writing pickle file: {file_path}")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as writer:
            pickle.dump(data, writer)
        logger.debug(f"Successfully wrote pickle file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing pickle file {file_path}: {e}")
        raise

@timing_decorator
def read_json(file_path: str) -> Any:
    """Read data from JSON file with timing and logging."""
    logger.info(f"Reading JSON file: {file_path}")
    try:
        with open(file_path, "r") as reader:
            data = json.load(reader)
        logger.debug(f"Successfully read JSON file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        raise

@timing_decorator
def write_json(data: Any, file_path: str):
    """Write data to JSON file with timing and logging."""
    logger.info(f"Writing JSON file: {file_path}")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as writer:
            json.dump(data, writer, indent=2)
        logger.debug(f"Successfully wrote JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        raise

@timing_decorator
def read_corpus_json(
    data_path: str, 
    subset_to_full_idx_map: Optional[Dict[int, int]] = None
) -> List[Dict]:
    """Read corpus from JSON file with enhanced logging and timing."""
    logger.info(f"Reading corpus from: {data_path}")
    corpus = []
    try:
        with open(data_path, "r") as f:
            for idx, record in enumerate(tqdm(ijson.items(f, "item"), desc="Reading corpus...")):
                if subset_to_full_idx_map and len(subset_to_full_idx_map) != 0:
                    record['full_corpus_idx'] = subset_to_full_idx_map[idx]
                else:
                    record['full_corpus_idx'] = idx
                corpus.append(record)
        logger.info(f"Successfully read {len(corpus)} documents from corpus")
        return corpus
    except Exception as e:
        logger.error(f"Error reading corpus {data_path}: {e}")
        raise

def read_subset_corpus_with_map(
    full_to_subset_path: str,
    subset_to_full_path: str,
    corpus_path: str
) -> Tuple[List[Dict], Dict[int, int]]:
    """Read subset corpus with mapping."""
    logger.info("Reading subset corpus with mapping")
    full_to_subset_idx_map = read_pickle(full_to_subset_path)
    subset_to_full_idx_map = read_pickle(subset_to_full_path)
    corpus = read_corpus_json(corpus_path, subset_to_full_idx_map)
    return corpus, full_to_subset_idx_map

# Original corpus reading functions with enhanced logging
def read_corpus_with_random():
    """Read corpus with random selection."""
    logger.info("Reading corpus with random selection")
    return read_subset_corpus_with_map(
        "data/mappings/full_to_subset_random_at60_in_corpus.pkl",
        "data/mappings/subset_to_full_random_at60_in_corpus.pkl",
        "data/processed/corpus_with_random_at60.json"
    )

def read_corpus_with_contriever():
    """Read corpus with Contriever."""
    logger.info("Reading corpus with Contriever")
    return read_subset_corpus_with_map(
        "data/mappings/full_to_subset_contriever_at150_in_corpus.pkl",
        "data/mappings/subset_to_full_contriever_at150_in_corpus.pkl",
        "data/processed/corpus_with_contriever_at150.json"
    )

def read_corpus_with_random_and_contriever():
    """Read corpus with random and Contriever."""
    logger.info("Reading corpus with random and Contriever")
    return read_subset_corpus_with_map(
        "data/mappings/full_to_subset_random_contriever_in_corpus.pkl",
        "data/mappings/subset_to_full_random_contriever_in_corpus.pkl",
        "data/processed/corpus_with_random_contriever.json"
    )

def read_test_corpus_with_random_and_bm25():
    """Read test corpus with random and BM25."""
    logger.info("Reading test corpus with random and BM25")
    return read_subset_corpus_with_map(
        "data/mappings/full_to_subset_test_random_bm25_in_corpus.pkl",
        "data/mappings/subset_to_full_test_random_bm25_in_corpus.pkl",
        "data/processed/test_corpus_with_random_bm25.json"
    )

def read_test_corpus_with_random_and_contriever():
    """Read test corpus with random and Contriever."""
    logger.info("Reading test corpus with random and Contriever")
    return read_subset_corpus_with_map(
        "data/mappings/full_to_subset_test_random_contriever_in_corpus.pkl",
        "data/mappings/subset_to_full_test_random_contriever_in_corpus.pkl",
        "data/processed/test_corpus_with_random_contriever.json"
    )

def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get appropriate device for computation."""
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def ensure_directory_exists(directory: Union[str, Path]):
    """Ensure directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def compute_overlap_score(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def compute_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    
    # Convert to sets for overlap computation
    pred_sets = [set(pred.lower().split()) for pred in predictions]
    ref_sets = [set(ref.lower().split()) for ref in references]
    
    # Compute overlap scores
    overlap_scores = [
        compute_overlap_score(pred, ref)
        for pred, ref in zip(pred_sets, ref_sets)
    ]
    
    metrics['mean_overlap'] = np.mean(overlap_scores)
    metrics['std_overlap'] = np.std(overlap_scores)
    
    return metrics

def setup_experiment_directories(
    experiment_name: str,
    base_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Setup directory structure for experiments."""
    if base_dir is None:
        base_dir = PROJECT_ROOT / 'results'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = base_dir / experiment_name / timestamp
    
    directories = {
        'root': exp_dir,
        'models': exp_dir / 'models',
        'results': exp_dir / 'results',
        'logs': exp_dir / 'logs',
        'plots': exp_dir / 'plots',
        'cache': exp_dir / 'cache'
    }
    
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
        
    return directories

class ExperimentTracker:
    """Track experiment progress and metrics."""
    def __init__(
        self, 
        experiment_name: str,
        save_dir: Optional[Path] = None
    ):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.metrics = {}
        self.save_dir = save_dir or PROJECT_ROOT / 'results' / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def log_metric(self, name: str, value: float):
        """Log a metric value."""
        self.metrics[name] = value
        self._save_metrics()
        
    def log_metrics(self, metrics: Dict[str, float]):
        """Log multiple metrics."""
        self.metrics.update(metrics)
        self._save_metrics()
        
    def _save_metrics(self):
        """Save metrics to file."""
        metrics_file = self.save_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def finish(self) -> Dict[str, Any]:
        """Finish experiment and return summary."""
        duration = datetime.now() - self.start_time
        summary = {
            'experiment_name': self.experiment_name,
            'duration': str(duration),
            'metrics': self.metrics
        }
        
        with open(self.save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary