import os
import json
import torch
import ijson
import pickle
import random
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import gc
from experiment_logger import ExperimentLogger

# Seed for reproducibility
def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# String-to-boolean converter
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Setup standardized logging
def setup_logger(experiment_name: str) -> logging.Logger:
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{experiment_name}_{timestamp}.log")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# Enhanced JSON utilities
def save_json(data: Any, file_path: str) -> None:
    logger = logging.getLogger()
    logger.info(f"Saving JSON to {file_path}")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        raise

def read_json(file_path: str) -> Any:
    logger = logging.getLogger()
    logger.info(f"Reading JSON from {file_path}")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON from {file_path}: {str(e)}")
        raise

# Memory-optimized corpus reading
def read_corpus_json(data_path: str, batch_size: int = 1000) -> List[Dict]:
    logger = logging.getLogger()
    logger.info(f"Reading corpus from {data_path}")
    corpus = []

    try:
        with open(data_path, "r") as f:
            parser = ijson.items(f, "item")
            batch = []

            for record in tqdm(parser, desc="Loading corpus in batches"):
                batch.append(record)
                if len(batch) >= batch_size:
                    corpus.extend(batch)
                    batch = []
                    gc.collect()

            if batch:
                corpus.extend(batch)

        logger.info(f"Loaded {len(corpus)} documents.")
        return corpus
    except Exception as e:
        logger.error(f"Error reading corpus from {data_path}: {str(e)}")
        raise

# Add the missing read_pickle function
def read_pickle(file_path: str) -> Any:
    """Reads a pickle file and returns its contents."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger = logging.getLogger()
        logger.error(f"Error reading pickle file from {file_path}: {str(e)}")
        raise

# Dynamic context creation with noise injection
def create_dynamic_context(
    context_ids: List[int],
    noise_ids: List[int],
    noise_ratio: float,
    logger: Optional[logging.Logger] = None
) -> List[int]:
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("Creating dynamic context with noise injection.")
    try:
        num_noise = int(len(context_ids) * noise_ratio)
        if num_noise == 0:
            return context_ids

        noise_positions = random.sample(range(len(context_ids)), num_noise)
        selected_noise = random.sample(noise_ids, num_noise)

        for pos, noise_id in zip(noise_positions, selected_noise):
            context_ids[pos] = noise_id

        logger.info(f"Injected {num_noise} noise documents.")
        return context_ids
    except Exception as e:
        logger.error(f"Error in dynamic context creation: {str(e)}")
        raise

# Validate retrieval results
def validate_retrieval_results(
    results: List[Tuple[List[int], List[float]]],
    corpus_size: int,
    logger: Optional[logging.Logger] = None
) -> bool:
    if logger is None:
        logger = logging.getLogger()

    try:
        logger.info("Validating retrieval results.")
        for doc_ids, scores in results:
            if len(doc_ids) != len(scores):
                raise ValueError("Mismatch in lengths of document IDs and scores.")
            if not all(isinstance(doc_id, int) and 0 <= doc_id < corpus_size for doc_id in doc_ids):
                raise ValueError("Invalid document IDs found.")
        logger.info("Retrieval results validation passed.")
        return True
    except Exception as e:
        logger.error(f"Error in validating retrieval results: {str(e)}")
        return False

# Memory cleanup
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Optimal batch size computation
def compute_batch_size(total_items: int, max_memory_gb: float = 0.8) -> int:
    if not torch.cuda.is_available():
        return 32  # Default batch size for CPU
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_per_item = total_memory / total_items
    optimal_batch_size = int((total_memory * max_memory_gb) / memory_per_item)
    return max(1, min(optimal_batch_size, 128))
