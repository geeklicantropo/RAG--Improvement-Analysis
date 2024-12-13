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
import psutil
import mmap

def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_json(data: Any, file_path: str, batch_size: int = 1000) -> None:
    logger = logging.getLogger()
    logger.info(f"Saving JSON to {file_path}")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            if isinstance(data, (list, tuple)):
                # Save large lists in batches
                f.write("[")
                for i, item in enumerate(tqdm(data, desc="Saving JSON")):
                    if i > 0:
                        f.write(",")
                    json.dump(item, f)
                    if (i + 1) % batch_size == 0:
                        f.flush()
                        clear_memory()
                f.write("]")
            else:
                json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        raise

def read_json_in_batches(file_path: str, batch_size: int = 1000) -> Any:
    logger = logging.getLogger()
    logger.info(f"Reading JSON from {file_path} in batches")
    try:
        with open(file_path, "rb") as f:
            parser = ijson.parse(f)
            batch = []
            result = []
            
            for prefix, event, value in tqdm(parser, desc="Reading JSON"):
                if prefix.endswith('.item') or prefix == 'item':
                    batch.append(value)
                    if len(batch) >= batch_size:
                        result.extend(batch)
                        batch = []
                        clear_memory()
            
            if batch:
                result.extend(batch)
            
            return result
    except Exception as e:
        logger.error(f"Error reading JSON from {file_path}: {str(e)}")
        raise

def read_json(file_path: str) -> Any:
    if os.path.getsize(file_path) > 100 * 1024 * 1024:  # If file > 100MB
        return read_json_in_batches(file_path)
    
    logger = logging.getLogger()
    logger.info(f"Reading JSON from {file_path}")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON from {file_path}: {str(e)}")
        raise

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
                    clear_memory()

            if batch:
                corpus.extend(batch)

        logger.info(f"Loaded {len(corpus)} documents.")
        return corpus
    except Exception as e:
        logger.error(f"Error reading corpus from {data_path}: {str(e)}")
        raise

def read_pickle_in_batches(file_path: str, batch_size: int = 1000):
    logger = logging.getLogger()
    logger.info(f"Reading pickle from {file_path} in batches")
    
    try:
        with open(file_path, 'rb') as f:
            while True:
                try:
                    batch = []
                    for _ in range(batch_size):
                        batch.append(pickle.load(f))
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                            clear_memory()
                except EOFError:
                    if batch:
                        yield batch
                    break
    except Exception as e:
        logger.error(f"Error reading pickle file from {file_path}: {str(e)}")
        raise

def read_pickle(file_path: str) -> Any:
    if os.path.getsize(file_path) > 100 * 1024 * 1024:  # If file > 100MB
        return list(read_pickle_in_batches(file_path))
        
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger = logging.getLogger()
        logger.error(f"Error reading pickle file from {file_path}: {str(e)}")
        raise

def write_pickle(data: Any, file_path: str, batch_size: int = 1000) -> None:
    logger = logging.getLogger()
    logger.info(f"Writing pickle to {file_path}")
    
    try:
        with open(file_path, 'wb') as f:
            if isinstance(data, (list, tuple)):
                for i in tqdm(range(0, len(data), batch_size), desc="Writing pickle"):
                    batch = data[i:i + batch_size]
                    pickle.dump(batch, f)
                    clear_memory()
            else:
                pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Error writing pickle file to {file_path}: {str(e)}")
        raise

def clear_memory():
    """Enhanced memory cleanup function"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force Python garbage collection
    for _ in range(3):
        gc.collect()

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    memory_stats = {
        'ram_used_gb': psutil.Process().memory_info().rss / (1024 ** 3),
        'ram_percent': psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        memory_stats.update({
            f'gpu_{i}_used_gb': torch.cuda.memory_allocated(i) / (1024 ** 3)
            for i in range(torch.cuda.device_count())
        })
    
    return memory_stats

def compute_batch_size(total_items: int, max_memory_gb: float = 0.8) -> int:
    """Compute optimal batch size based on available memory"""
    if not torch.cuda.is_available():
        return min(32, total_items)  # Default CPU batch size
        
    free_memory = torch.cuda.get_device_properties(0).total_memory * max_memory_gb
    estimated_item_memory = 1024 * 1024 * 1024  # 1GB estimate per item
    optimal_batch_size = int(free_memory / estimated_item_memory)
    
    return max(1, min(optimal_batch_size, min(128, total_items)))