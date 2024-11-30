import os
import json
import torch
import ijson
import pickle
import random
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

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
    

def read_pickle(file_path: str):
    with open(file_path, "rb") as reader:
        data = pickle.load(reader)
    return data


def write_pickle(data, file_path: str):
    with open(file_path, "wb") as writer:
        pickle.dump(data, writer)


def read_json(file_path: str):
    with open(file_path, "rb") as reader:
        data = json.load(reader)
    return data


def write_json(data, file_path: str):
    with open(file_path, "w") as writer:
        json.dump(data, writer)


def read_corpus_json(data_path: str, subset_to_full_idx_map: Optional[Dict[int, int]] = None) -> List[Dict]:
    """
    Reads documents from a JSON file silently, without debug output.
    
    Args:
        data_path (str): Path to the JSON file containing the corpus documents.
        subset_to_full_idx_map (Optional[Dict[int, int]]): Mapping from subset to full corpus indices.
        
    Returns:
        List[Dict]: List of corpus documents with adjusted indices.
    """
    corpus = []
    try:
        with open(data_path, "rb") as f:
            # Use a list comprehension instead of iterative appending to reduce output
            data = list(ijson.items(f, "item"))
            
        # Process all documents at once silently
        corpus = [
            {
                **record,
                'full_corpus_idx': (subset_to_full_idx_map[idx] if subset_to_full_idx_map 
                                  else idx)
            }
            for idx, record in enumerate(data)
        ]
        return corpus
        
    except Exception as e:
        logging.error(f"Error reading corpus from {data_path}: {str(e)}")
        raise


def read_subset_corupus_with_map(
    full_to_subset_path: str,
    subset_to_full_path: str,
    corpus_path: str
) -> Tuple[List[Dict], Dict[int, int]]:
    full_to_subset_idx_map = read_pickle(full_to_subset_path)
    subset_to_full_idx_map = read_pickle(subset_to_full_path)
    corpus = read_corpus_json(corpus_path, subset_to_full_idx_map)
    return corpus, full_to_subset_idx_map


def read_corpus_with_random():
    full_to_subset_path = "data/mappings/full_to_subset_random_at60_in_corpus.pkl"
    subset_to_full_path = "data/mappings/subset_to_full_random_at60_in_corpus.pkl"
    corpus_path = "data/processed/corpus_with_random_at60.json"
    return read_subset_corupus_with_map(
        full_to_subset_path,
        subset_to_full_path,
        corpus_path
    )


def read_corpus_with_adore():
    full_to_subset_path = "data/mappings/full_to_subset_adore_at200_in_corpus.pkl"
    subset_to_full_path = "data/mappings/subset_to_full_adore_at200_in_corpus.pkl"
    corpus_path = "data/processed/corpus_with_adore_at200.json"
    return read_subset_corupus_with_map(
        full_to_subset_path,
        subset_to_full_path,
        corpus_path
    )


def read_corpus_with_contriever():
    full_to_subset_path = "data/mappings/full_to_subset_contriever_at150_in_corpus.pkl"
    subset_to_full_path = "data/mappings/subset_to_full_contriever_at150_in_corpus.pkl"
    corpus_path = "data/processed/corpus_with_contriever_at150.json"
    return read_subset_corupus_with_map(
        full_to_subset_path,
        subset_to_full_path,
        corpus_path
    )


def read_corpus_with_random_and_contriever():
    full_to_subset_path = "data/mappings/full_to_subset_random_contriever_in_corpus.pkl"
    subset_to_full_path = "data/mappings/subset_to_full_random_contriever_in_corpus.pkl"
    corpus_path = "data/processed/corpus_with_random_contriever.json"
    return read_subset_corupus_with_map(
        full_to_subset_path,
        subset_to_full_path,
        corpus_path
    )


def read_test_corpus_with_random_and_bm25():
    full_to_subset_path = "data/mappings/full_to_subset_test_random_bm25_in_corpus.pkl"
    subset_to_full_path = "data/mappings/subset_to_full_test_random_bm25_in_corpus.pkl"
    corpus_path = "data/processed/test_corpus_with_random_bm25.json"
    return read_subset_corupus_with_map(
        full_to_subset_path,
        subset_to_full_path,
        corpus_path
    )

def read_test_corpus_with_random_and_contriever():
    full_to_subset_path = "data/mappings/full_to_subset_test_random_contriever_in_corpus.pkl"
    subset_to_full_path = "data/mappings/subset_to_full_test_random_contriever_in_corpus.pkl"
    corpus_path = "data/processed/test_corpus_with_random_contriever.json"
    return read_subset_corupus_with_map(
        full_to_subset_path,
        subset_to_full_path,
        corpus_path
    )