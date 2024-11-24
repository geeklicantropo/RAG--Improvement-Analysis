import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from experiment_logger import ExperimentLogger
import pickle
import json
import time 

def load_and_preprocess_data(
    corpus_path: str,
    queries_path: str,
    search_results_path: str,
    logger: ExperimentLogger
) -> Tuple[List[Dict], List[str], List[Tuple[List[int], List[float]]]]:
    """
    Load and preprocess experiment data with logging.
    
    Args:
        corpus_path: Path to corpus file
        queries_path: Path to queries file
        search_results_path: Path to search results file
        logger: Logger instance
        
    Returns:
        Tuple of (corpus, queries, search_results)
    """
    try:
        logger.log_step_start("Loading data")
        
        # Load corpus
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
        logger.log_metric("corpus_size", len(corpus))
        
        # Load queries
        with open(queries_path, 'r') as f:
            data = json.load(f)
        queries = [item['question'] for item in data]
        logger.log_metric("num_queries", len(queries))
        
        # Load search results
        with open(search_results_path, 'rb') as f:
            search_results = pickle.load(f)
        logger.log_metric("num_search_results", len(search_results))
        
        logger.log_step_end("Loading data", time.time())
        return corpus, queries, search_results
        
    except Exception as e:
        logger.log_error(e, "Error loading data")
        raise

def inject_random_documents(
    search_results: List[Tuple[List[int], List[float]]],
    random_docs_path: str,
    ratio: float,
    logger: ExperimentLogger
) -> List[Tuple[List[int], List[float]]]:
    """
    Inject random documents into search results.
    
    Args:
        search_results: Original search results
        random_docs_path: Path to random documents file
        ratio: Ratio of random documents to inject
        logger: Logger instance
        
    Returns:
        Modified search results
    """
    try:
        logger.log_step_start("Injecting random documents")
        
        # Load random documents
        with open(random_docs_path, 'rb') as f:
            random_docs = pickle.load(f)
            
        # Calculate number of random docs to inject
        num_random = int(len(search_results[0][0]) * ratio)
        logger.log_metric("num_random_docs", num_random)
        
        # Modify search results
        modified_results = []
        for (doc_ids, scores), random_result in zip(search_results, random_docs):
            # Select random documents
            random_doc_ids = random_result[0][:num_random]
            random_scores = random_result[1][:num_random]
            # Combine original and random documents
            combined_ids = doc_ids[:-num_random] + random_doc_ids
            combined_scores = scores[:-num_random] + random_scores
            
            modified_results.append((combined_ids, combined_scores))
            
        logger.log_step_end("Injecting random documents", time.time())
        return modified_results
        
    except Exception as e:
        logger.log_error(e, "Error injecting random documents")
        raise

def evaluate_clustering(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    logger: ExperimentLogger
) -> Dict[str, float]:
    """
    Evaluate clustering quality using multiple metrics.
    
    Args:
        embeddings: Document embeddings
        cluster_labels: Cluster assignments
        logger: Logger instance
        
    Returns:
        Dictionary of clustering metrics
    """
    try:
        logger.log_step_start("Evaluating clustering")
        
        metrics = {}
        
        # Calculate Silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        metrics['silhouette_score'] = silhouette_avg
        
        # Calculate Calinski-Harabasz score
        calinski_score = calinski_harabasz_score(embeddings, cluster_labels)
        metrics['calinski_harabasz_score'] = calinski_score
        
        # Calculate cluster sizes and distributions
        unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        metrics['cluster_sizes'] = {f'cluster_{i}': size for i, size in zip(unique_clusters, cluster_sizes)}
        metrics['cluster_distribution'] = {
            f'cluster_{i}': size/len(cluster_labels) 
            for i, size in zip(unique_clusters, cluster_sizes)
        }
        
        logger.log_metrics(metrics)
        logger.log_step_end("Evaluating clustering", time.time())
        return metrics
        
    except Exception as e:
        logger.log_error(e, "Error evaluating clustering")
        raise

def analyze_cluster_performance(
    results: List[Dict],
    cluster_assignments: Dict[int, List[int]],
    logger: ExperimentLogger
) -> Dict[str, Any]:
    """
    Analyze performance metrics per cluster.
    
    Args:
        results: Generation results
        cluster_assignments: Document cluster assignments
        logger: Logger instance
        
    Returns:
        Dictionary of cluster-specific metrics
    """
    try:
        logger.log_step_start("Analyzing cluster performance")
        
        # Initialize cluster metrics
        cluster_metrics = defaultdict(lambda: {
            'total_queries': 0,
            'correct_answers': 0,
            'response_length': [],
            'processing_time': []
        })
        
        # Analyze results by cluster
        for result in results:
            cluster_id = result['cluster_info']['primary_cluster']
            metrics = cluster_metrics[cluster_id]
            
            metrics['total_queries'] += 1
            if result['response']['correct']:
                metrics['correct_answers'] += 1
            metrics['response_length'].append(len(result['response']['text']))
            metrics['processing_time'].append(result['response']['generation_time'])
        
        # Calculate aggregate metrics
        analysis = {}
        for cluster_id, metrics in cluster_metrics.items():
            analysis[f'cluster_{cluster_id}'] = {
                'accuracy': metrics['correct_answers'] / metrics['total_queries'],
                'avg_response_length': np.mean(metrics['response_length']),
                'avg_processing_time': np.mean(metrics['processing_time']),
                'total_queries': metrics['total_queries']
            }
            
        logger.log_metrics({'cluster_performance': analysis})
        logger.log_step_end("Analyzing cluster performance", time.time())
        return analysis
        
    except Exception as e:
        logger.log_error(e, "Error analyzing cluster performance")
        raise

def save_experiment_artifacts(
    config: Dict,
    results: List[Dict],
    metrics: Dict,
    cluster_info: Dict,
    output_dir: str,
    logger: ExperimentLogger
) -> None:
    """
    Save all experiment artifacts with proper organization.
    
    Args:
        config: Experiment configuration
        results: Generation results
        metrics: Performance metrics
        cluster_info: Clustering information
        output_dir: Output directory
        logger: Logger instance
    """
    try:
        logger.log_step_start("Saving experiment artifacts")
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save results
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save cluster information
        cluster_path = os.path.join(output_dir, 'cluster_info.json')
        with open(cluster_path, 'w') as f:
            json.dump(cluster_info, f, indent=2)
        
        # Save paths to logger
        logger.log_metric('config_path', config_path)
        logger.log_metric('results_path', results_path)
        logger.log_metric('metrics_path', metrics_path)
        logger.log_metric('cluster_path', cluster_path)
        
        logger.log_step_end("Saving experiment artifacts", time.time())
        
    except Exception as e:
        logger.log_error(e, "Error saving experiment artifacts")
        raise

class ClusteringDataset(Dataset):
    """
    Dataset class for handling clustered documents.
    """
    def __init__(
        self,
        documents: List[Dict],
        clusters: Dict[int, List[int]],
        tokenizer: Any,
        max_length: int
    ):
        self.documents = documents
        self.clusters = clusters
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create reverse mapping from document to cluster
        self.doc_to_cluster = {}
        for cluster_id, doc_ids in clusters.items():
            for doc_id in doc_ids:
                self.doc_to_cluster[doc_id] = cluster_id

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc = self.documents[idx]
        cluster_id = self.doc_to_cluster.get(doc['id'], -1)
        
        # Tokenize document
        encoding = self.tokenizer(
            doc['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'doc_id': doc['id'],
            'cluster_id': cluster_id,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': doc['text']
        }

def create_cluster_dataloader(
    dataset: ClusteringDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for clustered documents.
    
    Args:
        dataset: ClusteringDataset instance
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )