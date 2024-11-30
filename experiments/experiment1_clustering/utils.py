import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from src.experiment_logger import ExperimentLogger
import pickle
import json
import time
from src.llm import LLM
from transformers import AutoTokenizer, AutoModel

class ClusteringExperimentUtils:
    """Utilities for clustering experiments."""

    def compute_embeddings(self, corpus: List[Dict], output_dir: str, logger: ExperimentLogger, batch_size: int = 32) -> np.ndarray:
        """
        Compute document embeddings using the Contriever model.

        Args:
            corpus: List of corpus documents
            output_dir: Directory to save the embeddings
            logger: ExperimentLogger instance
            batch_size: Batch size for embedding computation

        Returns:
            Document embeddings as a numpy array
        """
        try:
            logger.log_step_start("Computing document embeddings")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
            encoder = AutoModel.from_pretrained("facebook/contriever").to(device)

            all_embeddings = []
            num_steps = 0

            for i in logger.log_progress(range(0, len(corpus), batch_size), desc="Computing embeddings"):
                batch_docs = corpus[i:i + batch_size]
                texts = [doc.get("text", "") for doc in batch_docs]

                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = encoder(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                    all_embeddings.append(batch_embeddings)

                num_steps += 1
                if num_steps == 500 or i + batch_size >= len(corpus):
                    embeddings = np.vstack(all_embeddings)
                    file_index = i + batch_size - 1
                    file_path = os.path.join(output_dir, f"embeddings_{file_index}.npy")
                    np.save(file_path, embeddings)
                    logger.experiment_logger.info(f"Saved embeddings for {file_index} passages to {file_path}")
                    num_steps = 0
                    all_embeddings = []

            logger.log_step_end("Computing document embeddings", time.time())
            return np.vstack(all_embeddings)

        except Exception as e:
            logger.log_error(e, "Error computing document embeddings")
            raise

    def evaluate_clustering(self, embeddings: np.ndarray, cluster_labels: np.ndarray, logger: ExperimentLogger) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics.

        Args:
            embeddings: Document embeddings
            cluster_labels: Cluster assignments
            logger: ExperimentLogger instance

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

            # Calculate Davies-Bouldin score
            db_score = davies_bouldin_score(embeddings, cluster_labels)
            metrics['davies_bouldin_score'] = db_score

            # Calculate cluster sizes and distributions
            unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
            metrics['cluster_sizes'] = {f'cluster_{i}': size for i, size in zip(unique_clusters, cluster_sizes)}
            metrics['cluster_distribution'] = {
                f'cluster_{i}': size/len(cluster_labels)
                for i, size in zip(unique_clusters, cluster_sizes)
            }

            # Calculate intra-cluster and inter-cluster metrics
            cluster_centers = self._get_cluster_centers(embeddings, cluster_labels)
            intra_cluster_similarity, inter_cluster_distance = self._calculate_cluster_qualities(embeddings, cluster_labels, cluster_centers)
            metrics['intra_cluster_similarity'] = intra_cluster_similarity
            metrics['inter_cluster_distance'] = inter_cluster_distance

            logger.log_metrics(metrics)
            logger.log_step_end("Evaluating clustering", time.time())
            return metrics

        except Exception as e:
            logger.log_error(e, "Error evaluating clustering")
            raise

    def _get_cluster_centers(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers from the embeddings and cluster labels."""
        unique_labels = np.unique(labels[labels != -1])  # Exclude noise points
        centers = np.zeros((len(unique_labels), embeddings.shape[1]))

        for i, label in enumerate(unique_labels):
            centers[i] = np.mean(embeddings[labels == label], axis=0)

        return centers

    def _calculate_cluster_qualities(self, embeddings: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> Tuple[float, float]:
        """Calculate intra-cluster similarity and inter-cluster distance."""
        intra_cluster_similarities = []
        inter_cluster_distances = []

        for i in range(len(centers)):
            cluster_points = embeddings[labels == i]
            if len(cluster_points) > 1:
                intra_cluster_similarity = np.mean([
                    np.linalg.norm(p - centers[i])
                    for p in cluster_points
                ])
                intra_cluster_similarities.append(intra_cluster_similarity)

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                inter_cluster_distance = np.linalg.norm(centers[i] - centers[j])
                inter_cluster_distances.append(inter_cluster_distance)

        return np.mean(intra_cluster_similarities), np.mean(inter_cluster_distances)

    def analyze_cluster_performance(self, results: List[Dict], cluster_assignments: Dict[str, int], logger: ExperimentLogger) -> Dict[str, Any]:
        """
        Analyze performance metrics per cluster.

        Args:
            results: Generation results
            cluster_assignments: Document cluster assignments
            logger: ExperimentLogger instance

        Returns:
            Dictionary of cluster-specific metrics
        """
        try:
            logger.log_step_start("Analyzing cluster performance")

            cluster_metrics = defaultdict(lambda: {
                'total_queries': 0,
                'correct_answers': 0,
                'response_length': [],
                'processing_time': []
            })

            for result in results:
                cluster_id = result['cluster_info']['primary_cluster']
                metrics = cluster_metrics[cluster_id]

                metrics['total_queries'] += 1
                if result['response']['correct']:
                    metrics['correct_answers'] += 1
                metrics['response_length'].append(len(result['response']['text']))
                metrics['processing_time'].append(result['response']['generation_time'])

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

    def save_experiment_artifacts(self, config: Dict, results: List[Dict], metrics: Dict, cluster_info: Dict, output_dir: str, logger: ExperimentLogger) -> None:
        """
        Save all experiment artifacts with proper organization.

        Args:
            config: Experiment configuration
            results: Generation results
            metrics: Performance metrics
            cluster_info: Clustering information
            output_dir: Output directory
            logger: ExperimentLogger instance
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