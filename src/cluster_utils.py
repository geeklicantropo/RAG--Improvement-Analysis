import os
import gc
import json
import logging
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import torch
from experiment_logger import ExperimentLogger
from utils import seed_everything


class ClusteringMethod:
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"


class DocumentClusterer:
    def __init__(
        self,
        num_clusters: int,
        random_seed: int = 42,
        use_scaler: bool = True,
        min_cluster_size: int = 2,
        method: str = ClusteringMethod.KMEANS,
        batch_size: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[ExperimentLogger] = None
    ):
        self.num_clusters = num_clusters
        self.random_seed = random_seed
        self.use_scaler = use_scaler
        self.min_cluster_size = min_cluster_size
        self.method = method
        self.batch_size = batch_size
        self.device = device
        self.logger = logger

        self.clusterer = None
        self.scaler = StandardScaler() if use_scaler else None
        self.cluster_metrics = {}

        seed_everything(random_seed)

        if logger:
            logger.experiment_logger.info(
                f"Initialized DocumentClusterer with {num_clusters} clusters using {method}"
            )

    def _initialize_clusterer(self):
        if self.method == ClusteringMethod.KMEANS:
            self.clusterer = KMeans(
                n_clusters=self.num_clusters,
                random_state=self.random_seed,
                n_init=10
            )
        elif self.method == ClusteringMethod.HIERARCHICAL:
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.num_clusters,
                linkage='ward'
            )
        elif self.method == ClusteringMethod.DBSCAN:
            self.clusterer = DBSCAN(
                min_samples=self.min_cluster_size,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

    def fit_clusters(
        self,
        embeddings: np.ndarray,
        document_ids: Optional[List[int]] = None,
        use_batches: bool = True
    ) -> Dict[int, List[int]]:
        try:
            if self.logger:
                self.logger.log_step_start("Fitting clusters")

            self._initialize_clusterer()

            if use_batches and len(embeddings) > self.batch_size:
                clusters = self.process_in_batches(embeddings, document_ids)
            else:
                if self.use_scaler:
                    embeddings = self.scaler.fit_transform(embeddings)

                labels = self.clusterer.fit_predict(embeddings)
                clusters = self._labels_to_clusters(
                    labels,
                    document_ids if document_ids else range(len(embeddings))
                )

            if self.logger:
                self.logger.log_step_end("Fitting clusters")
                self._log_clustering_stats(clusters)

            return clusters

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error during cluster fitting")
            raise
        finally:
            self._cleanup_memory()

    def evaluate_clusters(
        self,
        embeddings: np.ndarray,
        noise_ratio: float = 0.0
    ) -> Dict[str, float]:
        try:
            if self.clusterer is None:
                raise ValueError("Must fit clusters before evaluation")

            # Add noise if specified
            if noise_ratio > 0:
                embeddings = self._inject_noise(embeddings, noise_ratio)

            labels = self.clusterer.labels_
            metrics = {
                "silhouette_score": silhouette_score(embeddings, labels),
                "calinski_harabasz_score": calinski_harabasz_score(embeddings, labels),
                "davies_bouldin_score": davies_bouldin_score(embeddings, labels)
            }

            if hasattr(self.clusterer, "inertia_"):
                metrics["inertia"] = self.clusterer.inertia_

            self.cluster_metrics = metrics
            if self.logger:
                self.logger.log_metrics(metrics)
            return metrics

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error evaluating clusters")
            raise

    def process_in_batches(
        self,
        embeddings: np.ndarray,
        document_ids: Optional[List[int]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[int, List[int]]:
        if batch_size is None:
            batch_size = self.batch_size

        try:
            if self.logger:
                self.logger.log_step_start("Processing in batches")

            total_batches = (len(embeddings) + batch_size - 1) // batch_size
            all_labels = []

            for i in tqdm(range(total_batches), desc="Processing batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(embeddings))

                batch_embeddings = embeddings[start_idx:end_idx]
                if self.use_scaler:
                    batch_embeddings = self.scaler.transform(batch_embeddings)

                batch_labels = self.clusterer.predict(batch_embeddings)
                all_labels.extend(batch_labels)

                self._cleanup_memory()

            clusters = self._labels_to_clusters(
                all_labels,
                document_ids if document_ids else range(len(embeddings))
            )

            if self.logger:
                self.logger.log_step_end("Batch processing")

            return clusters

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error in batch processing")
            raise

    def _labels_to_clusters(
        self,
        labels: List[int],
        document_ids: Union[List[int], range]
    ) -> Dict[int, List[int]]:
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # Exclude noise points from DBSCAN
                clusters[label].append(document_ids[idx])
        return clusters

    def save_clusters(self, output_path: str, include_metrics: bool = True):
        """Save clustering results and metrics to JSON."""
        output_data = {
            "clusters": self.clusterer.labels_.tolist() if self.clusterer else [],
            "config": {
                "num_clusters": self.num_clusters,
                "method": self.method,
                "min_cluster_size": self.min_cluster_size
            }
        }

        if include_metrics and self.cluster_metrics:
            output_data["metrics"] = self.cluster_metrics

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    def _cleanup_memory(self):
        """Clean up GPU memory and garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _inject_noise(self, embeddings: np.ndarray, noise_ratio: float) -> np.ndarray:
        """Add Gaussian noise to embeddings for diversity testing."""
        noise = np.random.normal(0, 0.1, embeddings.shape)
        noise_mask = np.random.random(len(embeddings)) < noise_ratio
        embeddings[noise_mask] += noise[noise_mask]
        return embeddings

    def _log_clustering_stats(self, clusters: Dict[int, List[int]]):
        """Log detailed clustering statistics."""
        stats = {
            "num_clusters": len(clusters),
            "total_documents": sum(len(docs) for docs in clusters.values()),
            "avg_cluster_size": np.mean([len(docs) for docs in clusters.values()]),
            "empty_clusters": self.num_clusters - len(clusters)
        }

        if self.logger:
            self.logger.log_metrics({"clustering_stats": stats})


def find_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int,
    min_clusters: int = 2,
    method: str = ClusteringMethod.KMEANS,
    batch_size: int = 1000,
    logger: Optional[ExperimentLogger] = None
) -> Tuple[int, Dict[str, List[float]]]:
    if logger:
        logger.log_step_start("Finding optimal clusters")

    metrics = {
        "silhouette_scores": [],
        "calinski_scores": [],
        "davies_scores": []
    }

    try:
        for k in tqdm(range(min_clusters, max_clusters + 1), desc="Evaluating cluster sizes"):
            clusterer = DocumentClusterer(
                num_clusters=k,
                method=method,
                batch_size=batch_size,
                logger=logger
            )

            clusterer.fit_clusters(embeddings)
            evaluation = clusterer.evaluate_clusters(embeddings)

            metrics["silhouette_scores"].append(evaluation.get("silhouette_score", 0))
            metrics["calinski_scores"].append(evaluation.get("calinski_harabasz_score", 0))
            metrics["davies_scores"].append(evaluation.get("davies_bouldin_score", float('inf')))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        combined_scores = np.array(metrics["silhouette_scores"])
        optimal_k = min_clusters + np.argmax(combined_scores)

        if logger:
            logger.log_step_end("Finding optimal clusters")
            logger.log_metric("optimal_clusters", optimal_k)
            logger.log_metric("clustering_metrics", metrics)

        return optimal_k, metrics

    except Exception as e:
        if logger:
            logger.log_error(e, "Error finding optimal clusters")
        raise
