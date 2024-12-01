import os
import gc
import json
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path


from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything


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
        seed_everything(random_seed)

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
                linkage="ward"
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
        """
        Fit clusters to the embeddings.

        Args:
            embeddings: Document embeddings.
            document_ids: IDs corresponding to documents.
            use_batches: Whether to process embeddings in batches.

        Returns:
            Dictionary of cluster assignments.
        """
        try:
            self._initialize_clusterer()

            if self.use_scaler:
                embeddings = self.scaler.fit_transform(embeddings)

            labels = self.clusterer.fit_predict(embeddings)
            clusters = self._labels_to_clusters(
                labels, document_ids or list(range(len(embeddings)))
            )
            return clusters
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error during clustering")
            raise

    @staticmethod
    def _labels_to_clusters(labels: np.ndarray, doc_ids: List[int]) -> Dict[int, List[int]]:
        clusters = defaultdict(list)
        for doc_id, label in zip(doc_ids, labels):
            clusters[label].append(doc_id)
        return dict(clusters)


class ClusteringExperimentUtils:
    def __init__(self, logger: Optional[ExperimentLogger] = None):
        self.logger = logger or ExperimentLogger("clustering_utils")

    def analyze_clustering_results(self, clustering_results: Dict, logger: Optional[ExperimentLogger] = None) -> Dict:
        """
        Analyze clustering results and compute metrics.
        Example: Count the number of clusters and the size of each cluster.
        """
        metrics = {}
        try:
            cluster_sizes = [len(cluster['documents']) for cluster in clustering_results]
            metrics['num_clusters'] = len(clustering_results)
            metrics['cluster_sizes'] = cluster_sizes
            metrics['avg_cluster_size'] = sum(cluster_sizes) / len(cluster_sizes)
            
            if logger:
                logger.log_metric("num_clusters", metrics['num_clusters'])
                logger.log_metric("avg_cluster_size", metrics['avg_cluster_size'])
                
            return metrics
        except Exception as e:
            if logger:
                logger.log_error(e, "Error analyzing clustering results")
            raise

    def calculate_cluster_quality_metrics(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.
        """
        try:
            if self.logger:
                self.logger.log_step_start("Calculating cluster quality metrics")

            metrics = {
                "silhouette_score": silhouette_score(embeddings, cluster_labels),
                "davies_bouldin_score": davies_bouldin_score(embeddings, cluster_labels),
                "calinski_harabasz_score": calinski_harabasz_score(embeddings, cluster_labels),
            }

            if self.logger:
                self.logger.log_metrics(metrics)
                self.logger.log_step_end("Calculated cluster quality metrics")

            return metrics
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error calculating cluster quality metrics")
            raise

    def inject_noise_into_clusters(
        self,
        clusters: Dict[int, List[int]],
        corpus: List[Dict[str, Any]],
        noise_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Dict[int, List[int]]:
        """
        Inject noise into clusters.
        """
        try:
            if self.logger:
                self.logger.log_step_start("Injecting noise into clusters")

            random = np.random.default_rng(seed=random_seed)
            noisy_clusters = {}

            for cluster_id, doc_ids in clusters.items():
                num_noise = max(1, int(len(doc_ids) * noise_ratio))
                noise_ids = random.choice(
                    [i for i in range(len(corpus)) if i not in doc_ids],
                    size=num_noise,
                    replace=False
                )
                noisy_clusters[cluster_id] = doc_ids + noise_ids.tolist()

            if self.logger:
                self.logger.log_metric("num_noisy_clusters", len(noisy_clusters))
                self.logger.log_step_end("Noise injection completed")

            return noisy_clusters
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error injecting noise into clusters")
            raise

    def save_clustered_contexts(
        self,
        clusters: Dict[int, List[int]],
        corpus: List[Dict[str, Any]],
        output_path: Path
    ):
        """
        Save clustered contexts to a file.
        """
        try:
            if self.logger:
                self.logger.log_step_start("Saving clustered contexts")

            clustered_contexts = {
                cluster_id: [corpus[doc_id] for doc_id in doc_ids]
                for cluster_id, doc_ids in clusters.items()
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(clustered_contexts, f, indent=4)

            if self.logger:
                self.logger.log_step_end("Clustered contexts saved")
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error saving clustered contexts")
            raise
