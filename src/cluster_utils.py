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
    davies_bouldin_score, 
    adjusted_rand_score
)
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel

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
        noise_ratio: float = 0.0,
        method: str = ClusteringMethod.KMEANS,
        batch_size: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[ExperimentLogger] = None
    ):
        self.num_clusters = num_clusters
        self.random_seed = random_seed
        self.use_scaler = use_scaler
        self.min_cluster_size = min_cluster_size
        self.noise_ratio = noise_ratio
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

    def add_noise_to_clusters(
        self,
        clusters: Dict[int, List[int]],
        random_docs: List[int],
        noise_ratio: Optional[float] = None,
        strategy: str = "uniform"
    ) -> Dict[int, List[int]]:
        if noise_ratio is None:
            noise_ratio = self.noise_ratio
            
        if noise_ratio <= 0 or not random_docs:
            return clusters
            
        try:
            if self.logger:
                self.logger.log_step_start("Adding noise to clusters")
            
            modified_clusters = {}
            num_noise_total = int(sum(len(docs) for docs in clusters.values()) * noise_ratio)
            
            if strategy == "uniform":
                noise_per_cluster = num_noise_total // len(clusters)
                remaining_noise = num_noise_total % len(clusters)
                
                for cluster_id, doc_ids in clusters.items():
                    cluster_noise_count = noise_per_cluster + (1 if remaining_noise > 0 else 0)
                    remaining_noise -= 1
                    
                    cluster_noise = np.random.choice(
                        random_docs,
                        size=min(cluster_noise_count, len(random_docs)),
                        replace=False
                    ).tolist()
                    
                    modified_clusters[cluster_id] = doc_ids + cluster_noise
                    
            elif strategy == "proportional":
                for cluster_id, doc_ids in clusters.items():
                    cluster_ratio = len(doc_ids) / sum(len(d) for d in clusters.values())
                    cluster_noise_count = int(num_noise_total * cluster_ratio)
                    
                    cluster_noise = np.random.choice(
                        random_docs,
                        size=min(cluster_noise_count, len(random_docs)),
                        replace=False
                    ).tolist()
                    
                    modified_clusters[cluster_id] = doc_ids + cluster_noise
            
            if self.logger:
                self.logger.log_step_end("Adding noise to clusters")
                self._log_clustering_stats(modified_clusters, noise_added=True)
            
            return modified_clusters
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error adding noise to clusters")
            raise

    def evaluate_clusters(
        self, 
        embeddings: np.ndarray,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        try:
            if self.clusterer is None:
                raise ValueError("Must fit clusters before evaluation")
                
            labels = self.clusterer.labels_
            metrics = {
                "silhouette_score": silhouette_score(embeddings, labels),
                "calinski_harabasz_score": calinski_harabasz_score(embeddings, labels),
                "davies_bouldin_score": davies_bouldin_score(embeddings, labels)
            }
            
            if hasattr(self.clusterer, "inertia_"):
                metrics["inertia"] = self.clusterer.inertia_
                
            cluster_centers = self._get_cluster_centers(embeddings, labels)
            metrics.update(self._calculate_cluster_qualities(embeddings, labels, cluster_centers))
            
            if save_dir:
                self._save_evaluation_results(metrics, save_dir)
            
            if self.logger:
                self.logger.log_metrics(metrics)
                
            self.cluster_metrics = metrics
            return metrics
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error evaluating clusters")
            raise

    def _calculate_cluster_qualities(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray
    ) -> Dict[str, float]:
        metrics = {}
        
        # Intra-cluster density
        densities = []
        for i in range(len(centers)):
            cluster_points = embeddings[labels == i]
            if len(cluster_points) > 1:
                density = np.mean([
                    np.linalg.norm(p - centers[i]) 
                    for p in cluster_points
                ])
                densities.append(density)
        
        metrics["avg_cluster_density"] = np.mean(densities)
        metrics["std_cluster_density"] = np.std(densities)
        
        # Inter-cluster separation
        separations = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                separation = np.linalg.norm(centers[i] - centers[j])
                separations.append(separation)
                
        metrics["avg_cluster_separation"] = np.mean(separations)
        metrics["std_cluster_separation"] = np.std(separations)
        
        return metrics

    def _get_cluster_centers(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        unique_labels = np.unique(labels[labels != -1])  # Exclude noise points
        centers = np.zeros((len(unique_labels), embeddings.shape[1]))
        
        for i, label in enumerate(unique_labels):
            centers[i] = np.mean(embeddings[labels == label], axis=0)
            
        return centers

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
                
                # Memory cleanup
                del batch_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            clusters = self._labels_to_clusters(
                all_labels,
                document_ids if document_ids else range(len(embeddings))
            )
            
            if self.logger:
                self.logger.log_step_end("Batch processing")
                self._log_clustering_stats(clusters)
            
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
            
        # Filter small clusters
        return {
            k: v for k, v in clusters.items() 
            if len(v) >= self.min_cluster_size
        }

    def save_cluster_results(
        self,
        output_dir: str,
        clusters: Dict[int, List[int]],
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        try:
            if self.logger:
                self.logger.log_step_start("Saving cluster results")
            
            os.makedirs(output_dir, exist_ok=True)
            
            results = {
                "clusters": clusters,
                "config": {
                    "num_clusters": self.num_clusters,
                    "min_cluster_size": self.min_cluster_size,
                    "noise_ratio": self.noise_ratio,
                    "method": self.method
                },
                "metrics": self.cluster_metrics
            }
            
            # Save main results
            output_path = os.path.join(output_dir, "cluster_results.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Optionally save embeddings
            if embeddings is not None:
                embeddings_path = os.path.join(output_dir, "cluster_embeddings.npy")
                np.save(embeddings_path, embeddings)
            
            # Save cluster centers if available
            if hasattr(self.clusterer, "cluster_centers_"):
                centers_path = os.path.join(output_dir, "cluster_centers.npy")
                np.save(centers_path, self.clusterer.cluster_centers_)
                
            if self.logger:
                self.logger.log_step_end("Saving cluster results")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error saving cluster results")
            raise

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

    def predict_clusters(
        self, 
        embeddings: np.ndarray,
        document_ids: Optional[List[int]] = None,
        use_batches: bool = True
    ) -> Dict[int, List[int]]:
        if self.clusterer is None:
            raise ValueError("Must call fit_clusters before predict_clusters")
        
        try:
            if self.logger:
                self.logger.log_step_start("Predicting clusters")
            
            if use_batches and len(embeddings) > self.batch_size:
                clusters = self.process_in_batches(embeddings, document_ids)
            else:
                if self.use_scaler:
                    embeddings = self.scaler.transform(embeddings)
                
                labels = self.clusterer.predict(embeddings)
                clusters = self._labels_to_clusters(
                    labels,
                    document_ids if document_ids else range(len(embeddings))
                )
            
            if self.logger:
                self.logger.log_step_end("Predicting clusters")
                self._log_clustering_stats(clusters)
            
            return clusters
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error during cluster prediction")
            raise

    def _log_clustering_stats(
        self,
        clusters: Dict[int, List[int]],
        noise_added: bool = False
    ):
        try:
            stats = {
                "num_clusters": len(clusters),
                "cluster_sizes": {i: len(docs) for i, docs in clusters.items()},
                "total_documents": sum(len(docs) for docs in clusters.values()),
                "avg_cluster_size": np.mean([len(docs) for docs in clusters.values()]),
                "std_cluster_size": np.std([len(docs) for docs in clusters.values()]),
                "min_cluster_size": min(len(docs) for docs in clusters.values()),
                "max_cluster_size": max(len(docs) for docs in clusters.values()),
            }
            
            if hasattr(self.clusterer, "inertia_"):
                stats["inertia"] = self.clusterer.inertia_
                
            if noise_added:
                stats["noise_ratio"] = self.noise_ratio
                
            self.logger.experiment_logger.info(f"Clustering Stats: {stats}")
            self.logger.log_metric("clustering_stats", stats)
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error logging clustering stats")
            raise

def find_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int,
    min_clusters: int = 2,
    method: str = ClusteringMethod.KMEANS,
    batch_size: int = 1000,
    logger: Optional[ExperimentLogger] = None
) -> Tuple[int, List[float], List[float], List[float]]:
    if logger:
        logger.log_step_start("Finding optimal clusters")
    
    metrics = {
        "inertias": [],
        "silhouette_scores": [],
        "calinski_scores": [],
        "davies_scores": []
    }
    
    try:
        for k in range(min_clusters, max_clusters + 1):
            clusterer = DocumentClusterer(
                num_clusters=k,
                method=method,
                batch_size=batch_size,
                logger=logger
            )
            
            clusters = clusterer.fit_clusters(embeddings)
            evaluation = clusterer.evaluate_clusters(embeddings)
            
            for metric_name, values in metrics.items():
                if metric_name[:-1] in evaluation:
                    values.append(evaluation[metric_name[:-1]])
            
        # Find optimal k using combined metrics
        normalized_scores = []
        for k in range(len(metrics["silhouette_scores"])):
            score = (
                metrics["silhouette_scores"][k] / max(metrics["silhouette_scores"]) +
                metrics["calinski_scores"][k] / max(metrics["calinski_scores"]) -
                metrics["davies_scores"][k] / max(metrics["davies_scores"])
            ) / 3
            normalized_scores.append(score)
            
        optimal_k = min_clusters + np.argmax(normalized_scores)
        
        if logger:
            logger.log_step_end("Finding optimal clusters")
            logger.log_metric("optimal_clusters", optimal_k)
            for metric_name, values in metrics.items():
                logger.log_metric(metric_name, values)
        
        return optimal_k, metrics["inertias"], metrics["silhouette_scores"], metrics["davies_scores"]
        
    except Exception as e:
        if logger:
            logger.log_error(e, "Error finding optimal clusters")
        raise

def compute_cluster_embeddings(
    documents: List[Dict],
    encoder: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    logger: Optional[ExperimentLogger] = None
) -> np.ndarray:
    try:
        if logger:
            logger.log_step_start("Computing document embeddings")
            
        encoder = encoder.to(device)
        embeddings = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Computing embeddings"):
            batch_docs = documents[i:i + batch_size]
            texts = [doc.get("text", "") for doc in batch_docs]
            
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = encoder(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                embeddings.append(batch_embeddings)
                
            del inputs, outputs, batch_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
                
        embeddings = np.vstack(embeddings)
        
        if logger:
            logger.log_step_end("Computing document embeddings")
            logger.log_metric("embeddings_shape", embeddings.shape)
            
        return embeddings
        
    except Exception as e:
        if logger:
            logger.log_error(e, "Error computing embeddings")
        raise