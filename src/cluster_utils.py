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
from src.utils.file_utils import seed_everything
import google.generativeai as genai
import time

class ClusteringMethod:
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"

class DocumentClusterer:
    def __init__(
        self,
        api_key: str,
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
        self.logger = logger or ExperimentLogger("DocumentClusterer", "logs")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        self.clusterer = None
        self.scaler = StandardScaler() if use_scaler else None
        self.cluster_metrics = {}
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

    def evaluate_cluster_quality(self, cluster_docs: List[Dict], query: str = None) -> Dict[str, float]:
        """Evaluate cluster coherence using LLM"""
        try:
            eval_docs = cluster_docs[:5] if len(cluster_docs) > 5 else cluster_docs
            doc_texts = "\n".join(d.get('text', '')[:200] for d in eval_docs)
            
            prompt = f"""
            {"Query: " + query if query else ""}
            Documents:
            {doc_texts}
            
            Rate document cluster coherence (0-100):
            Consider:
            1. Topic similarity
            2. Content relevance
            3. Information complementarity
            {"4. Query relevance" if query else ""}
            
            Provide only the numerical score (0-100):
            """
            
            response = self.model.generate_content(prompt)
            coherence_score = float(response.text.strip())
            
            avg_length = np.mean([len(d.get('text', '')) for d in cluster_docs])
            unique_terms = len(set(' '.join([d.get('text', '') for d in cluster_docs]).split()))
            
            return {
                "coherence_score": coherence_score / 100,
                "avg_doc_length": avg_length,
                "unique_terms": unique_terms,
                "docs_evaluated": len(eval_docs)
            }
            
        except Exception as e:
            self.logger.experiment_logger.error(f"Cluster evaluation failed: {str(e)}")
            return {"coherence_score": 0.0, "error": str(e)}

    def fit_clusters(
        self,
        embeddings: np.ndarray,
        document_ids: Optional[List[int]] = None,
        use_batches: bool = True
    ) -> Dict[int, List[int]]:
        try:
            self.logger.experiment_logger.info("Starting cluster fitting")
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

            self.logger.experiment_logger.info("Cluster fitting completed")
            self._log_clustering_stats(clusters)

            return clusters

        except Exception as e:
            self.logger.experiment_logger.error(f"Error during clustering: {str(e)}")
            raise

    def evaluate_clusters(self, embeddings: np.ndarray, noise_ratio: float = 0.0) -> Dict[str, float]:
        try:
            if self.clusterer is None:
                raise ValueError("Must fit clusters before evaluation")

            if noise_ratio > 0:
                embeddings = self._inject_noise(embeddings, noise_ratio)

            labels = self.clusterer.labels_
            
            with tqdm(total=3, desc="Computing metrics") as pbar:
                metrics = {
                    "silhouette_score": silhouette_score(embeddings, labels),
                    "calinski_harabasz_score": calinski_harabasz_score(embeddings, labels),
                    "davies_bouldin_score": davies_bouldin_score(embeddings, labels)
                }
                pbar.update(1)

                if hasattr(self.clusterer, "inertia_"):
                    metrics["inertia"] = self.clusterer.inertia_
                pbar.update(1)

                unique_labels = np.unique(labels)
                metrics.update({
                    "num_clusters": len(unique_labels),
                    "avg_cluster_size": np.mean([np.sum(labels == l) for l in unique_labels]),
                    "std_cluster_size": np.std([np.sum(labels == l) for l in unique_labels])
                })
                pbar.update(1)

                self.cluster_metrics = metrics
                self.logger.log_metrics(metrics)
                
            return metrics

        except Exception as e:
            self.logger.experiment_logger.error(f"Error evaluating clusters: {str(e)}")
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
            self.logger.experiment_logger.info("Starting batch processing")

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

            return clusters

        except Exception as e:
            self.logger.experiment_logger.error(f"Error in batch processing: {str(e)}")
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
        return dict(clusters)

    def save_clusters(self, output_path: str, include_metrics: bool = True):
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _inject_noise(self, embeddings: np.ndarray, noise_ratio: float) -> np.ndarray:
        noise = np.random.normal(0, 0.1, embeddings.shape)
        noise_mask = np.random.random(len(embeddings)) < noise_ratio
        embeddings[noise_mask] += noise[noise_mask]
        return embeddings

    def _log_clustering_stats(self, clusters: Dict[int, List[int]]):
        stats = {
            "num_clusters": len(clusters),
            "total_documents": sum(len(docs) for docs in clusters.values()),
            "avg_cluster_size": np.mean([len(docs) for docs in clusters.values()]),
            "empty_clusters": self.num_clusters - len(clusters)
        }
        self.logger.log_metrics({"clustering_stats": stats})
            
    def get_evaluation_summary(self) -> Dict[str, Any]:
        if not self.cluster_metrics:
            return {"error": "No evaluation metrics available"}
            
        summary = {
            "quality_metrics": self.cluster_metrics,
            "cluster_stats": {
                "num_clusters": len(set(self.clusterer.labels_)),
                "noise_points": np.sum(self.clusterer.labels_ == -1) if hasattr(self.clusterer.labels_, "__len__") else 0,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return summary

def fit_clusters(
    embeddings: np.ndarray, 
    num_clusters: int, 
    method: str = ClusteringMethod.KMEANS, 
    random_seed: int = 42, 
    use_scaler: bool = True,
    min_cluster_size: int = 2,
    batch_size: int = 1000,
    api_key: Optional[str] = None
) -> Dict[int, List[int]]:
    if api_key is None:
        api_key = os.getenv("GEMINI_TOKEN")
        if not api_key:
            raise ValueError("No API key provided for DocumentClusterer and GEMINI_TOKEN not set.")
    
    clusterer = DocumentClusterer(
        api_key=api_key,
        num_clusters=num_clusters,
        random_seed=random_seed,
        use_scaler=use_scaler,
        min_cluster_size=min_cluster_size,
        method=method,
        batch_size=batch_size
    )
    
    clusters = clusterer.fit_clusters(embeddings, use_batches=False)
    return clusters

def get_top_k_docs_from_cluster(
    clusters: Dict[int, List[int]], 
    doc_scores: Optional[Dict[int,float]] = None, 
    k: int = 5
) -> Dict[int, List[int]]:
    top_docs = {}
    for cid, docs in clusters.items():
        if doc_scores:
            docs = sorted(docs, key=lambda d: doc_scores.get(d, 0), reverse=True)
        top_docs[cid] = docs[:k]
    return top_docs