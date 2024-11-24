import os
import logging
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer

from experiment_logger import ExperimentLogger
from utils import seed_everything

class DocumentClusterer:
    """
    A class for clustering documents using K-means and organizing them for RAG systems.
    Supports both embedding-based and text-based clustering with various preprocessing options.
    """
    def __init__(
        self,
        num_clusters: int,
        random_seed: int = 42,
        use_scaler: bool = True,
        logger: Optional[ExperimentLogger] = None
    ):
        """
        Initialize the document clusterer.

        Args:
            num_clusters: Number of clusters for k-means
            random_seed: Seed for reproducibility
            use_scaler: Whether to standardize features before clustering
            logger: Optional experiment logger for tracking
        """
        self.num_clusters = num_clusters
        self.random_seed = random_seed
        self.use_scaler = use_scaler
        self.logger = logger
        self.kmeans = None
        self.scaler = StandardScaler() if use_scaler else None
        
        seed_everything(random_seed)
        
        if logger:
            logger.experiment_logger.info(
                f"Initialized DocumentClusterer with {num_clusters} clusters"
            )

    def fit_clusters(
        self, 
        embeddings: np.ndarray,
        document_ids: Optional[List[int]] = None
    ) -> Dict[int, List[int]]:
        """
        Fit k-means clustering on document embeddings.

        Args:
            embeddings: Document embeddings array
            document_ids: Optional list of document IDs corresponding to embeddings

        Returns:
            Dictionary mapping cluster IDs to lists of document indices
        """
        try:
            if self.logger:
                self.logger.log_step_start("Fitting clusters")
            
            # Scale features if requested
            if self.use_scaler:
                embeddings = self.scaler.fit_transform(embeddings)
            
            # Initialize and fit k-means
            self.kmeans = KMeans(
                n_clusters=self.num_clusters,
                random_state=self.random_seed,
                n_init=10
            )
            cluster_labels = self.kmeans.fit_predict(embeddings)
            
            # Organize documents by cluster
            clusters = {i: [] for i in range(self.num_clusters)}
            for idx, label in enumerate(cluster_labels):
                doc_id = document_ids[idx] if document_ids else idx
                clusters[label].append(doc_id)
            
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
        document_ids: Optional[List[int]] = None
    ) -> Dict[int, List[int]]:
        """
        Predict clusters for new embeddings using fitted k-means model.

        Args:
            embeddings: Document embeddings array
            document_ids: Optional list of document IDs corresponding to embeddings

        Returns:
            Dictionary mapping cluster IDs to lists of document indices
        """
        if self.kmeans is None:
            raise ValueError("Must call fit_clusters before predict_clusters")
        
        try:
            if self.logger:
                self.logger.log_step_start("Predicting clusters")
            
            # Scale features if scaler was used during fitting
            if self.use_scaler:
                embeddings = self.scaler.transform(embeddings)
            
            # Predict clusters
            cluster_labels = self.kmeans.predict(embeddings)
            
            # Organize documents by cluster
            clusters = {i: [] for i in range(self.num_clusters)}
            for idx, label in enumerate(cluster_labels):
                doc_id = document_ids[idx] if document_ids else idx
                clusters[label].append(doc_id)
            
            if self.logger:
                self.logger.log_step_end("Predicting clusters")
                self._log_clustering_stats(clusters)
            
            return clusters
        
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error during cluster prediction")
            raise

    def get_cluster_centroids(self) -> np.ndarray:
        """Get the centroid vectors for each cluster."""
        if self.kmeans is None:
            raise ValueError("Must call fit_clusters before accessing centroids")
        return self.kmeans.cluster_centers_

    def get_inertia(self) -> float:
        """Get the inertia (within-cluster sum of squares) of the clustering."""
        if self.kmeans is None:
            raise ValueError("Must call fit_clusters before accessing inertia")
        return self.kmeans.inertia_

    def _log_clustering_stats(self, clusters: Dict[int, List[int]]):
        """Log statistics about the clustering results."""
        stats = {
            "num_clusters": len(clusters),
            "cluster_sizes": {i: len(docs) for i, docs in clusters.items()},
            "total_documents": sum(len(docs) for docs in clusters.values()),
            "inertia": self.get_inertia() if self.kmeans else None
        }
        self.logger.experiment_logger.info(f"Clustering Stats: {stats}")
        self.logger.log_metric("clustering_stats", stats)

class ClusterOrganizer:
    """
    Class for organizing and preparing clustered documents for RAG prompts.
    Handles document sorting, selection, and formatting within clusters.
    """
    def __init__(
        self,
        max_docs_per_cluster: int,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        logger: Optional[ExperimentLogger] = None
    ):
        """
        Initialize the cluster organizer.

        Args:
            max_docs_per_cluster: Maximum number of documents to include per cluster
            tokenizer: Tokenizer for length checking
            max_length: Maximum token length for documents
            logger: Optional experiment logger
        """
        self.max_docs_per_cluster = max_docs_per_cluster
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logger

    def organize_clusters(
        self,
        clusters: Dict[int, List[int]],
        documents: List[Dict],
        scores: Optional[Dict[int, float]] = None
    ) -> List[Dict]:
        """
        Organize documents within clusters for prompt creation.

        Args:
            clusters: Mapping of cluster IDs to document indices
            documents: List of document dictionaries
            scores: Optional document relevance scores

        Returns:
            List of organized cluster dictionaries with selected documents
        """
        try:
            if self.logger:
                self.logger.log_step_start("Organizing clusters")
            
            organized_clusters = []
            
            for cluster_id, doc_indices in clusters.items():
                # Sort documents by score if available
                if scores:
                    doc_indices = sorted(
                        doc_indices,
                        key=lambda x: scores.get(x, 0),
                        reverse=True
                    )
                
                # Select documents while respecting length constraints
                selected_docs = self._select_documents(
                    [documents[i] for i in doc_indices]
                )
                
                if selected_docs:
                    cluster_info = {
                        "cluster_id": cluster_id,
                        "documents": selected_docs,
                        "size": len(selected_docs)
                    }
                    organized_clusters.append(cluster_info)
            
            if self.logger:
                self.logger.log_step_end("Organizing clusters")
                self._log_organization_stats(organized_clusters)
            
            return organized_clusters
        
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error during cluster organization")
            raise

    def _select_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Select documents from a cluster while respecting length constraints.
        
        Args:
            documents: List of document dictionaries

        Returns:
            List of selected documents
        """
        selected = []
        current_length = 0
        
        for doc in documents[:self.max_docs_per_cluster]:
            # Check document length in tokens
            tokens = self.tokenizer.encode(
                doc.get("text", ""),
                add_special_tokens=True,
                truncation=True
            )
            
            if current_length + len(tokens) <= self.max_length:
                selected.append(doc)
                current_length += len(tokens)
            else:
                break
                
        return selected

    def format_cluster_prompt(
        self,
        organized_clusters: List[Dict],
        include_cluster_info: bool = True
    ) -> str:
        """
        Format clustered documents into a prompt string.

        Args:
            organized_clusters: List of organized cluster dictionaries
            include_cluster_info: Whether to include cluster metadata in prompt

        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for cluster in organized_clusters:
            if include_cluster_info:
                prompt_parts.append(f"\nCluster {cluster['cluster_id']} (Size: {cluster['size']}):")
            
            for doc in cluster["documents"]:
                doc_str = f"Document [{doc.get('full_corpus_idx', '')}] "
                if doc.get("title"):
                    doc_str += f"(Title: {doc['title']}) "
                doc_str += doc.get("text", "")
                prompt_parts.append(doc_str)
        
        return "\n".join(prompt_parts)

    def _log_organization_stats(self, organized_clusters: List[Dict]):
        """Log statistics about the organized clusters."""
        stats = {
            "num_organized_clusters": len(organized_clusters),
            "docs_per_cluster": [c["size"] for c in organized_clusters],
            "total_selected_docs": sum(c["size"] for c in organized_clusters)
        }
        self.logger.experiment_logger.info(f"Cluster Organization Stats: {stats}")
        self.logger.log_metric("organization_stats", stats)

def find_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int,
    min_clusters: int = 2,
    logger: Optional[ExperimentLogger] = None
) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using elbow method.

    Args:
        embeddings: Document embeddings array
        max_clusters: Maximum number of clusters to try
        min_clusters: Minimum number of clusters to try
        logger: Optional experiment logger

    Returns:
        Tuple of (optimal number of clusters, list of inertia values)
    """
    if logger:
        logger.log_step_start("Finding optimal clusters")
    
    inertias = []
    
    try:
        for k in range(min_clusters, max_clusters + 1):
            clusterer = DocumentClusterer(num_clusters=k, logger=logger)
            _ = clusterer.fit_clusters(embeddings)
            inertias.append(clusterer.get_inertia())
            
        # Find elbow point using rate of change
        diffs = np.diff(inertias)
        rates_of_change = np.diff(diffs)
        optimal_idx = np.argmin(rates_of_change) + min_clusters
        
        if logger:
            logger.log_step_end("Finding optimal clusters")
            logger.log_metric("optimal_clusters", optimal_idx)
            logger.log_metric("inertia_values", inertias)
        
        return optimal_idx, inertias
    
    except Exception as e:
        if logger:
            logger.log_error(e, "Error finding optimal clusters")
        raise