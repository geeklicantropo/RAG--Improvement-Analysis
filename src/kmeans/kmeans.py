import logging
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import torch
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
import gc
import os
import tempfile
from collections import defaultdict
import joblib
import mmap

@dataclass
class ClusteringResult:
    """Stores clustering results and metrics with enhanced document tracking."""
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    doc_ids: List[str]  # Track document IDs
    silhouette: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    model: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

class KMeansClustering:
    """
    Enhanced KMeans clustering with memory-efficient processing and document tracking.
    """
    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        max_iter: int = 300,
        experiment_name: Optional[str] = None,
        embedding_model: str = 'sentence-transformers/all-mpnet-base-v2',
        scale_features: bool = True,
        batch_size: int = 32,
        cache_dir: Optional[str] = None
    ):
        self.setup_logging()
        self.logger = logging.getLogger(f"{__name__}.{experiment_name}")
        
        # Initialize parameters
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.experiment_name = experiment_name or f"kmeans_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up cache directory
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="kmeans_")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize embeddings model
        if embedding_model:
            self.embed_model = SentenceTransformer(embedding_model)
            if torch.cuda.is_available():
                self.embed_model.to('cuda')
        else:
            self.embed_model = None
        
        # Initialize clustering model with memory optimization
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            batch_size=min(batch_size * 10, 1024),  # Larger batches for MiniBatchKMeans
            compute_labels=True
        )
        
        self.logger.info(
            f"Initialized KMeansClustering with {n_clusters} clusters "
            f"and batch_size={batch_size}"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs/clustering")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(
            log_dir / f"{self.experiment_name}.log"
        )
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)

    def _compute_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> np.ndarray:
        """Compute embeddings with caching and memory optimization."""
        if not self.embed_model:
            raise ValueError("No embedding model initialized")
            
        # Create memory-mapped cache file
        cache_file = Path(self.cache_dir) / "embeddings_cache.mmap"
        if use_cache and cache_file.exists():
            return np.memmap(
                cache_file, dtype='float32', 
                mode='r', shape=(len(texts), 768)
            )
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                batch_embeddings = self.embed_model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
            
            # Clear GPU memory if needed
            if torch.cuda.is_available() and i % (self.batch_size * 5) == 0:
                torch.cuda.empty_cache()
            gc.collect()
        
        embeddings = np.vstack(all_embeddings)
        
        # Cache embeddings using memory mapping
        if use_cache:
            mmap_embeddings = np.memmap(
                cache_file, dtype='float32',
                mode='w+', shape=embeddings.shape
            )
            mmap_embeddings[:] = embeddings[:]
            mmap_embeddings.flush()
            
        return embeddings

    def fit(
        self,
        texts: Union[List[str], np.ndarray],
        doc_ids: Optional[List[str]] = None,
        compute_metrics: bool = True,
        validate_clustering: bool = True,
        batch_size: Optional[int] = None
    ) -> ClusteringResult:
        """
        Fit KMeans clustering model with memory-efficient batch processing.
        """
        try:
            start_time = datetime.now()
            batch_size = batch_size or self.batch_size
            
            # Convert texts to embeddings if needed
            if isinstance(texts[0], str):
                embeddings = self._compute_embeddings(texts)
            else:
                embeddings = texts
            
            # Assign document IDs if not provided
            if doc_ids is None:
                doc_ids = [str(i) for i in range(len(embeddings))]
            
            # Validate input if requested
            if validate_clustering:
                self._validate_input(embeddings)
            
            # Initialize progress tracking
            total_batches = (len(embeddings) - 1) // batch_size + 1
            
            # Fit MiniBatchKMeans in batches
            for i in tqdm(range(0, len(embeddings), batch_size), total=total_batches, desc="Clustering"):
                batch = embeddings[i:i + batch_size]
                self.model.partial_fit(batch)
                
                # Clear memory
                if torch.cuda.is_available() and i % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Get clustering results
            labels = self.model.predict(embeddings)
            centroids = self.model.cluster_centers_
            inertia = self.model.inertia_
            
            # Compute clustering metrics if requested
            silhouette = None
            calinski_harabasz = None
            if compute_metrics and len(np.unique(labels)) > 1:
                # Compute metrics in batches for memory efficiency
                silhouette_scores = []
                ch_scores = []
                
                for i in range(0, len(embeddings), batch_size):
                    batch_emb = embeddings[i:i + batch_size]
                    batch_labels = labels[i:i + batch_size]
                    
                    if len(np.unique(batch_labels)) > 1:
                        silhouette_scores.append(silhouette_score(batch_emb, batch_labels))
                        ch_scores.append(calinski_harabasz_score(batch_emb, batch_labels))
                
                silhouette = np.mean(silhouette_scores)
                calinski_harabasz = np.mean(ch_scores)
            
            # Create result with document tracking
            result = ClusteringResult(
                labels=labels,
                centroids=centroids,
                inertia=inertia,
                doc_ids=doc_ids,
                silhouette=silhouette,
                calinski_harabasz=calinski_harabasz,
                model=self.model,
                metadata={
                    'duration': (datetime.now() - start_time).total_seconds(),
                    'n_clusters': self.n_clusters,
                    'embedding_model': str(self.embed_model),
                    'cluster_sizes': [int(sum(labels == i)) for i in range(self.n_clusters)]
                }
            )
            
            # Log clustering statistics
            self._log_cluster_statistics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during clustering: {e}")
            raise

    def predict(
        self,
        texts: Union[List[str], np.ndarray],
        return_distances: bool = False,
        batch_size: Optional[int] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict cluster labels for new data with batch processing."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        try:
            batch_size = batch_size or self.batch_size
            
            # Convert texts to embeddings if needed
            if isinstance(texts[0], str):
                embeddings = self._compute_embeddings(texts)
            else:
                embeddings = texts
            
            # Predict in batches
            all_labels = []
            all_distances = [] if return_distances else None
            
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                
                if return_distances:
                    labels, distances = self.model.predict(batch, return_distances)
                    all_labels.extend(labels)
                    all_distances.extend(distances)
                else:
                    labels = self.model.predict(batch)
                    all_labels.extend(labels)
                
                # Clear memory
                if torch.cuda.is_available() and i % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
                gc.collect()
            
            if return_distances:
                return np.array(all_labels), np.array(all_distances)
            return np.array(all_labels)
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def _validate_input(self, X: np.ndarray):
        """Validate input data for clustering."""
        if len(X) < self.n_clusters:
            raise ValueError(
                f"Number of samples {len(X)} is less than number of clusters {self.n_clusters}"
            )
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("Input data contains infinite values")

    def _log_cluster_statistics(self, result: ClusteringResult):
        """Log detailed statistics about the clusters."""
        self.logger.info("\nCluster Statistics:")
        self.logger.info("-" * 50)
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = result.labels == cluster_id
            cluster_size = sum(cluster_mask)
            
            # Calculate cluster statistics
            if result.silhouette is not None:
                cluster_stats = {
                    'size': cluster_size,
                    'percentage': (cluster_size / len(result.labels)) * 100,
                    'silhouette': result.silhouette,
                    'calinski_harabasz': result.calinski_harabasz
                }
            else:
                cluster_stats = {
                    'size': cluster_size,
                    'percentage': (cluster_size / len(result.labels)) * 100
                }
            
            self.logger.info(f"\nCluster {cluster_id}:")
            for metric, value in cluster_stats.items():
                self.logger.info(f"- {metric}: {value:.4f}")

    def save_model(self, filepath: str):
        """Save the model and metadata with memory optimization."""
        try:
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare model data
            model_data = {
                'model': self.model,
                'n_clusters': self.n_clusters,
                'random_state': self.random_state,
                'max_iter': self.max_iter,
                'embedding_model': str(self.embed_model),
                'metadata': {
                    'experiment_name': self.experiment_name,
                    'timestamp': datetime.now().isoformat(),
                    'cache_dir': str(self.cache_dir)
                }
            }
            
            # Save in chunks using joblib
            joblib.dump(model_data, filepath, compress=3)
            
            # Save embeddings cache if it exists
            cache_file = Path(self.cache_dir) / "embeddings_cache.mmap"
            if cache_file.exists():
                cache_backup = save_path.parent / "embeddings_cache.mmap"
                with open(cache_file, 'rb') as src, open(cache_backup, 'wb') as dst:
                    for chunk in iter(lambda: src.read(1024*1024), b''):
                        dst.write(chunk)
            
            self.logger.info(f"Saved model to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filepath: str):
        """Load model with memory optimization."""
        try:
            model_data = joblib.load(filepath)
            
            # Load model components
            self.model = model_data['model']
            self.n_clusters = model_data['n_clusters']
            self.random_state = model_data['random_state']
            self.max_iter = model_data['max_iter']
            
            # Restore cache if available
            cache_backup = Path(filepath).parent / "embeddings_cache.mmap"
            if cache_backup.exists():
                cache_file = Path(self.cache_dir) / "embeddings_cache.mmap"
                with open(cache_backup, 'rb') as src, open(cache_file, 'wb') as dst:
                    for chunk in iter(lambda: src.read(1024*1024), b''):
                        dst.write(chunk)
            
            self.logger.info(
                f"Loaded model from {filepath} "
                f"(saved on {model_data['metadata']['timestamp']})"
            )
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def cleanup_cache(self):
        """Clean up cache files."""
        try:
            # Remove all files in cache directory
            for file in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    self.logger.warning(f"Error deleting {file_path}: {e}")
            
            # Try to remove cache directory
            try:
                os.rmdir(self.cache_dir)
            except OSError:
                pass
                
            self.logger.info("Cache cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get comprehensive clustering statistics."""
        try:
            if self.model is None:
                return {'status': 'Model not fitted'}
                
            stats = {
                'config': {
                    'n_clusters': self.n_clusters,
                    'batch_size': self.batch_size,
                    'random_state': self.random_state,
                    'max_iter': self.max_iter
                },
                'model_stats': {
                    'inertia': float(self.model.inertia_),
                    'n_iter': int(self.model.n_iter_),
                    'cluster_centers': self.model.cluster_centers_.shape
                },
                'cache': {
                    'location': self.cache_dir,
                    'size_bytes': sum(
                        os.path.getsize(os.path.join(self.cache_dir, f))
                        for f in os.listdir(self.cache_dir)
                    )
                },
                'runtime': {
                    'experiment_name': self.experiment_name,
                    'embedding_model': str(self.embed_model),
                    'gpu_available': torch.cuda.is_available(),
                    'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else None
                }
            }
            
            # Add cluster size distribution if model is fitted
            if hasattr(self.model, 'labels_'):
                cluster_sizes = np.bincount(self.model.labels_)
                stats['cluster_distribution'] = {
                    'sizes': cluster_sizes.tolist(),
                    'min_size': int(cluster_sizes.min()),
                    'max_size': int(cluster_sizes.max()),
                    'mean_size': float(cluster_sizes.mean()),
                    'std_size': float(cluster_sizes.std())
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cluster stats: {e}")
            return {'error': str(e)}

    def get_diverse_samples(
        self,
        texts: List[str],
        k: int,
        doc_ids: Optional[List[str]] = None,
        min_samples_per_cluster: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> List[str]:
        """Select diverse samples using clustering with memory efficiency."""
        try:
            batch_size = batch_size or self.batch_size
            
            # Get embeddings
            embeddings = self._compute_embeddings(texts)
            
            # Predict clusters
            labels = self.predict(embeddings)
            
            selected_indices = []
            remaining_slots = k
            
            # Ensure minimum samples per cluster if specified
            if min_samples_per_cluster is not None:
                for cluster_id in range(self.n_clusters):
                    cluster_mask = labels == cluster_id
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    if len(cluster_indices) > 0:
                        # Process cluster samples in batches
                        cluster_distances = []
                        
                        for i in range(0, len(cluster_indices), batch_size):
                            batch_indices = cluster_indices[i:i + batch_size]
                            batch_embeddings = embeddings[batch_indices]
                            
                            # Calculate distances to centroid
                            centroid = self.model.cluster_centers_[cluster_id]
                            distances = np.linalg.norm(
                                batch_embeddings - centroid,
                                axis=1
                            )
                            
                            cluster_distances.extend(list(zip(batch_indices, distances)))
                            
                            # Clear memory
                            if self.enable_gc and i % (batch_size * 5) == 0:
                                gc.collect()
                        
                        # Sort by distance and select samples
                        sorted_samples = sorted(cluster_distances, key=lambda x: x[1])
                        n_samples = min(min_samples_per_cluster, len(sorted_samples))
                        selected_indices.extend([idx for idx, _ in sorted_samples[:n_samples]])
                        remaining_slots -= n_samples
            
            # Fill remaining slots with most representative samples
            if remaining_slots > 0:
                # Calculate distances for remaining samples
                remaining_indices = list(set(range(len(texts))) - set(selected_indices))
                distances_to_centroid = []
                
                for i in range(0, len(remaining_indices), batch_size):
                    batch_indices = remaining_indices[i:i + batch_size]
                    batch_embeddings = embeddings[batch_indices]
                    batch_labels = labels[batch_indices]
                    
                    for idx, embedding, label in zip(batch_indices, batch_embeddings, batch_labels):
                        centroid = self.model.cluster_centers_[label]
                        distance = np.linalg.norm(embedding - centroid)
                        distances_to_centroid.append((idx, distance))
                    
                    # Clear memory
                    if self.enable_gc and i % (batch_size * 5) == 0:
                        gc.collect()
                
                # Sort by distance and select remaining samples
                sorted_samples = sorted(distances_to_centroid, key=lambda x: x[1])
                additional_indices = [x[0] for x in sorted_samples[:remaining_slots]]
                selected_indices.extend(additional_indices)
            
            # Return selected texts and optionally doc_ids
            if doc_ids:
                return [doc_ids[i] for i in selected_indices]
            return [texts[i] for i in selected_indices]
            
        except Exception as e:
            self.logger.error(f"Error selecting diverse samples: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_cache()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup_cache()
        except:
            pass  # Ignore cleanup errors during deletion