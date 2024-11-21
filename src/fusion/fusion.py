import logging
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import torch
import gc
import json
import mmap
import tempfile
import os
import pickle

@dataclass
class Document:
    """Represents a document with index-based identification."""
    id: str  # Document ID
    text: str  # Document content
    score: float  # Retrieval/ranking score
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class FusionResult:
    """Stores results from fusion process with index tracking."""
    documents: List[Document]
    scores: List[float]
    fusion_method: str
    original_ranking: List[int]  # Original document indices
    original_scores: List[float]  # Original document scores
    metadata: Dict[str, Any]

class RAGFusion:
    """
    Implements different fusion strategies for RAG with memory-efficient processing.
    """
    def __init__(
        self,
        embedding_model: Optional[str] = 'sentence-transformers/all-mpnet-base-v2',
        n_clusters: int = 5,
        fusion_weights: Optional[List[float]] = None,
        experiment_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        diversity_threshold: float = 0.3
    ):
        self.experiment_name = experiment_name or f"rag_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger(f"{__name__}.{self.experiment_name}")
        self._setup_logging()
        
        # Initialize parameters
        self.n_clusters = n_clusters
        self.fusion_weights = fusion_weights or [1.0, 0.8, 0.6, 0.4, 0.2]
        self.batch_size = batch_size
        self.diversity_threshold = diversity_threshold
        
        # Set up cache directory
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="rag_fusion_")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize embeddings model with memory optimization
        if embedding_model:
            self.embed_model = SentenceTransformer(embedding_model)
            if torch.cuda.is_available():
                self.embed_model.to('cuda')
        else:
            self.embed_model = None
            
        self.logger.info(
            f"Initialized RAGFusion with {n_clusters} clusters "
            f"and batch_size={batch_size}"
        )

    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs/fusion")
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

    def _process_in_batches(
        self,
        items: List[Any],
        process_fn: callable,
        desc: str = "Processing"
    ) -> List[Any]:
        """Process items in batches with memory cleanup."""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = process_fn(batch)
            results.extend(batch_results)
            
            if torch.cuda.is_available() and i % (self.batch_size * 5) == 0:
                torch.cuda.empty_cache()
            gc.collect()
            
        return results

    def _compute_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> np.ndarray:
        """Compute embeddings with caching and memory optimization."""
        if not self.embed_model:
            raise ValueError("No embedding model initialized")
            
        # Use memory mapping for cache
        cache_file = Path(self.cache_dir) / "embeddings_cache.mmap"
        if use_cache and cache_file.exists():
            embeddings = np.memmap(
                cache_file, dtype='float32', 
                mode='r', shape=(len(texts), 768)
            )
            return embeddings
            
        embeddings = []
        
        def process_batch(batch):
            with torch.no_grad():
                batch_embeddings = self.embed_model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return batch_embeddings
                
        embeddings = self._process_in_batches(
            texts,
            process_batch,
            desc="Computing embeddings"
        )
        embeddings = np.vstack(embeddings)
        
        # Cache embeddings using memory mapping
        if use_cache:
            mmap_embeddings = np.memmap(
                cache_file, dtype='float32',
                mode='w+', shape=embeddings.shape
            )
            mmap_embeddings[:] = embeddings[:]
            mmap_embeddings.flush()
            
        return embeddings

    def fuse_results(
        self,
        query: str,
        documents: List[Document],
        fusion_method: str = "reciprocal_rank",
        alpha: float = 0.5,
        enforce_diversity: bool = True
    ) -> FusionResult:
        """Fuse documents with memory-efficient processing."""
        try:
            start_time = datetime.now()
            
            # Process documents in batches
            if fusion_method == "reciprocal_rank":
                fused_docs, scores = self._reciprocal_rank_fusion(documents)
            elif fusion_method == "semantic":
                fused_docs, scores = self._semantic_fusion(query, documents)
            elif fusion_method == "hybrid":
                fused_docs, scores = self._hybrid_fusion(query, documents, alpha)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
                
            # Enforce diversity if requested
            if enforce_diversity:
                fused_docs, scores = self._enforce_diversity(fused_docs, scores)
            
            # Track original ordering
            original_ranking = [documents.index(doc) for doc in fused_docs]
            original_scores = [doc.score for doc in fused_docs]
            
            result = FusionResult(
                documents=fused_docs,
                scores=scores,
                fusion_method=fusion_method,
                original_ranking=original_ranking,
                original_scores=original_scores,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'fusion_duration': (datetime.now() - start_time).total_seconds(),
                    'num_documents': len(documents),
                    'fusion_params': {
                        'method': fusion_method,
                        'alpha': alpha,
                        'enforce_diversity': enforce_diversity,
                        'diversity_threshold': self.diversity_threshold
                    }
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in fusion process: {e}")
            raise

    def _enforce_diversity(
        self,
        documents: List[Document],
        scores: List[float]
    ) -> Tuple[List[Document], List[float]]:
        """Enforce diversity with memory efficiency."""
        try:
            if len(documents) <= 1:
                return documents, scores
            
            # Process in batches to avoid memory issues
            diverse_docs = [documents[0]]  # Keep highest scoring document
            diverse_scores = [scores[0]]
            
            def process_batch(batch_docs):
                if not hasattr(batch_docs[0], 'embedding'):
                    # Compute embeddings for batch
                    texts = [doc.text for doc in batch_docs]
                    embeddings = self._compute_embeddings(texts)
                    for doc, emb in zip(batch_docs, embeddings):
                        doc.embedding = emb
                return batch_docs
            
            remaining_docs = documents[1:]
            remaining_scores = scores[1:]
            
            # Process remaining documents in batches
            for i in range(0, len(remaining_docs), self.batch_size):
                batch_docs = remaining_docs[i:i + self.batch_size]
                batch_scores = remaining_scores[i:i + self.batch_size]
                
                # Process batch
                processed_batch = process_batch(batch_docs)
                
                # Check diversity for each document in batch
                for doc, score in zip(processed_batch, batch_scores):
                    # Compute max similarity with already selected documents
                    max_similarity = max(
                        float(cosine_similarity(
                            doc.embedding.reshape(1, -1),
                            selected_doc.embedding.reshape(1, -1)
                        )[0][0])
                        for selected_doc in diverse_docs
                    )
                    
                    if max_similarity < self.diversity_threshold:
                        diverse_docs.append(doc)
                        diverse_scores.append(score)
                
                # Clear memory
                if torch.cuda.is_available() and i % (self.batch_size * 5) == 0:
                    torch.cuda.empty_cache()
                gc.collect()
            
            return diverse_docs, diverse_scores
            
        except Exception as e:
            self.logger.error(f"Error enforcing diversity: {e}")
            raise

    def _reciprocal_rank_fusion(
        self,
        documents: List[Document],
        k: float = 60.0
    ) -> Tuple[List[Document], List[float]]:
        """Memory-efficient reciprocal rank fusion."""
        try:
            # Calculate RRF scores in batches
            doc_scores = {}
            
            def process_batch(batch_docs):
                batch_scores = {}
                for rank, doc in enumerate(batch_docs):
                    batch_scores[doc.id] = 1.0 / (k + rank)
                return batch_scores
            
            # Process in batches
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                batch_scores = process_batch(batch)
                doc_scores.update(batch_scores)
                
                if i % (self.batch_size * 5) == 0:
                    gc.collect()
            
            # Sort by scores
            sorted_items = sorted(
                ((doc, doc_scores[doc.id]) for doc in documents),
                key=lambda x: x[1],
                reverse=True
            )
            
            fused_docs, scores = zip(*sorted_items)
            return list(fused_docs), list(scores)
            
        except Exception as e:
            self.logger.error(f"Error in reciprocal rank fusion: {e}")
            raise

    def _semantic_fusion(
        self,
        query: str,
        documents: List[Document]
    ) -> Tuple[List[Document], List[float]]:
        """Memory-efficient semantic fusion."""
        try:
            if not self.embed_model:
                raise ValueError("Embedding model not initialized")
            
            # Get query embedding
            query_embedding = self._compute_embeddings([query])[0]
            
            # Process documents in batches
            all_similarities = []
            processed_docs = []
            
            def process_batch(batch):
                if not hasattr(batch[0], 'embedding'):
                    # Compute embeddings for batch
                    texts = [doc.text for doc in batch]
                    embeddings = self._compute_embeddings(texts)
                    for doc, emb in zip(batch, embeddings):
                        doc.embedding = emb
                
                # Calculate similarities
                batch_embeddings = np.vstack([doc.embedding for doc in batch])
                similarities = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    batch_embeddings
                )[0]
                
                return similarities
            
            # Process documents in batches
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                batch_similarities = process_batch(batch)
                
                all_similarities.extend(batch_similarities)
                processed_docs.extend(batch)
                
                if i % (self.batch_size * 5) == 0:
                    gc.collect()
            
            # Sort by similarity
            sorted_indices = np.argsort(all_similarities)[::-1]
            fused_docs = [processed_docs[i] for i in sorted_indices]
            scores = [all_similarities[i] for i in sorted_indices]
            
            return fused_docs, scores
            
        except Exception as e:
            self.logger.error(f"Error in semantic fusion: {e}")
            raise

    def _hybrid_fusion(
        self,
        query: str,
        documents: List[Document],
        alpha: float
    ) -> Tuple[List[Document], List[float]]:
        """Memory-efficient hybrid fusion."""
        try:
            # Get scores from both methods
            rrf_docs, rrf_scores = self._reciprocal_rank_fusion(documents)
            sem_docs, sem_scores = self._semantic_fusion(query, documents)
            
            # Normalize scores
            rrf_scores = np.array(rrf_scores)
            sem_scores = np.array(sem_scores)
            rrf_scores = rrf_scores / rrf_scores.sum()
            sem_scores = sem_scores / sem_scores.sum()
            
            # Combine scores
            hybrid_scores = {}
            for doc, rrf_score, sem_score in zip(documents, rrf_scores, sem_scores):
                hybrid_scores[doc.id] = alpha * rrf_score + (1 - alpha) * sem_score
            
            # Sort by combined scores
            sorted_items = sorted(
                ((doc, hybrid_scores[doc.id]) for doc in documents),
                key=lambda x: x[1],
                reverse=True
            )
            
            fused_docs, scores = zip(*sorted_items)
            return list(fused_docs), list(scores)
            
        except Exception as e:
            self.logger.error(f"Error in hybrid fusion: {e}")
            raise

    def save_fusion_state(self, filepath: str):
        """Save fusion state with memory optimization."""
        try:
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare state dictionary with only essential data
            state = {
                'n_clusters': self.n_clusters,
                'fusion_weights': self.fusion_weights,
                'batch_size': self.batch_size,
                'diversity_threshold': self.diversity_threshold,
                'cache_metadata': {
                    'cache_dir': str(self.cache_dir),
                    'files': [f for f in os.listdir(self.cache_dir) if f.endswith('.mmap')]
                },
                'metadata': {
                    'experiment_name': self.experiment_name,
                    'embedding_model': str(self.embed_model),
                    'timestamp': datetime.now().isoformat(),
                    'gpu_available': torch.cuda.is_available(),
                    'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else None
                }
            }
            
            # Save embeddings cache separately if it exists
            cache_file = Path(self.cache_dir) / "embeddings_cache.mmap"
            if cache_file.exists():
                # Create a new directory for cache backup
                cache_backup_dir = save_path.parent / "cache_backup"
                cache_backup_dir.mkdir(exist_ok=True)
                
                # Copy cache file in chunks
                with open(cache_file, 'rb') as src, open(cache_backup_dir / "embeddings_cache.mmap", 'wb') as dst:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        dst.write(chunk)
                
                state['cache_metadata']['backup_location'] = str(cache_backup_dir)
            
            # Save state in chunks using JSON for better memory efficiency
            temp_path = save_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                # Write header
                f.write('{\n')
                
                # Write main configuration
                for key in ['n_clusters', 'fusion_weights', 'batch_size', 'diversity_threshold']:
                    f.write(f'"{key}": {json.dumps(state[key])},\n')
                
                # Write cache metadata
                f.write('"cache_metadata": ')
                json.dump(state['cache_metadata'], f)
                f.write(',\n')
                
                # Write general metadata
                f.write('"metadata": ')
                json.dump(state['metadata'], f)
                f.write('\n}')
            
            # Rename temp file to final file
            os.replace(temp_path, save_path)
            
            # Log success
            self.logger.info(
                f"Saved fusion state to {filepath} "
                f"(cache backup: {state['cache_metadata'].get('backup_location', 'None')})"
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error saving fusion state: {e}")
            # Cleanup temp file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            raise

    def load_fusion_state(self, filepath: str):
        """Load fusion state with memory optimization."""
        try:
            # Read state in chunks
            with open(filepath, 'rb') as f:
                # Memory map the file
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                state = pickle.loads(mm)
                
            self.n_clusters = state['n_clusters']
            self.fusion_weights = state['fusion_weights']
            
            # Log loading
            self.logger.info(
                f"Loaded fusion state from {filepath} "
                f"(saved on {state['metadata']['timestamp']})"
            )
            
        except Exception as e:
            self.logger.error(f"Error loading fusion state: {e}")
            raise
        finally:
            if 'mm' in locals():
                mm.close()

    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion statistics."""
        try:
            stats = {
                'config': {
                    'n_clusters': self.n_clusters,
                    'fusion_weights': self.fusion_weights,
                    'batch_size': self.batch_size,
                    'diversity_threshold': self.diversity_threshold
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
                    'gpu_available': torch.cuda.is_available()
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting fusion stats: {e}")
            return {}

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