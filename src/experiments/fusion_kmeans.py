import logging
import logging.config
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from rich.progress import track, Progress
from rich.console import Console
import gc
import argparse
import json
from collections import defaultdict
import tempfile
import os
import mmap
from contextlib import contextmanager

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.llm import LLM, GenerationResult
from src.utils.retriever import Retriever, RetrievalResult
from src.kmeans.kmeans import KMeansClustering, ClusteringResult
from src.fusion.fusion import RAGFusion, Document, FusionResult
from src.utils.id_mapping import IDMapper
from src.core.env.env_loader import load_environment
from src.core.utils.system_check import perform_system_checks
from src.config import paths, settings

@dataclass
class FusionKMeansResult:
    """Stores fusion-kmeans experiment results with memory-efficient tracking."""
    query: str
    query_idx: int  # Dataset index
    query_cluster: int
    retrieved_docs: List[RetrievalResult]
    fused_docs: List[Document]
    cluster_docs: List[Document]
    final_docs: List[Document]
    generation: GenerationResult
    fusion: FusionResult
    clustering: Optional[ClusteringResult] = None
    original_doc_indices: List[int] = None  # Original corpus indices
    gold_doc_idx: Optional[int] = None
    metrics: Dict[str, Any] = None
    attention_metrics: Optional[Dict[str, Any]] = None

class FusionKMeansRAG:
    """
    Enhanced Fusion-KMeans RAG implementation with memory-efficient processing.
    """
    def __init__(
        self,
        retriever_type: str = "contriever",
        n_clusters: int = 5,
        fusion_weights: Optional[List[float]] = None,
        embedding_model: str = 'sentence-transformers/all-mpnet-base-v2',
        experiment_name: Optional[str] = None,
        batch_size: int = 8,
        id_mapper: Optional[IDMapper] = None,
        enable_gc: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.setup_logging()
        self.console = Console()
        self.logger = logging.getLogger(f"{__name__}.{experiment_name}")
        
        # Initialize parameters
        self.retriever_type = retriever_type
        self.n_clusters = n_clusters
        self.fusion_weights = fusion_weights or [1.0, 0.8, 0.6, 0.4, 0.2]
        self.batch_size = batch_size
        self.enable_gc = enable_gc
        self.experiment_name = experiment_name or f"fusion_kmeans_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        
        # Setup cache directory
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="fusion_kmeans_")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize components
        self.id_mapper = id_mapper
        
        # Setup metrics tracking
        self.metrics_file = Path(self.cache_dir) / "metrics.mmap"
        self.metrics = self._init_metrics()
        
        # Create experiment directories
        self.setup_directories()
        
        self.logger.info(
            f"Initialized FusionKMeansRAG with {retriever_type} retriever "
            f"(clusters={n_clusters}, batch_size={batch_size})"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        with open(paths.LOGGING_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)

    def setup_directories(self):
        """Create necessary directories."""
        try:
            self.results_dir = paths.FUSION_RESULTS_DIR / "kmeans" / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.models_dir = self.results_dir / "models"
            self.indices_dir = self.results_dir / "indices"
            self.fusion_dir = self.results_dir / "fusion"
            self.clusters_dir = self.results_dir / "clusters"
            self.temp_dir = self.results_dir / "temp"
            
            for dir_path in [
                self.results_dir,
                self.models_dir,
                self.indices_dir,
                self.fusion_dir,
                self.clusters_dir,
                self.temp_dir
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Created results directory: {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            raise

    def _init_metrics(self) -> Dict[str, Any]:
        """Initialize metrics with memory mapping."""
        base_metrics = {
            'experiment_type': 'fusion_kmeans_rag',
            'retriever_type': self.retriever_type,
            'n_clusters': self.n_clusters,
            'fusion_weights': self.fusion_weights,
            'start_time': self.start_time.isoformat(),
            'batch_size': self.batch_size,
            'processing_stats': defaultdict(float),
            'sample_metrics': []
        }
        
        # Save initial metrics to memory-mapped file
        with open(self.metrics_file, 'wb') as f:
            f.write(json.dumps(base_metrics).encode())
            
        return base_metrics

    @contextmanager
    def _update_metrics(self):
        """Context manager for thread-safe metrics updates."""
        try:
            with open(self.metrics_file, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                metrics = json.loads(mm.read().decode())
                yield metrics
                
                # Write updated metrics
                mm.seek(0)
                mm.write(json.dumps(metrics).encode())
                mm.flush()
        finally:
            if 'mm' in locals():
                mm.close()

    def initialize_components(
        self,
        index_dir: Optional[str] = None,
        validate_indices: bool = True
    ):
        """Initialize all required components with index validation."""
        try:
            self.logger.info("Initializing components...")
            
            # Initialize device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize retriever
            self.retriever = Retriever(
                retriever_type=self.retriever_type,
                device=device,
                batch_size=self.batch_size,
                cache_embeddings=True
            )
            
            # Initialize LLM
            self.generator = LLM(
                model_id=settings.LLM_MODEL_ID,
                device=device,
                quantization_bits=4,
                model_max_length=settings.MODEL_MAX_LENGTH,
                experiment_name=self.experiment_name
            )
            
            # Initialize KMeans
            self.kmeans = KMeansClustering(
                n_clusters=self.n_clusters,
                embedding_model=self.embedding_model,
                experiment_name=self.experiment_name,
                batch_size=self.batch_size,
                cache_dir=self.cache_dir
            )
            
            # Initialize RAG Fusion
            self.fusion = RAGFusion(
                embedding_model=self.embedding_model,
                n_clusters=self.n_clusters,
                fusion_weights=self.fusion_weights,
                experiment_name=self.experiment_name,
                batch_size=self.batch_size,
                cache_dir=self.cache_dir
            )
            
            # Load index mappings if provided
            if index_dir:
                index_dir = Path(index_dir)
                if validate_indices:
                    self._validate_indices(index_dir)
                self.id_mapper.load_mappings(index_dir)
                self.logger.info(f"Loaded and validated indices from {index_dir}")
            
            # Update metrics
            with self._update_metrics() as metrics:
                metrics.update({
                    'model_id': settings.LLM_MODEL_ID,
                    'model_max_length': settings.MODEL_MAX_LENGTH,
                    'device': device
                })
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise

    def _validate_indices(self, index_dir: Path) -> bool:
        """Validate index mappings before loading."""
        try:
            required_files = [
                'corpus_idx_mapping.json',
                'dataset_idx_mapping.json',
                'search_idx_mapping.json'
            ]
            
            for filename in required_files:
                file_path = index_dir / filename
                if not file_path.exists():
                    raise ValueError(f"Missing required index file: {filename}")
                
                # Validate file structure using streaming
                with open(file_path, 'r') as f:
                    # Read and validate header
                    header = ''
                    for _ in range(100):  # Read first 100 lines max
                        line = f.readline()
                        header += line
                        if '"mappings"' in header:
                            break
                    else:
                        raise ValueError(f"Invalid mapping format in {filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating indices: {e}")
            raise

    def run_experiment(
        self,
        dataset_path: str,
        num_samples: int = -1,
        batch_size: Optional[int] = None,
        inject_random: bool = False,
        random_ratio: float = 0.3,
        index_dir: Optional[str] = None
    ) -> List[FusionKMeansResult]:
        """
        Run the Fusion-KMeans RAG experiment with memory-efficient batch processing.
        """
        try:
            # Initialize components
            self.initialize_components(index_dir)
            
            # Initialize dataset iterator
            dataset_iterator = self._get_dataset_iterator(
                dataset_path, 
                batch_size or self.batch_size
            )
            
            results = []
            total_samples = 0
            batch_idx = 0
            
            # Process dataset in batches
            for batch_data in track(
                dataset_iterator,
                description="Processing batches"
            ):
                # Process batch
                batch_results = self._process_batch(
                    batch_data,
                    inject_random=inject_random,
                    random_ratio=random_ratio
                )
                
                if batch_results:
                    # Save batch results
                    self._save_batch_results(batch_results, batch_idx)
                    results.extend(batch_results)
                    
                    # Update progress
                    total_samples += len(batch_results)
                    if num_samples > 0 and total_samples >= num_samples:
                        break
                    
                    # Memory cleanup
                    if self.enable_gc and batch_idx % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                batch_idx += 1
            
            # Update final metrics
            with self._update_metrics() as metrics:
                metrics.update({
                    'total_samples': len(results),
                    'end_time': datetime.now().isoformat(),
                    'total_duration': (
                        datetime.now() - self.start_time
                    ).total_seconds()
                })
            
            # Save final results
            self._save_final_results()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in experiment: {e}")
            raise

    def _get_dataset_iterator(self, dataset_path: str, batch_size: int):
        """Get memory-efficient dataset iterator."""
        try:
            current_batch = []
            
            with open(dataset_path, 'r') as f:
                # Skip header
                f.readline()
                
                for line in f:
                    if line.strip() == ']':  # End of array
                        break
                    
                    # Remove trailing comma
                    line = line.rstrip(',\n')
                    
                    try:
                        item = json.loads(line)
                        current_batch.append(item)
                        
                        if len(current_batch) >= batch_size:
                            yield current_batch
                            current_batch = []
                            
                    except json.JSONDecodeError:
                        self.logger.warning(f"Error parsing line: {line}")
                        continue
            
            # Yield remaining items
            if current_batch:
                yield current_batch
                
        except Exception as e:
            self.logger.error(f"Error in dataset iterator: {e}")
            raise

    def _process_batch(
        self,
        batch_data: List[Dict],
        inject_random: bool = False,
        random_ratio: float = 0.3
    ) -> List[FusionKMeansResult]:
        """Process a batch of queries with memory efficiency."""
        batch_results = []
        batch_start = datetime.now()
        
        for item in batch_data:
            try:
                # Extract query information
                query = item['question']
                query_idx = int(item.get('example_id', -1))
                gold_doc_idx = int(item.get('idx_gold_in_corpus', -1))
                
                # Track original corpus index
                if gold_doc_idx >= 0:
                    gold_doc_idx = self.id_mapper.get_index_mapping(
                        gold_doc_idx,
                        'dataset_idx_to_corpus'
                    ) or gold_doc_idx
                
                # Initial retrieval
                retrieval_start = datetime.now()
                retrieved_docs = self.retriever.retrieve(
                    query,
                    top_k=self.n_clusters * 3,
                    include_metadata=True
                )
                retrieval_time = (datetime.now() - retrieval_start).total_seconds()
                
                # Track corpus indices
                original_indices = []
                for doc in retrieved_docs:
                    corpus_idx = self.id_mapper.get_index_mapping(
                        doc.corpus_idx,
                        'search_to_corpus_idx'
                    ) or doc.corpus_idx
                    original_indices.append(corpus_idx)
                
                # Convert to fusion Document format
                documents = [
                    Document(
                        id=doc.corpus_idx,
                        text=doc.text,
                        score=doc.score,
                        metadata={
                            **doc.metadata,
                            'corpus_idx': self.id_mapper.get_index_mapping(
                                doc.corpus_idx,
                                'search_to_corpus_idx'
                            ) or doc.corpus_idx
                        }
                    )
                    for doc in retrieved_docs
                ]
                
                # Apply fusion
                fusion_start = datetime.now()
                fusion_result = self.fusion.fuse_results(
                    query,
                    documents,
                    fusion_method="reciprocal_rank"
                )
                fusion_time = (datetime.now() - fusion_start).total_seconds()
                
                # Perform clustering
                clustering_start = datetime.now()
                clustering_result = self.kmeans.fit(
                    np.array([doc.embedding for doc in fusion_result.documents]),
                    doc_ids=[doc.id for doc in fusion_result.documents]
                )
                
                # Assign clusters to documents
                for doc, label in zip(fusion_result.documents, clustering_result.labels):
                    doc.cluster_id = int(label)
                
                # Get query cluster
                query_embedding = self.fusion._get_query_embedding(query)
                query_cluster = self.kmeans.predict(query_embedding.reshape(1, -1))[0]
                
                # Get documents from query cluster
                cluster_docs = [
                    doc for doc in fusion_result.documents
                    if doc.cluster_id == query_cluster
                ]
                cluster_docs = sorted(cluster_docs, key=lambda x: x.score, reverse=True)[:self.n_clusters]
                
                clustering_time = (datetime.now() - clustering_start).total_seconds()
                
                # Final document set
                final_docs = cluster_docs.copy()
                num_random = 0
                
                # Add random documents if requested
                if inject_random and random_ratio > 0:
                    num_random = int(len(cluster_docs) * random_ratio)
                    if num_random > 0:
                        random_docs = self.retriever.get_random_documents(num_random)
                        random_fusion_docs = [
                            Document(
                                id=doc.corpus_idx,
                                text=doc.text,
                                score=0.0,
                                cluster_id=-1,  # Special cluster for random docs
                                metadata={
                                    'corpus_idx': self.id_mapper.get_index_mapping(
                                        doc.corpus_idx,
                                        'random_to_corpus_idx'
                                    ) or doc.corpus_idx,
                                    'is_random': True
                                }
                            )
                            for doc in random_docs
                        ]
                        final_docs.extend(random_fusion_docs)
                        
                        # Track random document indices
                        for doc in random_docs:
                            corpus_idx = self.id_mapper.get_index_mapping(
                                doc.corpus_idx,
                                'random_to_corpus_idx'
                            ) or doc.corpus_idx
                            original_indices.append(corpus_idx)
                
                # Generate response
                generation_start = datetime.now()
                context = self._prepare_context(final_docs, query_cluster)
                generation_result = self.generator.generate(
                    f"Question: {query}\n\nContext: {context}\n\nAnswer:",
                    doc_ids=[doc.id for doc in final_docs]
                )
                generation_time = (datetime.now() - generation_start).total_seconds()
                
                # Analyze attention patterns if available
                attention_metrics = self._analyze_attention(
                    generation_result,
                    final_docs
                ) if hasattr(generation_result, 'attention_weights') else None
                
                # Record metrics
                metrics = {
                    'query_idx': query_idx,
                    'query_cluster': query_cluster,
                    'retrieval_time': retrieval_time,
                    'fusion_time': fusion_time,
                    'clustering_time': clustering_time,
                    'generation_time': generation_time,
                    'total_time': (datetime.now() - batch_start).total_seconds(),
                    'num_retrieved_docs': len(retrieved_docs),
                    'num_fused_docs': len(fusion_result.documents),
                    'num_cluster_docs': len(cluster_docs),
                    'num_random_docs': num_random,
                    'retrieved_doc_scores': [doc.score for doc in retrieved_docs],
                    'fusion_scores': fusion_result.scores,
                    'cluster_doc_scores': [doc.score for doc in cluster_docs],
                    'gold_doc_retrieved': gold_doc_idx in original_indices,
                    'gold_doc_cluster': next(
                        (doc.cluster_id for doc in fusion_result.documents
                         if self.id_mapper.get_index_mapping(doc.id, 'search_to_corpus_idx') == gold_doc_idx),
                        None
                    ),
                    'clustering_metrics': {
                        'silhouette': clustering_result.silhouette,
                        'calinski_harabasz': clustering_result.calinski_harabasz,
                        'cluster_sizes': clustering_result.metadata.get('cluster_sizes', [])
                    },
                    'fusion_metrics': fusion_result.metadata
                }
                
                # Create result object
                result = FusionKMeansResult(
                    query=query,
                    query_idx=query_idx,
                    query_cluster=query_cluster,
                    retrieved_docs=retrieved_docs,
                    fused_docs=fusion_result.documents,
                    cluster_docs=cluster_docs,
                    final_docs=final_docs,
                    generation=generation_result,
                    fusion=fusion_result,
                    clustering=clustering_result,
                    original_doc_indices=original_indices,
                    gold_doc_idx=gold_doc_idx,
                    metrics=metrics,
                    attention_metrics=attention_metrics
                )
                
                batch_results.append(result)
                
                # Clean memory after each query
                if self.enable_gc:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            except Exception as e:
                self.logger.error(f"Error processing query '{query}': {e}")
                continue
        
        return batch_results

    def _prepare_context(
        self,
        documents: List[Document],
        query_cluster: int
    ) -> str:
        """Prepare context string with memory efficiency."""
        context_parts = [f"[Cluster {query_cluster}] Retrieved and processed documents:"]
        
        for idx, doc in enumerate(documents, 1):
            # Get corpus index mapping
            corpus_idx = self.id_mapper.get_index_mapping(
                doc.id,
                'search_to_corpus_idx'
            ) or doc.id
            
            # Add cluster and document information
            cluster_info = f"Cluster {doc.cluster_id}" if hasattr(doc, 'cluster_id') else "No Cluster"
            context_parts.append(
                f"Document [ID: {corpus_idx}] "
                f"(Score: {doc.score:.3f}, {cluster_info}): {doc.text}"
            )
        
        return "\n\n".join(context_parts)

    def _analyze_attention(
        self,
        generation: GenerationResult,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """Analyze attention patterns with memory efficiency."""
        try:
            # Calculate document lengths
            doc_lengths = [len(doc.text.split()) for doc in documents]
            
            # Get attention analysis from LLM
            attention_analysis = self.generator.analyze_attention_patterns(
                generation,
                doc_lengths=doc_lengths
            )
            
            # Add document-specific metrics
            if 'doc_attentions' in attention_analysis:
                for doc_idx, attention in enumerate(attention_analysis['doc_attentions']):
                    doc = documents[doc_idx]
                    corpus_idx = self.id_mapper.get_index_mapping(
                        doc.id,
                        'search_to_corpus_idx'
                    ) or doc.id
                    
                    attention_analysis[f'doc_{corpus_idx}'] = {
                        'attention': float(attention),
                        'score': doc.score,
                        'position': doc_idx,
                        'length': doc_lengths[doc_idx],
                        'cluster_info': {
                            'cluster_id': getattr(doc, 'cluster_id', None),
                            'fusion_score': getattr(doc, 'fusion_score', None)
                        }
                    }
            
            return attention_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing attention patterns: {e}")
            return {}

    def _save_batch_results(
        self,
        results: List[FusionKMeansResult],
        batch_idx: int
    ):
        """Save batch results with memory efficiency."""
        try:
            # Save to temporary file first
            temp_file = self.temp_dir / f"batch_{batch_idx}_temp.json"
            results_file = self.results_dir / f"batch_{batch_idx}.json"
            
            # Convert results to serializable format in chunks
            with open(temp_file, 'w') as f:
                f.write('[\n')
                
                for idx, result in enumerate(results):
                    result_dict = {
                        'query': result.query,
                        'query_idx': result.query_idx,
                        'query_cluster': result.query_cluster,
                        'original_doc_indices': result.original_doc_indices,
                        'gold_doc_idx': result.gold_doc_idx,
                        'metrics': result.metrics,
                        'attention_metrics': result.attention_metrics,
                        'generation': {
                            'text': result.generation.text,
                            'tokens_used': result.generation.tokens_used,
                            'generation_time': result.generation.generation_time
                        }
                    }
                    
                    f.write(json.dumps(result_dict))
                    if idx < len(results) - 1:
                        f.write(',\n')
                    
                f.write('\n]')
            
            # Rename temp file to final file
            os.replace(temp_file, results_file)
            
            # Update metrics
            with self._update_metrics() as metrics:
                batch_metrics = {
                    'batch_idx': batch_idx,
                    'batch_size': len(results),
                    'timestamp': datetime.now().isoformat()
                }
                metrics['sample_metrics'].append(batch_metrics)
                
            self.logger.info(f"Saved results for batch {batch_idx}")
            
        except Exception as e:
            self.logger.error(f"Error saving batch results: {e}")
            # Clean up temp file if it exists
            if 'temp_file' in locals():
                try:
                    temp_file.unlink()
                except:
                    pass
            raise

    def _save_final_results(self):
        """Save final results and metadata."""
        try:
            # Save metrics
            metrics_path = self.results_dir / "metrics.json"
            with open(self.metrics_file, 'rb') as f:
                metrics = json.loads(f.read().decode())
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save models
            self.kmeans.save_model(self.models_dir / "kmeans_model.joblib")
            self.fusion.save_fusion_state(self.models_dir / "fusion_state.pkl")
            
            # Save experiment configuration
            config = {
                'retriever_type': self.retriever_type,
                'n_clusters': self.n_clusters,
                'fusion_weights': self.fusion_weights,
                'batch_size': self.batch_size,
                'model_config': {
                    'model_id': settings.LLM_MODEL_ID,
                    'model_max_length': settings.MODEL_MAX_LENGTH
                },
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat()
            }
            
            config_path = self.results_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            self.logger.info(f"Saved final results to {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving final results: {e}")
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            # Remove temp directory
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                for file in self.temp_dir.iterdir():
                    file.unlink()
                self.temp_dir.rmdir()
            
            # Remove cache directory
            if hasattr(self, 'cache_dir') and os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    os.unlink(os.path.join(self.cache_dir, file))
                os.rmdir(self.cache_dir)
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run Fusion-KMeans RAG experiment"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/gen_res',
        help='Output directory'
    )
    parser.add_argument(
        '--experiment_mode',
        type=str,
        default='fusion_kmeans',
        help='Experiment mode'
    )
    parser.add_argument(
        '--retriever_type',
        type=str,
        choices=['contriever', 'bm25', 'adore'],
        default='contriever',
        help='Type of retriever to use'
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=5,
        help='Number of clusters'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--index_dir',
        type=str,
        help='Directory containing index mapping files'
    )
    parser.add_argument(
        '--enable_gc',
        type=bool,
        default=True,
        help='Enable garbage collection for memory optimization'
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    try:
        # Load environment variables
        load_environment()
        
        # Perform system checks
        perform_system_checks()
        
        # Initialize experiment
        experiment = FusionKMeansRAG(
            retriever_type=args.retriever_type,
            n_clusters=args.n_clusters,
            batch_size=args.batch_size,
            enable_gc=args.enable_gc
        )
        
        # Run experiment with context manager for automatic cleanup
        with experiment:
            results = experiment.run_experiment(
                dataset_path=str(paths.TEST_DATASET_PATH),
                num_samples=settings.get('NUM_SAMPLES', -1),
                inject_random=settings.get('RANDOM_INJECTION', {}).get('enabled', False),
                random_ratio=settings.get('RANDOM_INJECTION', {}).get('injection_ratio', 0.3),
                index_dir=args.index_dir
            )
        
        # Log completion
        duration = datetime.now() - start_time
        experiment.logger.info(f"Fusion-KMeans experiment completed in {duration}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()