from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import os

@dataclass
class ClusteringConfig:
    """Configuration for clustering experiment."""
    
    # Experiment identification
    experiment_name: str = "clustering_experiment"
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Model configuration
    llm_id: str = "meta-llama/Llama-2-7b-chat-hf"
    model_max_length: int = 4096
    max_new_tokens: int = 15
    
    # Clustering configuration
    num_clusters: int = 5
    cluster_seed: int = 42
    use_scaler: bool = True
    min_cluster_size: int = 3
    min_docs_per_cluster: int = 2
    max_docs_per_category: int = 3
    
    # Embeddings configuration
    compute_new_embeddings: bool = True
    embeddings_path: Optional[str] = None
    max_length_encoder: int = 512
    embeddings_output_dir: str = "data/embeddings"
    
    # Data configuration
    base_data_dir: Path = Path("data")
    processed_dir: Path = base_data_dir / "processed"
    use_bm25: bool = False
    use_adore: bool = False
    use_random: bool = False
    use_test: bool = False
    
    # Processing configuration
    batch_size: int = 32
    use_gpu: bool = True
    normalize_embeddings: bool = True
    lower_case: bool = False 
    do_normalize_text: bool = True
    
    # Noise injection configuration
    use_random_docs: bool = False
    random_doc_ratio: float = 0.2
    noise_ratio: float = 0.0
    
    # Output configuration 
    output_dir: str = field(default_factory=lambda: f"experiments/experiment1_clustering/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_intermediates: bool = True
    save_every: int = 250
    save_checkpoints: bool = True

    @property
    def corpus_path(self) -> Path:
        if self.use_random:
            return self.processed_dir / "corpus_with_random_at60.json"
        elif self.use_adore:
            return self.processed_dir / "corpus_with_adore_at200.json"
        return self.processed_dir / "corpus_with_contriever_at150.json"
    
    @property
    def train_dataset_path(self) -> Path:
        return self.base_data_dir / "10k_train_dataset.json"
    
    @property
    def test_dataset_path(self) -> Path:
        return self.base_data_dir / "test_dataset.json"
    
    @property
    def data_path(self) -> Path:
        return self.test_dataset_path if self.use_test else self.train_dataset_path
        
    @property
    def search_results_path(self) -> Path:
        if self.use_random:
            return self.base_data_dir / "10k_random_results_at60.pkl"
        elif self.use_adore:
            return self.base_data_dir / "adore_search_results_at200.pkl"
        elif self.use_bm25:
            return self.base_data_dir / "bm25_test_search_results_at250.pkl"
        return self.base_data_dir / "contriever_search_results_at150.pkl"
    
    @property
    def random_doc_source(self) -> Path:
        return self.base_data_dir / "10k_random_results_at60.pkl"
    
    # Batch size management
    batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 64
    batch_size_reduction_factor: float = 0.5
    clustering_batch_size: int = 1000
    
    def adjust_batch_sizes(self, memory_usage: float):
        """Adjust batch sizes for generation and clustering"""
        if memory_usage > 0.9:
            self.batch_size = max(
                self.min_batch_size,
                int(self.batch_size * self.batch_size_reduction_factor)
            )
            self.clustering_batch_size = max(
                100,
                int(self.clustering_batch_size * self.batch_size_reduction_factor)
            )
        elif memory_usage < 0.7:
            self.batch_size = min(
                self.max_batch_size,
                int(self.batch_size / self.batch_size_reduction_factor)
            )
            self.clustering_batch_size = min(
                5000,
                int(self.clustering_batch_size / self.batch_size_reduction_factor)
            )

    def validate(self):
        """Validate configuration parameters."""
        if self.num_clusters < 2:
            raise ValueError("num_clusters must be at least 2")
            
        if not self.compute_new_embeddings and not self.embeddings_path:
            self.embeddings_path = os.path.join(self.embeddings_output_dir, "document_embeddings.npy")
            self.compute_new_embeddings = True

        if not self.compute_new_embeddings and not os.path.exists(self.embeddings_path):
            raise ValueError(f"Embeddings path does not exist: {self.embeddings_path}")
        
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus not found: {self.corpus_path}")
            
        if not self.data_path.exists():
            raise ValueError(f"Dataset not found: {self.data_path}")
            
        if not self.search_results_path.exists():
            raise ValueError(f"Search results not found: {self.search_results_path}")
            
        if self.use_random_docs and not self.random_doc_source.exists():
            raise ValueError(f"Random document source not found: {self.random_doc_source}")
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.embeddings_output_dir, exist_ok=True)

# Factory for common configurations
class ClusteringConfigFactory:
    @staticmethod
    def get_base_config() -> ClusteringConfig:
        """Get base clustering configuration."""
        return ClusteringConfig()
    
    @staticmethod
    def get_random_config() -> ClusteringConfig:
        """Get configuration with random document injection."""
        return ClusteringConfig(
            use_random_docs=True,
            random_doc_ratio=0.3,
            noise_ratio=0.2
        )
        
    @staticmethod
    def get_test_config() -> ClusteringConfig:
        """Get configuration for test dataset."""
        return ClusteringConfig(
            use_test=True,
            use_bm25=True
        )