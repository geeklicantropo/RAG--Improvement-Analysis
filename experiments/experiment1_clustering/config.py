from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import os
import json

@dataclass
class ClusteringConfig:
    """Configuration for clustering experiments."""
    
    # Experiment identification
    experiment_name: str = "clustering_experiment"
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Model Configuration  
    llm_id: str = "gemini-1.5-flash"
    embedding_model: str = "gemini-1.5-flash"
    model_max_length: int = 30720
    max_new_tokens: int = 1024
    base_corpus_size: int = 1000

    # Encoder configuration 
    encoder_id: str = 'facebook/contriever'
    max_length_encoder: int = 512
    normalize_embeddings: bool = True
    
    # Memory Configuration
    gpu_memory_utilization: float = 0.8 
    gpu_memory_threshold: float = 0.9
    cpu_memory_threshold: float = 0.9
    memory_warning_threshold: float = 0.7
    memory_error_threshold: float = 0.9
    memory_logging_interval: int = 100

    # Clustering Configuration
    num_clusters: int = 10
    cluster_seed: int = 42
    method: str = "kmeans"
    use_scaler: bool = True
    min_cluster_size: int = 5
    min_docs_per_cluster: int = 5
    max_docs_per_category: int = 10

    # Document Configuration
    num_documents_in_context: int = 50
    max_doc_length: int = 512

    # Embeddings Configuration
    compute_new_embeddings: bool = True
    embeddings_path: Optional[str] = None
    max_length_encoder: int = 512
    embeddings_output_dir: str = "data/embeddings"

    # Noise Configuration
    inject_noise: bool = False
    noise_ratio: float = 0.2
    noise_seed: int = 42
    min_noise_docs: int = 1
    max_noise_docs: int = 5

    # Retrieval Configuration
    use_bm25: bool = False
    use_adore: bool = False
    use_random: bool = False
    use_test: bool = False

    # Evaluation Metrics
    silhouette_enabled: bool = True
    calinski_harabasz_enabled: bool = True
    davies_bouldin_enabled: bool = True

    # Data Paths
    base_data_dir: Path = Path("data")
    processed_dir: Path = base_data_dir / "processed"

    # Processing Configuration
    output_dir: Path = Path("experiments/experiment1_clustering/results")
    batch_size: int = 256
    min_batch_size: int = 8
    max_batch_size: int = 512
    batch_size_reduction_factor: float = 0.5
    clustering_batch_size: int = 100
    save_every: int = 100
    memory_usage_check_interval: int = 10

    @property
    def corpus_path(self) -> Path:
        """Path to the document corpus."""
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
    def contriever_results_path(self) -> Path:
        return self.base_data_dir / "contriever_search_results_at150.pkl"
    
    @property
    def bm25_results_path(self) -> Path:
        return self.base_data_dir / "bm25_test_search_results_at250.pkl"
    
    @property
    def random_results_path(self) -> Path:
        return self.base_data_dir / "10k_random_results_at60.pkl"
    
    @property
    def data_paths(self) -> Dict[str, Path]:
        return {
            'train': self.train_dataset_path,
            'test': self.test_dataset_path,
            'corpus': self.corpus_path,
            'contriever_results': self.contriever_results_path,
            'bm25_results': self.bm25_results_path,
            'random_results': self.random_results_path
    }

    def adjust_batch_sizes(self, memory_usage: float):
        """Adjust batch sizes based on memory usage."""
        if memory_usage > self.memory_error_threshold:
            self.batch_size = max(
                self.min_batch_size,
                int(self.batch_size * self.batch_size_reduction_factor)
            )
            self.clustering_batch_size = max(
                100,
                int(self.clustering_batch_size * self.batch_size_reduction_factor)  
            )
        elif memory_usage < self.memory_warning_threshold:
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

        if self.noise_ratio < 0 or self.noise_ratio > 1:
            raise ValueError("noise_ratio must be between 0 and 1")

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        os.makedirs(self.embeddings_output_dir, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class ClusteringConfigFactory:
    @staticmethod
    def from_json(config_path: str) -> ClusteringConfig:
        """Create configuration from JSON file."""
        with open(config_path) as f:
            config_dict = json.load(f)
            return ClusteringConfig(**config_dict)

    @staticmethod
    def get_base_config() -> ClusteringConfig:
        """Get base clustering configuration.""" 
        return ClusteringConfig()

    @staticmethod
    def get_noise_config(noise_ratio: float = 0.2) -> ClusteringConfig:
        """Get configuration with noise injection."""
        return ClusteringConfig(
            inject_noise=True,
            noise_ratio=noise_ratio,
            noise_seed=42
        )
    
    @staticmethod
    def get_config_for_type(experiment_type: str) -> ClusteringConfig:
        """
        Returns the appropriate config based on the experiment type.
        """
        if experiment_type == 'clustering':
            return ClusteringConfig()  # Assuming `ClusteringConfig` is already defined
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")