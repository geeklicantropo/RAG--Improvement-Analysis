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
    
    # Model configuration
    llm_id: str = "gemini-1.5-flash"
    embedding_model: str = "gemini-1.5-flash"
    model_max_length: int = 30720
    max_new_tokens: int = 1024
    base_corpus_size: int = 1000

    # Document Configuration
    num_random_docs: int = 1000
    num_adversarial_docs: int = 1000
    random_seed: int = 42
    adversarial_seed: int = 42
    use_random: bool = False
    use_adore: bool = False
    use_test: bool = False

    # Encoder Configuration
    encoder_id: str = 'facebook/contriever'
    max_length_encoder: int = 512
    normalize_embeddings: bool = True
    
    # Memory Configuration
    gpu_memory_threshold: float = 0.8 
    gpu_memory_threshold: float = 0.9
    cpu_memory_threshold: float = 0.9
    memory_warning_threshold: float = 0.7
    memory_error_threshold: float = 0.9
    memory_logging_interval: int = 100

    # Clustering Configuration
    num_clusters: int = 10
    cluster_seed: int = 42
    method: str = "kmeans"  # Options: "kmeans", "hierarchical", "dbscan"
    use_scaler: bool = True
    min_cluster_size: int = 5
    min_docs_per_cluster: int = 5
    max_docs_per_category: int = 10
    enable_cluster_evaluation: bool = True
    max_docs_for_evaluation: int = 100

    # Document Context Configuration
    num_documents_in_context: int = 50
    max_doc_length: int = 512

    # Embeddings Configuration
    compute_new_embeddings: bool = True
    embeddings_path: Optional[str] = None
    embeddings_output_dir: str = "data/embeddings"

    # Noise Configuration
    inject_noise: bool = False
    noise_ratio: float = 0.2
    noise_seed: int = 42
    min_noise_docs: int = 1
    max_noise_docs: int = 5

    # Evaluation Metrics
    silhouette_enabled: bool = True
    calinski_harabasz_enabled: bool = True
    davies_bouldin_enabled: bool = True

    # Data Paths
    base_data_dir: Path = Path("data")
    processed_dir: Path = base_data_dir / "processed"
    random_docs_path: str = 'data/processed/corpus_with_random_50_words.pkl'
    adversarial_docs_path: str = 'data/processed/reddit_corpus.pkl'

    # Processing Configuration
    output_dir_path: str = "experiments/experiment1_clustering/results"
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

    def validate(self):
        """Validate configuration parameters."""
        if self.num_clusters < 2:
            raise ValueError("num_clusters must be at least 2")

        if self.enable_cluster_evaluation and self.max_docs_for_evaluation < 1:
            raise ValueError("max_docs_for_evaluation must be at least 1 when cluster evaluation is enabled")

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