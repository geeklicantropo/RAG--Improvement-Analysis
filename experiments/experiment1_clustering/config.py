import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

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
    
    # Embeddings configuration
    compute_new_embeddings: bool = False
    embeddings_path: Optional[str] = None
    max_length_encoder: int = 512
    
    # Data configuration
    base_data_dir: Path = Path("data")
    processed_dir: Path = base_data_dir / "processed"
    use_bm25: bool = False
    use_adore: bool = False
    use_random: bool = False
    
    @property
    def corpus_path(self) -> Path:
        """Get appropriate corpus path based on configuration."""
        if self.use_random:
            return self.processed_dir / "corpus_with_random_at60.json"
        elif self.use_adore:
            return self.processed_dir / "corpus_with_adore_at200.json"
        else:
            return self.processed_dir / "corpus_with_contriever_at150.json"
    
    @property
    def queries_path(self) -> Path:
        return self.base_data_dir / "10k_train_dataset.json"
        
    @property
    def search_results_path(self) -> Path:
        if self.use_random:
            return self.base_data_dir / "10k_random_results_at60.pkl"
        elif self.use_adore:
            return self.base_data_dir / "adore_search_results_at200.pkl"
        else:
            return self.base_data_dir / "contriever_search_results_at150.pkl"
    
    # Processing configuration
    batch_size: int = 32
    use_gpu: bool = True
    
    # Random document injection
    use_random_docs: bool = False
    random_doc_ratio: float = 0.2
    
    @property
    def random_doc_source(self) -> Path:
        return self.base_data_dir / "10k_random_results_at60.pkl"
    
    # Output configuration 
    output_dir: str = field(default_factory=lambda: f"experiments/experiment1_clustering/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_intermediates: bool = True

    @classmethod
    def load(cls, config_path: str) -> "ClusteringConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, config_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
            
    def validate(self):
        """Validate configuration parameters."""
        if self.num_clusters < 2:
            raise ValueError("num_clusters must be at least 2")
            
        if not self.compute_new_embeddings and not self.embeddings_path:
            raise ValueError("Must either compute new embeddings or provide embeddings_path")
            
        # Validate file existence
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus path does not exist: {self.corpus_path}")
            
        if not self.queries_path.exists():
            raise ValueError(f"Queries path does not exist: {self.queries_path}")
            
        if not self.search_results_path.exists():
            raise ValueError(f"Search results path does not exist: {self.search_results_path}")
            
        if self.use_random_docs and not self.random_doc_source.exists():
            raise ValueError(f"Random document source does not exist: {self.random_doc_source}")
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        os.makedirs(self.output_dir, exist_ok=True)