import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
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
    corpus_path: str = "data/corpus.json"
    queries_path: str = "data/train_dataset.json"
    search_results_path: str = "data/contriever_search_results_at150.pkl"
    
    # Processing configuration
    batch_size: int = 32
    use_gpu: bool = True
    
    # Random document injection
    use_random_docs: bool = False
    random_doc_ratio: float = 0.2
    random_doc_source: str = "10k_random_results_at60.pkl"
    
    # Output configuration
    output_dir: str = field(default_factory=lambda: f"results/clustering/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
            
        if not os.path.exists(self.corpus_path):
            raise ValueError(f"Corpus path does not exist: {self.corpus_path}")
            
        if not os.path.exists(self.queries_path):
            raise ValueError(f"Queries path does not exist: {self.queries_path}")
            
        if not os.path.exists(self.search_results_path):
            raise ValueError(f"Search results path does not exist: {self.search_results_path}")
            
        if self.use_random_docs:
            random_doc_path = os.path.join("data", self.random_doc_source)
            if not os.path.exists(random_doc_path):
                raise ValueError(f"Random document source does not exist: {random_doc_path}")
                
    def get_random_doc_path(self) -> str:
        """Get full path to random document source."""
        return os.path.join("data", self.random_doc_source)
    
    def get_experiment_dir(self) -> str:
        """Get experiment-specific output directory."""
        return os.path.join(self.output_dir, self.experiment_id)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        os.makedirs(self.output_dir, exist_ok=True)