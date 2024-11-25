from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

@dataclass
class BaselineConfig:
    """Configuration for baseline RAG experiment."""
    
    # Model Configuration
    llm_id: str = "meta-llama/Llama-2-7b-chat-hf"
    model_max_length: int = 4096
    batch_size: int = 16
    max_new_tokens: int = 15
    
    # Retrieval Configuration 
    use_bm25: bool = False
    use_adore: bool = False
    normalize_embeddings: bool = False
    
    # Document Configuration
    num_documents_in_context: int = 5
    gold_position: Optional[int] = None
    get_documents_without_answer: bool = True
    
    # Random Document Configuration
    use_random: bool = False

    # Clustering Configuration (New)
    use_clustering: bool = False
    num_clusters: Optional[int] = None
    cluster_seed: int = 42

    # Experimental Features
    use_fusion: bool = False
    use_categories: bool = False
    
    # Data Paths
    base_data_dir: Path = Path("data")
    processed_dir: Path = base_data_dir / "processed"
    
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
    def train_dataset_path(self) -> Path:
        return self.base_data_dir / "10k_train_dataset.json"
    
    @property
    def test_dataset_path(self) -> Path:
        return self.base_data_dir / "test_dataset.json"
    
    @property
    def search_results_path(self) -> Path:
        if self.use_random:
            return self.base_data_dir / "10k_random_results_at60.pkl"
        elif self.use_adore:
            return self.base_data_dir / "adore_search_results_at200.pkl"
        else:
            return self.base_data_dir / "contriever_search_results_at150.pkl"
        
    @property
    def contriever_results_path(self) -> Path:
        return self.base_data_dir / "contriever_search_results_at150.pkl"
        
    @property  
    def bm25_results_path(self) -> Path:
        return self.base_data_dir / "bm25_test_search_results_at250.pkl"

    @property
    def random_results_path(self) -> Path:
        return self.base_data_dir / "10k_random_results_at60.pkl"
    
    # Output Configuration
    output_dir: Path = Path("experiments/experiment0_baseline/results")
    save_every: int = 250
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_parameters()
        self._create_directories()
    
    def _validate_parameters(self):
        if self.num_documents_in_context <= 0:
            raise ValueError("num_documents_in_context must be positive")
            
        if self.gold_position is not None:
            if self.gold_position < 0 or self.gold_position >= self.num_documents_in_context:
                raise ValueError("gold_position must be within document context range")
                
        # Validate file existence
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus file not found: {self.corpus_path}")
            
        if not self.search_results_path.exists():
            raise ValueError(f"Search results file not found: {self.search_results_path}")
    
    def _create_directories(self):
        """Create necessary directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in self.__dict__.items()
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaselineConfig':
        """Create configuration from dictionary."""
        # Convert string paths back to Path objects
        for key, value in config_dict.items():
            if isinstance(value, str) and key.endswith('_path') or key.endswith('_dir'):
                config_dict[key] = Path(value)
        return cls(**config_dict)   

# Default configurations for different scenarios
class BaselineConfigFactory:
    @staticmethod 
    def get_config_for_retriever(retriever_type: str) -> BaselineConfig:
        """Get configuration for specified retriever type."""
        if retriever_type == 'contriever':
            return BaselineConfigFactory.get_contriever_config()
        elif retriever_type == 'bm25':
            return BaselineConfigFactory.get_bm25_config() 
        elif retriever_type == 'random':
            return BaselineConfigFactory.get_random_config()
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
    @staticmethod
    def get_contriever_config() -> BaselineConfig:
        """Get configuration for Contriever experiment."""
        return BaselineConfig(
            use_bm25=False,
            normalize_embeddings=True,
            num_documents_in_context=7,
            get_documents_without_answer=True
        )
    
    @staticmethod
    def get_bm25_config() -> BaselineConfig:
        """Get configuration for BM25 experiment."""
        return BaselineConfig(
            use_bm25=True,
            num_documents_in_context=7,
            get_documents_without_answer=True
        )
    
    @staticmethod
    def get_random_config() -> BaselineConfig:
        """Get configuration for random document experiment."""
        return BaselineConfig(
            use_random=True,
            random_doc_percentage=0.4,
            num_documents_in_context=7,
            get_documents_without_answer=True
        )