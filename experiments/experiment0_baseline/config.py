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
    use_bm25: bool = False  # If False, uses Contriever
    top_k_docs: int = 100
    normalize_embeddings: bool = False
    
    # Document Configuration
    num_documents_in_context: int = 5
    gold_position: Optional[int] = None
    get_documents_without_answer: bool = True
    
    # Random Document Configuration
    use_random: bool = False
    random_doc_percentage: float = 0.3  # Percentage of random documents to include
    random_doc_source: str = "wikipedia"  # Options: "wikipedia", "reddit", "synthetic"
    
    # Data Paths
    base_data_dir: Path = Path("data")
    corpus_path: Path = base_data_dir / "corpus.json"
    train_dataset_path: Path = base_data_dir / "10k_train_dataset.json"
    test_dataset_path: Path = base_data_dir / "test_dataset.json"
    
    # Output Configuration
    output_dir: Path = Path("experiments/experiment0_baseline/results")
    save_every: int = 250
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_parameters()
        self._create_directories()
    
    def _validate_parameters(self):
        """Validate configuration parameters."""
        if self.num_documents_in_context <= 0:
            raise ValueError("num_documents_in_context must be positive")
        
        if self.gold_position is not None:
            if self.gold_position < 0 or self.gold_position >= self.num_documents_in_context:
                raise ValueError("gold_position must be within document context range")
        
        if not (0 <= self.random_doc_percentage <= 1):
            raise ValueError("random_doc_percentage must be between 0 and 1")
        
        if self.random_doc_source not in ["wikipedia", "reddit", "synthetic"]:
            raise ValueError("Invalid random_doc_source")
    
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