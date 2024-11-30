from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import os

@dataclass
class BaselineConfig:
    """Configuration for baseline RAG experiment."""
    
    # Model Configuration
    llm_id: str = "meta-llama/Llama-2-7b-chat-hf"
    model_max_length: int = 2048 
    batch_size: int = 4
    max_new_tokens: int = 15
    
    # Memory Management
    gpu_memory_utilization: float = 0.7
    empty_cache_every_n_batches: int = 1
    gradient_checkpointing: bool = True
    cpu_offload: bool = True
    use_8bit_quantization: bool = True
    batch_size_reduction_factor: float = 0.5
    
    # Batch Processing
    min_batch_size: int = 1
    max_batch_size: int = 8
    dynamic_batch_size: bool = True
    
    # Document Configuration
    num_documents_in_context: int = 5
    max_doc_length: int = 512
    gold_position: Optional[int] = None
    get_documents_without_answer: bool = True
    use_random: bool = False
    use_adore: bool = False
    use_bm25: bool = False
    normalize_embeddings: bool = False
    use_test: bool = False
    random_doc_percentage: Optional[float] = None
    
    # Data Paths
    base_data_dir: Path = Path("data")
    processed_dir: Path = base_data_dir / "processed"
    output_dir: Path = Path("experiments/experiment0_baseline/results")
    save_every: int = 100

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
    def contriever_results_path(self) -> Path:
        return self.base_data_dir / "contriever_search_results_at150.pkl"
    
    @property
    def bm25_results_path(self) -> Path:
        return self.base_data_dir / "bm25_test_search_results_at250.pkl"
    
    @property
    def random_results_path(self) -> Path:
        return self.base_data_dir / "10k_random_results_at60.pkl"

    def __post_init__(self):
        self._validate_parameters()
        self._create_directories()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def _validate_parameters(self):
        """
        Validate the parameters of the BaselineConfig to ensure consistency and that
        required files and directories exist.
        """
        # Validate number of documents in context
        if self.num_documents_in_context <= 0:
            raise ValueError("num_documents_in_context must be positive")
        
        # Validate gold position if specified
        if self.gold_position is not None and (
            self.gold_position < 0 or self.gold_position >= self.num_documents_in_context
        ):
            raise ValueError("gold_position must be within the document context range")
        
        # Ensure the corpus path exists
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus file not found: {self.corpus_path}")
        
        # Dynamically select the appropriate search results path
        search_results_path = (
            self.contriever_results_path if not self.use_bm25 and not self.use_random else
            self.bm25_results_path if self.use_bm25 else
            self.random_results_path
        )
        
        # Ensure the selected search results path exists
        if not search_results_path.exists():
            raise ValueError(f"Search results file not found: {search_results_path}")
    
    def _create_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def to_dict(self) -> Dict[str, Any]:
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaselineConfig':
        for key, value in config_dict.items():
            if isinstance(value, str) and (key.endswith('_path') or key.endswith('_dir')):
                config_dict[key] = Path(value)
        return cls(**config_dict)

    def adjust_batch_size(self, current_memory_usage: float):
        if not self.dynamic_batch_size:
            return

        if current_memory_usage > 0.85:
            self.batch_size = max(self.min_batch_size, 
                                int(self.batch_size * self.batch_size_reduction_factor))
        elif current_memory_usage < 0.6:
            self.batch_size = min(self.max_batch_size,
                                int(self.batch_size / self.batch_size_reduction_factor)) 


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
            get_documents_without_answer=True,
            use_test=False
        )
    
    @staticmethod
    def get_bm25_config() -> BaselineConfig:
        """Get configuration for BM25 experiment."""
        return BaselineConfig(
            use_bm25=True,
            num_documents_in_context=7,
            get_documents_without_answer=True,
            use_test=True  # BM25 results are only available for test set
        )
    
    @staticmethod
    def get_random_config() -> BaselineConfig:
        """Get configuration for random document experiment."""
        return BaselineConfig(
            use_random=True,
            random_doc_percentage=0.4,
            num_documents_in_context=7, 
            get_documents_without_answer=True,
            use_test=False
        )