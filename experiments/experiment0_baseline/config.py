from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import os
import json
from src.utils.file_utils import read_pickle

@dataclass 
class BaselineConfig:
    """Configuration for baseline RAG experiment."""
    
    # Model Configuration
    llm_id: str = "gemini-1.5-flash"
    model_max_length: int = 30720  
    batch_size: int = 256
    max_new_tokens: int = 1024

    # Encoder configuration
    encoder_id: str = 'facebook/contriever'
    max_length_encoder: int = 512
    normalize_embeddings: bool = True
    
    # Memory Management
    gpu_memory_utilization: float = 0.8
    empty_cache_every_n_batches: int = 1
    gradient_checkpointing: bool = True
    cpu_offload: bool = True
    batch_size_reduction_factor: float = 0.5
    
    # Batch Processing 
    min_batch_size: int = 8
    max_batch_size: int = 512
    dynamic_batch_size: bool = True
    
    # Document Configuration
    num_documents_in_context: int = 50
    max_doc_length: int = 512
    gold_position: Optional[int] = None
    get_documents_without_answer: bool = True
    use_random: bool = False
    use_adore: bool = False
    use_bm25: bool = False
    normalize_embeddings: bool = False
    use_test: bool = False
    random_doc_percentage: Optional[float] = None
    noise_ratio: float = 0.2
    
    # Data Paths 
    base_data_dir: Path = Path("data")
    processed_dir: Path = base_data_dir / "processed"
    output_dir: Path = Path("experiments/experiment0_baseline/results")
    save_every: int = 100

    # Corpus Configuration
    base_corpus_size: int = 1000 
    num_random_docs: int = 1000
    num_adversarial_docs: int = 1000
    random_seed: int = 42
    adversarial_seed: int = 42

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
    
    @property
    def total_batches(self) -> int:
        """Calculate total number of batches based on dataset size."""
        if not hasattr(self, '_total_batches'):
            try:
                with open(self.data_path) as f:
                    dataset = json.load(f)
                self._total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
            except Exception as e:
                return 0
        return self._total_batches

    def __post_init__(self):
        self._validate_parameters()
        self._create_directories()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def _validate_parameters(self):
        """Validate configuration parameters."""
        if self.num_documents_in_context <= 0:
            raise ValueError("num_documents_in_context must be positive")
        
        if self.gold_position is not None and (
            self.gold_position < 0 or 
            self.gold_position >= self.num_documents_in_context
        ):
            raise ValueError("gold_position must be within the document context range")
        
        self._validate_paths()
        self._validate_search_results()

    def _validate_paths(self):
        """Validate required data paths exist."""
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus file not found: {self.corpus_path}")
            
        if not self.data_path.exists():
            raise ValueError(f"Dataset not found: {self.data_path}")
            
        search_results_path = self._get_search_results_path()
        if not search_results_path.exists():
            raise ValueError(f"Search results not found: {search_results_path}")

    def _validate_search_results(self) -> bool:
        """Validate search results have proper format."""
        try:
            search_results_path = self._get_search_results_path()
            if not search_results_path.exists():
                raise ValueError(f"Search results file not found: {search_results_path}")

            results = read_pickle(str(search_results_path))

            if not isinstance(results, list):
                raise ValueError("Search results must be a list")

            # Validate first result structure
            if not results:
                raise ValueError("Empty search results")

            # Allow both tuple and list formats
            first_result = results[0]
            if isinstance(first_result, (tuple, list)) and len(first_result) == 2:
                doc_ids, scores = first_result
                if len(doc_ids) == len(scores):
                    return True
                    
            raise ValueError("Invalid search result format")

        except Exception as e:
            raise ValueError(f"Invalid search results: {str(e)}")
    
    def _get_search_results_path(self) -> Path:
        """Get appropriate search results path based on configuration."""
        if self.use_bm25:
            return self.bm25_results_path
        elif self.use_random:
            return self.random_results_path
        return self.contriever_results_path
    
    def _create_directories(self):
        """Create required output directories."""
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

    @staticmethod
    def load_from_json(config_path: str) -> 'BaselineConfig':
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return BaselineConfig.from_dict(config_dict)

# Factory for common configurations
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
            noise_ratio=0.3,  
            num_documents_in_context=7, 
            get_documents_without_answer=True,
            use_test=False
        )