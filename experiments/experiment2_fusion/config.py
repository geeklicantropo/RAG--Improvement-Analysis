from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

@dataclass
class FusionConfig:
    """Configuration for RAG-Fusion experiment."""
    
    # Experiment identification
    experiment_name: str = "fusion_experiment"
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Model configuration
    llm_id: str = "meta-llama/Llama-2-7b-chat-hf"
    model_max_length: int = 4096
    max_new_tokens: int = 15
    
    # Fusion configuration
    fusion_strategy: str = "rrf"  # 'rrf' or 'linear'
    fusion_k: float = 60.0  # k parameter for RRF fusion
    normalize_scores: bool = True
    contriever_weight: float = 0.7
    bm25_weight: float = 0.3
    
    # Retrieval configuration
    top_docs: int = 150
    num_documents_in_context: int = 7
    
    # Data paths
    base_data_dir: Path = Path("data")
    processed_dir: Path = base_data_dir / "processed"
    
    @property
    def contriever_results_path(self) -> Path:
        return self.base_data_dir / "contriever_search_results_at150.pkl"
    
    @property
    def bm25_results_path(self) -> Path:
        return self.base_data_dir / "bm25_test_search_results_at250.pkl"
    
    @property
    def corpus_path(self) -> Path:
        return self.processed_dir / "corpus_with_contriever_at150.json"
    
    @property
    def train_dataset_path(self) -> Path:
        return self.base_data_dir / "10k_train_dataset.json"
    
    @property
    def test_dataset_path(self) -> Path:
        return self.base_data_dir / "test_dataset.json"
    
    # Random document configuration
    use_random_docs: bool = False
    random_doc_ratio: float = 0.2
    
    @property
    def random_doc_source(self) -> Path:
        return self.base_data_dir / "10k_random_results_at60.pkl"
    
    # Processing configuration
    batch_size: int = 32
    use_gpu: bool = True
    
    # Output configuration
    output_dir: str = field(default_factory=lambda: f"experiments/experiment2_fusion/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_intermediates: bool = True
    save_every: int = 250

    def validate(self):
        """Validate configuration parameters."""
        if self.fusion_strategy not in ['rrf', 'linear']:
            raise ValueError("fusion_strategy must be either 'rrf' or 'linear'")
            
        if self.fusion_k <= 0:
            raise ValueError("fusion_k must be positive")
            
        if not 0 <= self.contriever_weight <= 1 or not 0 <= self.bm25_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")
            
        if abs(self.contriever_weight + self.bm25_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
            
        # Validate file existence
        if not self.contriever_results_path.exists():
            raise ValueError(f"Contriever results not found: {self.contriever_results_path}")
            
        if not self.bm25_results_path.exists():
            raise ValueError(f"BM25 results not found: {self.bm25_results_path}")
            
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus not found: {self.corpus_path}")
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in self.__dict__.items()
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FusionConfig':
        """Create configuration from dictionary."""
        # Convert string paths back to Path objects
        for key, value in config_dict.items():
            if isinstance(value, str) and (key.endswith('_path') or key.endswith('_dir')):
                config_dict[key] = Path(value)
        return cls(**config_dict)
    
     # Batch size management
    batch_size: int = 32
    min_batch_size: int = 1 
    max_batch_size: int = 64
    batch_size_reduction_factor: float = 0.5
    fusion_batch_size: int = 100
    
    def adjust_batch_sizes(self, memory_usage: float):
        """Adjust batch sizes for generation and fusion"""
        if memory_usage > 0.9:
            self.batch_size = max(
                self.min_batch_size,
                int(self.batch_size * self.batch_size_reduction_factor)
            )
            self.fusion_batch_size = max(
                10,
                int(self.fusion_batch_size * self.batch_size_reduction_factor)
            )
        elif memory_usage < 0.7:
            self.batch_size = min(
                self.max_batch_size,
                int(self.batch_size / self.batch_size_reduction_factor)
            )
            self.fusion_batch_size = min(
                500,
                int(self.fusion_batch_size / self.batch_size_reduction_factor)
            )



# Factory for common configurations
class FusionConfigFactory:
    @staticmethod
    def get_rrf_config() -> FusionConfig:
        """Get configuration for RRF fusion experiment."""
        return FusionConfig(
            fusion_strategy="rrf",
            fusion_k=60.0,
            normalize_scores=True,
            contriever_weight=0.7,
            bm25_weight=0.3
        )
    
    @staticmethod
    def get_linear_config() -> FusionConfig:
        """Get configuration for linear fusion experiment."""
        return FusionConfig(
            fusion_strategy="linear",
            normalize_scores=True,
            contriever_weight=0.6,
            bm25_weight=0.4
        )
    
    @staticmethod
    def get_random_config() -> FusionConfig:
        """Get configuration for fusion with random documents."""
        config = FusionConfig(
            fusion_strategy="rrf",
            use_random_docs=True,
            random_doc_ratio=0.3
        )
        return config
    
    @staticmethod
    def get_config_for_strategy(strategy: str, use_random: bool = False) -> 'FusionConfig':
        """Get configuration for specified fusion strategy."""
        if strategy == 'rrf':
            if use_random:
                return FusionConfigFactory.get_random_config()
            return FusionConfigFactory.get_rrf_config()
        elif strategy == 'linear':
            return FusionConfigFactory.get_linear_config()
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")