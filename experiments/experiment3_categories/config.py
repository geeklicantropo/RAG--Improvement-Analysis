from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

@dataclass
class CategoriesConfig:
    """Configuration for category-based RAG experiment."""
    
    # Experiment identification
    experiment_name: str = "categories_experiment"
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Model configuration
    llm_id: str = "meta-llama/Llama-2-7b-chat-hf"
    model_max_length: int = 4096
    max_new_tokens: int = 15
    
    # Category configuration
    score_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high_confidence": 0.8,
        "medium_confidence": 0.5,
        "low_confidence": 0.0
    })
    max_docs_per_category: int = 3
    min_category_score: float = 0.2
    
    # Document Source Configuration
    use_bm25: bool = False
    use_fusion: bool = True
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        'contriever': 0.7,
        'bm25': 0.3
    })
    
    # Data paths
    base_data_dir: Path = Path("data")
    processed_dir: Path = base_data_dir / "processed"
    
    @property
    def corpus_path(self) -> Path:
        return self.processed_dir / "corpus_with_contriever_at150.json"
    
    @property
    def train_dataset_path(self) -> Path:
        return self.base_data_dir / "10k_train_dataset.json"
    
    @property
    def test_dataset_path(self) -> Path:
        return self.base_data_dir / "test_dataset.json"
    
    @property
    def contriever_results_path(self) -> Path:
        return self.base_data_dir / "contriever_search_results_at150.pkl"
        
    @property
    def bm25_results_path(self) -> Path:
        return self.base_data_dir / "bm25_test_search_results_at250.pkl"
    
    # Random document configuration
    use_random_docs: bool = False
    random_doc_ratio: float = 0.2
    random_doc_category: str = "supplementary"
    
    @property
    def random_doc_source(self) -> Path:
        return self.base_data_dir / "10k_random_results_at60.pkl"
    
    # Processing configuration
    batch_size: int = 32
    use_gpu: bool = True
    
    # Output configuration
    output_dir: str = field(default_factory=lambda: f"experiments/experiment3_categories/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_intermediates: bool = True
    save_every: int = 250
    
    def validate(self):
        """Validate configuration parameters."""
        # Validate score thresholds
        if not all(0 <= v <= 1 for v in self.score_thresholds.values()):
            raise ValueError("Score thresholds must be between 0 and 1")
            
        # Validate fusion weights if using fusion
        if self.use_fusion:
            if not all(0 <= v <= 1 for v in self.fusion_weights.values()):
                raise ValueError("Fusion weights must be between 0 and 1")
            if abs(sum(self.fusion_weights.values()) - 1.0) > 1e-6:
                raise ValueError("Fusion weights must sum to 1")
                
        # Validate file existence
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus not found: {self.corpus_path}")
            
        if not self.contriever_results_path.exists():
            raise ValueError(f"Contriever results not found: {self.contriever_results_path}")
            
        if self.use_bm25 and not self.bm25_results_path.exists():
            raise ValueError(f"BM25 results not found: {self.bm25_results_path}")
    
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CategoriesConfig':
        """Create configuration from dictionary."""
        # Convert string paths back to Path objects
        for key, value in config_dict.items():
            if isinstance(value, str) and (key.endswith('_path') or key.endswith('_dir')):
                config_dict[key] = Path(value)
        return cls(**config_dict)

# Factory for common configurations
class CategoriesConfigFactory:
    @staticmethod
    def get_confidence_based_config() -> CategoriesConfig:
        """Get configuration for confidence-based categorization."""
        return CategoriesConfig(
            score_thresholds={
                "high_confidence": 0.8,
                "medium_confidence": 0.5,
                "low_confidence": 0.2
            },
            max_docs_per_category=3
        )
    
    @staticmethod
    def get_fusion_based_config() -> CategoriesConfig:
        """Get configuration using both retrievers with fusion."""
        return CategoriesConfig(
            use_fusion=True,
            fusion_weights={
                'contriever': 0.7,
                'bm25': 0.3
            }
        )
    
    @staticmethod
    def get_random_augmented_config() -> CategoriesConfig:
        """Get configuration with random document augmentation."""
        return CategoriesConfig(
            use_random_docs=True,
            random_doc_ratio=0.3,
            random_doc_category="supplementary"
        )
    
    @staticmethod
    def get_config_for_type(config_type: str) -> 'CategoriesConfig':
        """Get configuration for specified category type."""
        if config_type == 'confidence':
            return CategoriesConfigFactory.get_confidence_based_config()
        elif config_type == 'fusion':
            return CategoriesConfigFactory.get_fusion_based_config() 
        elif config_type == 'random':
            return CategoriesConfigFactory.get_random_augmented_config()
        else:
            raise ValueError(f"Unknown category type: {config_type}")