import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class IndexTrackingConfig:
    """Configuration for index tracking functionality."""
    enabled: bool = True
    validate_indices: bool = True
    track_unused_indices: bool = False 
    max_index_memory: int = 10000000000
    cleanup_interval: int = 3600
    persistence: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'format': 'json',
        'compression': True
    })
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'alert_threshold': 0.9,
        'metrics_interval': 300
    })

class Settings:
    """Enhanced settings handler with index tracking support."""
    def __init__(self):
        self._settings = self._load_settings()
        self._defaults = self._configure_defaults()
        self._index_tracking = self._configure_index_tracking()

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from settings.yaml"""
        config_dir = Path(__file__).resolve().parent
        settings_path = config_dir / 'settings.yaml'
        with open(settings_path, 'r') as f:
            return yaml.safe_load(f)

    def _configure_defaults(self) -> Dict[str, Any]:
        """Configure default values for settings."""
        return {
            'NUM_SAMPLES': -1,
            'TOP_K': 5,
            'BATCH_SIZE': 8,
            'RANDOM_RATIO': 0.3,
            'INDEX_MAPPING': {
                'cache_size': 10000,
                'batch_size': 1000,
                'validate_mappings': True,
                'mapping_types': [
                    'corpus_idx_to_search',
                    'search_to_corpus_idx',
                    'dataset_idx_to_corpus',
                    'corpus_to_dataset_idx'
                ]
            },
            'NAIVE_RAG': {
                'num_documents': 5,
                'index_tracking': True
            },
            'RANDOM_INJECTION': {
                'enabled': False,
                'injection_ratio': 0.3,
                'maintain_indices': True
            },
            'DIVERSITY': {
                'min_similarity_threshold': 0.3,
                'index_based_selection': True
            },
            'KMEANS_CLUSTERING': {
                'n_clusters': 5,
                'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
                'random_state': 42,
                'max_iter': 300,
                'index_tracking': True
            },
            'RAG_FUSION': {
                'fusion_weights': [1.0, 0.8, 0.6, 0.4, 0.2],
                'fusion_strategy': 'reciprocal_rank',
                'top_k_per_cluster': 3,
                'index_tracking': True
            },
            'RETRIEVERS': {
                'contriever': {
                    'enabled': True,
                    'index_type': 'corpus_idx',
                    'max_index_size': 25000000
                },
                'bm25': {
                    'enabled': True,
                    'index_type': 'corpus_idx',
                    'max_index_size': 25000000
                },
                'adore': {
                    'enabled': True,
                    'index_type': 'corpus_idx',
                    'max_index_size': 25000000
                }
            }
        }

    def _configure_index_tracking(self) -> IndexTrackingConfig:
        """Configure index tracking settings."""
        index_config = self._settings.get('INDEX_TRACKING', {})
        return IndexTrackingConfig(
            enabled=index_config.get('enabled', True),
            validate_indices=index_config.get('validate_indices', True),
            track_unused_indices=index_config.get('track_unused_indices', False),
            max_index_memory=index_config.get('max_index_memory', 10000000000),
            cleanup_interval=index_config.get('cleanup_interval', 3600),
            persistence=index_config.get('persistence', {
                'enabled': True,
                'format': 'json',
                'compression': True
            }),
            monitoring=index_config.get('monitoring', {
                'enabled': True,
                'alert_threshold': 0.9,
                'metrics_interval': 300
            })
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value with fallback to defaults and then to provided default."""
        # First check in loaded settings
        if key in self._settings:
            return self._settings[key]
        # Then check in defaults
        if key in self._defaults:
            return self._defaults[key]
        # Finally return provided default
        return default

    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access to settings."""
        if key == 'index_tracking':
            return self._index_tracking
        if key in self._settings:
            return self._settings[key]
        if key in self._defaults:
            return self._defaults[key]
        raise AttributeError(f"Setting '{key}' not found")

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to settings."""
        return self.get(key)

    def update(self, key: str, value: Any):
        """Update a setting value."""
        self._settings[key] = value
        # If updating index tracking settings, reconfigure
        if key == 'INDEX_TRACKING':
            self._index_tracking = self._configure_index_tracking()

    def validate_index_settings(self) -> bool:
        """Validate index-related settings."""
        try:
            # Check if required index mapping types are configured
            required_mappings = {'corpus_idx_to_search', 'dataset_idx_to_corpus'}
            configured_mappings = set(self.get('INDEX_MAPPING', {}).get('mapping_types', []))
            if not required_mappings.issubset(configured_mappings):
                return False

            # Check if retrievers have valid index configurations
            for retriever_config in self.get('RETRIEVERS', {}).values():
                if retriever_config.get('enabled'):
                    if 'index_type' not in retriever_config:
                        return False
                    if retriever_config['index_type'] not in {'corpus_idx', 'doc_id'}:
                        return False

            # Validate index tracking configuration
            if self._index_tracking.enabled:
                if self._index_tracking.max_index_memory <= 0:
                    return False
                if self._index_tracking.cleanup_interval <= 0:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating index settings: {e}")
            return False

    def get_index_config(self, component: str) -> Dict[str, Any]:
        """Get index tracking configuration for a specific component."""
        base_config = {
            'enabled': self._index_tracking.enabled,
            'validate_indices': self._index_tracking.validate_indices
        }

        # Add component-specific settings
        if component == 'retriever':
            retriever_settings = self.get('RETRIEVERS', {})
            base_config.update({
                'index_type': retriever_settings.get('index_type', 'corpus_idx'),
                'max_index_size': retriever_settings.get('max_index_size', 25000000)
            })
        elif component == 'fusion':
            base_config.update({
                'track_unused_indices': self._index_tracking.track_unused_indices,
                'persistence': self._index_tracking.persistence
            })
        elif component == 'clustering':
            base_config.update({
                'monitoring': self._index_tracking.monitoring
            })

        return base_config

# Create singleton instance
_instance = Settings()

# Create function to get settings instance
def get_settings():
    return _instance

# Export the settings instance
settings = _instance

# Make settings available at module level
globals().update({key: settings.get(key) for key in settings._settings})

# Export settings instance
__all__ = ['settings', 'get_settings', 'IndexTrackingConfig']