# src/config/paths.py
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Config directories
CONFIG_DIR = PROJECT_ROOT / 'src' / 'config'
LOGGING_CONFIG_PATH = CONFIG_DIR / 'logging_config.yaml'
SETTINGS_PATH = CONFIG_DIR / 'settings.yaml'

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'

# Corpus directories
CORPUS_DIR = DATA_DIR / 'corpus'
CORPUS_JSON = CORPUS_DIR / 'corpus.json'
CORPUS_ARROW_DIR = CORPUS_DIR / 'train'  # Contains arrow files
CORPUS_INDEX_MAP = CORPUS_DIR / 'index_mapping.json'  # Maps between indices

# Dataset directories
DATASETS_DIR = DATA_DIR / 'datasets'
TRAIN_DATASET_PATH = DATASETS_DIR / '10k_train_dataset.json'
TEST_DATASET_PATH = DATASETS_DIR / 'test_dataset.json'
DATASET_INDEX_MAP = DATASETS_DIR / 'index_mapping.json'  # Dataset index mapping

# Embeddings directory
EMBEDDINGS_DIR = DATA_DIR / 'embeddings'
EMBEDDINGS_INDEX_MAP = EMBEDDINGS_DIR / 'index_mapping.json'  # Embeddings index mapping

# Search results directory
RETRIEVAL_DIR = DATA_DIR / 'retrieval_results'
SEARCH_INDEX_MAP = RETRIEVAL_DIR / 'index_mapping.json'  # Search results index mapping

# Retrieval results paths
CONTRIEVER_RESULTS_PATH = RETRIEVAL_DIR / 'contriever_search_results_at150.pkl'
BM25_RESULTS_PATH = RETRIEVAL_DIR / 'bm25_test_search_results_at250.pkl'
ADORE_RESULTS_PATH = RETRIEVAL_DIR / 'adore_search_results_at200.pkl'

# Random results directory
RANDOM_RESULTS_DIR = DATA_DIR / 'random_results'
RANDOM_INDEX_MAP = RANDOM_RESULTS_DIR / 'index_mapping.json'  # Random results index mapping

# Results directories
RESULTS_DIR = PROJECT_ROOT / 'results'
BASELINE_RESULTS_DIR = RESULTS_DIR / 'baseline'
KMEANS_RESULTS_DIR = RESULTS_DIR / 'kmeans'
FUSION_RESULTS_DIR = RESULTS_DIR / 'fusion'
VISUALIZATION_DIR = RESULTS_DIR / 'visualizations'

# Experiment results paths
EXPERIMENT_RESULTS = {
    'baseline': BASELINE_RESULTS_DIR,
    'kmeans': KMEANS_RESULTS_DIR / 'clustering',
    'fusion': FUSION_RESULTS_DIR / 'fusion',
    'categories': FUSION_RESULTS_DIR / 'categories'
}

# Experiment directories (added this)
EXPERIMENT_DIRS = {
    'baseline': BASELINE_RESULTS_DIR,
    'kmeans': KMEANS_RESULTS_DIR,
    'fusion_kmeans': FUSION_RESULTS_DIR,
    'fusion_categories': FUSION_RESULTS_DIR / 'categories'
}

# Temp directories for experiment processing
TEMP_DIR = DATA_DIR / 'temp'
TEMP_CLUSTERING_DIR = TEMP_DIR / 'clustering'
TEMP_FUSION_DIR = TEMP_DIR / 'fusion'

# Index mapping directory
MAPPINGS_DIR = DATA_DIR / 'mappings'  # Stores all index mapping files
INDEX_MAPPINGS = {
    'corpus': CORPUS_INDEX_MAP,
    'dataset': DATASET_INDEX_MAP,
    'embeddings': EMBEDDINGS_INDEX_MAP,
    'search': SEARCH_INDEX_MAP,
    'random': RANDOM_INDEX_MAP
}

# Logs directory
LOGS_DIR = PROJECT_ROOT / 'logs'
ANALYSIS_LOGS_DIR = LOGS_DIR / 'analysis'
EXPERIMENT_LOGS_DIR = LOGS_DIR / 'experiments'
INDEXING_LOGS_DIR = LOGS_DIR / 'indexing'
FUSION_LOGS_DIR = LOGS_DIR / 'fusion'
CLUSTERING_LOGS_DIR = LOGS_DIR / 'clustering'
GENERATION_LOGS_DIR = LOGS_DIR / 'generation'
EVALUATION_LOGS_DIR = LOGS_DIR / 'evaluation'

# Cache directory
CACHE_DIR = DATA_DIR / 'cache'

# Required directories list
REQUIRED_DIRS = [
    DATASETS_DIR,
    CORPUS_DIR,
    CORPUS_ARROW_DIR,
    EMBEDDINGS_DIR,
    RETRIEVAL_DIR,
    RANDOM_RESULTS_DIR,
    MAPPINGS_DIR,
    CACHE_DIR,
    BASELINE_RESULTS_DIR,
    KMEANS_RESULTS_DIR,
    FUSION_RESULTS_DIR,
    VISUALIZATION_DIR,
    TEMP_DIR,
    TEMP_CLUSTERING_DIR,
    TEMP_FUSION_DIR,
    EXPERIMENT_RESULTS['kmeans'],
    EXPERIMENT_RESULTS['fusion'],
    EXPERIMENT_RESULTS['categories'],
    LOGS_DIR,
    ANALYSIS_LOGS_DIR,
    EXPERIMENT_LOGS_DIR,
    INDEXING_LOGS_DIR,
    FUSION_LOGS_DIR,
    CLUSTERING_LOGS_DIR,
    GENERATION_LOGS_DIR,
    EVALUATION_LOGS_DIR
]

# Create required directories
for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)

# Export paths
__all__ = [
    'PROJECT_ROOT',
    'CONFIG_DIR',
    'LOGGING_CONFIG_PATH',
    'SETTINGS_PATH',
    'DATA_DIR',
    'CORPUS_DIR',
    'CORPUS_JSON',
    'CORPUS_ARROW_DIR',
    'DATASETS_DIR',
    'TRAIN_DATASET_PATH',
    'TEST_DATASET_PATH',
    'EMBEDDINGS_DIR',
    'RETRIEVAL_DIR',
    'RANDOM_RESULTS_DIR',
    'RESULTS_DIR', 
    'MAPPINGS_DIR',
    'LOGS_DIR',
    'CACHE_DIR',
    'INDEX_MAPPINGS',
    'EXPERIMENT_RESULTS',
    'EXPERIMENT_DIRS',
    'TEMP_DIR',
    'TEMP_CLUSTERING_DIR',
    'TEMP_FUSION_DIR',
    'CONTRIEVER_RESULTS_PATH',
    'BM25_RESULTS_PATH',
    'ADORE_RESULTS_PATH'
]