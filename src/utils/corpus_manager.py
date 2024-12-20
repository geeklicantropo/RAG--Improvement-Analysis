import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import logging
from tqdm import tqdm
import gc
import mmap


class CorpusManager:
    def __init__(self, base_corpus_path: str, cache_dir: str = "cache/corpus"):
        self.base_corpus_path = Path(base_corpus_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.doc_cache = {}
        self.corpus_variants = {}
        self.base_corpus = None
        
        self.random_docs_path = 'data/processed/corpus_with_random_50_words.pkl'
        self.adversarial_docs_path = 'data/processed/reddit_corpus.pkl'

    def to_dict(self) -> Dict:
        """Make CorpusManager JSON-serializable."""
        return {
            "base_corpus_path": str(self.base_corpus_path),
            "cache_dir": str(self.cache_dir),
            "corpus_variants": {
                k: len(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in self.corpus_variants.items()
            },
            "doc_cache_size": len(self.doc_cache),
        }

    def __getstate__(self):
        """Support for pickle serialization."""
        state = self.__dict__.copy()
        # Remove logger as it's not serializable
        del state["logger"]
        return state

    def __setstate__(self, state):
        """Support for pickle deserialization."""
        self.__dict__.update(state)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("CorpusManager")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        return logger

    def get_corpus(self) -> List[Dict]:
        """Get the base corpus."""
        return self._lazy_load_corpus()

    def _lazy_load_corpus(self) -> List[Dict]:
        """Lazily load the base corpus."""
        if self.base_corpus is None:
            self.logger.info("Loading base corpus")
            
            try:
                with open(self.base_corpus_path, "r") as f:
                    if str(self.base_corpus_path).endswith('.pkl'):
                        self.base_corpus = [{'text': doc, 'title': ''} for doc in pickle.load(f)]
                    else:
                        self.base_corpus = json.load(f)
                        
                self.logger.info(f"Loaded {len(self.base_corpus)} documents")
                
            except Exception as e:
                self.logger.error(f"Error loading corpus: {str(e)}")
                raise
                
        return self.base_corpus

    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Retrieve a single document by its ID."""
        corpus = self._lazy_load_corpus()
        if 0 <= doc_id < len(corpus):
            return corpus[doc_id]
        return None

    def get_documents(self, doc_ids: List[int]) -> List[Dict]:
        """Retrieve multiple documents by their IDs."""
        return [doc for doc_id in doc_ids if (doc := self.get_document(doc_id))]

    def get_random_subset(
    self, 
    corpus_type: str, 
    num_docs: int, 
    seed: int = 42
    ) -> List[Dict]:
        """
        Retrieve a random subset of documents from the specified corpus.

        Args:
            corpus_type (str): Type of corpus ('base', 'random', or 'adversarial').
            num_docs (int): Number of documents to retrieve.
            seed (int): Random seed for reproducibility.
            
        Returns:
            List[Dict]: Random subset of documents without embeddings.
        """
        if corpus_type == 'base':
            corpus = self._lazy_load_corpus()
        elif corpus_type == 'random':
            if not Path(self.random_docs_path).exists():
                raise ValueError(f"Random docs file not found: {self.random_docs_path}")
            with open(self.random_docs_path, "rb") as f:
                corpus = [{'text': doc, 'title': ''} for doc in pickle.load(f)]
        elif corpus_type == 'adversarial':
            if not Path(self.adversarial_docs_path).exists():
                raise ValueError(f"Adversarial docs file not found: {self.adversarial_docs_path}")
            with open(self.adversarial_docs_path, "rb") as f:
                corpus = [{'text': doc, 'title': ''} for doc in pickle.load(f)]
        else:
            raise ValueError(f"Invalid corpus type: {corpus_type}")

        if num_docs > len(corpus):
            self.logger.warning(f"Requested {num_docs} documents, but corpus only contains {len(corpus)}. Returning all documents.")
            return corpus

        np.random.seed(seed)
        subset_indices = np.random.choice(len(corpus), size=num_docs, replace=False)
        return [corpus[idx] for idx in subset_indices]
    

    def add_noise_documents(self, noise_ratio: float = 0.2, seed: int = 42) -> List[Dict]:
        """
            Add noise documents to the corpus.

        Args:
            noise_ratio (float): Proportion of documents to add as noise.
            seed (int): Random seed for reproducibility.

        Returns:
            List[Dict]: Corpus with added noise documents.
        """
        key = f"noisy_{noise_ratio}_{seed}"
        if key not in self.corpus_variants:
            np.random.seed(seed)
            corpus = self._lazy_load_corpus()
            num_noise = int(len(corpus) * noise_ratio)

            self.logger.info(f"Adding {num_noise} noise documents.")
            noisy_corpus = corpus.copy()

            for idx in tqdm(range(num_noise), desc="Generating noise docs"):
                noisy_doc = {
                    "text": " ".join(np.random.choice(["random", "noise", "distractor"], size=50)),
                    "title": f"Noise_{idx}",
                    "id": len(corpus) + idx,
                    "is_noise": True,
                }
                noisy_corpus.append(noisy_doc)

            self.corpus_variants[key] = noisy_corpus

        return self.corpus_variants[key]

    def get_gold_documents(self) -> List[Dict]:
        """Retrieve gold documents."""
        corpus = self._lazy_load_corpus()
        self.logger.info("Searching for gold documents...")
        
        # Check multiple possible gold indicators
        gold_docs = []
        for doc in corpus:
            if any([
                'answers' in doc,
                'gold_answer' in doc,
                doc.get('is_gold', False),
                doc.get('type', '') == 'gold'
            ]):
                gold_docs.append(doc)
        
        if not gold_docs:
            with open('data/10k_train_dataset.json') as f:
                train_data = json.load(f)
                gold_docs = [{'text': item['text'], 'answers': item['answers']} 
                            for item in train_data]
        
        self.logger.info(f"Found {len(gold_docs)} gold documents")
        return gold_docs

    def clear_cache(self):
        """Clear the document cache."""
        self.doc_cache.clear()
        gc.collect()