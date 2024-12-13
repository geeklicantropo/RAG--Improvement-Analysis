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
        self.corpus_variants = {}
        self.logger = self._setup_logger()
        self.doc_cache = {}
        
    def _setup_logger(self):
        logger = logging.getLogger("CorpusManager")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger
    
    def _lazy_load_corpus(self) -> List[Dict]:
        if not hasattr(self, 'base_corpus'):
            self.logger.info("Loading base corpus")
            batch_size = 1000
            self.base_corpus = []
            
            with open(self.base_corpus_path) as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    batch = []
                    total_docs = mm.count(b'\n')
                    
                    for line in tqdm(mm, total=total_docs, desc="Loading corpus"):
                        try:
                            if line.strip():
                                doc = json.loads(line)
                                batch.append(doc)
                                
                                if len(batch) >= batch_size:
                                    self.base_corpus.extend(batch)
                                    batch = []
                                    gc.collect()
                                    
                        except json.JSONDecodeError:
                            self.logger.warning("Skipping invalid JSON line")
                            continue
                            
                    if batch:
                        self.base_corpus.extend(batch)
                        
            self.logger.info(f"Loaded {len(self.base_corpus)} documents")
            
        return self.base_corpus

    def get_document(self, doc_id: int) -> Optional[Dict]:
        if doc_id in self.doc_cache:
            return self.doc_cache[doc_id]
            
        corpus = self._lazy_load_corpus()
        if 0 <= doc_id < len(corpus):
            doc = corpus[doc_id]
            self.doc_cache[doc_id] = doc
            return doc
        return None

    def get_documents(self, doc_ids: List[int]) -> List[Dict]:
        return [doc for doc_id in doc_ids if (doc := self.get_document(doc_id))]

    def add_noise_documents(self, noise_ratio: float = 0.2, seed: int = 42) -> List[Dict]:
        key = f'noisy_{noise_ratio}_{seed}'
        if key not in self.corpus_variants:
            np.random.seed(seed)
            corpus = self._lazy_load_corpus()
            num_noise = int(len(corpus) * noise_ratio)
            
            self.logger.info(f"Adding {num_noise} noise documents")
            noisy_corpus = corpus.copy()
            
            for idx in tqdm(range(num_noise), desc="Generating noise docs"):
                noisy_doc = {
                    'text': ' '.join(np.random.choice(['random', 'noise', 'distractor'], size=50)),
                    'title': f'Noise_{idx}',
                    'id': len(corpus) + idx,
                    'is_noise': True
                }
                noisy_corpus.append(noisy_doc)
                
            self.corpus_variants[key] = noisy_corpus
            
        return self.corpus_variants[key]

    def clear_cache(self):
        self.doc_cache.clear()
        gc.collect()