import logging
import time
import hashlib
import json
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from pathlib import Path
import torch
import gc
from collections import defaultdict
import google.generativeai as genai
from .llm_evaluator import LLMEvaluator

class DocumentClassifier:
    def __init__(
        self,
        llm_evaluator: LLMEvaluator,
        cache_dir: str = "cache/document_classifier",
        batch_size: int = 10,
        max_memory_usage: float = 0.9
    ):
        self.llm_evaluator = llm_evaluator
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_memory_usage = max_memory_usage
        self.logger = self._setup_logger()
        self.classification_cache = {}
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_classifications": 0,
            "gold_hits": 0
        }

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DocumentClassifier")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def classify_documents(
    self,
    documents: List[Dict],
    query: str,
    gold_answer: str,
    use_cache: bool = True
    ) -> Dict[str, List[Dict]]:
        categories = {"gold": [], "distracting": [], "random": []}
        
        if not documents:
            return categories
            
        try:
            self.logger.info(f"Starting classification of {len(documents)} documents")
            
            for i in range(0, len(documents), self.batch_size):
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    max_allocated = torch.cuda.max_memory_allocated()
                    if max_allocated > 0:
                        current_memory = allocated / max_allocated
                        if current_memory > self.max_memory_usage:
                            self.batch_size = max(1, self.batch_size // 2)
                            self.logger.info(f"Reduced batch size to {self.batch_size}")

                batch = documents[i:i + self.batch_size]
                batch_classifications = self._classify_batch(
                    batch,
                    query,
                    gold_answer,
                    use_cache
                )

                for cat, docs in batch_classifications.items():
                    if cat == 'gold':
                        self.stats["gold_hits"] += len(docs)
                        for doc in docs:
                            self.logger.debug(f"Gold document content: {doc.get('text', '')[:200]}")
                    categories[cat].extend(docs)

                self._cleanup_memory()

            self._log_classification_stats(categories)
            return categories
                
        except Exception as e:
            self.logger.error(f"Error classifying documents: {str(e)}")
            raise

    def _classify_batch(self, batch: List[Dict], query: str, gold_answer: str, use_cache: bool) -> Dict[str, List[Dict]]:
        batch_categories = defaultdict(list)

        for doc in batch:
            try:
                # First check if document contains gold answer
                doc_text = doc.get('text', '')
                contains_answer = self._contains_answer(doc_text, gold_answer)
                
                if contains_answer:
                    self.logger.info(f"Document {doc.get('id', 'unknown')} identified as gold")
                    doc = doc.copy()
                    doc['category'] = 'gold'
                    doc['is_gold'] = True
                    batch_categories['gold'].append(doc)
                    continue

                cache_key = None
                if use_cache:
                    cache_key = self._get_cache_key(doc, query, gold_answer)
                    cached_category = self._check_cache(cache_key)
                    if cached_category:
                        doc = doc.copy()
                        batch_categories[cached_category].append(doc)
                        self.stats["cache_hits"] += 1
                        continue

                category = self.llm_evaluator.classify_document(
                    question=query,
                    gold_answer=gold_answer,
                    doc_text=doc_text[:500]
                )

                doc = doc.copy()
                doc['category'] = category
                batch_categories[category].append(doc)

                if use_cache and cache_key:
                    self._save_to_cache(cache_key, category)
                    self.stats["cache_misses"] += 1

                self.stats["total_classifications"] += 1
                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                self.logger.error(f"Error classifying document: {str(e)}")
                doc = doc.copy() 
                batch_categories["random"].append(doc)

        return batch_categories

    def _contains_answer(self, text: str, answer: str) -> bool:
        """Check if document text contains the gold answer."""
        if not text or not answer:
            return False
            
        # Normalize text and answer
        text = text.lower().strip()
        answer = answer.lower().strip()
        
        # Check exact match
        if answer in text:
            return True
            
        # Check word overlap
        answer_words = set(answer.split())
        text_words = set(text.split())
        overlap = len(answer_words & text_words)
        
        # If more than 80% of answer words are in text
        if overlap >= len(answer_words) * 0.8:
            return True
            
        return False

    def validate_gold_documents(
        self,
        documents: List[Dict],
        query: str,
        gold_answer: str,
        batch_size: Optional[int] = None
    ) -> List[Dict]:
        if batch_size is None:
            batch_size = self.batch_size

        validated_docs = []
        self.logger.info("Validating gold documents")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                try:
                    cache_key = self._get_cache_key(doc, query, gold_answer, prefix="validation")
                    cached_result = self._check_cache(cache_key)
                    
                    if cached_result is not None:
                        if cached_result.get("is_valid", False):
                            validated_docs.append(doc)
                        continue

                    is_valid = self.llm_evaluator.validate_gold_document(
                        question=query,
                        gold_answer=gold_answer,
                        doc_text=doc.get('text', '')
                    )

                    if is_valid:
                        validated_docs.append(doc)
                        
                    self._save_to_cache(cache_key, {"is_valid": is_valid})
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.error(f"Validation error: {str(e)}")
                    continue

            self._cleanup_memory()

        self.logger.info(f"Validated {len(validated_docs)} gold documents")
        return validated_docs

    def _get_cache_key(
        self,
        doc: Dict,
        query: str,
        gold_answer: str,
        prefix: str = "classification"
    ) -> str:
        content = f"{doc.get('text', '')}:{query}:{gold_answer}"
        return f"{prefix}_{hashlib.md5(content.encode()).hexdigest()}"

    def _check_cache(self, cache_key: str) -> Optional[str]:
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Cache read error: {str(e)}")
        return None

    def _save_to_cache(self, cache_key: str, category: str):
        self.classification_cache[cache_key] = category
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(category, f)
        except Exception as e:
            self.logger.error(f"Cache write error: {str(e)}")

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _log_classification_stats(self, categories: Dict[str, List[Dict]]):
        stats = {
            cat: len(docs) for cat, docs in categories.items()
        }
        self.logger.info(f"Classification results: {stats}")
        self.logger.info(f"Cache stats - Hits: {self.stats['cache_hits']}, Misses: {self.stats['cache_misses']}")

    def get_classification_stats(self) -> Dict[str, Any]:
        return {
            "classifications": self.stats["total_classifications"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": self.stats["cache_hits"] / max(1, self.stats["total_classifications"])
        }