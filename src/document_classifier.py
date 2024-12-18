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
from .llm_evaluator import LLMEvaluator
from src.utils.rate_limit import rate_limit

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

    @rate_limit
    def classify_document(
    self,
    question: str,
    gold_answer: str,
    doc_text: str,
    retry_count: int = 3
    ) -> str:
        cache_key = hashlib.md5(f"doc_class:{question}:{gold_answer}:{doc_text}".encode()).hexdigest()
    
        cached = self._check_cache(cache_key)
        if cached and 'category' in cached:
            return cached['category']

        prompt = f"""You are categorizing a document's relevance and correctness with respect to a given question and its gold answer.

        Follow these steps carefully:
        1. Read the question and the gold answer.
        2. Examine the document for information that directly or indirectly (through paraphrase or close factual match) matches the gold answer. If the document clearly supports or provides the gold answer, it is "GOLD".
        3. If the document does not contain the gold answer, check if it is still about the same topic or context as the question. If it is relevant to the question but does NOT provide the correct answer, it is "DISTRACTING".
        4. If the document is off-topic or unrelated to the question, it is "RANDOM".

        Definitions:
        - GOLD: The document provides substantial evidence or content that answers the question correctly (it either explicitly states the gold answer or contains enough information from which the gold answer can be confidently derived).
        - DISTRACTING: The document stays on-topic (discusses something related to the question) but does not provide the gold answer. It might mention related concepts, hints, or partially correct info, but not the actual needed answer.
        - RANDOM: The document is unrelated, off-topic, or does not help answer the question at all.

        Now classify the document strictly as one of [GOLD, DISTRACTING, RANDOM].

        Question: {question}
        Gold Answer: {gold_answer}
        Document: {doc_text}

        Your classification (just one word: GOLD, DISTRACTING, or RANDOM):"""

        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(prompt)
                if not response.text:
                    raise ValueError("Empty response")
                    
                category = response.text.strip().upper()
                if category in ['GOLD', 'DISTRACTING', 'RANDOM']:
                    self._save_to_cache(cache_key, {'category': category.lower()})
                    return category.lower()
                    
                raise ValueError(f"Invalid category: {category}")

            except Exception as e:
                self.logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == retry_count - 1:
                    # Try one final time with simplified prompt
                    try:
                        simple_prompt = f"Is this document: {doc_text[:200]} relevant to answering: {question}? Say only DISTRACTING or RANDOM"
                        response = self.model.generate_content(simple_prompt)
                        category = 'DISTRACTING' if 'DISTRACTING' in response.text.upper() else 'RANDOM'
                        return category.lower()
                    except:
                        self.logger.error("Final fallback failed")
                        raise
                time.sleep(1)

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

    @rate_limit
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