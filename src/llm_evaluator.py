import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import hashlib
from datetime import datetime
from tqdm import tqdm
import google.generativeai as genai
from src.utils.rate_limit import rate_limit



class LLMEvaluator:
    def __init__(self, api_key: str, cache_dir: str = "cache/evaluations"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.eval_cache = {}
        self.stats = {"hits": 0, "misses": 0}
        
        # Initialize Gemini
        #genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self._setup_templates()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("LLMEvaluator")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _setup_templates(self):
        self.templates = {
            'answer_evaluation': """
            Question: {question}
            Generated Answer: {generated_answer}
            Gold Answer: {gold_answer}
            Context: {context}

            Evaluate the answer considering:
            1. Factual accuracy vs gold answer
            2. Information completeness
            3. Context utilization
            4. Semantic equivalence

            Format:
            Score (0-100):
            Correct (Yes/No):
            Reasoning:""",

            'document_relevance': """
            Question: {question}
            Document: {document}
            Gold Answer: {gold_answer}

            Rate document relevance (0-100) considering:
            1. Answer presence
            2. Information completeness
            3. Context relevance

            Provide only the numerical score:""",
            
            'semantic_similarity': """
            Compare these texts and rate their semantic similarity (0-100):
            Text 1: {text1}
            Text 2: {text2}
            Score:""",

            'document_classification': """
            Question: {question}
            Gold Answer: {gold_answer}
            Document: {doc_text}

            Classify this document as either:
            1. "gold" if it contains the answer
            2. "distracting" if semantically similar but doesn't contain answer
            3. "random" if unrelated

            Classification:""",

            'gold_validation': """
            Question: {question}
            Expected Answer: {gold_answer}
            Document: {doc_text}

            Does this document contain enough information to answer the question correctly?
            Answer 'yes' or 'no' and explain why:
            """
        }

    @rate_limit
    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        gold_answer: str,
        context: Optional[str] = None,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        cache_key = hashlib.md5(f"ans_eval:{question}:{generated_answer}:{gold_answer}:{context}".encode()).hexdigest()
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        prompt = self.templates['answer_evaluation'].format(
            question=question,
            generated_answer=generated_answer,
            gold_answer=gold_answer,
            context=context or ""
        )

        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(prompt)
                lines = response.text.strip().split('\n')
                
                score_line = lines[0]
                correct_line = lines[1]
                reasoning_line = lines[2]

                score = float(score_line.split(':', 1)[1].strip())
                correct = 'yes' in correct_line.lower()
                reasoning = reasoning_line.split(':', 1)[1].strip()
                
                result = {
                    'score': score / 100,
                    'correct': correct,
                    'reasoning': reasoning,
                    'timestamp': datetime.now().isoformat()
                }
                
                self._save_to_cache(cache_key, result)
                return result

            except Exception as e:
                if attempt == retry_count - 1:
                    self.logger.error(f"Evaluation failed: {str(e)}")
                    result = {
                        'score': 0,
                        'correct': False,
                        'reasoning': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    self._save_to_cache(cache_key, result)
                    return result
                time.sleep(1)

    def evaluate_batch(
        self,
        items: List[Dict],
        batch_size: int = 10
    ) -> List[Dict]:
        results = []
        
        for i in tqdm(range(0, len(items), batch_size), desc="Evaluating"):
            batch = items[i:i + batch_size]
            batch_results = []
            
            for item in batch:
                eval_result = self.evaluate_answer(
                    item['question'],
                    item['generated_answer'],
                    item['gold_answer'],
                    item.get('context')
                )
                item['evaluation'] = eval_result
                batch_results.append(item)
                time.sleep(0.1)  # Rate limiting
                
            results.extend(batch_results)
            
        return results

    @rate_limit
    def compute_semantic_similarity(
        self,
        text1: str,
        text2: str,
        cache_key: Optional[str] = None
    ) -> float:
        if not cache_key:
            cache_key = hashlib.md5(f"sem_sim:{text1}:{text2}".encode()).hexdigest()
            
        cached = self._check_cache(cache_key)
        if cached and 'similarity' in cached:
            return cached['similarity']
            
        prompt = self.templates['semantic_similarity'].format(
            text1=text1,
            text2=text2
        )
        
        try:
            response = self.model.generate_content(prompt)
            similarity = float(response.text.strip()) / 100
            self._save_to_cache(cache_key, {'similarity': similarity})
            return similarity
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {str(e)}")
            return 0.0

    def evaluate_document_relevance(
        self,
        document: Dict,
        question: str,
        gold_answer: str
    ) -> float:
        doc_id = document.get('id', 'no_id')
        cache_key = hashlib.md5(f"doc_rel:{doc_id}:{question}:{gold_answer}".encode()).hexdigest()
        
        cached = self._check_cache(cache_key)
        if cached and 'relevance' in cached:
            return cached['relevance']
            
        prompt = self.templates['document_relevance'].format(
            question=question,
            document=document['text'],
            gold_answer=gold_answer
        )
        
        try:
            response = self.model.generate_content(prompt)
            relevance = float(response.text.strip()) / 100
            self._save_to_cache(cache_key, {'relevance': relevance})
            return relevance
        except Exception as e:
            self.logger.error(f"Relevance evaluation failed: {str(e)}")
            return 0.0

    def classify_document(
    self,
    question: str,
    gold_answer: str,
    doc_text: str,
    retry_count: int = 3
    ) -> str:
        """Classify document as gold/distracting/random"""
        cache_key = hashlib.md5(f"doc_class:{question}:{gold_answer}:{doc_text}".encode()).hexdigest()
        
        cached = self._check_cache(cache_key)
        if cached and 'category' in cached:
            return cached['category']

        # Added to existing LLMEvaluator class
        prompt = f"""Evaluate if this document contains the answer to the question:

    Question: {question}
    Correct Answer: {gold_answer}
    Document: {doc_text}

    Choose ONE category:
    GOLD - Document contains the correct answer
    DISTRACTING - Document is related but doesn't contain the answer
    RANDOM - Document is unrelated

    Output only one word (GOLD/DISTRACTING/RANDOM):"""

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
                    return 'random'
                time.sleep(1)

    def validate_gold_document(
    self,
    question: str,
    gold_answer: str,
    doc_text: str
    ) -> bool:
        """Validate if document contains the gold answer"""
        cache_key = hashlib.md5(f"gold_val:{question}:{gold_answer}:{doc_text}".encode()).hexdigest()
        
        cached = self._check_cache(cache_key)
        if cached and 'is_valid' in cached:
            return cached['is_valid']

        prompt = f"""
        Question: {question}
        Expected Answer: {gold_answer}
        Document: {doc_text}

        Does this document contain enough information to answer the question correctly?
        Answer 'yes' or 'no' and explain why:
        """

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip().lower()
            is_valid = answer.startswith('yes')
            
            self._save_to_cache(cache_key, {'is_valid': is_valid})
            return is_valid

        except Exception as e:
            self.logger.error(f"Gold validation failed: {str(e)}")
            return False

    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        if cache_key in self.eval_cache:
            self.stats["hits"] += 1
            return self.eval_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    result = json.load(f)
                    self.eval_cache[cache_key] = result
                    self.stats["hits"] += 1
                    return result
            except Exception as e:
                self.logger.warning(f"Cache read failed: {str(e)}")

        self.stats["misses"] += 1
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        self.eval_cache[cache_key] = data
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Cache write failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            'total_evaluations': len(self.eval_cache),
            'cache_hits': self.stats["hits"],
            'cache_misses': self.stats["misses"],
            'cache_hit_rate': hit_rate
        }
