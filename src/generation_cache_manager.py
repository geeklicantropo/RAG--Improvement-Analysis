import os
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime
import shutil
from tqdm import tqdm
import google.generativeai as genai
import time

class GenerationCacheManager:
    def __init__(
        self,
        api_key: str,
        cache_dir: str = "cache/llm_responses",
        max_cache_size_gb: float = 10.0,
        batch_size: int = 1000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_gb = max_cache_size_gb
        self.batch_size = batch_size
        self.logger = self._setup_logger()
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}

        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("GenerationCacheManager")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def evaluate_generation(self, prompt: str, response: str, gold_answer: str) -> Dict[str, Any]:
        """Evaluate generation using Gemini"""
        eval_prompt = f"""
        Question: {prompt}
        Generated Answer: {response}
        Gold Answer: {gold_answer}
        
        Rate the answer's correctness on a scale of 0-100 and explain why:
        """
        
        try:
            response = self.model.generate_content(eval_prompt)
            eval_text = response.text.strip()
            
            # Extract score and explanation
            score_line = [l for l in eval_text.split('\n') if any(x in l.lower() for x in ['score:', 'rating:'])][0]
            score = float(score_line.split(':')[1].strip().split()[0])
            
            return {
                "score": score,
                "explanation": eval_text,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {"score": 0, "explanation": str(e)}

    def get_response(self, prompt: str) -> Optional[Dict]:
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    self.memory_cache[cache_key] = data
                    self.cache_stats["hits"] += 1
                    return data
            except Exception as e:
                self.logger.error(f"Cache read error: {str(e)}")
                
        self.cache_stats["misses"] += 1
        return None

    def save_batch(self, items: List[Dict[str, str]], show_progress: bool = True):
        """Save batch of prompts and responses with evaluation"""
        self.logger.info(f"Processing batch of {len(items)} items")
        
        iterator = tqdm(items) if show_progress else items
        for item in iterator:
            try:
                # Evaluate if gold answer provided
                if "gold_answer" in item:
                    evaluation = self.evaluate_generation(
                        item["prompt"],
                        item["response"],
                        item["gold_answer"]
                    )
                    item["evaluation"] = evaluation

                self.save_response(item)
                
            except Exception as e:
                self.logger.error(f"Batch processing error: {str(e)}")
                continue
                
            time.sleep(0.1)  # Rate limiting

    def save_response(self, item: Dict[str, Any]):
        cache_key = hashlib.md5(item["prompt"].encode()).hexdigest()
        
        # Save to memory cache
        self.memory_cache[cache_key] = item
        
        # Save to disk cache
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(item, f)
                
            self._check_cache_size()
            
        except Exception as e:
            self.logger.error(f"Cache write error: {str(e)}")

    def _check_cache_size(self):
        """Monitor and manage cache size"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        cache_size_gb = total_size / (1024 ** 3)
        
        if cache_size_gb > self.max_cache_size_gb:
            self.logger.warning(f"Cache size ({cache_size_gb:.2f}GB) exceeds limit")
            self._clean_cache()

    def _clean_cache(self):
        """Clean cache with progress bar"""
        cache_files = list(self.cache_dir.glob("*.json"))
        cache_files.sort(key=lambda x: x.stat().st_mtime)
        
        target_size = self.max_cache_size_gb * 0.8 * (1024 ** 3)
        current_size = sum(f.stat().st_size for f in cache_files)
        
        with tqdm(total=len(cache_files), desc="Cleaning cache") as pbar:
            for file in cache_files:
                if current_size <= target_size:
                    break
                    
                try:
                    file_size = file.stat().st_size
                    file.unlink()
                    current_size -= file_size
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {str(e)}")
                
                pbar.update(1)