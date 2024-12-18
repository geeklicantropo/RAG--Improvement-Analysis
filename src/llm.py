from src.utils.rate_limit import rate_limit
import os
import gc
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import logging
import google.generativeai as genai
from tqdm import tqdm
import torch
import time

class LLM:
    def __init__(self, api_key: str, cache_dir: str = "cache/llm_responses"):
        self.generation_config = {
            "temperature": 0.1,  
            "top_p": 0.95,      
            "top_k": 3,         
            "max_output_tokens": 1024,
        }

        self.safety_settings = [
            {"category": cat, "threshold": "BLOCK_NONE"}
            for cat in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH", 
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT"
            ]
        ]

        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "cache_hits": 0,
            "retries": 0,
            "failures": 0
        }

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("LLM")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    @rate_limit
    def generate_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate text with retry logic."""
        cache_key = self._get_cache_key(prompt)
        cached = self._check_cache(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            return cached

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    self._save_to_cache(cache_key, response.text)
                    return response.text
                
                modified_prompt = f"Respond with a specific answer based on the context: {prompt}"
                response = self.model.generate_content(modified_prompt)
                
                if response.text:
                    self._save_to_cache(cache_key, response.text)
                    return response.text

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(1)

        return None

    def _modify_prompt(self, prompt: str, attempt: int) -> str:
        """Modify prompt slightly when retrying to avoid copyright issues."""
        modifications = [
            lambda p: f"Please provide an original response to: {p}",
            lambda p: f"Using your own words, address: {p}",
            lambda p: f"Generate a unique response for: {p}"
        ]
        
        if attempt < len(modifications):
            return modifications[attempt](prompt)
        return prompt

    @rate_limit
    def generate(self, prompts: Union[str, List[str]], max_new_tokens: int = 15) -> List[str]:
        """Generate responses for prompts."""
        if isinstance(prompts, str):
            prompts = [prompts]
            
        results = []
        for prompt in prompts:
            augmented_prompt = f"""You are an expert at answering questions using provided context. 
    Rules:
    - Use ONLY information from the given context
    - If the answer is in the context, provide it directly and concisely
    - If uncertain about context information, point out the specific ambiguity
    - If the answer isn't in the context, briefly explain which information is missing

    Context and Question:
    {prompt}

    Provide a direct answer based on the context above:"""

            result = self.generate_with_retry(augmented_prompt)
            results.append(result if result else "")
                    
        return results

    @rate_limit
    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        gold_answer: str,
        context: Optional[str] = None,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """Evaluate generated answer against gold answer."""
        cache_key = hashlib.md5(f"{question}:{generated_answer}:{gold_answer}:{context}".encode()).hexdigest()
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        eval_prompt = f"""
        Question: {question}
        Generated Answer: {generated_answer}
        Gold Answer: {gold_answer}
        {'Context: ' + context if context else ''}
        
        Rate the answer's correctness (0-100) and explain why:
        """
        
        for attempt in range(retry_count):
            try:
                result = self.model.generate_content(eval_prompt)
                lines = result.text.strip().split('\n')
                score_line = next((l for l in lines if any(x in l.lower() for x in ['score:', 'rating:'])), None)
                
                if score_line:
                    score = float(score_line.split(':')[1].strip().split()[0])
                    eval_result = {
                        "score": score / 100,
                        "correct": score >= 70,
                        "reasoning": '\n'.join(lines[1:]),
                        "timestamp": datetime.now().isoformat()
                    }
                    self._save_to_cache(cache_key, eval_result)
                    return eval_result
                    
                raise ValueError("Could not parse evaluation score")
                
            except Exception as e:
                if attempt == retry_count - 1:
                    self.logger.error(f"Final evaluation attempt failed: {str(e)}")
                    break
                time.sleep(2 ** attempt)
        
        failed_result = {
            "score": 0,
            "correct": False,
            "reasoning": "Evaluation failed after multiple retries",
            "timestamp": datetime.now().isoformat()
        }
        self._save_to_cache(cache_key, failed_result)
        return failed_result

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[str]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)["response"]
            except Exception as e:
                self.logger.error(f"Cache read error: {str(e)}")
        return None

    def _save_to_cache(self, cache_key: str, response: str):
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({"response": response, "timestamp": datetime.now().isoformat()}, f)
        except Exception as e:
            self.logger.error(f"Cache write error: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        total = self.stats["total_calls"]
        if total > 0:
            success_rate = (self.stats["successful_calls"] / total) * 100
            cache_rate = (self.stats["cache_hits"] / total) * 100
            failure_rate = (self.stats["failures"] / total) * 100
            
            return {
                **self.stats,
                "success_rate": f"{success_rate:.2f}%",
                "cache_hit_rate": f"{cache_rate:.2f}%",
                "failure_rate": f"{failure_rate:.2f}%"
            }
        return self.stats