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
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        cache_dir: Optional[str] = "cache/llm_responses"
    ):
        if not api_key:
            raise ValueError("API key cannot be empty")
            
        genai.configure(api_key=api_key)
        
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config,
            safety_settings=safety_settings
        )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        gold_answer: str
    ) -> Dict[str, Any]:
        """Evaluate answer using LLM."""
        if not question or not generated_answer or not gold_answer:
            return {
                'score': 0,
                'correct': False,
                'reasoning': "Missing required input",
                'semantic_similarity': 0
            }
            
        prompt = self.evaluation_template.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer
        )
        
        cache_key = self._get_cache_key(prompt)
        cached = self._check_cache(cache_key)
        if cached:
            return cached
            
        try:
            # Use LLM for evaluation
            response = self.eval_model.generate_content(prompt)
            if not response.parts:
                raise ValueError("Empty response from model")
            response_text = response.parts[0].text
            
            # Parse LLM evaluation response
            lines = response_text.strip().split('\n')
            score = float([l for l in lines if 'Score' in l][0].split(':')[1].strip())
            correct = 'yes' in [l for l in lines if 'Correct' in l][0].lower()
            reasoning = [l for l in lines if 'Reasoning' in l][0].split(':')[1].strip()
            
            # Use LLM for semantic similarity
            similarity = self.compute_semantic_similarity(generated_answer, gold_answer)
            
            result = {
                'score': score,
                'correct': correct,
                'reasoning': reasoning,
                'semantic_similarity': similarity,
                'evaluation_method': 'llm',
                'timestamp': str(datetime.now())
            }
            
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {str(e)}")
            return {
                'score': 0,
                'correct': False,
                'reasoning': f"LLM evaluation error: {str(e)}",
                'semantic_similarity': 0,
                'evaluation_method': 'failed',
                'timestamp': str(datetime.now())
            }

    def compute_semantic_similarity(
        self,
        answer1: str,
        answer2: str
    ) -> float:
        prompt = self.similarity_template.format(
            answer1=answer1,
            answer2=answer2
        )
        
        cache_key = self._get_cache_key(prompt)
        cached = self._check_cache(cache_key)
        if cached:
            return cached['similarity']
            
        try:
            response = self.model.generate_content(prompt)
            if not response.parts:
                raise ValueError("Empty response from model")
            response_text = response.parts[0].text
            
            # Extract similarity score from response
            similarity = float(response_text.split('\n')[0].split(':')[1].strip())
            self._save_to_cache(cache_key, {'similarity': similarity})
            return similarity
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {str(e)}")
            return 0.0

    def generate(self, prompts: Union[str, List[str]], max_new_tokens: int = 15) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
            
        results = []
        for prompt in prompts:
            cache_key = self._get_cache_key(prompt)
            cached = self._check_cache(cache_key)
            
            if cached:
                results.append(cached['response'])
                continue
                
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                self._save_to_cache(cache_key, {'response': response_text})
                results.append(response_text)
                
            except Exception as e:
                self.logger.error(f"Generation failed: {str(e)}")
                results.append("")
                
        return results

    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 10
    ) -> List[str]:
        results = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch:
                result = self.generate(prompt)
                batch_results.append(result)
                time.sleep(0.1)  # Rate limiting
                
            results.extend(batch_results)
            
        return results

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("LLM")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'total_evaluations': len(os.listdir(self.cache_dir)),
            'cache_size_mb': sum(f.stat().st_size for f in self.cache_dir.glob('*.json')) / (1024 * 1024)
        }
        return stats