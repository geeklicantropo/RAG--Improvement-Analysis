import os
import gc
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import google.generativeai as genai
from tqdm import tqdm
import torch
import numpy as np

class LLM:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        cache_dir: Optional[str] = "cache/llm_responses",
        memory_threshold: float = 0.9,
        max_retries: int = 3,
        evaluation_threshold: float = 0.8
    ):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Configure Gemini model settings
        generation_config = {
            "temperature": 0.3,
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
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        self.eval_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_threshold = memory_threshold
        self.max_retries = max_retries
        self.evaluation_threshold = evaluation_threshold
        self.logger = self._setup_logger()

        # Templates
        self.generation_template = """
        Question: {question}
        Context: {context}
        Task: Extract the precise answer from the context.
        Answer:"""
        
        self.evaluation_template = """
        You are an expert evaluator judging answer correctness.
        
        Question: {question}
        Gold Answer: {gold_answer}
        Generated Answer: {generated_answer}
        
        Task: Evaluate if the generated answer is correct. Consider:
        1. Factual accuracy
        2. Semantic equivalence
        3. Completeness
        
        Provide your evaluation in the following format:
        Score (0-100): <numerical score>
        Correct (Yes/No): <yes/no>
        Reasoning: <detailed explanation>
        """
        
        self.similarity_template = """
        Task: Evaluate the semantic similarity between these two answers on a scale of 0-100.
        
        Answer 1: {answer1}
        Answer 2: {answer2}
        
        Provide:
        Similarity Score (0-100): 
        Explanation:
        """

    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        gold_answer: str
    ) -> Dict[str, Any]:
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
            response = self.eval_model.generate_content(prompt)
            if not response.parts:
                raise ValueError("Empty response from model")
            response_text = response.parts[0].text
            
            # Parse evaluation response
            lines = response_text.strip().split('\n')
            score = float([l for l in lines if 'Score' in l][0].split(':')[1].strip())
            correct = 'yes' in [l for l in lines if 'Correct' in l][0].lower()
            reasoning = [l for l in lines if 'Reasoning' in l][0].split(':')[1].strip()
            
            similarity = self.compute_semantic_similarity(generated_answer, gold_answer)
            
            result = {
                'score': score,
                'correct': correct,
                'reasoning': reasoning,
                'semantic_similarity': similarity
            }
            
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {
                'score': 0,
                'correct': False,
                'reasoning': f"Evaluation error: {str(e)}",
                'semantic_similarity': 0
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

    def generate(
        self,
        questions: Union[str, List[str]],
        contexts: Union[str, List[str]],
        batch_size: int = 8
    ) -> List[str]:
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(contexts, str):
            contexts = [contexts]

        self.logger.info(f"Generating answers for {len(questions)} questions")
        results = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
            batch_questions = questions[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            batch_results = []
            for q, c in zip(batch_questions, batch_contexts):
                prompt = self.generation_template.format(question=q, context=c)
                cache_key = self._get_cache_key(prompt)
                
                cached = self._check_cache(cache_key)
                if cached:
                    batch_results.append(cached['response'])
                    continue
                
                for attempt in range(self.max_retries):
                    try:
                        response = self.model.generate_content(prompt)
                        if not response.parts:
                            raise ValueError("Empty response from model")
                        response_text = response.parts[0].text
                        self._save_to_cache(cache_key, {'response': response_text})
                        batch_results.append(response_text)
                        break
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            self.logger.error(f"Generation failed: {str(e)}")
                            batch_results.append("")
                        else:
                            self.logger.warning(f"Attempt {attempt+1} failed, retrying...")
            
            results.extend(batch_results)
            self._cleanup_memory()
            
        return results

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("LLM")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
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