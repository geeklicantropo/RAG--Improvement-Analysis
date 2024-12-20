import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import hashlib
from datetime import datetime
from tqdm import tqdm
import google.generativeai as genai
from src.utils.rate_limit import rate_limit
import time


class LLM:
    def __init__(
        self,
        api_key: str,
        cache_dir: str = "cache/llm_responses"
    ):
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 3,
            "candidate_count": 1
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

        genai.configure(api_key=api_key)
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
    def generate(self, prompts: Union[str, List[str]], max_new_tokens: int = 15) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            result = self.generate_with_retry(prompt)
            results.append(result if result else "")

        return results

    @rate_limit
    def generate_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
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
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(1)

        return None

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
        """
        Get generation statistics.

        Returns:
            Dict[str, Any]: Statistics of LLM operations.
        """
        total = self.stats["total_calls"]
        if total > 0:
            success_rate = (self.stats["successful_calls"] / total) * 100
            cache_rate = (self.stats["cache_hits"] / total) * 100
            failure_rate = (self.stats["failures"] / total) * 100

            return {
                **self.stats,
                "success_rate": f"{success_rate:.2f}%",
                "cache_hit_rate": f"{cache_rate:.2f}%",
                "failure_rate": f"{failure_rate:.2f}%",
                "cluster_eval_stats": self.stats.get("cluster_eval", {})
            }
        return self.stats