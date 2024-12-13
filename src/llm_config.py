import os
import time
import logging
from typing import Dict, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta
import google.generativeai as genai
from ratelimit import limits, sleep_and_retry

class LLMConfig:
    def __init__(self, api_key: str, cache_dir: str = "cache/llm"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
        self.rate_limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 3000,
            'tokens_per_minute': 60000
        }
        
        self.usage_tracking = {
            'minute': {'count': 0, 'reset_time': datetime.now()},
            'hour': {'count': 0, 'reset_time': datetime.now()},
            'token_count': 0
        }
        
        self._initialize_gemini()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("LLMConfig")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _initialize_gemini(self):
        genai.configure(api_key=self.api_key)
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

    @sleep_and_retry
    @limits(calls=60, period=60)
    def check_rate_limit(self):
        self._update_usage_tracking()

    def _update_usage_tracking(self):
        now = datetime.now()
        
        # Reset minute counters
        if now - self.usage_tracking['minute']['reset_time'] > timedelta(minutes=1):
            self.usage_tracking['minute'] = {'count': 0, 'reset_time': now}
            self.usage_tracking['token_count'] = 0
            
        # Reset hourly counters    
        if now - self.usage_tracking['hour']['reset_time'] > timedelta(hours=1):
            self.usage_tracking['hour'] = {'count': 0, 'reset_time': now}

        self.usage_tracking['minute']['count'] += 1
        self.usage_tracking['hour']['count'] += 1

    def add_token_usage(self, tokens: int):
        self.usage_tracking['token_count'] += tokens
        if self.usage_tracking['token_count'] > self.rate_limits['tokens_per_minute']:
            sleep_time = 60 - (datetime.now() - self.usage_tracking['minute']['reset_time']).seconds
            self.logger.warning(f"Token limit reached, sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
            self.usage_tracking['token_count'] = tokens

    def get_cache_key(self, prompt: str) -> str:
        return f"{hash(prompt)}.json"

    def save_to_cache(self, prompt: str, response: str):
        cache_key = self.get_cache_key(prompt)
        cache_path = self.cache_dir / cache_key
        with open(cache_path, 'w') as f:
            json.dump({'prompt': prompt, 'response': response, 'timestamp': str(datetime.now())}, f)

    def load_from_cache(self, prompt: str) -> Optional[str]:
        cache_key = self.get_cache_key(prompt)
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
                return data['response']
        return None

    def get_model_fallback_chain(self) -> list:
        return [
            "gemini-pro",  # Primary model
            "chat-bison",  # Fallback model
            "text-bison"   # Emergency fallback
        ]