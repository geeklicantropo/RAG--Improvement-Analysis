import torch
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import psutil
import gc
import google.generativeai as genai
import time
from datetime import datetime

class BatchProcessor:
    def __init__(
        self,
        api_key: str,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        gpu_memory_threshold: float = 0.9,
        cpu_memory_threshold: float = 0.9
    ):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.gpu_memory_threshold = gpu_memory_threshold
        self.cpu_memory_threshold = cpu_memory_threshold
        self.logger = self._setup_logger()
        self.stats = {
            "processed_batches": 0,
            "total_items": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0
        }

        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("BatchProcessor")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def evaluate_batch(self, items: List[Dict]) -> List[Dict]:
        """Evaluate a batch of items using LLM"""
        evaluated_items = []

        for item in tqdm(items, desc="Evaluating batch"):
            try:
                eval_prompt = f"""
                Question: {item.get('question', '')}
                Generated Answer: {item.get('generated_answer', '')}
                Gold Answer: {item.get('gold_answer', '')}
                
                Rate the answer's correctness (0-100) and explain why:
                """

                response = self.model.generate_content(eval_prompt)
                eval_text = response.text.strip()

                # Extract score
                score_line = [l for l in eval_text.split('\n') if any(x in l.lower() for x in ['score:', 'rating:'])][0]
                score = float(score_line.split(':')[1].strip().split()[0])

                item["evaluation"] = {
                    "score": score,
                    "explanation": eval_text,
                    "timestamp": datetime.now().isoformat()
                }

                self.stats["successful_evaluations"] += 1

            except Exception as e:
                self.logger.error(f"Evaluation failed: {str(e)}")
                item["evaluation"] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.stats["failed_evaluations"] += 1

            evaluated_items.append(item)
            time.sleep(0.1)  # Rate limiting

        return evaluated_items

    def process_batches(
        self,
        items: List[Dict],
        processing_fn: callable,
        show_progress: bool = True,
        evaluate: bool = True
    ) -> List[Dict]:
        self.logger.info(f"Processing {len(items)} items in batches")
        results = []

        batch_iterator = tqdm(
            range(0, len(items), self.batch_size),
            desc="Processing batches"
        ) if show_progress else range(0, len(items), self.batch_size)

        for batch_start in batch_iterator:
            try:
                # Memory monitoring and batch size adjustment
                self._adjust_batch_size()

                batch_end = min(batch_start + self.batch_size, len(items))
                batch = items[batch_start:batch_end]

                # Process batch
                batch_results = processing_fn(batch)

                # Evaluate if requested
                if evaluate:
                    batch_results = self.evaluate_batch(batch_results)

                results.extend(batch_results)

                # Update stats
                self.stats["processed_batches"] += 1
                self.stats["total_items"] += len(batch)

                # Memory cleanup
                self._cleanup_memory()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning("OOM error - reducing batch size")
                    self.batch_size = max(self.min_batch_size, self.batch_size // 2)

                    # Retry with smaller batch
                    batch_results = processing_fn(batch)
                    if evaluate:
                        batch_results = self.evaluate_batch(batch_results)
                    results.extend(batch_results)
                else:
                    raise

        self.log_final_stats()
        return results

    def _adjust_batch_size(self):
        """Monitor and adjust batch size based on memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.logger.debug(f"GPU memory usage: {gpu_memory:.2%}")

            if gpu_memory > self.gpu_memory_threshold:
                self.batch_size = max(self.min_batch_size, self.batch_size // 2)
                self.logger.info(f"Reduced batch size to {self.batch_size}")

        cpu_percent = psutil.Process().memory_info().rss / psutil.virtual_memory().total
        self.logger.debug(f"CPU memory usage: {cpu_percent:.2%}")

        if cpu_percent > self.cpu_memory_threshold:
            self.batch_size = max(self.min_batch_size, self.batch_size // 2)
            self.logger.info(f"Reduced batch size to {self.batch_size}")

    def _cleanup_memory(self):
        """Memory cleanup with logging"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.debug("Memory cleaned up")

    def log_final_stats(self):
        """Log final processing statistics"""
        self.logger.info("Final Statistics:")
        self.logger.info(f"Total batches processed: {self.stats['processed_batches']}")
        self.logger.info(f"Total items processed: {self.stats['total_items']}")
        self.logger.info(f"Successful evaluations: {self.stats['successful_evaluations']}")
        self.logger.info(f"Failed evaluations: {self.stats['failed_evaluations']}")

        if self.stats['total_items'] > 0:
            success_rate = (self.stats['successful_evaluations'] / self.stats['total_items']) * 100
            self.logger.info(f"Evaluation success rate: {success_rate:.2f}%")