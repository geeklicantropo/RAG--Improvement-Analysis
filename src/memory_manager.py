import torch
import psutil
import gc
import logging
from typing import Dict, Optional, Tuple
from tqdm import tqdm

class MemoryManager:
    def __init__(
        self,
        gpu_memory_threshold: float = 0.9,
        cpu_memory_threshold: float = 0.9,
        min_batch_size: int = 1,
        max_batch_size: int = 64
    ):
        self.gpu_memory_threshold = gpu_memory_threshold
        self.cpu_memory_threshold = cpu_memory_threshold
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.logger = self._setup_logger()
        self.memory_stats = {"peak_gpu": 0, "peak_cpu": 0}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("MemoryManager")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def monitor_memory(self) -> Dict[str, float]:
        stats = {
            'cpu_percent': psutil.Process().memory_info().rss / psutil.virtual_memory().total,
            'ram_gb': psutil.Process().memory_info().rss / (1024 ** 3)
        }
        
        if torch.cuda.is_available():
            stats.update({
                f'gpu_{i}_percent': torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            })
            
            current_gpu = stats.get('gpu_0_percent', 0)
            if current_gpu > self.memory_stats["peak_gpu"]:
                self.memory_stats["peak_gpu"] = current_gpu
                
        current_cpu = stats['cpu_percent']
        if current_cpu > self.memory_stats["peak_cpu"]:
            self.memory_stats["peak_cpu"] = current_cpu
            
        return stats

    def compute_optimal_batch_size(self, current_memory: Dict[str, float]) -> int:
        gpu_mem = max([v for k, v in current_memory.items() if 'gpu_' in k], default=0)
        cpu_mem = current_memory['cpu_percent']
        
        if gpu_mem > self.gpu_memory_threshold or cpu_mem > self.cpu_memory_threshold:
            self.logger.warning("Memory threshold exceeded, reducing batch size")
            return max(self.min_batch_size, self.max_batch_size // 2)
        return self.max_batch_size

    def cleanup(self, full: bool = False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if full:
                torch.cuda.synchronize()

    def log_memory_stats(self):
        stats = self.monitor_memory()
        self.logger.info(f"Current Memory Usage - CPU: {stats['cpu_percent']:.2%}, RAM: {stats['ram_gb']:.2f}GB")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"GPU {i}: {stats[f'gpu_{i}_percent']:.2%}")
                
    def get_memory_summary(self) -> Dict[str, float]:
        return {
            'peak_gpu_memory': self.memory_stats["peak_gpu"],
            'peak_cpu_memory': self.memory_stats["peak_cpu"],
            'current_memory': self.monitor_memory()
        }