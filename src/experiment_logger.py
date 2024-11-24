import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import psutil
import torch
from tqdm import tqdm

class ExperimentLogger:
    """
    A comprehensive logger for RAG experiments that handles both console and file logging,
    progress tracking, and resource monitoring.
    """
    def __init__(
        self,
        experiment_name: str,
        base_log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        self.experiment_name = experiment_name
        self.start_time = time.time()
        
        # Setup log directories
        self.experiment_log_dir = os.path.join(base_log_dir, "experiment_logs", experiment_name)
        self.system_log_dir = os.path.join(base_log_dir, "system_logs")
        Path(self.experiment_log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.system_log_dir).mkdir(parents=True, exist_ok=True)

        # Create loggers
        self.experiment_logger = self._setup_logger(
            f"{experiment_name}_experiment",
            os.path.join(self.experiment_log_dir, f"{experiment_name}_{self._get_timestamp()}.log"),
            console_level,
            file_level
        )
        
        self.system_logger = self._setup_logger(
            f"{experiment_name}_system",
            os.path.join(self.system_log_dir, f"{experiment_name}_system_{self._get_timestamp()}.log"),
            console_level,
            file_level
        )

        # Initialize metrics storage
        self.metrics: Dict[str, Any] = {}
        
    def _get_timestamp(self) -> str:
        """Generate a timestamp string."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _setup_logger(
        self,
        name: str,
        log_file: str,
        console_level: int,
        file_level: int
    ) -> logging.Logger:
        """Setup individual logger with console and file handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file)
        
        # Set levels
        console_handler.setLevel(console_level)
        file_handler.setLevel(file_level)
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger

    def log_experiment_params(self, params: Dict[str, Any]):
        """Log experiment parameters."""
        self.experiment_logger.info(f"Experiment Parameters: {json.dumps(params, indent=2)}")
        
    def log_progress(self, iterable, desc: str = "", **kwargs):
        """Create a progress bar for an iterable."""
        return tqdm(iterable, desc=desc, **kwargs)

    def log_system_info(self):
        """Log system information including GPU if available."""
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        system_info = {
            "cpu_usage": f"{cpu_usage}%",
            "memory_used": f"{memory.percent}%",
            "memory_available": f"{memory.available / (1024 ** 3):.2f} GB"
        }
        
        if torch.cuda.is_available():
            gpu_info = {
                f"gpu_{i}": {
                    "name": torch.cuda.get_device_name(i),
                    "memory_used": f"{torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB",
                    "memory_total": f"{torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB"
                }
                for i in range(torch.cuda.device_count())
            }
            system_info["gpu"] = gpu_info
            
        self.system_logger.info(f"System Information: {json.dumps(system_info, indent=2)}")

    def log_error(self, error: Exception, additional_info: Optional[str] = None):
        """Log an error with optional additional information."""
        error_msg = f"Error: {str(error)}"
        if additional_info:
            error_msg += f"\nAdditional Info: {additional_info}"
        self.system_logger.error(error_msg, exc_info=True)

    def log_metric(self, metric_name: str, value: Any):
        """Log a metric value."""
        self.metrics[metric_name] = value
        self.experiment_logger.info(f"Metric - {metric_name}: {value}")

    def save_metrics(self):
        """Save all collected metrics to a JSON file."""
        metrics_file = os.path.join(
            self.experiment_log_dir,
            f"metrics_{self._get_timestamp()}.json"
        )
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.experiment_logger.info(f"Metrics saved to {metrics_file}")

    def get_experiment_duration(self) -> float:
        """Get the duration of the experiment in seconds."""
        return time.time() - self.start_time

    def log_step_start(self, step_name: str):
        """Log the start of an experiment step."""
        self.experiment_logger.info(f"Starting step: {step_name}")
        return time.time()

    def log_step_end(self, step_name: str, start_time: float):
        """Log the end of an experiment step with duration."""
        duration = time.time() - start_time
        self.experiment_logger.info(f"Completed step: {step_name} - Duration: {duration:.2f}s")

    def __enter__(self):
        """Context manager entry."""
        self.log_system_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.log_error(exc_val)
        self.save_metrics()
        duration = self.get_experiment_duration()
        self.experiment_logger.info(f"Experiment completed in {duration:.2f}s")