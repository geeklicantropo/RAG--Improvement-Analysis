import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import warnings
import psutil
import torch
from tqdm import tqdm

class ExperimentLogger:
    _instances = {}  # Class variable to store instances
    
    def __new__(cls, experiment_name: str, base_log_dir: str = "logs"):
        # Create singleton instance per experiment name
        if experiment_name not in cls._instances:
            cls._instances[experiment_name] = super().__new__(cls)
        return cls._instances[experiment_name]
        
    def __init__(self, experiment_name: str, base_log_dir: str = "logs"):
        # Only initialize if this is a new instance
        if not hasattr(self, 'initialized'):
            self.experiment_name = experiment_name
            self.start_time = time.time()
            
            # Setup log directories
            self.experiment_log_dir = os.path.join(base_log_dir, "experiment_logs", experiment_name)
            self.system_log_dir = os.path.join(base_log_dir, "system_logs")
            Path(self.experiment_log_dir).mkdir(parents=True, exist_ok=True)
            Path(self.system_log_dir).mkdir(parents=True, exist_ok=True)
            
            # Suppress all warnings
            warnings.filterwarnings('ignore')
            logging.getLogger('transformers').setLevel(logging.ERROR)
            logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
            
            # Initialize loggers only once with stricter logging levels
            self.experiment_logger = self._get_logger(
                f"{experiment_name}_experiment",
                os.path.join(self.experiment_log_dir, f"{experiment_name}_{self._get_timestamp()}.log"),
                console_level=logging.INFO,  # More restrictive console output
                file_level=logging.DEBUG     # Keep detailed logs in file
            )
            self.system_logger = self._get_logger(
                f"{experiment_name}_system",
                os.path.join(self.system_log_dir, f"{experiment_name}_system_{self._get_timestamp()}.log"),
                console_level=logging.WARNING,  # Only show warnings/errors on console
                file_level=logging.INFO        # Keep important info in file
            )
            
            self.metrics = {}
            self.initialized = True
            
            # Set up output filtering
            self._setup_output_filtering()
    
    def _setup_output_filtering(self):
        """Configure output filtering rules."""
        self.skip_patterns = [
            'token indices sequence',
            'Setting `pad_token_id`',
            'Special tokens have been added',
            'Using device:',
            'corpus_idx',
            'Document [',
            'embedding',
            'processing batch',
            'tokenizer'
        ]
        
        # Only show these patterns on console
        self.console_patterns = [
            'Error:',
            'WARNING:',
            'Starting experiment',
            'Completed experiment',
            'Progress:',
            'Accuracy:'
        ]
    
    def _should_log_to_console(self, message: str) -> bool:
        """Determine if message should be shown on console."""
        if any(pattern in message for pattern in self.skip_patterns):
            return False
            
        if any(pattern in message for pattern in self.console_patterns):
            return True
            
        return False

    def _get_logger(
        self,
        name: str,
        log_file: str,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ) -> logging.Logger:
        """Get or create logger with proper configuration and filtering."""
        logger = logging.getLogger(name)
        
        # Remove existing handlers if any
        if logger.hasHandlers():
            logger.handlers.clear()
            
        logger.setLevel(logging.DEBUG)  # Base level for logger
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message).500s'  # Limit message length
        )
        
        # Console handler with filtering
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(lambda record: self._should_log_to_console(record.getMessage()))
        
        # File handler for complete logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    def log_experiment_params(self, params: Dict[str, Any]):
        """Log experiment parameters."""
        # Only log to file, not console
        if isinstance(params, dict):
            self.experiment_logger.debug(f"Experiment Parameters: {json.dumps(params, indent=2)}")
        
    def log_progress(self, iterable, desc: str = "", **kwargs):
        """Create a progress bar with minimal output."""
        return tqdm(
            iterable,
            desc=desc,
            ncols=80,  # Fixed width
            leave=False,  # Don't leave progress bars
            **kwargs
        )

    def log_system_info(self):
        """Log system information with minimal console output."""
        try:
            system_info = {
                "cpu_usage": f"{psutil.cpu_percent()}%",
                "memory_used": f"{psutil.virtual_memory().percent}%"
            }
            
            if torch.cuda.is_available():
                system_info["gpu"] = {
                    f"gpu_{i}": {
                        "name": torch.cuda.get_device_name(i),
                        "memory_used": f"{torch.cuda.memory_allocated(i) / 1e9:.2f}GB"
                    }
                    for i in range(torch.cuda.device_count())
                }
            
            # Log to file only
            self.system_logger.debug(f"System Information: {json.dumps(system_info, indent=2)}")
            
        except Exception as e:
            self.log_error(e, "Error logging system info")

    def log_error(self, error: Exception, additional_info: Optional[str] = None):
        """Log errors prominently."""
        error_msg = f"Error: {str(error)}"
        if additional_info:
            error_msg += f"\nInfo: {additional_info}"
        self.system_logger.error(error_msg, exc_info=True)

    def log_metric(self, metric_name: str, value: Any):
        """Log metrics with controlled output."""
        self.metrics[metric_name] = value
        # Only log certain metrics to console
        if any(pattern in metric_name for pattern in ['accuracy', 'loss', 'error']):
            #self.experiment_logger.info(f"{metric_name}: {value}")
            self.experiment_logger.info(f"{metric_name}")
        else:
            #self.experiment_logger.debug(f"{metric_name}: {value}")
            self.experiment_logger.debug(f"{metric_name}")

    def log_step_start(self, step_name: str):
        """Log step start with minimal output."""
        self.experiment_logger.info(f"Starting: {step_name}")
        return time.time()

    def log_step_end(self, step_name: str, start_time: float):
        """Log step completion with timing."""
        duration = time.time() - start_time
        self.experiment_logger.info(f"Completed: {step_name} ({duration:.2f}s)")

    def save_metrics(self):
        """Save metrics to file without console output."""
        metrics_file = os.path.join(
            self.experiment_log_dir,
            f"metrics_{self._get_timestamp()}.json"
        )
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.experiment_logger.debug(f"Metrics saved to {metrics_file}")

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_experiment_duration(self) -> float:
        return time.time() - self.start_time

    def __enter__(self):
        self.log_system_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log_error(exc_val)
        self.save_metrics()
        duration = self.get_experiment_duration()
        self.experiment_logger.info(f"Experiment completed in {duration:.2f}s")