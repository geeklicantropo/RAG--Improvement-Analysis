import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

class ExperimentLogger:
    def __init__(self, output_dir: Path, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.log_dir = output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.memory_logger = self._setup_memory_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # File handler with detailed formatting
        fh = logging.FileHandler(self.log_dir / f"{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        # Console handler with progress bar friendly formatting
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(ch)

        return logger

    def _setup_memory_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.experiment_name}_memory")
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(self.log_dir / f"memory_{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - Memory Usage: %(message)s'))
        logger.addHandler(fh)
        
        return logger

    def log_progress(self, current: int, total: int, desc: str = ""):
        percentage = (current / total) * 100
        self.logger.info(f"{desc} - Progress: {percentage:.1f}% ({current}/{total})")

    def log_memory(self, usage_dict: dict):
        message = ", ".join(f"{k}: {v:.2f}GB" for k, v in usage_dict.items())
        self.memory_logger.info(message)

    def log_error(self, error: Exception, context: str = ""):
        self.logger.error(f"{context}: {str(error)}", exc_info=True)

    def log_experiment_start(self, config: dict):
        self.logger.info("Experiment started")
        self.logger.info(f"Configuration: {config}")

    def log_experiment_end(self, metrics: dict):
        self.logger.info("Experiment completed")
        self.logger.info(f"Final metrics: {metrics}")