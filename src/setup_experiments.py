import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

from src.experiment_logger import ExperimentLogger
from src.utils.file_utils import clear_memory

import torch
import google.generativeai as genai
import numpy as np
from transformers import AutoTokenizer

class ExperimentSetup:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ExperimentSetup")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def validate_data_files(self) -> bool:
        required_files = [
            'data/10k_train_dataset.json',
            'data/test_dataset.json',
            'data/contriever_search_results_at150.pkl',
            'data/bm25_test_search_results_at250.pkl',
            'data/10k_random_results_at60.pkl',
            'data/processed/corpus_with_contriever_at150.json'
        ]
        
        for file_path in tqdm(required_files, desc="Validating data files"):
            if not Path(file_path).exists():
                self.logger.error(f"Missing required file: {file_path}")
                return False
        return True

    def create_experiment_structure(self):
        experiments = [
            'experiment0_baseline',
            'experiment1_clustering',
            'experiment2_fusion',
            'experiment3_categories'
        ]
        
        for exp in tqdm(experiments, desc="Creating experiment structure"):
            exp_dir = self.base_dir / exp
            results_dir = exp_dir / "results"
            checkpoint_dir = exp_dir / "checkpoints"
            
            for directory in [exp_dir, results_dir, checkpoint_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            (exp_dir / "__init__.py").touch()
            
            required_files = ["config.py", "main.py", "utils.py"]
            for file in required_files:
                path = exp_dir / file
                if not path.exists():
                    path.touch()

    def create_logging_structure(self):
        log_dirs = [
            'logs/experiment_logs/baseline',
            'logs/experiment_logs/clustering',
            'logs/experiment_logs/fusion',
            'logs/experiment_logs/categories',
            'logs/system_logs',
            'logs/memory_logs'
        ]
        
        for log_dir in tqdm(log_dirs, desc="Creating log directories"):
            Path(log_dir).mkdir(parents=True, exist_ok=True)

    def validate_environment(self) -> bool:
        try:
            if torch.cuda.is_available():
                self.logger.info(f"Found GPU: {torch.cuda.get_device_name(0)}")
            
            api_key = os.getenv("GEMINI_TOKEN")
            if not api_key:
                self.logger.error("GEMINI_TOKEN not found in environment")
                return False
                
            return True
            
        except ImportError as e:
            self.logger.error(f"Missing required package: {str(e)}")
            return False

    def setup(self) -> bool:
        try:
            if not self.validate_environment():
                return False
                
            if not self.validate_data_files():
                return False
                
            self.create_experiment_structure()
            self.create_logging_structure()
            
            self.logger.info("Experiment setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            return False
        finally:
            clear_memory()

def main():
    setup = ExperimentSetup()
    success = setup.setup()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()