import os
import sys
from pathlib import Path
import logging
from typing import List, Dict
from datetime import datetime

class ExperimentSetup:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ExperimentSetup")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs/setup")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(f"{log_dir}/setup_{self.timestamp}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
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
        
        for file_path in required_files:
            if not Path(file_path).exists():
                self.logger.error(f"Missing required file: {file_path}")
                return False
        
        self.logger.info("All required data files present")
        return True

    def create_experiment_structure(self):
        experiments = [
            'experiment0_baseline',
            'experiment1_clustering',
            'experiment2_fusion',
            'experiment3_categories'
        ]
        
        for exp in experiments:
            exp_dir = self.base_dir / exp
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create results directory
            (exp_dir / "results").mkdir(exist_ok=True)
            
            # Create __init__.py
            (exp_dir / "__init__.py").touch()
            
            # Check/Create required files if they don't exist
            self._ensure_file_exists(exp_dir / "config.py")
            self._ensure_file_exists(exp_dir / "main.py")
            self._ensure_file_exists(exp_dir / "utils.py")
            
            self.logger.info(f"Created/Verified structure for {exp}")

    def create_logging_structure(self):
        log_dirs = [
            'logs/experiment_logs/baseline',
            'logs/experiment_logs/clustering',
            'logs/experiment_logs/fusion',
            'logs/experiment_logs/categories',
            'logs/system_logs'
        ]
        
        for log_dir in log_dirs:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created log directory: {log_dir}")

    def _ensure_file_exists(self, file_path: Path):
        if not file_path.exists():
            file_path.touch()
            self.logger.info(f"Created file: {file_path}")

    def setup(self):
        self.logger.info("Starting experiment setup")
        
        if not self.validate_data_files():
            self.logger.error("Data validation failed")
            return False
            
        try:
            self.create_experiment_structure()
            self.create_logging_structure()
            self.logger.info("Experiment setup completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            return False

def main():
    setup = ExperimentSetup()
    success = setup.setup()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()