from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional
import jsonschema
import psutil
import os

class ConfigValidator:
    def __init__(self, config_path: str = "experiments/experiments_config.json"):
        self.config_path = Path(config_path)
        self.logger = self._setup_logger()
        self.config = self._load_config()

    def validate_config(self) -> bool:
        """Main validation method that checks all configuration aspects."""
        try:
            if not self.config:
                self.logger.error("No configuration loaded")
                return False

            validation_steps = [
                self._validate_global_config,
                self._validate_experiment_configs,
                self._validate_data_paths,
                self._validate_memory_settings,
                self._validate_cross_experiment_consistency
            ]

            for step in validation_steps:
                if not step():
                    return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
        
    def load_and_validate_config(self) -> Optional[Dict]:
        """Load and validate the configuration."""
        try:
            config = self._load_config()
            if config is None:
                raise ValueError("Failed to load configuration")

            if not self.validate_config():
                raise ValueError("Configuration validation failed")

            return config

        except Exception as e:
            self.logger.error(f"Error loading and validating configuration: {str(e)}")
            return None

    def _validate_global_config(self) -> bool:
        """Validate global configuration settings."""
        required_keys = [
            'random_seed', 
            'output_dir', 
            'log_dir', 
            'max_memory_usage',
            'save_every',
            'gpu_memory_threshold',
            'batch_size_reduction_factor'
        ]

        if 'global' not in self.config:
            self.logger.error("Missing global configuration section")
            return False

        global_config = self.config['global']
        for key in required_keys:
            if key not in global_config:
                self.logger.error(f"Missing required global config key: {key}")
                return False

        return True

    def _validate_experiment_configs(self) -> bool:
        """Validate configuration for each experiment type."""
        experiment_types = ['baseline', 'clustering', 'fusion', 'categories']
        
        for exp_type in experiment_types:
            if exp_type not in self.config['experiments']:
                if exp_type in ['baseline', 'clustering']:  # Baseline and clustering experiments are required
                    self.logger.error(f"Missing {exp_type} configuration section")
                    return False
                else:
                    continue

            exp_config = self.config['experiments'][exp_type]
            if not self._validate_experiment_section(exp_type, exp_config):
                return False

        return True

    def _validate_baseline_config(self, config: Dict) -> bool:
        required_baseline_keys = ['retrieval_config']
        for key in required_baseline_keys:
            if key not in config:
                self.logger.error(f"Missing required baseline config key: {key}")
                return False
        return True

    def _validate_clustering_config(self, config: Dict) -> bool:
        required_clustering_keys = ['clustering_config']
        for key in required_clustering_keys:
            if key not in config:
                self.logger.error(f"Missing required clustering config key: {key}")
                return False
        return True

    def _validate_fusion_config(self, config: Dict) -> bool:
        required_fusion_keys = ['fusion_config']
        for key in required_fusion_keys:
            if key not in config:
                self.logger.error(f"Missing required fusion config key: {key}")
                return False
        return True

    def _validate_categories_config(self, config: Dict) -> bool:
        required_categories_keys = ['category_config']
        for key in required_categories_keys:
            if key not in config:
                self.logger.error(f"Missing required categories config key: {key}")
                return False
        return True

    def _validate_experiment_section(self, exp_type: str, config: Dict) -> bool:
        """Validate individual experiment configuration section."""
        required_base_keys = ['experiment_name', 'llm_config', 'data_paths']
        
        for key in required_base_keys:
            if key not in config:
                self.logger.error(f"Missing {key} in {exp_type} config")
                return False

        # Validate LLM config
        if not self._validate_llm_config(config['llm_config']):
            return False

        # Validate data paths
        if not self._validate_data_paths():
            return False

        # Experiment-specific validations
        validation_methods = {
            'baseline': self._validate_baseline_config,
            'clustering': self._validate_clustering_config,
            'fusion': self._validate_fusion_config,
            'categories': self._validate_categories_config
        }

        if exp_type in validation_methods:
            if not validation_methods[exp_type](config):
                return False

        return True

    def _validate_llm_config(self, config: Dict) -> bool:
        """Validate LLM configuration."""
        required_keys = [
            'model_id',
            'model_max_length',
            'max_new_tokens',
            'use_8bit',
            'batch_size',
            'min_batch_size',
            'max_batch_size'
        ]

        for key in required_keys:
            if key not in config:
                self.logger.error(f"Missing {key} in llm_config")
                return False

        return True

    def _validate_data_paths(self) -> bool:
        """Validate existence of required data files."""
        required_paths = [
            'data/processed/corpus_with_contriever_at150.json',
            'data/10k_train_dataset.json',
            'data/test_dataset.json',
            'data/contriever_search_results_at150.pkl',
            'data/bm25_test_search_results_at250.pkl',
            'data/10k_random_results_at60.pkl'
        ]

        for path in required_paths:
            if not Path(path).exists():
                self.logger.error(f"Required file not found: {path}")
                return False

        return True

    def _validate_memory_settings(self) -> bool:
        """Validate memory-related settings."""
        try:
            max_memory = self.config['global'].get('max_memory_usage', 0.8)
            if not 0 < max_memory <= 1:
                self.logger.error(f"Invalid max_memory_usage value: {max_memory}")
                return False

            if psutil.virtual_memory().available / psutil.virtual_memory().total < 0.2:
                self.logger.warning("Low system memory available")

            return True
            
        except Exception as e:
            self.logger.error(f"Error validating memory settings: {str(e)}")
            return False

    def _validate_cross_experiment_consistency(self) -> bool:
        """Validate consistency across experiment configurations."""
        try:
            model_ids = set()
            batch_sizes = set()
            
            for exp_type in ['baseline', 'clustering', 'fusion', 'categories']:
                if exp_type in self.config['experiments']:
                    llm_config = self.config['experiments'][exp_type].get('llm_config', {})
                    model_ids.add(llm_config.get('model_id'))
                    batch_sizes.add(llm_config.get('batch_size'))

            # Check if all experiments use the same model
            if len(model_ids) > 1:
                self.logger.warning("Different model IDs used across experiments")
                
            # Check for consistent batch sizes
            if len(batch_sizes) > 1:
                self.logger.warning("Different batch sizes used across experiments")

            return True

        except Exception as e:
            self.logger.error(f"Error validating cross-experiment consistency: {str(e)}")
            return False

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("ConfigValidator")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        log_dir = Path("logs/config_validation")
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "config_validation.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def _load_config(self) -> Optional[Dict]:
        """Load configuration from file."""
        try:
            if not self.config_path.exists():
                self.logger.error(f"Configuration file not found: {self.config_path}")
                return None

            with open(self.config_path) as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return None

def validate_experiment_setup() -> bool:
    validator = ConfigValidator()
    if validator.validate_config():
        print("Configuration is valid")
    else:
        print("Configuration validation failed")
    config = validator.load_and_validate_config()
    
    if config is None:
        return False
        
    exp_dirs = [
        'experiments/experiment0_baseline',
        'experiments/experiment1_clustering',
        'experiments/experiment2_fusion',
        'experiments/experiment3_categories'
    ]
    
    for dir_path in exp_dirs:
        for file in ['config.py', 'main.py']:
            if not os.path.exists(f"{dir_path}/{file}"):
                validator.logger.error(f"Missing {file} in {dir_path}")
                return False
    
    validator.logger.info("Experiment setup validation successful")
    return True


def main():
    validator = ConfigValidator()
    if validator.validate_config():
        print("Configuration is valid")
    else:
        print("Configuration validation failed")
        
    config = validator.load_and_validate_config()
    if config is None or not validate_experiment_setup():
        raise RuntimeError("Validation failed")
    validator.logger.info("All validation checks passed")

if __name__ == "__main__":
    main()