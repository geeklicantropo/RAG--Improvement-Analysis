import logging
from pathlib import Path
import sys

def setup_experiment_logging(output_dir: Path, experiment_name: str) -> logging.Logger:
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_dir / f'{experiment_name}.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger