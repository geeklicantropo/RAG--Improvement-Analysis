import os
from pathlib import Path
from typing import List, Dict

def validate_data_paths() -> bool:
    """Validate required data files exist."""
    required_files = [
        'data/corpus.json',
        'data/10k_train_dataset.json',
        'data/test_dataset.json',
        'data/contriever_search_results_at150.pkl',
        'data/bm25_test_search_results_at250.pkl',
        'data/10k_random_results_at60.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    return True

def validate_experiment_setup() -> bool:
    """Validate experiment directories and configurations."""
    # Check experiment directories
    exp_dirs = [
        'experiments/experiment0_baseline',
        'experiments/experiment1_clustering', 
        'experiments/experiment2_fusion',
        'experiments/experiment3_categories'
    ]
    
    for dir_path in exp_dirs:
        if not os.path.exists(f"{dir_path}/config.py"):
            print(f"Missing config.py in {dir_path}")
            return False
            
        if not os.path.exists(f"{dir_path}/main.py"):
            print(f"Missing main.py in {dir_path}")
            return False
    
    return True

def main():
    """Run all validation checks."""
    if not validate_data_paths():
        raise RuntimeError("Data validation failed")
        
    if not validate_experiment_setup():
        raise RuntimeError("Experiment setup validation failed")
        
    print("All validation checks passed")

if __name__ == "__main__":
    main()