import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import torch
from tqdm import tqdm

def _make_serializable(obj: Any, depth: int = 0) -> Any:
    if depth > 10:  #Prevent infinite recursion
        return str(obj)
        
    if torch.is_tensor(obj):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v, depth + 1) for k, v in obj.items() if not k.startswith('_')}
    elif isinstance(obj, list):
        return [_make_serializable(item, depth + 1) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: _make_serializable(v, depth + 1) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return str(obj) if not isinstance(obj, (int, float, str, bool, type(None))) else obj

def save_checkpoint(results: List[Dict], batch_idx: int, output_dir: Path, experiment_type: str = "baseline") -> None:
    """
    Save checkpoint results for an experiment.

    Args:
        results (List[Dict]): List of results to save in the checkpoint.
        batch_idx (int): The batch index for checkpointing.
        output_dir (Path): The directory to save checkpoints.
        experiment_type (str): The type of experiment ('baseline' or 'clustering').
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_{experiment_type}_batch_{batch_idx}.json"

    with open(checkpoint_path, "w") as f:
        json.dump(results, f, indent=2)

def load_checkpoints(checkpoint_dir: Path, experiment_type: str = "baseline") -> List[Dict]:
    """
    Load all checkpoint results for an experiment.

    Args:
        checkpoint_dir (Path): The directory containing checkpoints.
        experiment_type (str): The type of experiment ('baseline' or 'clustering').

    Returns:
        List[Dict]: Combined results from all checkpoints.
    """
    if not checkpoint_dir.exists():
        return []

    checkpoints = sorted(
        checkpoint_dir.glob(f"checkpoint_{experiment_type}_batch_*.json"),
        key=lambda p: int(p.stem.split('_')[-1])
    )

    all_results = []
    seen_example_ids = set()

    for cp in checkpoints:
        with open(cp) as f:
            batch_results = json.load(f)
            for result in batch_results:
                if result["example_id"] not in seen_example_ids:
                    all_results.append(result)
                    seen_example_ids.add(result["example_id"])

    return all_results

def get_last_checkpoint_batch(checkpoint_dir: Path) -> Optional[Dict]:
    if not checkpoint_dir.exists():
        return None
        
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_batch_*.json"),
        key=lambda p: int(p.stem.split('_')[2])
    )
    
    if not checkpoints:
        return None
        
    with open(checkpoints[-1]) as f:
        return json.load(f)

def merge_checkpoint_results(results: List[Dict]) -> List[Dict]:
    """
    Merge checkpoint results, ensuring no duplicates.

    Args:
        results (List[Dict]): List of results to merge.

    Returns:
        List[Dict]: Merged and sorted results.
    """
    return sorted(results, key=lambda x: x["example_id"])

def save_final_results(results: List[Dict], output_dir: Path, experiment_type: str = "baseline") -> None:
    """
    Save final results for an experiment.

    Args:
        results (List[Dict]): List of final results.
        output_dir (Path): The directory to save results.
        experiment_type (str): The type of experiment ('baseline' or 'clustering').
    """
    final_results_path = output_dir / f"final_results_{experiment_type}.json"
    with open(final_results_path, "w") as f:
        json.dump(results, f, indent=2)

def save_cluster_specific_results(
    cluster_results: Dict[int, List[Dict]],
    output_dir: Path,
    mode: str,
    batch_idx: int
) -> None:
    """
    Save clustering-specific results.

    Args:
        cluster_results (Dict[int, List[Dict]]): Cluster-specific results.
        output_dir (Path): The output directory.
        mode (str): The experiment mode ('gold_only', 'gold_random', etc.).
        batch_idx (int): Batch index for checkpointing.
    """
    cluster_dir = output_dir / f"cluster_results/{mode}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    cluster_checkpoint_path = cluster_dir / f"cluster_checkpoint_batch_{batch_idx}.json"
    with open(cluster_checkpoint_path, "w") as f:
        json.dump(cluster_results, f, indent=2)