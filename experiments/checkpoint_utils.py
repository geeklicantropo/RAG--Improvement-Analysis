import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import torch
from tqdm import tqdm
import logging


def _make_serializable(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items() if not k.startswith('_')}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: _make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return str(obj) if not isinstance(obj, (int, float, str, bool, type(None))) else obj

def save_checkpoint(results: List[Dict], checkpoint_dir: Path, batch_idx: int) -> None:
    """Save checkpoint with proper error handling and validation."""
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"checkpoint_batch_{batch_idx}_{timestamp}.json"

        # Validate results before saving
        if not results:
            logging.error("Attempting to save empty results")
            return
            
        # Clean results for serialization
        clean_results = []
        for result in results:
            try:
                clean_result = {
                    "query": result.get("query", ""),
                    "example_id": result.get("example_id", ""),
                    "generated_answer": result.get("generated_answer", ""),
                    "gold_answer": result.get("gold_answer", ""),
                    "llm_evaluation": result.get("llm_evaluation", {}),
                    "context": result.get("context", []),
                    "clusters": result.get("clusters", {})
                }
                clean_results.append(clean_result)
            except Exception as e:
                logging.error(f"Error cleaning result: {str(e)}")
                continue

        with open(checkpoint_file, 'w') as f:
            json.dump(clean_results, f, indent=2)

    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
        raise

def load_checkpoints(checkpoint_dir: Path) -> List[Dict]:
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_batch_*.json"),
        key=lambda p: int(p.stem.split('_')[3])
    )
    
    all_results = []
    seen_example_ids = set()
    
    for cp in tqdm(checkpoints, desc="Loading checkpoints"):
        with open(cp) as f:
            batch_results = json.load(f)
            for result in batch_results:
                if result['example_id'] not in seen_example_ids:
                    all_results.append(result)
                    seen_example_ids.add(result['example_id'])
    
    return all_results

def merge_checkpoint_results(all_results: List[Dict]) -> List[Dict]:
    return sorted(all_results, key=lambda x: x['example_id'])

def get_last_checkpoint_batch(checkpoint_dir: Path) -> int:
    if not checkpoint_dir.exists():
        return 0
        
    checkpoints = list(checkpoint_dir.glob("results_checkpoint_batch_*.json"))
    if not checkpoints:
        return 0
        
    last_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[3]))
    return int(last_checkpoint.stem.split('_')[3])

def find_latest_results_and_checkpoints(mode_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find latest results file and checkpoint for a given mode directory."""
    result_files = list(mode_dir.glob("results_*_*.json"))
    latest_result = max(result_files, key=lambda x: x.stat().st_mtime) if result_files else None
    
    checkpoint_dir = mode_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_batch_*.json"))
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime) if checkpoint_files else None
    else:
        latest_checkpoint = None
    
    return latest_result, latest_checkpoint

def load_experiment_state(base_dir: Path, experiment_type: str, modes: List[str]) -> Dict[str, Dict]:
    """Load experiment state for all modes."""
    experiment_dir = base_dir / experiment_type / "results"
    state = {}
    
    for mode in tqdm(modes, desc="Loading experiment state"):
        mode_dir = experiment_dir / mode
        if not mode_dir.exists():
            continue
            
        latest_result, latest_checkpoint = find_latest_results_and_checkpoints(mode_dir)
        
        if latest_result:
            with open(latest_result) as f:
                state[mode] = {
                    "completed": True,
                    "results": json.load(f),
                    "checkpoint_batch": None
                }
        elif latest_checkpoint:
            with open(latest_checkpoint) as f:
                checkpoint_data = json.load(f)
                batch_num = int(latest_checkpoint.stem.split('_')[-2])
                state[mode] = {
                    "completed": False,
                    "results": checkpoint_data,
                    "checkpoint_batch": batch_num
                }
    
    return state

def get_next_batch_number(checkpoint_dir: Path) -> int:
    """Get next batch number based on existing checkpoints."""
    if not checkpoint_dir.exists():
        return 0
        
    checkpoints = list(checkpoint_dir.glob("checkpoint_batch_*.json"))
    if not checkpoints:
        return 0
        
    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[-2]))
    return int(latest.stem.split('_')[-2]) + 1

def save_final_results(results: List[Dict], mode_dir: Path) -> None:
    """Save final results for a mode."""
    if results:
        results_path = mode_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with tqdm(total=1, desc="Saving final results") as pbar:
            with open(results_path, 'w') as f:
                json.dump(_make_serializable(results), f, indent=2)
            pbar.update(1)