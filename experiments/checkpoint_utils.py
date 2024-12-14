import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import torch
from tqdm import tqdm

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

def save_checkpoint(results: List[Dict], batch_idx: int, output_dir: Path) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"checkpoint_batch_{batch_idx}_{timestamp}.json"
    
    with tqdm(total=1, desc=f"Saving checkpoint {batch_idx}") as pbar:
        serializable_results = _make_serializable(results)
        with open(checkpoint_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        pbar.update(1)

def load_checkpoints(checkpoint_dir: Path) -> List[Dict]:
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = sorted(
        checkpoint_dir.glob("results_checkpoint_batch_*.json"),
        key=lambda p: int(p.stem.split('_')[3])
    )
    
    all_results = []
    seen_example_ids = set()
    
    for cp in checkpoints:
        with open(cp) as f:
            batch_results = json.load(f)
            for result in batch_results:
                if result['example_id'] not in seen_example_ids:
                    all_results.append(result)
                    seen_example_ids.add(result['example_id'])
    
    return all_results

def merge_checkpoint_results(all_results: List[Dict]) -> List[Dict]:
    # Sort by example_id to ensure consistent ordering
    return sorted(all_results, key=lambda x: x['example_id'])

def get_last_checkpoint_batch(checkpoint_dir: Path) -> int:
    if not checkpoint_dir.exists():
        return 0
        
    checkpoints = list(checkpoint_dir.glob("results_checkpoint_batch_*.json"))
    if not checkpoints:
        return 0
        
    last_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[3]))
    return int(last_checkpoint.stem.split('_')[3])