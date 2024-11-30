import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import argparse
import warnings
import logging
import json
import gc
import numpy as np
from torch.utils.data import DataLoader

from experiments.checkpoint_utils import (
    save_checkpoint,
    load_checkpoints, 
    get_last_checkpoint_batch,
    merge_checkpoint_results
)

from experiments.experiment0_baseline.config import BaselineConfig, BaselineConfigFactory
from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything, read_corpus_json, read_pickle
from src.llm import LLM
from src.prompt_dataset import PromptDataset
import matplotlib.pyplot as plt

class ExperimentBase:
    """Base class for all experiments with checkpoint handling"""
    
    def _get_phase_checkpoint(self, phase: str) -> Optional[Dict]:
        """Check for phase-specific checkpoint"""
        checkpoint_dir = self.output_dir / "checkpoints" / phase
        if not checkpoint_dir.exists():
            return None
            
        checkpoints = list(checkpoint_dir.glob("*.json"))
        if not checkpoints:
            return None
            
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            return json.load(f)

    def _save_phase_checkpoint(self, phase: str, data: Dict):
        """Save phase-specific checkpoint"""
        checkpoint_dir = self.output_dir / "checkpoints" / phase
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"{phase}_checkpoint_{timestamp}.json"
        
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

class BaselineExperiment(ExperimentBase):
    def __init__(self, config: BaselineConfig, logger: Optional[ExperimentLogger] = None):
        self.config = config
        self.logger = logger or ExperimentLogger("baseline", str(Path(__file__).parent.parent.parent / "logs"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {"contriever": None, "bm25": None, "random": None}
        self.output_dir = Path(config.output_dir)  # Add output_dir attribute
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_memory_management()

    def _setup_memory_management(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_utilization)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def run_retrieval_phase(self):
        try:
            self.logger.log_step_start("Retrieval Phase")
            
            # Contriever Retrieval
            self.logger.log_step_start("Contriever Retrieval")
            contriever_results = self._run_retrieval("contriever")
            self._validate_retrieval_results(contriever_results, "contriever")
            self._cleanup_memory()
            
            # BM25 Retrieval
            self.logger.log_step_start("BM25 Retrieval")
            bm25_results = self._run_retrieval("bm25")
            self._validate_retrieval_results(bm25_results, "bm25")
            self._cleanup_memory()
            
            self.logger.log_step_end("Retrieval Phase")
            return contriever_results, bm25_results
            
        except Exception as e:
            self.logger.log_error(e, "Error in retrieval phase")
            raise

    def run_evaluation_phase(self):
        try:
            self.logger.log_step_start("Evaluation Phase")
            metrics = {}
            
            for retriever_type in ["contriever", "bm25", "random"]:
                self.logger.log_step_start(f"Evaluating {retriever_type}")
                metrics[retriever_type] = self._evaluate_results(
                    self.results[retriever_type], 
                    retriever_type
                )
                self._cleanup_memory()
                self.logger.log_step_end(f"Evaluating {retriever_type}")
            
            self._plot_baseline_comparisons(metrics)
            self.logger.log_step_end("Evaluation Phase")
            return metrics
            
        except Exception as e:
            self.logger.log_error(e, "Error in evaluation phase")
            raise

    def _initialize_llm(self):
        """
        Initialize the LLM, handling device and quantization settings.
        """
        quantization_bits = 8 if self.config.use_8bit_quantization else None
        return LLM(
            model_id=self.config.llm_id,
            device=self.device,
            quantization_bits=quantization_bits,
            model_max_length=self.config.model_max_length
        )

    def _run_retrieval(self, retriever_type: str) -> List[Tuple[List[int], List[float]]]:
        if retriever_type == "contriever":
            search_results_path = self.config.contriever_results_path
        elif retriever_type == "bm25":
            search_results_path = self.config.bm25_results_path
        else:  # random
            search_results_path = self.config.random_results_path
            
        return read_pickle(str(search_results_path))

    def _validate_retrieval_results(self, results: List, retriever_type: str):
        if not results:
            raise ValueError(f"No results found for {retriever_type}")
        self.logger.log_metric(f"{retriever_type}_retrieval_count", len(results))

    def _load_dataset(self, retriever_type: str) -> PromptDataset:
        """
        Loads the dataset for the given retriever type with all necessary configurations.

        Args:
            retriever_type (str): The type of retriever (e.g., "BM25", "Contriever").

        Returns:
            PromptDataset: An instance of the dataset prepared for the experiment.
        """
        try:
            # Read corpus and run retrieval for search results
            corpus = read_corpus_json(str(self.config.corpus_path))
            search_results = self._run_retrieval(retriever_type)

            # Prepare dataset arguments
            dataset_kwargs = {
                "corpus": corpus,
                "data_path": str(self.config.data_path),
                "tokenizer": self.llm.tokenizer,
                "max_tokenized_length": self.config.model_max_length - 2,  # Adjust for special tokens
                "search_results": search_results,
                "num_documents_in_context": self.config.num_documents_in_context,
                "gold_position": self.config.gold_position,
                "get_documents_without_answer": self.config.get_documents_without_answer,
                "max_doc_length": self.config.max_doc_length,  # Pass document truncation length
            }

            # Ensure the PromptDataset constructor accepts all arguments
            return PromptDataset(**dataset_kwargs)

        except Exception as e:
            self.logger.log_error(e, f"Error loading dataset for {retriever_type}")
            raise

    def _create_dataloader(self, dataset: PromptDataset) -> DataLoader:
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.config.adjust_batch_size(current_memory)
            
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def run_generation_phase(self):
        try:
            self.logger.log_step_start("Generation Phase")
            
            # Check for existing generation phase checkpoint
            checkpoint = self._get_phase_checkpoint("generation")
            if checkpoint:
                self.logger.experiment_logger.info("Resuming from generation checkpoint")
                self.results = checkpoint
                return merge_checkpoint_results(self.results)
            
            # Check for batch-level checkpoints
            checkpoint_dir = self.output_dir / "checkpoints" / "batches"
            if checkpoint_dir.exists():
                existing_results = load_checkpoints(checkpoint_dir)
                if existing_results:
                    self.logger.experiment_logger.info(f"Resuming from batch checkpoint {len(existing_results)}")
                    self.results = merge_checkpoint_results(existing_results)
                    return self.results
            
            # No checkpoints found, run full generation
            self.llm = self._initialize_llm()
            dataset = self._load_dataset()
            dataloader = self._create_dataloader(dataset)
            results = self._run_generation(dataloader)
            
            # Save final phase checkpoint
            self._save_phase_checkpoint("generation", results)
            
            return merge_checkpoint_results(results)
            
        except Exception as e:
            self.logger.log_error(e, "Error in generation phase")
            raise

    def _run_generation(self, dataloader: DataLoader) -> List[Dict]:
        results = []
        last_checkpoint = get_last_checkpoint_batch(Path(self.config.output_dir) / "checkpoints")
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch_idx <= last_checkpoint:
                continue
                
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            if (batch_idx + 1) % self.config.save_every == 0:
                save_checkpoint(results, batch_idx + 1, Path(self.config.output_dir))
                results = []  # Clear processed results after checkpoint
                
        if results:  # Save any remaining results
            save_checkpoint(results, len(dataloader), Path(self.config.output_dir))
            
        return load_checkpoints(Path(self.config.output_dir) / "checkpoints")

    def _process_batch(self, batch: Dict) -> List[Dict]:
        """Process batches with improved error handling and checkpointing."""
        results = []
        for batch_idx, batch in enumerate(tqdm(DataLoader, desc="Generating")):
            try:
                batch_results = self._process_single_batch(batch)
                results.extend(batch_results)
                
                if (batch_idx + 1) % self.config.save_every == 0:
                    self._save_checkpoint(results, batch_idx + 1)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._handle_oom_error(batch_idx)
                    continue
                raise
            except Exception as e:
                self.logger.log_error(e, f"Error processing batch {batch_idx}")
                continue
                
        return results

    def _process_single_batch(self, batch: Dict) -> List[Dict]:
        prompts = batch['prompt']
        outputs = self.llm.generate(prompts, max_new_tokens=self.config.max_new_tokens)
        results = self._process_batch_outputs(batch, outputs)
        self._cleanup_batch_tensors(batch)
        return results

    def _process_batch_outputs(self, batch: Dict[str, Any], outputs: List[str]) -> List[Dict[str, Any]]:
        processed_results = []
        answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
        
        for idx, output in enumerate(outputs):
            try:
                start_idx = output.find(answer_string) + len(answer_string)
                
                # Convert tensor values to native Python types
                document_indices = [int(i) for i in batch['document_indices'][idx].cpu().numpy()]
                prompt_tokens_len = int(batch['prompt_tokens_len'][idx])
                
                result = {
                    'query': batch['query'][idx],
                    'generated_answer': output[start_idx:].strip(),
                    'document_indices': document_indices,
                    'prompt_tokens_len': prompt_tokens_len
                }
                processed_results.append(result)
                
            except Exception as e:
                self.logger.log_error(e, f"Error processing batch item {idx}")
                continue

        return processed_results

    def _cleanup_batch_tensors(self, batch: Dict):
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                del v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _handle_oom_error(self, batch_idx: int):
        self._cleanup_memory()
        self.config.batch_size = max(1, self.config.batch_size // 2)
        self.logger.experiment_logger.warning(f"OOM at batch {batch_idx}. Reduced batch size to {self.config.batch_size}")

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _evaluate_results(self, results: List[Dict], retriever_type: str) -> Dict:
        metrics = self._compute_metrics(results)
        self._save_results(results, metrics, retriever_type)
        return metrics

    def _compute_metrics(self, results: List[Dict]) -> Dict:
        metrics = {
            'total_examples': len(results),
            'correct_answers': sum(1 for r in results if r.get('ans_match_after_norm', False)),
            'avg_context_length': sum(len(r['document_indices']) for r in results) / len(results)
        }
        metrics['accuracy'] = metrics['correct_answers'] / metrics['total_examples']
        return metrics

    def _save_results(self, results: List[Dict], metrics: Dict, retriever_type: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / f"{retriever_type}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.experiment_logger.info(f"Saved {retriever_type} results to {output_dir}")

    def _save_checkpoint(self, results: List[Dict], batch_idx: int):
        """Save results checkpoint with proper error handling."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"results_checkpoint_batch_{batch_idx}_{timestamp}.json"
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.experiment_logger.info(f"Saved checkpoint at batch {batch_idx}")
            
        except Exception as e:
            self.logger.log_error(e, f"Error saving checkpoint at batch {batch_idx}")
            raise  # Re-raise the error to stop execution

    def _make_json_serializable(self, data):
        """
        Recursively convert non-serializable objects in the data to JSON-serializable types.
        
        Args:
            data: Any Python data structure.

        Returns:
            A JSON-serializable version of the data.
        """
        if isinstance(data, torch.Tensor):
            return data.tolist()  # Convert Tensor to list
        elif isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._make_json_serializable(item) for item in data)
        else:
            return data

    def _plot_baseline_comparisons(self, metrics: Dict):
        try:
            accuracies = [metrics[r]["accuracy"] for r in ["contriever", "bm25", "random"]]
            plt.figure(figsize=(10, 6))
            plt.bar(["Contriever", "BM25", "Random"], accuracies)
            plt.title("Retriever Accuracy Comparison")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(self.output_dir / "accuracy_comparison.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.log_error(e, "Error plotting baseline comparisons")

def main(args=None):
    try:
        args = parse_arguments() if args is None else argparse.Namespace(**args)
        config = BaselineConfigFactory.get_config_for_retriever(args.retriever)
        
        experiment = BaselineExperiment(config)
        
        # Run phases sequentially
        experiment.run_retrieval_phase()
        experiment.run_generation_phase()
        metrics = experiment.run_evaluation_phase()
        
        return experiment.results, metrics
        
    except Exception as e:
        logging.error(f"Error in baseline experiment: {str(e)}", exc_info=True)
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        '--retriever',
        type=str,
        choices=['contriever', 'bm25', 'random'],
        required=True,
        help="Type of retriever to use"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="experiments/experiment0_baseline/results",
        help="Output directory"
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()