import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import argparse
import json
import logging
import gc
from torch.utils.data import DataLoader

from experiments.checkpoint_utils import (
    save_checkpoint,
    load_checkpoints,
    get_last_checkpoint_batch
)

from experiments.experiment0_baseline.config import BaselineConfig, BaselineConfigFactory
from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything, read_corpus_json, read_pickle
from src.llm import LLM
from src.prompt_dataset import PromptDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

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
    def __init__(self, config: BaselineConfig, logger: Optional[ExperimentLogger] = None, output_dir: Optional[str] = None):
        self.config = config
        self.logger = logger or ExperimentLogger("baseline", str(Path(__file__).parent.parent.parent / "logs"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {"contriever": None, "bm25": None, "random": None}
        self.output_dir = Path(output_dir or config.output_dir) 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_memory_management()
        
        # Initialize LLM during setup
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM with proper configuration."""
        quantization_bits = 8 if self.config.use_8bit_quantization else None
        llm = LLM(
            model_id=self.config.llm_id,
            device=self.device,
            quantization_bits=quantization_bits,
            model_max_length=self.config.model_max_length
        )
        return llm

    def _setup_memory_management(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_utilization)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def _load_corpus(self, file_path: Union[str, Path]) -> List[Dict]:
        try:
            return read_corpus_json(str(file_path))
        except Exception as e:
            self.logger.error(f"Error loading corpus from {file_path}: {str(e)}")
            raise

    def merge_generation_checkpoints(self):
        """Merge all generation checkpoints after verifying completeness."""
        checkpoint_dir = self.output_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            return []
            
        # Get sorted checkpoint files
        checkpoint_files = sorted(
            checkpoint_dir.glob("generation_checkpoint_*.json"), 
            key=lambda p: int(p.stem.split('_')[-1]))
            
        actual_batch_count = len(checkpoint_files)
        
        # Check for incomplete checkpoints
        if actual_batch_count < self.config.total_batches:
            self.logger.experiment_logger.warning(
                f"Found {actual_batch_count}/{self.config.total_batches} checkpoints. "
                "Resuming from last checkpoint."
            )
            if checkpoint_files:
                last_checkpoint = checkpoint_files[-1]
                with open(last_checkpoint) as f:
                    return json.load(f)
            return []
            
        # Merge all checkpoints
        all_results = []
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file) as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
                
        return all_results

    def run_pipeline(self):
        """Suppress debug output and run the experiment pipeline."""
        with self.logger.experiment_logger.handlers[0]:  # Suppress non-essential output
            # Run phases sequentially
            self.run_retrieval_phase()
            self.run_generation_phase()
            metrics = self.run_evaluation_phase()
            return metrics

    def inject_noise_into_contexts(
    self,
    corpus: List[Dict],
    search_results: List[Tuple[List[int], List[float]]],
    noise_ratio: float,
    seed: int = 42
    ) -> List[Tuple[List[int], List[float]]]:
        try:
            if self.logger:
                self.logger.log_step_start("Injecting noise into contexts")

            random.seed(seed)
            noisy_results = []

            # Fix the tqdm usage here
            for doc_indices, scores in tqdm(search_results, desc="Injecting Noise"):
                num_noise_docs = max(1, int(len(doc_indices) * noise_ratio))
                noise_indices = random.sample(range(len(corpus)), num_noise_docs)
                doc_indices = doc_indices[:-num_noise_docs] + noise_indices
                scores = scores[:-num_noise_docs] + [0.0] * len(noise_indices)
                noisy_results.append((doc_indices, scores))

            if self.logger:
                self.logger.log_metric("num_noisy_results", len(noisy_results))
                self.logger.log_step_end("Injecting noise into contexts")

            return noisy_results

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error injecting noise into contexts")
            raise

    def run_retrieval_phase(self):
        # Load the training corpus
        train_corpus = self._load_corpus(self.config.train_dataset_path)

        # Use the train_corpus for retrieval 
        contriever_results = self._run_retrieval("contriever")
        bm25_results = self._run_retrieval("bm25")
        
        # Inject noise if required
        if self.config.noise_ratio > 0.0:
            self.logger.log_step_start("Injecting noise into retrieval contexts")
            contriever_results = self.inject_noise_into_contexts(
                train_corpus, contriever_results, self.config.noise_ratio
            )
            bm25_results = self.inject_noise_into_contexts(
                train_corpus, bm25_results, self.config.noise_ratio
            )
            self.logger.log_step_end("Injecting noise into retrieval contexts")

        # Save retrieval results and proceed
        self._validate_retrieval_results(contriever_results, "contriever")
        self._validate_retrieval_results(bm25_results, "bm25")
        return contriever_results, bm25_results

    def run_evaluation_phase(self):
        try:
            self.logger.log_step_start("Evaluation Phase")
            metrics = {}
            
            for retriever_type in ["contriever", "bm25", "random"]:
                self.logger.log_step_start(f"Evaluating {retriever_type}")
                
                # Ensure results exist before evaluating
                if retriever_type not in self.results or not self.results[retriever_type]:
                    self.logger.experiment_logger.warning(f"No results found for {retriever_type}")
                    metrics[retriever_type] = {
                        'accuracy': 0,
                        'total_examples': 0,
                        'correct_answers': 0
                    }
                    continue
                    
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
        """Run retrieval for the specified retriever type.
        
        Args:
            retriever_type: Type of retriever to use ('contriever', 'bm25', or 'random')
            
        Returns:
            List of tuples containing document IDs and scores
        """
        search_results_path = None
        
        if retriever_type == "contriever":
            search_results_path = self.config.contriever_results_path
        elif retriever_type == "bm25":
            search_results_path = self.config.bm25_results_path
        else:  # random
            search_results_path = self.config.random_results_path
                
        if not Path(search_results_path).exists():
            raise ValueError(f"Search results not found: {search_results_path}")
            
        return read_pickle(str(search_results_path))

    def _validate_retrieval_results(self, results: List, retriever_type: str):
        if not results:
            raise ValueError(f"No results found for {retriever_type}")
        self.logger.log_metric(f"{retriever_type}_retrieval_count", len(results))

    def _load_dataset(self, retriever_type: str) -> PromptDataset:
        try:
            corpus = read_corpus_json(str(self.config.corpus_path))
            search_results = self._run_retrieval(retriever_type)

            dataset_kwargs = {
                "corpus": corpus,
                "data_path": str(self.config.data_path),
                "tokenizer": self.llm.tokenizer,
                "max_tokenized_length": self.config.model_max_length - 2,
                "search_results": search_results,
                "num_documents_in_context": self.config.num_documents_in_context,
                "gold_position": self.config.gold_position,
                "get_documents_without_answer": self.config.get_documents_without_answer,
                "max_doc_length": self.config.max_doc_length,
                "logger": self.logger
            }

            return PromptDataset(**dataset_kwargs)

        except Exception as e:
            self.logger.error(f"Error loading dataset for {retriever_type}: {str(e)}")
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
        """Run generation phase with test dataset."""
        try:
            self.logger.log_step_start("Generation Phase")

            # Check for existing generation results or merge checkpoints
            if not self.results.get('generation'):
                self.results['generation'] = self.merge_generation_checkpoints()

            # If no previous results exist, run from scratch 
            if not self.results['generation']:
                # Use test_dataset_path instead of test_dataset
                test_corpus = read_corpus_json(str(self.config.test_dataset_path))
                dataset = self._load_dataset(test_corpus)
                dataloader = self._create_dataloader(dataset)
                results = self._run_generation(dataloader)

                # Save final phase checkpoint
                self._save_phase_checkpoint("generation", results)
                self.results['generation'] = results

            return self.results['generation']

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
        try:
            results = []
            prompts = batch['prompt']
            outputs = self.llm.generate(prompts, max_new_tokens=self.config.max_new_tokens)

            for idx, output in enumerate(outputs):
                try:
                    answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
                    start_idx = output.find(answer_string) + len(answer_string)
                    
                    # Convert tensors to native Python types
                    doc_indices = batch['document_indices'][idx].cpu().numpy().tolist()
                    prompt_tokens = int(batch['prompt_tokens_len'][idx])
                    
                    result = {
                        'query': batch['query'][idx],
                        'generated_answer': output[start_idx:].strip(),
                        'document_indices': doc_indices,
                        'prompt_tokens_len': prompt_tokens
                    }
                    results.append(result)
                    
                except Exception as e:
                    self.logger.log_error(e, f"Error processing batch item {idx}")
                    continue

            return results

        except Exception as e:
            self.logger.log_error(e, "Error processing batch")
            return []

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

    
    def _evaluate_results(self, results: List[Dict], retriever_type: str) -> Dict[str, float]:
        if not results:
            return {
                'accuracy': 0,
                'total_examples': 0, 
                'correct_answers': 0
            }
        
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

def main(experiment_args=None):
    try:
        # Set up args if not passed
        if isinstance(experiment_args, dict):
            parser = argparse.ArgumentParser()
            parser.add_argument('--retriever', type=str)
            parser.add_argument('--output_dir', type=str)
            namespace = parser.parse_args([])  # Avoid parsing args directly
            for k, v in experiment_args.items():
                setattr(namespace, k, v)
            experiment_args = namespace
        else:
            experiment_args = parse_arguments()

        # Load baseline config
        config = BaselineConfigFactory.get_config_for_retriever(experiment_args.retriever) 
        
        # Load global config and set GPU memory settings
        try:
            with open("experiments/experiments_config.json") as f:
                global_config = json.load(f)['global']
                config.gpu_memory_utilization = global_config.get('max_memory_usage', 0.8)
        except Exception as e:
            logging.error(f"Error loading global config: {str(e)}")
            raise
            
        # Initialize and run experiment
        try:
            experiment = BaselineExperiment(config, output_dir=experiment_args.output_dir)
            
            # Run phases sequentially
            experiment.run_retrieval_phase()
            experiment.run_generation_phase()
            metrics = experiment.run_evaluation_phase()
            
            return experiment.results, metrics
            
        except Exception as e:
            logging.error(f"Error in experiment execution: {str(e)}")
            raise
            
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
