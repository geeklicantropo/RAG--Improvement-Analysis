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

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything
from src.llm import LLM
from src.prompt_dataset import PromptDataset
from .config import BaselineConfig, BaselineConfigFactory

class BaselineExperiment:
    def __init__(
        self,
        config: BaselineConfig,
        experiment_name: str,
        retriever_type: str = None,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.retriever_type = retriever_type or ('bm25' if config.use_bm25 else 'contriever')
        self.logger = logger or ExperimentLogger(
            experiment_name=experiment_name,
            base_log_dir=str(Path(project_root) / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup(self):
        """Set up experiment components."""
        try:
            self.logger.log_step_start("Setup")
            
            # Initialize LLM
            self.llm = LLM(
                self.config.llm_id,
                self.device,
                quantization_bits=4,
                model_max_length=self.config.model_max_length
            )
            
            # Load dataset
            self.dataset = self._load_dataset()
            
            self.logger.log_step_end("Setup")
            
        except Exception as e:
            self.logger.log_error(e, "Error in experiment setup")
            raise
            
    def _load_dataset(self) -> PromptDataset:
        """Load and prepare dataset."""
        try:
            # Load corpus
            from src.utils import read_corpus_json
            corpus = read_corpus_json(str(self.config.corpus_path))
            
            # Load search results
            from src.utils import read_pickle
            search_results = read_pickle(str(self.config.search_results_path))
            
            return PromptDataset(
                corpus=corpus,
                data_path=str(self.config.train_dataset_path),
                tokenizer=self.llm.tokenizer,
                max_tokenized_length=self.config.model_max_length - 2,
                search_results=search_results,
                num_documents_in_context=self.config.num_documents_in_context,
                gold_position=self.config.gold_position,
                get_documents_without_answer=self.config.get_documents_without_answer
            )
        except Exception as e:
            self.logger.log_error(e, "Error loading dataset")
            raise


    def run(self):
        """Execute the baseline experiment workflow with enhanced memory management."""
        try:
            self.logger.log_step_start("Experiment execution")
            
            if torch.cuda.is_available():
                gpu = torch.cuda.current_device()
                initial_memory = torch.cuda.memory_allocated(gpu) / torch.cuda.max_memory_allocated(gpu)
                self.config.adjust_batch_size(initial_memory)

            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )

            all_metrics = []
            checkpoint_counter = 0
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated(gpu) / torch.cuda.max_memory_allocated(gpu)
                    self.config.adjust_batch_size(current_memory)
                    
                    if current_memory > 0.85:  # Memory threshold
                        torch.cuda.empty_cache()
                        gc.collect()

                try:
                    # Process in smaller sub-batches if needed
                    sub_batch_size = min(len(batch['prompt']), self.config.batch_size)
                    results = []
                    
                    for i in range(0, len(batch['prompt']), sub_batch_size):
                        end_idx = min(i + sub_batch_size, len(batch['prompt']))
                        sub_batch = {k: v[i:end_idx] for k, v in batch.items()}
                        
                        # Generate answers
                        generated_outputs = self.llm.generate(
                            sub_batch['prompt'],
                            max_new_tokens=self.config.max_new_tokens
                        )
                        
                        # Process outputs
                        batch_results = self._process_batch_outputs(sub_batch, generated_outputs)
                        results.extend(batch_results)
                        
                        # Clear memory
                        del generated_outputs
                        torch.cuda.empty_cache()

                    # Save checkpoint
                    if self.config.save_every and (batch_idx + 1) % self.config.save_every == 0:
                        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        
                        checkpoint_counter += 1
                        checkpoint_path = checkpoint_dir / f"checkpoint_{checkpoint_counter}.json"
                        
                        # Convert tensors and save
                        serializable_results = [{k: v.tolist() if torch.is_tensor(v) else v for k, v in r.items()} for r in results]
                        with open(checkpoint_path, 'w') as f:
                            json.dump(serializable_results, f, indent=2)
                        
                        # Compute metrics for this batch
                        batch_metrics = self._compute_metrics(results)
                        all_metrics.append(batch_metrics)
                        
                        # Clear results after saving
                        results = []
                        gc.collect()
                    
                except Exception as e:
                    self.logger.log_error(e, f"Error processing batch {batch_idx}")
                    continue

            # Compute final metrics
            final_metrics = self._aggregate_metrics(all_metrics)
            
            # Save final results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir) / f"run_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "metrics.json", 'w') as f:
                json.dump(final_metrics, f, indent=2)

            self.logger.log_step_end("Experiment execution")
            return None, final_metrics
            
        except Exception as e:
            self.logger.log_error(e, "Error during experiment execution")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _run_generation(self, dataloader):
        results = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
            # Generate answers
            prompts = batch['prompt']
            generated_outputs = self.llm.generate(
                prompts,
                max_new_tokens=self.config.max_new_tokens
            )
            
            # Process outputs
            batch_results = self._process_batch_outputs(batch, generated_outputs)
            results.extend(batch_results)
            
            # Save checkpoint if configured
            if self.config.save_every and (batch_idx + 1) % self.config.save_every == 0:
                self._save_checkpoint(results, batch_idx + 1)
                
        return results

    def _process_batch_outputs(
    self,
    batch: Dict[str, Any],
    outputs: List[str]
    ) -> List[Dict[str, Any]]:
        """Process generation outputs for a batch."""
        processed_results = []
        answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
        
        # Ensure batch and outputs match in length
        min_len = min(len(outputs), len(batch['document_indices']))
        
        for idx in range(min_len):
            try:
                start_idx = outputs[idx].find(answer_string) + len(answer_string)
                answer = outputs[idx][start_idx:].strip()
                
                # Convert tensors to regular Python types
                doc_indices = batch['document_indices'][idx]
                if torch.is_tensor(doc_indices):
                    doc_indices = doc_indices.cpu().numpy().tolist()
                elif isinstance(doc_indices, np.ndarray):
                    doc_indices = doc_indices.tolist()
                    
                result = {
                    'example_id': batch['example_id'][idx] if 'example_id' in batch else idx,
                    'query': batch['query'][idx],
                    'generated_answer': answer,
                    'document_indices': doc_indices,
                    'prompt_tokens_len': int(batch['prompt_tokens_len'][idx].item()) if torch.is_tensor(batch['prompt_tokens_len'][idx]) else batch['prompt_tokens_len'][idx]
                }
                processed_results.append(result)
                
            except Exception as e:
                self.logger.log_error(e, f"Error processing batch item {idx}")
                continue
                    
        return processed_results

    def _compute_metrics(self, results):
        """Compute evaluation metrics."""
        metrics = {
            'total_examples': len(results),
            'correct_answers': sum(1 for r in results if r.get('ans_match_after_norm', False)),
            'avg_context_length': sum(len(r['document_indices']) for r in results) / len(results),
            'retriever_type': self.retriever_type
        }
        
        metrics['accuracy'] = metrics['correct_answers'] / metrics['total_examples']
        return metrics

    def _save_results(self, results, metrics, output_dir):
        """Save experiment results and metrics."""
        import json
        
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.experiment_logger.info(f"Saved results to {output_dir}")

    def _save_checkpoint(self, results, batch_idx):
        """Save generation checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{batch_idx}_{timestamp}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.experiment_logger.info(f"Saved checkpoint at batch {batch_idx}")

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

def main(args=None):
    try:
        # Set up args
        if isinstance(args, dict):
            parser = argparse.ArgumentParser()
            parser.add_argument('--retriever', type=str)
            namespace = parser.parse_args([])
            for k, v in args.items():
                setattr(namespace, k, v)
            args = namespace
        else:
            args = parse_arguments()
            
        # Run experiment    
        config = BaselineConfigFactory.get_config_for_retriever(args.retriever)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            experiment = BaselineExperiment(
                config=config,
                experiment_name=f"baseline_{args.retriever}",
                retriever_type=args.retriever
            )
            experiment.setup()
            results = experiment.run()
            
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return results
            
    except Exception as e:
        logging.error(f"Error in baseline experiment: {str(e)}", exc_info=True)
        raise
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()