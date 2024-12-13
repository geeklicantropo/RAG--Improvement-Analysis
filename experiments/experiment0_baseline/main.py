import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import argparse
import json
import logging
import gc
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
from src.utils.corpus_manager import CorpusManager

from experiments.checkpoint_utils import (
    save_checkpoint,
    load_checkpoints,
    get_last_checkpoint_batch,
    merge_checkpoint_results
)

from experiments.experiment0_baseline.config import BaselineConfig, BaselineConfigFactory
from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything, read_corpus_json, read_pickle, clear_memory
from src.llm import LLM
from src.prompt_dataset import PromptDataset

class BaselineExperiment:
    def __init__(
        self, 
        config: BaselineConfig, 
        corpus_manager: CorpusManager, 
        llm_evaluator: Optional[LLM] = None
    ):
        self.config = config
        self.corpus_manager = corpus_manager
        self.logger = ExperimentLogger("baseline", str(Path(__file__).parent.parent.parent / "logs"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_evaluator = llm_evaluator
        self._setup_memory_management()
        self.llm = self._initialize_llm()
        self.results = {"contriever": None, "bm25": None, "random": None}
        self.output_dir = Path(config.output_dir) 
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def run_retrieval_phase(self):
        """Run retrieval phase with proper error handling."""
        try:
            # Load the training corpus
            train_corpus = self._load_corpus()

            # Get retrieval results
            contriever_results = self._run_retrieval("contriever")
            bm25_results = self._run_retrieval("bm25")
            
            # Inject noise if configured
            if self.config.noise_ratio > 0.0:
                self.logger.log_step_start("Injecting noise into retrieval contexts")
                
                contriever_results = self.inject_noise_into_contexts(
                    train_corpus, 
                    contriever_results,
                    self.config.noise_ratio
                )
                
                bm25_results = self.inject_noise_into_contexts(
                    train_corpus,
                    bm25_results,
                    self.config.noise_ratio
                )
                
                self.logger.log_step_end("Noise injection completed")

            # Validate results
            self._validate_retrieval_results(contriever_results, "contriever")
            self._validate_retrieval_results(bm25_results, "bm25")
            
            return contriever_results, bm25_results

        except Exception as e:
            self.logger.log_error(e, "Error in retrieval phase")
            raise

    def inject_noise_into_contexts(
        self,
        corpus: List[Dict],
        search_results: List[Tuple[List[int], List[float]]],
        noise_ratio: float,
        seed: int = 42
    ) -> List[Tuple[List[int], List[float]]]:
        """Inject noise into retrieval contexts."""
        try:
            if self.logger:
                self.logger.log_step_start("Injecting noise into contexts")

            random.seed(seed)
            noisy_results = []

            for doc_indices, scores in tqdm(search_results, desc="Injecting Noise"):
                # Calculate number of noise documents to inject
                num_docs = len(doc_indices)
                num_noise_docs = max(1, int(num_docs * noise_ratio))
                
                # Keep original documents up to available space
                keep_docs = num_docs - num_noise_docs
                retained_indices = doc_indices[:keep_docs]
                retained_scores = scores[:keep_docs]
                
                # Sample noise documents from available corpus
                available_indices = list(set(range(len(corpus))) - set(retained_indices))
                if num_noise_docs > len(available_indices):
                    num_noise_docs = len(available_indices)
                    
                noise_indices = random.sample(available_indices, num_noise_docs)
                noise_scores = [0.0] * num_noise_docs
                
                # Combine original and noise documents
                new_indices = retained_indices + noise_indices
                new_scores = retained_scores + noise_scores
                
                noisy_results.append((new_indices, new_scores))

            if self.logger:
                self.logger.log_metric("num_noisy_results", len(noisy_results))
                self.logger.log_step_end("Noise injection completed")

            return noisy_results

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error injecting noise into contexts")
            raise

    def run_generation_phase(self):
        """Run generation phase with test dataset and position testing."""
        try:
            self.logger.log_step_start("Generation Phase")

            # Check for existing generation results or merge checkpoints
            if not self.results.get('generation'):
                self.results['generation'] = self.merge_generation_checkpoints()

            # If no previous results exist, run position tests
            if not self.results['generation']:
                position_results = {}
                
                # Test different positions for gold document
                for position in range(self.config.num_documents_in_context):
                    self.logger.log_step_start(f"Testing position {position}")
                    
                    self.config.gold_position = position
                    corpus = self._load_corpus()
                    dataset = self._load_dataset(corpus)
                    dataloader = self._create_dataloader(dataset)
                    
                    # Run generation for this position
                    results = self._run_generation(dataloader)
                    position_results[position] = results
                    
                    # Save position checkpoint
                    save_checkpoint(
                        results,
                        position,
                        self.config.output_dir / "position_checkpoints"
                    )
                    
                    clear_memory()
                
                # Run noise injection tests
                noise_results = {}
                noise_ratios = [0.1, 0.2, 0.3]
                
                for ratio in noise_ratios:
                    self.logger.log_step_start(f"Testing noise ratio {ratio}")
                    
                    # Inject noise into corpus
                    corpus = self._load_corpus()
                    noisy_corpus = self.inject_noise_into_contexts(
                        corpus,
                        self._get_search_results(),
                        ratio
                    )
                    
                    dataset = self._load_dataset(noisy_corpus)
                    dataloader = self._create_dataloader(dataset)
                    results = self._run_generation(dataloader)
                    
                    noise_results[ratio] = results
                    
                    # Save noise checkpoint
                    save_checkpoint(
                        results,
                        int(ratio * 100),
                        self.config.output_dir / "noise_checkpoints"
                    )
                    
                    clear_memory()
                
                # Test document combinations
                combination_results = {}
                
                # Gold + Random
                gold_positions = [0, self.config.num_documents_in_context // 2, -1]
                for pos in gold_positions:
                    self.config.gold_position = pos
                    
                    corpus = self._load_corpus()
                    dataset = self._load_dataset(corpus)
                    dataloader = self._create_dataloader(dataset)
                    
                    results = self._run_generation(dataloader)
                    combination_results[f'gold_{pos}_random'] = results
                    
                    save_checkpoint(
                        results,
                        f'gold_{pos}_random',
                        self.config.output_dir / "combinations"
                    )
                
                # Gold + Distracting
                for pos in gold_positions:
                    self.config.gold_position = pos
                    self.config.get_documents_without_answer = True
                    
                    corpus = self._load_corpus()  
                    dataset = self._load_dataset(corpus)
                    dataloader = self._create_dataloader(dataset)
                    
                    results = self._run_generation(dataloader) 
                    combination_results[f'gold_{pos}_distracting'] = results
                    
                    save_checkpoint(
                        results,
                        f'gold_{pos}_distracting',  
                        self.config.output_dir / "combinations"
                    )
                    
                # Combine all results
                final_results = {
                    'position_results': position_results,
                    'noise_results': noise_results,
                    'combination_results': combination_results
                }
                
                # Save final checkpoint
                save_checkpoint(
                    final_results,
                    len(position_results) + len(noise_results) + len(combination_results),
                    self.config.output_dir / "final_checkpoint"  
                )
                
                self.results['generation'] = final_results

            return self.results['generation']

        except Exception as e:
            self.logger.log_error(e, "Error in generation phase")
            raise

    def _run_generation(self, dataloader: DataLoader) -> List[Dict]:
        """Run generation with proper evaluation and checkpointing."""
        results = []
        last_checkpoint = get_last_checkpoint_batch(Path(self.config.output_dir) / "checkpoints")
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch_idx <= last_checkpoint:
                continue
                
            batch_results = self._process_batch(batch)
            
            # Evaluate with LLM if available
            if self.llm_evaluator:
                for result in batch_results:
                    eval_result = self.llm_evaluator.evaluate_answer(
                        question=result['query'],
                        generated_answer=result['generated_answer'],
                        gold_answer=result['gold_answer']
                    )
                    result['llm_evaluation'] = eval_result
            
            results.extend(batch_results)
            
            if (batch_idx + 1) % self.config.save_every == 0:
                save_checkpoint(results, batch_idx + 1, Path(self.config.output_dir))
                results = []  # Clear processed results after checkpoint
                clear_memory()
                 
        if results:  # Save any remaining results
            save_checkpoint(results, len(dataloader), Path(self.config.output_dir))
            
        return load_checkpoints(Path(self.config.output_dir) / "checkpoints")

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        results = []
        for item in batch:
            context = "\n\n".join([
                f"Document [{i+1}]:\n{doc['text']}" 
                for i, doc in enumerate(item['documents'])
            ])
            
            eval_result = self.llm_evaluator.evaluate_answer(
                question=item['query'],
                context=context,
                gold_answer=item['gold_answer']
            )
            
            results.append({
                'query': item['query'],
                'generated_answer': eval_result['generated_answer'],
                'evaluation': eval_result,
                'document_indices': [d['id'] for d in item['documents']],
                'gold_position': self.config.gold_position,
                'noise_ratio': self.config.noise_ratio
            })
        return results

    def _load_corpus(self) -> List[Dict]:
        """Load appropriate corpus based on experiment type."""
        if self.config.use_random:
            return self.corpus_manager.get_noisy_corpus(noise_ratio=0.2)
        return self.corpus_manager.get_baseline_corpus()

    def _get_search_results(self) -> List[Tuple[List[int], List[float]]]:
        """Get appropriate search results based on configuration."""
        if self.config.use_bm25:
            return read_pickle(str(self.config.bm25_results_path))
        elif self.config.use_random:
            return read_pickle(str(self.config.random_results_path))
        return read_pickle(str(self.config.contriever_results_path))

    def _load_dataset(self, corpus: List[Dict]) -> List[Dict]:
        search_results = self._get_search_results()
        if self.config.gold_position is not None:
            search_results = self._position_documents(search_results, self.config.gold_position)
            
        formatted_data = []
        for doc_set in search_results:
            documents = [corpus[idx] for idx in doc_set[0][:self.config.num_documents_in_context]]
            formatted_data.append({
                'documents': documents,
                'scores': doc_set[1][:self.config.num_documents_in_context]
            })
        return formatted_data

    def _position_documents(
        self,
        search_results: List[Tuple[List[int], List[float]]],
        target_position: int
    ) -> List[Tuple[List[int], List[float]]]:
        """Position documents at specified location."""
        positioned_results = []
        
        for doc_indices, scores in search_results:
            # Ensure we have enough documents
            if len(doc_indices) <= target_position:
                positioned_results.append((doc_indices, scores))
                continue
                
            # Find answer-containing document
            gold_idx = self._find_gold_document(doc_indices)
            if gold_idx is None:
                positioned_results.append((doc_indices, scores))
                continue
                
            # Reposition document
            new_indices = doc_indices.copy()
            new_scores = scores.copy()
            
            doc_id = new_indices.pop(gold_idx)
            doc_score = new_scores.pop(gold_idx)
            
            new_indices.insert(target_position, doc_id)
            new_scores.insert(target_position, doc_score)
            
            positioned_results.append((new_indices, new_scores))
            
        return positioned_results

    def _find_gold_document(self, doc_indices: List[int]) -> Optional[int]:
        """Find index of document containing answer."""
        for idx, doc_id in enumerate(doc_indices):
            doc = self.corpus_manager.get_baseline_corpus()[doc_id]
            if any(answer in doc['text'] for answer in self.config.gold_answers):
                return idx
        return None

    def _create_dataloader(self, dataset: PromptDataset) -> DataLoader:
        """Create data loader with dynamic batch size adjustment."""
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

    def _run_retrieval(self, retriever_type: str) -> List[Tuple[List[int], List[float]]]:
        """Run retrieval for the specified retriever type."""
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

    def merge_generation_checkpoints(self):
        """Merge all generation checkpoints after verifying completeness."""
        checkpoint_dir = self.output_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            return []
            
        checkpoint_files = sorted(checkpoint_dir.glob("generation_checkpoint_*.json"), 
            key=lambda p: int(p.stem.split('_')[-1]))
            
        actual_batch_count = len(checkpoint_files)
        
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
            
        all_results = []
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file) as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
                
        return all_results

    def run(self):
        """Run complete baseline experiment with position and noise testing."""
        try:
            # Load last checkpoint if exists
            last_checkpoint = get_last_checkpoint_batch(Path(self.config.output_dir) / "checkpoints")
            if last_checkpoint > 0:
                results = load_checkpoints(Path(self.config.output_dir) / "checkpoints")
            else:
                # Run retrieval phase
                contriever_results, bm25_results = self.run_retrieval_phase()
                
                # Run generation phase
                generation_results = self.run_generation_phase()
                
                results = {
                    'contriever': contriever_results,
                    'bm25': bm25_results,
                    'generation': generation_results
                }
            
            # Evaluate with LLM
            if self.llm_evaluator:
                self._evaluate_with_llm(results)
            
            # Save final results
            self._save_final_results(results)
            
            return results
            
        except Exception as e:
            self.logger.log_error(e, "Error in baseline experiment")
            raise
        finally:
            clear_memory()

    def _evaluate_with_llm(self, results: Dict):
        """Evaluate results using LLM."""
        for result_type, result_set in results.items():
            if isinstance(result_set, dict):
                for key, subset in result_set.items():
                    self._evaluate_result_subset(subset)
            else:
                self._evaluate_result_subset(result_set)

    def _evaluate_result_subset(self, results: List[Dict]):
        """Evaluate a subset of results using LLM."""
        for result in results:
            eval_result = self.llm_evaluator.evaluate_answer(
                question=result['query'],
                generated_answer=result['generated_answer'],
                gold_answer=result['gold_answer']
            )
            result['llm_evaluation'] = eval_result

    def _save_final_results(self, results: Dict):
        """Save final experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self.config.output_dir / f"final_results_{timestamp}.json"
        
        with open(final_path, 'w') as f:
            json.dump(results, f, indent=2)

def main(experiment_args=None):
    try:
        # Set up args if not passed
        if isinstance(experiment_args, dict):
            parser = argparse.ArgumentParser()
            parser.add_argument('--retriever', type=str)
            parser.add_argument('--output_dir', type=str)
            namespace = parser.parse_args([])
            for k, v in experiment_args.items():
                setattr(namespace, k, v)
            experiment_args = namespace
        else:
            experiment_args = parse_arguments()

        # Load baseline config
        config = BaselineConfigFactory.get_config_for_retriever(experiment_args.retriever)
        
        # Initialize corpus manager and LLM evaluator
        corpus_manager = CorpusManager(
            base_corpus_path="data/processed/corpus_with_contriever_at150.json"
        )
        llm_evaluator = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        
        # Initialize and run experiment
        experiment = BaselineExperiment(
            config, 
            corpus_manager,
            llm_evaluator=llm_evaluator
        )
        results = experiment.run()
        
        return results
        
    except Exception as e:
        logging.error(f"Error in baseline experiment: {str(e)}", exc_info=True)
        raise
    finally:
        clear_memory()

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
    seed_everything(10)
    main()