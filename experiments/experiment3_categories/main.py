import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
import warnings
import json
import gc

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.utils import read_corpus_json

from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything
from src.llm import LLM
from src.prompt_dataset import PromptDataset
from src.rag_fusion_utils import RAGFusionRanker
from .config import CategoriesConfig, CategoriesConfigFactory
from .utils import CategoriesExperimentUtils

class CategoriesExperiment:
    """
    Implements experiment workflow for testing category-based document organization in RAG systems.
    """
    
    def __init__(
        self,
        config: CategoriesConfig,
        experiment_name: str,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.logger = logger or ExperimentLogger(
            experiment_name=experiment_name,
            base_log_dir=str(Path(project_root) / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize utilities
        self.utils = CategoriesExperimentUtils()
        
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
            
            # Load retrieval results
            self.retriever_results = self.utils.load_retrieval_results(
                str(self.config.contriever_results_path),
                str(self.config.bm25_results_path) if self.config.use_bm25 else None,
                self.config.use_fusion,
                self.logger
            )
            
            # Initialize fusion ranker if needed
            if self.config.use_fusion:
                self.fusion_ranker = RAGFusionRanker(
                    strategy="rrf",
                    normalize_scores=True,
                    score_weights=self.config.fusion_weights
                )
            
            # Load dataset
            self.dataset = self._load_dataset()
            
            self.logger.log_step_end("Setup")
            
        except Exception as e:
            self.logger.log_error(e, "Error in experiment setup")
            raise
            
    def _load_dataset(self) -> PromptDataset:
        """Load and prepare dataset with safe corpus handling."""
        try:
            # Load corpus
            corpus = read_corpus_json(str(self.config.corpus_path))
            
            # Load or perform fusion on search results
            if self.config.use_fusion:
                search_results = self.fusion_ranker.fuse_search_results(self.retriever_results)
                
                # Validate corpus indices
                max_idx = len(corpus) - 1
                filtered_results = []
                for doc_ids, scores in search_results:
                    valid_ids = [idx for idx in map(int, doc_ids) if 0 <= idx <= max_idx]
                    if valid_ids:
                        filtered_results.append((valid_ids, scores[:len(valid_ids)]))
                    else:
                        filtered_results.append(([], []))
                search_results = filtered_results
            else:
                search_results = self.retriever_results['contriever']
            
            return PromptDataset(
                corpus=corpus,
                data_path=str(self.config.train_dataset_path),
                tokenizer=self.llm.tokenizer,
                max_tokenized_length=self.config.model_max_length - 2,
                search_results=search_results,
                category_info={'score_thresholds': self.config.score_thresholds}
            )
        except Exception as e:
            self.logger.log_error(e, "Error loading dataset")
            raise
    
    def run(self):
        """Run the categories experiment."""
        try:
            self.logger.log_step_start("Experiment execution")
            
            # Monitor initial GPU memory
            if torch.cuda.is_available():
                gpu = torch.cuda.current_device()
                initial_memory = torch.cuda.memory_allocated(gpu) / torch.cuda.max_memory_allocated(gpu)
                self.config.adjust_batch_sizes(initial_memory)
            
            # Log configuration
            self.logger.log_experiment_params(self.config.to_dict())
            
            # Create data loader with adjusted batch size
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
            
            # Run generation
            results = []
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
                # Monitor memory during processing
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated(gpu) / torch.cuda.max_memory_allocated(gpu)
                    self.config.adjust_batch_sizes(current_memory)

                # Categorize documents
                categorized_docs = self.utils.categorize_documents(
                    batch['documents'],
                    batch['scores'],
                    self.config.score_thresholds,
                    self.config.max_docs_per_category,
                    batch_size=self.config.categorization_batch_size,
                    logger=self.logger
                )
                
                # Generate answers
                prompts = [
                    self.utils.format_category_prompt(q, docs, self.logger)
                    for q, docs in zip(batch['query'], categorized_docs)
                ]
                
                generated_outputs = self.llm.generate(
                    prompts,
                    max_new_tokens=self.config.max_new_tokens
                )
                
                # Process outputs
                batch_results = self._process_batch_outputs(
                    batch,
                    generated_outputs,
                    categorized_docs
                )
                results.extend(batch_results)
                
                # Save checkpoint if needed
                if self.config.save_intermediates and (batch_idx + 1) % self.config.save_every == 0:
                    self._save_checkpoint(results, batch_idx + 1)

            # Analyze results
            metrics = self.utils.analyze_category_impact(results, self.logger)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir) / f"run_{timestamp}"
            
            self.utils.save_category_artifacts(
                results,
                metrics,
                categorized_docs,
                str(output_dir),
                self.logger
            )
            
            self.logger.log_step_end("Experiment execution")
            return results, metrics
            
        except Exception as e:
            self.logger.log_error(e, "Error during experiment execution")
            raise
            
    def _run_generation(
        self, 
        dataloader: torch.utils.data.DataLoader
    ) -> List[Dict[str, Any]]:
        """Run generation on batches."""
        results = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
            # Categorize documents and prepare prompts
            categorized_docs = self.utils.categorize_documents(
                batch['documents'],
                batch['scores'],
                self.config.score_thresholds,
                self.config.max_docs_per_category,
                self.logger
            )
            
            # Generate answers
            prompts = [
                self.utils.format_category_prompt(
                    q, categorized_docs, self.logger
                )
                for q in batch['query']
            ]
            
            generated_outputs = self.llm.generate(
                prompts,
                max_new_tokens=self.config.max_new_tokens
            )
            
            # Process outputs
            batch_results = self._process_batch_outputs(
                batch,
                generated_outputs,
                categorized_docs
            )
            results.extend(batch_results)
            
            # Save checkpoint if configured
            if self.config.save_intermediates and (batch_idx + 1) % self.config.save_every == 0:
                self._save_checkpoint(results, batch_idx + 1)
                
        return results
    
    def _process_batch_outputs(
        self,
        batch: Dict[str, Any],
        outputs: List[str],
        categorized_docs: Dict[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """Process generation outputs for a batch."""
        processed_results = []
        answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
        
        for idx, output in enumerate(outputs):
            # Extract answer
            start_idx = output.find(answer_string) + len(answer_string)
            answer = output[start_idx:].strip()
            
            # Create result dictionary
            result = {
                'query': batch['query'][idx],
                'generated_answer': answer,
                'category_info': categorized_docs,
                'document_indices': batch['document_indices'][idx],
                'prompt_tokens_len': batch['prompt_tokens_len'][idx]
            }
            processed_results.append(result)
            
        return processed_results
    
    def _save_checkpoint(
        self,
        results: List[Dict],
        batch_idx: int
    ):
        """Save generation checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{batch_idx}_{timestamp}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.experiment_logger.info(f"Saved checkpoint at batch {batch_idx}")

def parse_arguments():
    """Parse command line arguments for categories experiment."""
    parser = argparse.ArgumentParser(description="Run categories experiment")
    
    parser.add_argument(
        '--config_type',
        type=str,
        choices=['confidence', 'fusion', 'random'],
        default='confidence',
        help='Type of categorization to use'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/experiment3_categories/results',
        help='Output directory for results'
    )
    
    return parser.parse_args()

def main(args=None):
    try:
        # Set up args
        if isinstance(args, dict):
            parser = argparse.ArgumentParser()
            parser.add_argument('--config_type', type=str)
            namespace = parser.parse_args([])
            for k, v in args.items():
                setattr(namespace, k, v)
            args = namespace
        else:
            args = parse_arguments()

        # Run experiment
        config = CategoriesConfigFactory.get_config_for_type(args.config_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            experiment = CategoriesExperiment(
                config=config,
                experiment_name=f"categories_{args.config_type}",
                logger=ExperimentLogger(f"categories_{args.config_type}")
            )
            experiment.setup()
            results = experiment.run()
            
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        gc.collect()
            
        return results
            
    except Exception as e:
        logging.error(f"Error in categories experiment: {str(e)}", exc_info=True)
        raise
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()