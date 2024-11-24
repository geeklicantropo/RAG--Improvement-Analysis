import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import argparse

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything
from src.llm import LLM
from src.prompt_dataset import PromptDataset
from src.rag_fusion_utils import RAGFusionRanker
from config import CategoriesConfig, CategoriesConfigFactory
from utils import CategoriesExperimentUtils

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
        """Load and prepare dataset with categorized documents."""
        # Load corpus
        from src.utils import read_corpus_json
        corpus = read_corpus_json(str(self.config.corpus_path))
        
        # Prepare search results
        if self.config.use_fusion:
            search_results = self.fusion_ranker.fuse_search_results(
                self.retriever_results
            )
        else:
            search_results = self.retriever_results['contriever']
        
        return PromptDataset(
            corpus=corpus,
            data_path=str(self.config.train_dataset_path),
            tokenizer=self.llm.tokenizer,
            max_tokenized_length=self.config.model_max_length - 2,
            search_results=search_results,
            category_info={
                'score_thresholds': self.config.score_thresholds,
                'max_docs_per_category': self.config.max_docs_per_category
            }
        )
    
    def run(self):
        """Run the categories experiment."""
        try:
            self.logger.log_step_start("Experiment execution")
            
            # Log configuration
            self.logger.log_experiment_params(self.config.to_dict())
            
            # Create data loader
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
            
            # Run generation
            results = self._run_generation(dataloader)
            
            # Analyze results
            metrics = self.utils.analyze_category_impact(results, self.logger)
            
            # Create category info
            category_info = {
                'thresholds': self.config.score_thresholds,
                'max_docs_per_category': self.config.max_docs_per_category,
                'statistics': self.utils.compute_category_statistics(
                    results[0]['category_info'],
                    self.logger
                ),
                'diversity_metrics': self.utils.evaluate_retrieval_diversity(
                    results[0]['category_info'],
                    self.logger
                )
            }
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir) / f"run_{timestamp}"
            
            self.utils.save_category_artifacts(
                results,
                metrics,
                category_info,
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

def main():
    parser = argparse.ArgumentParser(description="Run categories experiment")
    parser.add_argument(
        "--config_type",
        type=str,
        choices=['confidence', 'fusion', 'random'],
        default='confidence',
        help="Type of configuration to use"
    )
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(42)
    
    # Create appropriate configuration
    if args.config_type == 'confidence':
        config = CategoriesConfigFactory.get_confidence_based_config()
    elif args.config_type == 'fusion':
        config = CategoriesConfigFactory.get_fusion_based_config()
    else:
        config = CategoriesConfigFactory.get_random_augmented_config()
    
    # Initialize logger
    logger = ExperimentLogger(
        experiment_name=f"categories_{args.config_type}",
        base_log_dir="logs"
    )
    
    try:
        # Run experiment
        with logger:
            experiment = CategoriesExperiment(
                config=config,
                experiment_name=f"categories_{args.config_type}"
            )
            experiment.setup()
            results, metrics = experiment.run()
            
    except Exception as e:
        logger.log_error(e, "Error in main execution")
        raise

if __name__ == "__main__":
    main()