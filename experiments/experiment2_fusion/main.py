import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import argparse
import json
import warnings
import logging

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything
from src.llm import LLM
from src.prompt_dataset import PromptDataset
from src.rag_fusion_utils import RAGFusionRanker
#from config import FusionConfig, FusionConfigFactory
#from utils import FusionExperimentUtils

from experiments.experiment2_fusion.config import FusionConfig, FusionConfigFactory
from experiments.experiment2_fusion.utils import FusionExperimentUtils


class FusionExperiment:
    """
    Implements experiment workflow for testing RAG-Fusion approaches.
    """
    
    def __init__(
        self,
        config: FusionConfig,
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
        self.utils = FusionExperimentUtils()
        
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
                str(self.config.bm25_results_path),
                self.logger
            )
            
            # Initialize fusion ranker
            self.fusion_ranker = self.utils.initialize_fusion_ranker(
                strategy=self.config.fusion_strategy,
                k=self.config.fusion_k,
                normalize_scores=self.config.normalize_scores,
                weights={
                    'contriever': self.config.contriever_weight,
                    'bm25': self.config.bm25_weight
                },
                logger=self.logger
            )
            
            # Load dataset
            self.dataset = self._load_dataset()
            
            self.logger.log_step_end("Setup")
            
        except Exception as e:
            self.logger.log_error(e, "Error in experiment setup")
            raise
            
    def _load_dataset(self) -> PromptDataset:
        """Load and prepare dataset."""
        # Load corpus
        from src.utils import read_corpus_json
        corpus = read_corpus_json(str(self.config.corpus_path))
        
        # Perform fusion on search results
        fused_results = self.fusion_ranker.fuse_search_results(self.retriever_results)
        
        # Inject random documents if configured
        if self.config.use_random_docs:
            fused_results = self.utils.inject_random_documents(
                fused_results,
                str(self.config.random_doc_source),
                self.config.random_doc_ratio,
                self.logger
            )
        
        return PromptDataset(
            corpus=corpus,
            data_path=str(self.config.train_dataset_path),
            tokenizer=self.llm.tokenizer,
            max_tokenized_length=self.config.model_max_length - 2,
            search_results=fused_results,
            num_documents_in_context=self.config.num_documents_in_context
        )
    
    def run(self):
        """Run the fusion experiment."""
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
            metrics = self.utils.analyze_fusion_results(results, self.logger)
            
            # Create fusion info
            fusion_info = {
                'strategy': self.config.fusion_strategy,
                'weights': {
                    'contriever': self.config.contriever_weight,
                    'bm25': self.config.bm25_weight
                },
                'normalize_scores': self.config.normalize_scores,
                'fusion_k': self.config.fusion_k
            }
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir) / f"run_{timestamp}"
            
            self.utils.save_fusion_artifacts(
                results,
                metrics,
                fusion_info,
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
            # Generate answers
            prompts = batch['prompt']
            generated_outputs = self.llm.generate(
                prompts,
                max_new_tokens=self.config.max_new_tokens
            )
            
            # Process outputs
            batch_results = self._process_batch_outputs(batch, generated_outputs)
            results.extend(batch_results)
            
            # Save intermediate results if configured
            if self.config.save_intermediates and (batch_idx + 1) % self.config.save_every == 0:
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
        
        for idx, output in enumerate(outputs):
            # Extract answer from generation
            start_idx = output.find(answer_string) + len(answer_string)
            answer = output[start_idx:].strip()
            
            # Create result dictionary
            result = {
                'query': batch['query'][idx],
                'generated_answer': answer,
                'retriever_scores': {
                    'contriever': batch.get('contriever_scores', [None])[idx],
                    'bm25': batch.get('bm25_scores', [None])[idx]
                },
                'document_indices': batch['document_indices'][idx],
                'prompt_tokens_len': batch['prompt_tokens_len'][idx]
            }
            processed_results.append(result)
            
        return processed_results
    
    def _save_checkpoint(self, results: List[Dict], batch_idx: int):
        """Save intermediate results checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{batch_idx}_{timestamp}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.experiment_logger.info(f"Saved checkpoint at batch {batch_idx}")

def parse_arguments():
    """Parse command line arguments for fusion experiment."""
    parser = argparse.ArgumentParser(description="Run fusion experiment")
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['rrf', 'linear'],
        default='rrf',
        help='Fusion strategy to use'
    )
    parser.add_argument(
        '--use_random',
        action='store_true',
        help='Whether to use random documents'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/experiment2_fusion/results',
        help='Output directory for results'
    )
    
    return parser.parse_args()

def main(args=None):
    """Main entry point with silent execution for fusion experiment."""
    try:
        if isinstance(args, dict):
            parser = argparse.ArgumentParser()
            parser.add_argument('--strategy', type=str)
            parser.add_argument('--use_random', type=bool, default=False)
            namespace = parser.parse_args([])
            for k, v in args.items():
                setattr(namespace, k, v)
            args = namespace
        else:
            args = parse_arguments()

        config = FusionConfigFactory.get_config_for_strategy(args.strategy, args.use_random)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            experiment = FusionExperiment(
                config=config,
                experiment_name=f"fusion_{args.strategy}",
                logger=ExperimentLogger(f"fusion_{args.strategy}")
            )
            experiment.setup()
            return experiment.run()
            
    except Exception as e:
        logging.error(f"Error in fusion experiment: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()