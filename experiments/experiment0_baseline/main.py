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
        """Run the baseline experiment."""
        try:
            self.logger.log_step_start("Experiment execution")
            
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
            metrics = self._compute_metrics(results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir) / f"run_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self._save_results(results, metrics, output_dir)
            
            self.logger.log_step_end("Experiment execution")
            return results, metrics
            
        except Exception as e:
            self.logger.log_error(e, "Error during experiment execution")
            raise

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

    def _process_batch_outputs(self, batch, outputs):
        """Process generation outputs for a batch."""
        processed_results = []
        answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
        
        for idx, output in enumerate(outputs):
            start_idx = output.find(answer_string) + len(answer_string)
            answer = output[start_idx:].strip()
            
            result = {
                'query': batch['query'][idx],
                'generated_answer': answer,
                'document_indices': batch['document_indices'][idx],
                'prompt_tokens_len': batch['prompt_tokens_len'][idx]
            }
            processed_results.append(result)
            
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
        if isinstance(args, dict):
            parser = argparse.ArgumentParser()
            parser.add_argument('--retriever', type=str)
            namespace = parser.parse_args([])
            for k, v in args.items():
                setattr(namespace, k, v)
            args = namespace
        else:
            args = parse_arguments()
            
        config = BaselineConfigFactory.get_config_for_retriever(args.retriever)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            experiment = BaselineExperiment(
                config=config,
                experiment_name=f"baseline_{args.retriever}",
                retriever_type=args.retriever
            )
            experiment.setup()
            return experiment.run()
            
    except Exception as e:
        logging.error(f"Error in baseline experiment: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()