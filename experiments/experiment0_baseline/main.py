import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from src.retriever import Encoder
from transformers import AutoConfig, AutoTokenizer
from pyserini.search.lucene import LuceneSearcher

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.utils import seed_everything
from src.llm import LLM
from src.retriever import Retriever
from src.experiment_logger import ExperimentLogger
from config import BaselineConfig, BaselineConfigFactory

class BaselineExperiment:
    """
    Runner for baseline RAG experiments with comprehensive logging and evaluation.
    """
    def __init__(
        self,
        config: BaselineConfig,
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
            
            # Initialize retriever based on config
            if not self.config.use_bm25:
                self.retriever = self._setup_contriever()
            else:
                self.retriever = self._setup_bm25()
                
            # Load dataset
            self.dataset = self._load_dataset()
            
            self.logger.log_step_end("Setup")
            
        except Exception as e:
            self.logger.log_error(e, "Error in experiment setup")
            raise
            
    def _setup_contriever(self) -> Retriever:
        """Set up Contriever retriever."""
        
        config = AutoConfig.from_pretrained('facebook/contriever')
        encoder = Encoder(config).eval()
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        
        return Retriever(
            device=self.device,
            tokenizer=tokenizer,
            query_encoder=encoder,
            norm_doc_emb=self.config.normalize_embeddings
        )
    
    def _setup_bm25(self) -> Any:
        """Set up BM25 retriever."""
        
        return LuceneSearcher.from_prebuilt_index('wikipedia-dpr')
    
    def _load_dataset(self):
        """Load appropriate dataset based on configuration."""
        from torch.utils.data import DataLoader
        from src.prompt_dataset import PromptDataset
        
        dataset_path = self.config.test_dataset_path if self.config.use_test else self.config.train_dataset_path
        
        return PromptDataset(
            corpus=self._load_corpus(),
            data_path=str(dataset_path),
            tokenizer=self.llm.tokenizer,
            max_tokenized_length=self.config.model_max_length - 2,
            search_results=self._load_search_results(),
            num_documents_in_context=self.config.num_documents_in_context,
            gold_position=self.config.gold_position,
            get_documents_without_answer=self.config.get_documents_without_answer
        )
    
    def _load_corpus(self):
        """Load document corpus."""
        from src.utils import read_corpus_json
        return read_corpus_json(str(self.config.corpus_path))
    
    def _load_search_results(self):
        """Load appropriate search results based on configuration."""
        from src.utils import read_pickle
        
        if self.config.use_random:
            return read_pickle(str(self.config.base_data_dir / "10k_random_results_at60.pkl"))
        elif self.config.use_bm25:
            return read_pickle(str(self.config.base_data_dir / "bm25_test_search_results_at250.pkl"))
        else:
            return read_pickle(str(self.config.base_data_dir / "contriever_search_results_at150.pkl"))
    
    def run(self):
        """Run the experiment."""
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
            
            # Save results
            self._save_results(results)
            
            # Compute and log metrics
            metrics = self._compute_metrics(results)
            for metric_name, value in metrics.items():
                self.logger.log_metric(metric_name, value)
                
            self.logger.log_step_end("Experiment execution")
            
        except Exception as e:
            self.logger.log_error(e, "Error during experiment execution")
            raise
    
    def _run_generation(self, dataloader):
        """Run generation on batches."""
        all_results = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Generate answers
            prompts = batch['prompt']
            generated_outputs = self.llm.generate(
                prompts,
                max_new_tokens=self.config.max_new_tokens
            )
            
            # Process outputs
            processed_results = self._process_batch_outputs(batch, generated_outputs)
            all_results.extend(processed_results)
            
            # Periodic saving
            if (batch_idx + 1) % self.config.save_every == 0:
                self._save_checkpoint(all_results, batch_idx + 1)
                
        return all_results
    
    def _process_batch_outputs(self, batch, generated_outputs):
        """Process batch outputs into structured results."""
        processed_results = []
        answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
        
        for idx, output in enumerate(generated_outputs):
            start_idx = output.find(answer_string) + len(answer_string)
            answer = output[start_idx:].strip()
            
            result = {
                'example_id': batch['example_id'][idx],
                'query': batch['query'][idx],
                'generated_answer': answer,
                'prompt': batch['prompt'][idx],
                'document_indices': batch['document_indices'][idx],
                'gold_document_idx': batch['gold_document_idx'][idx],
            }
            processed_results.append(result)
            
        return processed_results
    
    def _save_checkpoint(self, results, batch_idx):
        """Save intermediate results."""
        from src.utils import write_pickle
        
        checkpoint_path = self.config.output_dir / f"checkpoint_{batch_idx}.pkl"
        write_pickle(results, str(checkpoint_path))
        self.logger.experiment_logger.info(f"Saved checkpoint at batch {batch_idx}")
    
    def _save_results(self, results):
        """Save final results."""
        import json
        
        results_path = self.config.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.experiment_logger.info(f"Saved final results to {results_path}")
    
    def _compute_metrics(self, results):
        """Compute experiment metrics."""
        metrics = {
            'total_examples': len(results),
            'avg_prompt_length': sum(len(r['prompt']) for r in results) / len(results),
        }
        
        # Add more metrics as needed
        return metrics

def main():
    # Set random seed
    seed_everything(42)
    
    # Create experiment configurations
    configs = {
        'contriever': BaselineConfigFactory.get_contriever_config(),
        'bm25': BaselineConfigFactory.get_bm25_config(),
        'random': BaselineConfigFactory.get_random_config()
    }
    
    # Run experiments
    for exp_name, config in configs.items():
        try:
            # Initialize experiment
            experiment = BaselineExperiment(
                config=config,
                experiment_name=f"baseline_{exp_name}"
            )
            
            # Run experiment phases
            with experiment.logger:
                experiment.setup()
                experiment.run()
                
        except Exception as e:
            logging.error(f"Error in {exp_name} experiment: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()