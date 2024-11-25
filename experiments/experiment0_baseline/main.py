import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.utils import seed_everything
from src.llm import LLM
from src.retriever import Retriever
from src.experiment_logger import ExperimentLogger
from config import BaselineConfig, BaselineConfigFactory

class BaselineExperiment:
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
        """Set up experiment components without re-downloading files."""
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
        """Set up Contriever retriever using existing files."""
        from transformers import AutoConfig, AutoTokenizer
        from src.retriever import Encoder
        
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
        """Set up BM25 retriever using existing index."""
        from pyserini.search.lucene import LuceneSearcher
        return LuceneSearcher.from_prebuilt_index('wikipedia-dpr')
    
    def _load_dataset(self):
        """Load dataset using existing files."""
        from src.prompt_dataset import PromptDataset
        
        # Load corpus and search results from existing files
        corpus = self._load_corpus()
        search_results = self._load_search_results()
        
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
    
    def _load_corpus(self):
        """Load corpus from existing file."""
        from src.utils import read_corpus_json
        return read_corpus_json(str(self.config.corpus_path))
    
    def _load_search_results(self):
        """Load search results from existing file."""
        from src.utils import read_pickle
        return read_pickle(str(self.config.search_results_path))
    
    def run(self):
        """Run experiment generating new output files."""
        try:
            self.logger.log_step_start("Experiment execution")
            
            # Log configuration
            self.logger.log_experiment_params(self.config.to_dict())
            
            # Create data loader
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
            
            # Generate timestamp for new results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config.output_dir / f"run_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run generation
            results = self._run_generation(dataloader)
            
            # Save results with timestamp
            self._save_results(results, output_dir)
            
            # Compute and log metrics
            metrics = self._compute_metrics(results)
            self._save_metrics(metrics, output_dir)
            
            self.logger.log_step_end("Experiment execution")
            
        except Exception as e:
            self.logger.log_error(e, "Error during experiment execution")
            raise

    def _run_generation(self, dataloader: DataLoader) -> List[Dict]:
        """Run generation on batches."""
        all_results = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._save_checkpoint(all_results, batch_idx + 1, timestamp)
                
        return all_results
    
    def _process_batch_outputs(
        self,
        batch: Dict[str, Any],
        generated_outputs: List[str]
    ) -> List[Dict]:
        """Process batch outputs into structured results."""
        processed_results = []
        answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
        
        for idx, output in enumerate(generated_outputs):
            # Extract answer from generated output
            start_idx = output.find(answer_string) + len(answer_string)
            answer = output[start_idx:].strip()
            
            # Create result dictionary
            result = {
                'example_id': batch['example_id'][idx],
                'query': batch['query'][idx],
                'generated_answer': answer,
                'prompt': batch['prompt'][idx],
                'document_indices': batch['document_indices'][idx],
                'gold_document_idx': batch['gold_document_idx'][idx],
                'prompt_tokens_len': batch['prompt_tokens_len'][idx],
                'ans_match_after_norm': False,  # Will be updated in metrics computation
                'gold_in_retrieved': str(batch['gold_document_idx'][idx]) in batch['document_indices'][idx]
            }
            processed_results.append(result)
            
        return processed_results
    
    def _save_checkpoint(
        self,
        results: List[Dict],
        batch_idx: int,
        timestamp: str
    ):
        """Save intermediate results."""
        from src.utils import write_pickle
        
        checkpoint_dir = self.config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{batch_idx}_{timestamp}.pkl"
        write_pickle(results, str(checkpoint_path))
        self.logger.experiment_logger.info(f"Saved checkpoint at batch {batch_idx}")
    
    def _save_results(self, results: List[Dict], output_dir: Path):
        """Save final results."""
        import json
        
        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.experiment_logger.info(f"Saved results to {results_path}")
        
    def _compute_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute comprehensive experiment metrics."""
        metrics = {
            'total_examples': len(results),
            'avg_prompt_length': sum(r['prompt_tokens_len'] for r in results) / len(results),
            'gold_retrieval_rate': sum(1 for r in results if r['gold_in_retrieved']) / len(results)
        }
        
        # Compute accuracy and gold document metrics
        correct_answers = 0
        gold_helpful = 0
        
        for result in results:
            if result['ans_match_after_norm']:
                correct_answers += 1
                if result['gold_in_retrieved']:
                    gold_helpful += 1
                    
        metrics['accuracy'] = correct_answers / len(results)
        metrics['gold_helpfulness'] = gold_helpful / sum(1 for r in results if r['gold_in_retrieved']) if sum(1 for r in results if r['gold_in_retrieved']) > 0 else 0
        
        # Log metrics
        for name, value in metrics.items():
            self.logger.log_metric(name, value)
            
        return metrics
    
    def _save_metrics(self, metrics: Dict[str, Any], output_dir: Path):
        """Save metrics to file."""
        import json
        
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.experiment_logger.info(f"Saved metrics to {metrics_path}")

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