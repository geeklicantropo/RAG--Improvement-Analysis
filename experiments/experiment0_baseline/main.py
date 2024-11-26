import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import argparse
import warnings

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.experiment_logger import ExperimentLogger
from src.utils import (
    seed_everything, 
    read_pickle,
    read_corpus_json,
    read_test_corpus_with_random_and_bm25,
    read_corpus_with_contriever
)
from src.llm import LLM
from src.prompt_dataset import PromptDataset
from src.utils import read_corpus_json, read_pickle, write_pickle
from src.cluster_utils import DocumentClusterer
from src.experiment_logger import ExperimentLogger
from .config import BaselineConfig, BaselineConfigFactory
from collections import defaultdict

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
            
            # Initialize clusterer if needed
            if self.config.use_clustering:
                self.clusterer = DocumentClusterer(
                    num_clusters=self.config.num_clusters,
                    random_seed=self.config.cluster_seed,
                    use_scaler=True,
                    logger=self.logger
                )
            
            self.logger.log_step_end("Setup")
            
        except Exception as e:
            self.logger.log_error(e, "Error in experiment setup")
            raise

    def _load_search_results(self):
        """Load appropriate search results based on config."""
        if self.config.use_random:
            return read_pickle(str(self.config.random_results_path))
        elif self.config.use_bm25:
            return read_pickle(str(self.config.bm25_results_path))
        else:
            return read_pickle(str(self.config.contriever_results_path))

    def run(self):
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
            
            # Process clustering if used
            if self.config.use_clustering:
                results = self._process_clustering_results(results)
            
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

    def _run_generation(self, dataloader) -> List[Dict]:
        results = []
        
        for batch in tqdm(dataloader, desc="Generating answers"):
            # Generate answers
            outputs = self.llm.generate(
                batch['prompt'],
                max_new_tokens=self.config.max_new_tokens
            )
            
            # Process outputs
            batch_results = self._process_batch_outputs(batch, outputs)
            results.extend(batch_results)
            
            # Save checkpoint if needed
            if self.config.save_intermediates and len(results) % self.config.save_every == 0:
                self._save_checkpoint(results)
                
        return results

    def _process_batch_outputs(
        self,
        batch: Dict[str, Any],
        outputs: List[str]
    ) -> List[Dict[str, Any]]:
        processed_results = []
        answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
        
        for idx, output in enumerate(outputs):
            start_idx = output.find(answer_string) + len(answer_string)
            answer = output[start_idx:].strip()
            
            result = {
                'query': batch['query'][idx],
                'generated_answer': answer,
                'document_indices': batch['document_indices'][idx],
                'gold_document_idx': batch['gold_document_idx'][idx],
                'prompt_tokens_len': batch['prompt_tokens_len'][idx]
            }
            
            if self.config.use_clustering:
                result['cluster_assignments'] = batch.get('cluster_assignments', {})[idx]
                
            processed_results.append(result)
            
        return processed_results

    def _process_clustering_results(self, results: List[Dict]) -> List[Dict]:
        """Add clustering-specific metrics to results if clustering was used."""
        if not self.config.use_clustering:
            return results
            
        for result in results:
            if 'cluster_assignments' in result:
                cluster_stats = self._compute_cluster_stats(result['cluster_assignments'])
                result['cluster_metrics'] = cluster_stats
                
        return results

    def _compute_cluster_stats(self, cluster_assignments: Dict) -> Dict:
        """Compute statistics for document clusters."""
        stats = {
            'num_clusters': len(set(cluster_assignments.values())),
            'cluster_sizes': {},
            'cluster_distribution': {}
        }
        
        # Count documents per cluster
        for cluster_id in set(cluster_assignments.values()):
            size = sum(1 for v in cluster_assignments.values() if v == cluster_id)
            stats['cluster_sizes'][str(cluster_id)] = size
            stats['cluster_distribution'][str(cluster_id)] = size / len(cluster_assignments)
            
        return stats

    def _compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        metrics = {
            'total_examples': len(results),
            'correct_answers': sum(1 for r in results if r.get('ans_match_after_norm', False)),
            'avg_context_length': sum(len(r['document_indices']) for r in results) / len(results),
            'retriever_type': self.retriever_type
        }
        
        metrics['accuracy'] = metrics['correct_answers'] / metrics['total_examples']
        
        # Add clustering metrics if used
        if self.config.use_clustering:
            cluster_metrics = self._compute_clustering_metrics(results)
            metrics.update(cluster_metrics)
            
        return metrics

    def _compute_clustering_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute clustering-specific metrics."""
        cluster_metrics = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for result in results:
            if 'cluster_assignments' in result:
                for cluster_id in set(result['cluster_assignments'].values()):
                    metrics = cluster_metrics[str(cluster_id)]
                    metrics['total'] += 1
                    if result.get('ans_match_after_norm'):
                        metrics['correct'] += 1
        
        return {
            f'cluster_{k}_accuracy': v['correct']/v['total'] if v['total'] > 0 else 0
            for k, v in cluster_metrics.items()
        }

    def _save_checkpoint(self, results: List[Dict]):
        """Save intermediate results checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{len(results)}_{timestamp}.pkl"
        write_pickle(results, str(checkpoint_path))
        
        self.logger.experiment_logger.info(f"Saved checkpoint with {len(results)} results")

    def _save_results(
        self,
        results: List[Dict],
        metrics: Dict[str, Any],
        output_dir: Path
    ):
        """Save experiment results and metrics."""
        import json
        
        # Save results
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save metrics
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.experiment_logger.info(f"Saved results to {output_dir}")

    def _load_dataset(self) -> PromptDataset:
        """Load and prepare dataset with proper configuration."""
        try:
            self.logger.log_step_start("Loading dataset")
            
            # Load corpus based on retriever type
            self.logger.experiment_logger.info(f"Loading corpus for {self.retriever_type}")
            
            if self.config.use_bm25:
                self.logger.experiment_logger.info("Using BM25 corpus...")
                corpus, full_to_subset_idx_map = read_test_corpus_with_random_and_bm25()
            else:
                self.logger.experiment_logger.info("Using Contriever corpus...")
                corpus, full_to_subset_idx_map = read_corpus_with_contriever()
            
            self.logger.log_metric("corpus_size", len(corpus))
            
            # Load search results based on retriever type
            self.logger.experiment_logger.info("Loading search results...")
            
            if self.config.use_bm25:
                search_results = read_pickle(str(self.config.bm25_results_path))
            else:
                search_results = read_pickle(str(self.config.contriever_results_path))
                
            self.logger.log_metric("search_results_size", len(search_results))
            
            # Log key configuration parameters
            self.logger.log_metric("corpus_mapping_size", 
                                 len(full_to_subset_idx_map) if full_to_subset_idx_map else 0)
            
            # Prepare dataset arguments
            dataset_kwargs = {
                'corpus': corpus,
                'data_path': str(self.config.train_dataset_path),
                'tokenizer': self.llm.tokenizer,
                'max_tokenized_length': self.config.model_max_length - 2,
                'search_results': search_results,
                'num_documents_in_context': self.config.num_documents_in_context,
                'gold_position': self.config.gold_position,
                'get_documents_without_answer': self.config.get_documents_without_answer,
                'full_to_subset_idx_map': full_to_subset_idx_map
            }
            
            # Add clustering configuration if enabled
            if getattr(self.config, 'use_clustering', False):
                dataset_kwargs.update({
                    'use_clustering': True,
                    'num_clusters': self.config.num_clusters,
                    'cluster_seed': self.config.cluster_seed
                })
            
            # Log dataset configuration
            self.logger.log_metric("dataset_config", {
                k: str(v) if isinstance(v, Path) else v
                for k, v in dataset_kwargs.items()
            })
            
            # Create dataset
            dataset = PromptDataset(**dataset_kwargs)
            self.logger.experiment_logger.info(f"Created dataset with {len(dataset)} samples")
            
            self.logger.log_step_end("Loading dataset")
            return dataset
            
        except Exception as e:
            self.logger.log_error(e, "Error loading dataset")
            raise

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=['contriever', 'bm25', 'random'],
        required=True,
        help="Type of retriever to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/experiment0_baseline/results",
        help="Output directory"
    )
    return parser.parse_args()

def main(args=None):
    """Main entry point with silent execution."""
    try:
        if isinstance(args, dict):
            parser = argparse.ArgumentParser()
            parser.add_argument('--retriever', type=str)
            parser.add_argument('--gold_position', type=int)
            parser.add_argument('--num_documents', type=int)
            parser.add_argument('--use_random', type=bool, default=False)
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
    args = parse_arguments()
    config = BaselineConfigFactory.get_config_for_retriever(args.retriever)
    config.output_dir = Path(args.output_dir)
    main(config)