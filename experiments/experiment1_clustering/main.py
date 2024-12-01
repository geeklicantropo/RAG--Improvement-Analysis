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
import gc
import numpy as np

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from experiments.checkpoint_utils import (
    save_checkpoint,
    load_checkpoints, 
    get_last_checkpoint_batch,
    merge_checkpoint_results
)

from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything, read_corpus_json, read_pickle
from src.llm import LLM
from src.prompt_dataset import PromptDataset
from src.cluster_utils import DocumentClusterer
from .config import ClusteringConfig, ClusteringConfigFactory
from .utils import ClusteringExperimentUtils
from src.compute_corpus_embeddings import compute_embeddings_with_memory_tracking
from experiments.experiment1_clustering.utils import ClusteringExperimentUtils

class ClusteringExperiment:
    def __init__(
        self,
        config: ClusteringConfig,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.logger = logger or ExperimentLogger("clustering", str(Path(__file__).parent.parent.parent / "logs"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_memory_management()

        # Initialize LLM (e.g., Llama, Meta-llama)
        self.llm = LLM(model_id=config.llm_id, device=self.device)  # Ensure LLM initialization

    def _setup_memory_management(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_utilization)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def run_embedding_phase(self):
        """Compute document embeddings phase."""
        try:
            self.logger.log_step_start("Embedding Phase")
            
            # Load training dataset to compute embeddings
            train_corpus = read_corpus_json(self.config.train_dataset)  # Load the training dataset

            # Compute embeddings for the training data
            self.embeddings = compute_embeddings_with_memory_tracking(
                corpus=train_corpus, 
                subset_indices=list(range(len(train_corpus))),  # Or sample your corpus
                args=self.config,  # Include your config settings
                logger=self.logger
            )

            # Save embeddings to disk
            embeddings_path = Path(self.config.embeddings_output_dir) / f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
            np.save(embeddings_path, self.embeddings)
            
            self._save_phase_checkpoint("embeddings", {'embeddings_path': str(embeddings_path)})

            self.logger.log_step_end("Embedding Phase")
            return self.embeddings
        except Exception as e:
            self.logger.log_error(e, "Error in embedding phase")
            raise

    def run_clustering_phase(self):
        """Perform document clustering phase."""
        try:
            self.logger.log_step_start("Clustering Phase")
            
            # Check for clustering checkpoint
            checkpoint = self._get_phase_checkpoint("clustering")
            if checkpoint:
                self.logger.experiment_logger.info("Using existing clusters")
                self.clusters = checkpoint
                return self.clusters
            
            # Initialize clusterer
            self.clusterer = DocumentClusterer(
                num_clusters=self.config.num_clusters,
                random_seed=self.config.cluster_seed,
                use_scaler=self.config.use_scaler,
                min_cluster_size=self.config.min_cluster_size,
                logger=self.logger
            )
            
            # Perform clustering
            self.clusters = self.clusterer.fit_clusters(self.embeddings)
            
            # Handle noise injection into clusters if required
            if self.config.inject_noise:
                self.logger.log_step_start("Injecting Noise into Clusters")
                self.clusters = self._inject_noise_into_clusters(self.clusters, self._load_corpus())
                self.logger.log_step_end("Completed Noise Injection")

            self._save_phase_checkpoint("clustering", self.clusters)
            
            return self.clusters
            
        except Exception as e:
            self.logger.log_error(e, "Error in clustering phase")
            raise

    def run_generation_phase(self):
        """Run generation phase with clustering-aware prompts."""
        try:
            self.logger.log_step_start("Generation Phase")
            
            # Check for generation checkpoint
            checkpoint = self._get_phase_checkpoint("generation")
            if checkpoint:
                self.logger.experiment_logger.info("Using existing generation results")
                self.results = checkpoint
                return merge_checkpoint_results(self.results)
            
            # Initialize LLM
            self.llm = self._initialize_llm()  # Ensure LLM initialization before using
            
            # Create dataset with clustered documents
            dataset = self._create_dataset()
            dataloader = self._create_dataloader(dataset)
            
            # Run generation
            results = self._run_generation(dataloader)
            self._save_phase_checkpoint("generation", results)
            
            return merge_checkpoint_results(results)
            
        except Exception as e:
            self.logger.log_error(e, "Error in generation phase")
            raise

    def run_evaluation_phase(self):
        """Evaluate clustering experiment results using test dataset."""
        try:
            self.logger.log_step_start("Evaluation Phase")

            # Load test dataset
            train_corpus = read_corpus_json(str(self.config.train_dataset_path))

            # Initialize ClusteringEvaluation
            clustering_utils = ClusteringExperimentUtils(logger=self.logger)

            # Analyze clustering results (this could be expanded to include more metrics)
            cluster_analysis = clustering_utils.analyze_clustering_results(self.clusters, logger=self.logger)

            # Calculate clustering quality metrics
            clustering_metrics = clustering_utils.calculate_cluster_quality_metrics(
                embeddings=self.embeddings,  # Embeddings used in clustering
                cluster_labels=np.array([label for cluster in self.clusters.values() for label in [cluster['cluster_id']] * len(cluster['documents'])])
            )

            # Log clustering evaluation results
            self.logger.log_metric("clustering_metrics", clustering_metrics)
            self.logger.log_metric("cluster_analysis", cluster_analysis)

            self.logger.log_step_end("Evaluation Phase")
            return clustering_metrics

        except Exception as e:
            self.logger.log_error(e, "Error in evaluation phase")
            raise

    def _initialize_llm(self):
        """Initialize LLM with memory optimization settings."""
        return LLM(
            model_id=self.config.llm_id,
            device=self.device,
            quantization_bits=4,
            model_max_length=self.config.model_max_length
        )

    def _load_corpus(self, file_path: Path) -> List[Dict]:
        try:
            return read_corpus_json(str(file_path))
        except Exception as e:
            self.logger.error(f"Error loading corpus from {file_path}: {str(e)}")
            raise

    def _create_dataset(self) -> PromptDataset:
        """Create dataset with clustered documents."""
        return PromptDataset(
            corpus=self._load_corpus(),
            data_path=str(self.config.data_path),
            tokenizer=self.llm.tokenizer,
            max_tokenized_length=self.config.model_max_length - 2,
            use_clustering=True,
            cluster_assignments=self.clusters
        )

    def _create_dataloader(self, dataset: PromptDataset) -> torch.utils.data.DataLoader:
        """Create dataloader with memory monitoring."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.config.adjust_batch_sizes(current_memory)
            
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def _run_generation(self, dataloader: torch.utils.data.DataLoader) -> List[Dict]:
        results = []
        last_checkpoint = get_last_checkpoint_batch(self.output_dir / "checkpoints")
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch_idx <= last_checkpoint:
                continue
                
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            if (batch_idx + 1) % self.config.save_every == 0:
                save_checkpoint(results, batch_idx + 1, self.output_dir)
                results = []  # Clear processed results after checkpoint
                 
        if results:  # Save any remaining results
            save_checkpoint(results, len(dataloader), self.output_dir)
            
        return load_checkpoints(self.output_dir / "checkpoints")

    def _process_batch(self, batch: Dict) -> List[Dict]:
        """Process batch with memory optimization.""" 
        try:
            prompts = batch['prompt']
            outputs = self.llm.generate(prompts, max_new_tokens=self.config.max_new_tokens)
            
            processed_results = []
            answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
            
            for idx, output in enumerate(outputs):
                try:
                    start_idx = output.find(answer_string) + len(answer_string)
                    
                    result = {
                        'query': batch['query'][idx],
                        'generated_answer': output[start_idx:].strip(),
                        'cluster_assignments': batch['cluster_assignments'][idx],
                        'document_indices': [int(i) for i in batch['document_indices'][idx]],
                        'prompt_tokens_len': int(batch['prompt_tokens_len'][idx])
                    }
                    processed_results.append(result)
                    
                except Exception as e:
                    self.logger.log_error(e, f"Error processing batch item {idx}")
                    continue
                    
            # Memory cleanup
            if torch.cuda.is_available():
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        del v
                torch.cuda.empty_cache()
            
            return processed_results
            
        except Exception as e:
            self.logger.log_error(e, f"Error processing batch")
            return []

    def _get_phase_checkpoint(self, phase: str) -> Optional[Dict]:
        """Check for phase-specific checkpoint."""
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
        """Save phase-specific checkpoint.""" 
        checkpoint_dir = self.output_dir / "checkpoints" / phase
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"{phase}_checkpoint_{timestamp}.json"
        
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _inject_noise_into_clusters(self, cluster_results: List[Dict], corpus: List[Dict]) -> List[Dict]:
        """Inject noise into clusters."""
        noisy_clusters = []
        for cluster in cluster_results:
            cluster_docs = cluster["documents"]
            noise_docs = self._sample_noise(corpus, len(cluster_docs))
            noisy_clusters.append({
                "cluster_id": cluster["cluster_id"],
                "documents": cluster_docs + noise_docs
            })
        return noisy_clusters

    def _sample_noise(self, corpus: List[Dict], num_samples: int) -> List[Dict]:
        """Sample noise documents from the corpus.""" 
        return np.random.choice(corpus, num_samples, replace=False).tolist()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run clustering experiment")
    parser.add_argument(
        '--num_clusters',
        type=int,
        default=5,
        help='Number of clusters'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/experiment1_clustering/results',
        help='Output directory for results'
    )
    return parser.parse_args()

def main(experiment_args=None):
    try:
        if experiment_args is None:
            args = parse_arguments()
        elif isinstance(experiment_args, dict):
            parser = argparse.ArgumentParser()
            parser.add_argument('--num_clusters', type=int)
            args = parser.parse_args([])
            for k, v in experiment_args.items():
                setattr(args, k, v)
        else:
            args = experiment_args

        config = ClusteringConfig()
        config.num_clusters = args.num_clusters
        experiment = ClusteringExperiment(config)
        
        # Run phases sequentially
        experiment.run_embedding_phase()
        experiment.run_clustering_phase()
        experiment.run_generation_phase()
        metrics = experiment.run_evaluation_phase()
        
        return experiment.results, metrics

    except Exception as e:
        logging.error(f"Error in clustering experiment: {str(e)}")
        raise

if __name__ == "__main__":
    main()
