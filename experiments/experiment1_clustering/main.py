import os
import argparse
import warnings
from typing import Dict, List, Any, Tuple, Optional
import torch
from transformers import AutoConfig, AutoModel

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import sys
import logging
import gc

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.experiment_logger import ExperimentLogger
from src.utils import seed_everything
from src.cluster_utils import DocumentClusterer
from experiments.experiment1_clustering.config import ClusteringConfig

import time
from collections import defaultdict
import numpy as np
import json

from src.llm import LLM
from src.prompt_dataset import PromptDataset
from src.retriever import Retriever


class ClusteringExperiment:
    def __init__(
        self,
        config: ClusteringConfig,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.logger = logger or ExperimentLogger(
            experiment_name=f"clustering_{config.num_clusters}clusters",
            base_log_dir="logs"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize core components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize LLM, retriever, and clustering components."""
        self.logger.log_step_start("Initializing components")
        
        try:
            # Initialize LLM
            self.llm = LLM(
                self.config.llm_id,
                self.device,
                quantization_bits=4,
                model_max_length=self.config.model_max_length
            )
            
            # Initialize encoder for retriever
            config = AutoConfig.from_pretrained('facebook/contriever')
            self.encoder = AutoModel.from_config(config).eval()
            
            # Initialize retriever with encoder
            if self.config.compute_new_embeddings:
                self.retriever = Retriever(
                    device=self.device,
                    tokenizer=self.llm.tokenizer,
                    query_encoder=self.encoder,
                    max_length=self.config.max_length_encoder
                )
            
            # Initialize clusterer
            self.clusterer = DocumentClusterer(
                num_clusters=self.config.num_clusters,
                random_seed=self.config.cluster_seed,
                use_scaler=self.config.use_scaler,
                logger=self.logger
            )
            
            self.logger.log_step_end("Initializing components")
            
        except Exception as e:
            self.logger.log_error(e, "Error initializing components")
            raise

    def run(self):
        """Run the clustering experiment with memory optimization."""
        try:
            self.logger.log_step_start("Experiment execution")
            
            # Monitor GPU memory
            if torch.cuda.is_available():
                gpu = torch.cuda.current_device()
                initial_memory = torch.cuda.memory_allocated(gpu) / torch.cuda.max_memory_allocated(gpu)
                self.config.adjust_batch_sizes(initial_memory)

            # Create timestamp for results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir) / f"run_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load or compute embeddings
            embeddings = self._prepare_embeddings(output_dir)
            
            # Cluster documents in batches
            clusters = self._cluster_documents(embeddings)
            
            # Initialize dataset with clustering
            dataset = self._initialize_dataset(clusters)
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
            
            results = []
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
                # Monitor memory
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated(gpu) / torch.cuda.max_memory_allocated(gpu)
                    self.config.adjust_batch_sizes(current_memory)
                
                # Generate answers
                prompts = batch['prompt']
                generated_outputs = self.llm.generate(
                    prompts,
                    max_new_tokens=self.config.max_new_tokens
                )
                
                # Process batch results
                batch_results = self._process_batch_outputs(batch, generated_outputs)
                results.extend(batch_results)
                
                # Save checkpoints
                if self.config.save_intermediates and (batch_idx + 1) % self.config.save_every == 0:
                    self._save_checkpoint(results, batch_idx + 1)

                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Adjust batch size if memory high
                if self.logger.check_memory_threshold():
                    self.config.batch_size = max(1, self.config.batch_size // 2)
            
            # Evaluate and save results
            metrics = self._evaluate_results(results)
            self._save_experiment_artifacts(results, metrics, clusters, output_dir)
            
            self.logger.log_step_end("Experiment execution")
            return results, metrics
            
        except Exception as e:
            self.logger.log_error(e, "Error during experiment execution")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _initialize_dataset(self, clusters: Dict[int, List[int]]) -> PromptDataset:
        """Initialize dataset with clustering."""
        return PromptDataset(
            corpus=self._load_corpus(),
            data_path=str(self.config.queries_path),
            tokenizer=self.llm.tokenizer,
            max_tokenized_length=self.config.model_max_length - 2,
            search_results=self._load_search_results(),
            use_clustering=True,
            cluster_assignments=clusters,
            num_clusters=self.config.num_clusters
        )

    def _prepare_embeddings(self, output_dir: Path) -> np.ndarray:
        """Load existing embeddings or compute new ones."""
        self.logger.log_step_start("Preparing embeddings")
        
        try:
            if self.config.compute_new_embeddings:
                # Compute new embeddings with timestamp
                embeddings = self._compute_new_embeddings()
                embeddings_path = output_dir / f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
                np.save(embeddings_path, embeddings)
                self.logger.experiment_logger.info(f"Saved new embeddings to {embeddings_path}")
            else:
                embeddings = np.load(self.config.embeddings_path)
            
            self.logger.log_metric("embeddings_shape", embeddings.shape)
            self.logger.log_step_end("Preparing embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.log_error(e, "Error preparing embeddings")
            raise

    def _cluster_documents(self, embeddings: np.ndarray) -> Dict[int, List[int]]:
        """Perform clustering with memory management."""
        self.logger.log_step_start("Clustering documents")
        
        try:
            clusters = {}
            embeddings_batch_size = self.config.clustering_batch_size

            # Process embeddings in batches
            for start_idx in range(0, len(embeddings), embeddings_batch_size):
                end_idx = min(start_idx + embeddings_batch_size, len(embeddings))
                batch_embeddings = embeddings[start_idx:end_idx]
                
                # Fit clusters for batch
                batch_clusters = self.clusterer.fit_clusters(batch_embeddings)
                
                # Merge with existing clusters
                for cluster_id, doc_indices in batch_clusters.items():
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].extend([idx + start_idx for idx in doc_indices])
                
                # Memory cleanup
                del batch_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Log cluster stats
            cluster_sizes = {k: len(v) for k, v in clusters.items()}
            self.logger.log_metric("cluster_sizes", cluster_sizes)
            
            return clusters
        except Exception as e:
            self.logger.log_error(e, "Error during clustering")
            raise

    def _compute_cluster_stats(
        self, 
        clusters: Dict[int, List[int]], 
        embeddings: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistical measures for clusters."""
        stats = {}
        
        # Calculate intra-cluster distances
        for cluster_id, doc_indices in clusters.items():
            cluster_embeddings = embeddings[doc_indices]
            centroid = cluster_embeddings.mean(axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            stats[f"cluster_{cluster_id}_avg_distance"] = float(distances.mean())
            stats[f"cluster_{cluster_id}_std_distance"] = float(distances.std())
            
        return stats

    def _generate_answers(self, clusters: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        """Generate answers using clustered document organization."""
        self.logger.log_step_start("Generating answers")
        
        try:
            # Initialize dataset with clustering
            dataset = PromptDataset(
                corpus=self._load_corpus(),
                data_path=str(self.config.queries_path),
                tokenizer=self.llm.tokenizer,
                max_tokenized_length=self.config.model_max_length - 2,
                search_results=self._load_search_results(),
                use_clustering=True,
                cluster_assignments=clusters,
                num_clusters=self.config.num_clusters
            )
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
            
            results = []
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating answers")):
                # Generate answers
                responses = self.llm.generate(
                    batch["prompt"],
                    max_new_tokens=self.config.max_new_tokens
                )
                
                # Process batch results
                batch_results = self._process_batch_outputs(batch, responses)
                results.extend(batch_results)
                
                # Save checkpoint if needed
                if self.config.save_intermediates and (batch_idx + 1) % 100 == 0:
                    self._save_checkpoint(results, batch_idx + 1)
            
            self.logger.log_step_end("Generating answers")
            return results
            
        except Exception as e:
            self.logger.log_error(e, "Error generating answers")
            raise

    def _process_batch_outputs(
        self,
        batch: Dict[str, torch.Tensor],
        responses: List[str]
    ) -> List[Dict[str, Any]]:
        """Process generation outputs for a batch."""
        processed_results = []
        answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
        
        for idx, response in enumerate(responses):
            # Extract answer
            start_idx = response.find(answer_string) + len(answer_string)
            answer = response[start_idx:].strip()
            
            # Create result dictionary
            result = {
                "query": batch["query"][idx],
                "generated_answer": answer,
                "cluster_info": {
                    "primary_cluster": batch["cluster_assignments"][idx],
                    "document_clusters": batch["document_clusters"][idx]
                },
                "prompt_tokens_len": batch["prompt_tokens_len"][idx],
                "document_indices": batch["document_indices"][idx]
            }
            processed_results.append(result)
            
        return processed_results

    def _evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate generation results with clustering-specific metrics."""
        self.logger.log_step_start("Evaluating results")
        
        try:
            metrics = {}
            
            # Overall metrics
            metrics["total_examples"] = len(results)
            metrics["avg_prompt_length"] = sum(r["prompt_tokens_len"] for r in results) / len(results)
            
            # Per-cluster metrics
            cluster_metrics = defaultdict(lambda: {"total": 0, "correct": 0, "avg_length": 0})
            
            for result in results:
                cluster_id = result["cluster_info"]["primary_cluster"]
                metrics = cluster_metrics[cluster_id]
                metrics["total"] += 1
                metrics["avg_length"] += len(result["generated_answer"])
                if result.get("ans_match_after_norm", False):
                    metrics["correct"] += 1
                    
            # Calculate aggregate metrics
            for cluster_id, metrics in cluster_metrics.items():
                accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
                avg_length = metrics["avg_length"] / metrics["total"] if metrics["total"] > 0 else 0
                
                metrics[f"cluster_{cluster_id}_accuracy"] = accuracy
                metrics[f"cluster_{cluster_id}_avg_length"] = avg_length
            
            self.logger.log_metrics(metrics)
            self.logger.log_step_end("Evaluating results")
            return metrics
            
        except Exception as e:
            self.logger.log_error(e, "Error evaluating results")
            raise

    def _save_experiment_artifacts(
        self,
        results: List[Dict],
        metrics: Dict[str, Any],
        clusters: Dict[int, List[int]],
        output_dir: Path
    ):
        """Save all experiment artifacts."""
        try:
            # Save results
            results_path = output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            
            # Save metrics
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save cluster assignments
            clusters_path = output_dir / "clusters.json"
            with open(clusters_path, "w") as f:
                json.dump({str(k): v for k, v in clusters.items()}, f, indent=2)
            
            # Save configuration
            config_path = output_dir / "config.yaml"
            self.config.save(str(config_path))
            
            self.logger.experiment_logger.info(f"Saved all artifacts to {output_dir}")
            
        except Exception as e:
            self.logger.log_error(e, "Error saving artifacts")
            raise

    def _load_corpus(self):
        """Load corpus from existing file."""
        from src.utils import read_corpus_json
        return read_corpus_json(str(self.config.corpus_path))
    
    def _load_search_results(self):
        """Load search results from existing file."""
        from src.utils import read_pickle
        return read_pickle(str(self.config.search_results_path))

    def _save_checkpoint(self, results: List[Dict], batch_idx: int):
        """Save generation checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{batch_idx}_{timestamp}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=2)
            
        self.logger.experiment_logger.info(f"Saved checkpoint at batch {batch_idx}")

def parse_arguments():
    """Parse command line arguments for clustering experiment."""
    parser = argparse.ArgumentParser(description="Run clustering experiment")
    
    parser.add_argument(
        '--num_clusters',
        type=int,
        default=5,
        help='Number of clusters'
    )
    parser.add_argument(
        '--use_random',
        action='store_true',
        help='Whether to use random documents'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/experiment1_clustering/results',
        help='Output directory for results'
    )
    
    return parser.parse_args()


class ClusteringConfigFactory:
    """Factory class for creating clustering configurations."""
    @staticmethod
    def get_config(num_clusters: int, use_random: bool) -> ClusteringConfig:
        """Create clustering configuration."""
        return ClusteringConfig(
            num_clusters=num_clusters,
            use_random=use_random
        )

def main(args=None):
    try:
        # Set up args
        if isinstance(args, dict):
            parser = argparse.ArgumentParser()
            parser.add_argument('--num_clusters', type=int)
            parser.add_argument('--use_random', type=bool, default=False)
            namespace = parser.parse_args([])
            for k, v in args.items():
                setattr(namespace, k, v)
            args = namespace
        else:
            args = parse_arguments()

        # Run experiment
        config = ClusteringConfigFactory.get_config(args.num_clusters, args.use_random)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            experiment = ClusteringExperiment(
                config=config,
                logger=ExperimentLogger(f"clustering_{args.num_clusters}")
            )
            experiment.setup()
            results = experiment.run()
            
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
        return results
            
    except Exception as e:
        logging.error(f"Error in clustering experiment: {str(e)}", exc_info=True) 
        raise
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()