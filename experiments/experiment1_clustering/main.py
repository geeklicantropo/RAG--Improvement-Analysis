import os
import argparse
import warnings
from typing import Dict, List, Any
import torch
from tqdm import tqdm
from experiment_logger import ExperimentLogger
from utils import seed_everything
from cluster_utils import DocumentClusterer
from config import ClusteringConfig

import time
from collections import defaultdict
import numpy as np
import json

import sys
sys.path.append('../..')
from src.llm import LLM
from src.prompt_dataset import PromptDataset
from src.retriever import Retriever
from src.compute_corpus_embeddings import compute_embeddings

class ClusteringExperiment:
    """
    Implements experiment workflow for testing K-means clustering impact on RAG systems.
    """
    def __init__(self, config: ClusteringConfig, logger: ExperimentLogger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize core components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize LLM, retriever, and clustering components."""
        self.logger.log_step_start("Initializing components")
        
        # Initialize LLM
        self.llm = LLM(
            self.config.llm_id,
            self.device,
            quantization_bits=4,
            model_max_length=self.config.model_max_length
        )
        
        # Initialize clusterer
        self.clusterer = DocumentClusterer(
            num_clusters=self.config.num_clusters,
            random_seed=self.config.cluster_seed,
            use_scaler=self.config.use_scaler,
            logger=self.logger
        )
        
        # Initialize retriever if needed
        if self.config.compute_new_embeddings:
            self.retriever = Retriever(
                device=self.device,
                tokenizer=self.llm.tokenizer,
                max_length=self.config.max_length_encoder
            )
            
        self.logger.log_step_end("Initializing components", time.time())

    def run_experiment(self):
        """Execute the clustering experiment workflow."""
        try:
            # Load or compute document embeddings
            embeddings = self._prepare_embeddings()
            
            # Perform document clustering
            clusters = self._cluster_documents(embeddings)
            
            # Generate answers with clustered documents
            results = self._generate_answers(clusters)
            
            # Evaluate results
            metrics = self._evaluate_results(results)
            
            return results, metrics
            
        except Exception as e:
            self.logger.log_error(e, "Error during experiment execution")
            raise

    def _prepare_embeddings(self) -> np.ndarray:
        """Load or compute document embeddings for clustering."""
        self.logger.log_step_start("Preparing embeddings")
        
        if self.config.compute_new_embeddings:
            embeddings = compute_embeddings(
                self.retriever,
                self.config.corpus_path,
                self.config.batch_size,
                self.logger
            )
        else:
            embeddings = np.load(self.config.embeddings_path)
            
        self.logger.log_step_end("Preparing embeddings", time.time())
        return embeddings

    def _cluster_documents(self, embeddings: np.ndarray) -> Dict[int, List[int]]:
        """Perform K-means clustering on document embeddings."""
        self.logger.log_step_start("Clustering documents")
        
        clusters = self.clusterer.fit_clusters(embeddings)
        
        # Log clustering statistics
        self.logger.log_metric("num_clusters", len(clusters))
        for cluster_id, docs in clusters.items():
            self.logger.log_metric(f"cluster_{cluster_id}_size", len(docs))
            
        self.logger.log_step_end("Clustering documents", time.time())
        return clusters

    def _generate_answers(self, clusters: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        """Generate answers using clustered document organization."""
        self.logger.log_step_start("Generating answers")
        
        # Initialize dataset with clustering
        dataset = PromptDataset(
            corpus=self.config.corpus,
            data_path=self.config.queries_path,
            tokenizer=self.llm.tokenizer,
            max_tokenized_length=self.config.model_max_length - 2,
            search_results=self.config.search_results,
            cluster_assignments=clusters,
            num_clusters=self.config.num_clusters,
            use_clustering=True
        )
        
        results = []
        for batch in tqdm(dataset, desc="Generating answers"):
            responses = self.llm.generate(
                batch["prompts"],
                max_new_tokens=self.config.max_new_tokens
            )
            
            for response in responses:
                results.append({
                    "query": batch["query"],
                    "response": response,
                    "cluster_info": batch["cluster_assignments"]
                })
                
        self.logger.log_step_end("Generating answers", time.time())
        return results

    def _evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate generation results with clustering-specific metrics."""
        self.logger.log_step_start("Evaluating results")
        
        metrics = {}
        
        # Calculate overall accuracy
        correct = sum(1 for r in results if r["response"]["correct"])
        metrics["accuracy"] = correct / len(results)
        
        # Calculate per-cluster metrics
        cluster_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in results:
            cluster_id = result["cluster_info"]["primary_cluster"]
            cluster_metrics[cluster_id]["total"] += 1
            if result["response"]["correct"]:
                cluster_metrics[cluster_id]["correct"] += 1
                
        for cluster_id, stats in cluster_metrics.items():
            accuracy = stats["correct"] / stats["total"]
            metrics[f"cluster_{cluster_id}_accuracy"] = accuracy
            
        self.logger.log_metrics(metrics)
        self.logger.log_step_end("Evaluating results", time.time())
        return metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run clustering experiment")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to experiment configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = ClusteringConfig.load(args.config_path)
    
    # Initialize logger
    logger = ExperimentLogger(
        experiment_name="clustering_experiment",
        base_log_dir="logs"
    )
    
    try:
        with logger:
            # Run experiment
            experiment = ClusteringExperiment(config, logger)
            results, metrics = experiment.run_experiment()
            
            # Save results
            os.makedirs(config.output_dir, exist_ok=True)
            with open(os.path.join(config.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
                
            with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
                
    except Exception as e:
        logger.log_error(e, "Error in main execution")
        raise

if __name__ == "__main__":
    seed_everything(10)
    main()