import os
import json
import pickle
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import numpy as np
from collections import defaultdict
import time

from src.experiment_logger import ExperimentLogger
from src.rag_fusion_utils import RAGFusionRanker

class FusionExperimentUtils:
    """Utilities for RAG-Fusion experiments."""
    
    @staticmethod
    def load_retrieval_results(
        contriever_path: str,
        bm25_path: str,
        logger: ExperimentLogger
    ) -> Dict[str, List[Tuple[List[str], List[float]]]]:
        """
        Load retrieval results from both retrievers.
        
        Args:
            contriever_path: Path to Contriever results
            bm25_path: Path to BM25 results
            logger: Logger instance
            
        Returns:
            Dictionary mapping retriever names to their results
        """
        try:
            logger.log_step_start("Loading retrieval results")
            
            # Load Contriever results
            with open(contriever_path, 'rb') as f:
                contriever_results = pickle.load(f)
            logger.log_metric("contriever_results_size", len(contriever_results))
            
            # Load BM25 results
            with open(bm25_path, 'rb') as f:
                bm25_results = pickle.load(f)
            logger.log_metric("bm25_results_size", len(bm25_results))
            
            retriever_results = {
                'contriever': contriever_results,
                'bm25': bm25_results
            }
            
            logger.log_step_end("Loading retrieval results", time.time())
            return retriever_results
            
        except Exception as e:
            logger.log_error(e, "Error loading retrieval results")
            raise
            
    @staticmethod
    def initialize_fusion_ranker(
        strategy: str,
        k: float,
        normalize_scores: bool,
        weights: Dict[str, float],
        logger: ExperimentLogger
    ) -> RAGFusionRanker:
        """
        Initialize RAG-Fusion ranker with specified parameters.
        
        Args:
            strategy: Fusion strategy ('rrf' or 'linear')
            k: RRF k parameter
            normalize_scores: Whether to normalize scores
            weights: Retriever score weights
            logger: Logger instance
            
        Returns:
            Configured RAGFusionRanker instance
        """
        try:
            logger.log_step_start("Initializing fusion ranker")
            
            ranker = RAGFusionRanker(
                k=k,
                strategy=strategy,
                normalize_scores=normalize_scores,
                score_weights=weights
            )
            
            logger.log_metric("fusion_config", {
                "strategy": strategy,
                "k": k,
                "normalize_scores": normalize_scores,
                "weights": weights
            })
            
            logger.log_step_end("Initializing fusion ranker", time.time())
            return ranker
            
        except Exception as e:
            logger.log_error(e, "Error initializing fusion ranker")
            raise
            
    @staticmethod
    def analyze_fusion_results(
        results: List[Dict],
        logger: ExperimentLogger
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze results of fusion experiment.
        
        Args:
            results: List of generation results
            logger: Logger instance
            
        Returns:
            Dictionary of analysis metrics
        """
        try:
            logger.log_step_start("Analyzing fusion results")
            
            analysis = {
                'overall': defaultdict(float),
                'per_retriever': defaultdict(lambda: defaultdict(float))
            }
            
            total_queries = len(results)
            correct_answers = 0
            retriever_contributions = defaultdict(int)
            
            for result in results:
                # Track overall accuracy
                if result['ans_match_after_norm']:
                    correct_answers += 1
                
                # Track which retriever contributed most
                if 'retriever_scores' in result:
                    max_retriever = max(
                        result['retriever_scores'].items(),
                        key=lambda x: x[1]
                    )[0]
                    retriever_contributions[max_retriever] += 1
            
            # Calculate overall metrics
            analysis['overall'] = {
                'accuracy': correct_answers / total_queries,
                'total_queries': total_queries,
                'correct_answers': correct_answers
            }
            
            # Calculate per-retriever metrics
            for retriever, count in retriever_contributions.items():
                analysis['per_retriever'][retriever] = {
                    'contribution_rate': count / total_queries
                }
            
            logger.log_metrics(analysis)
            logger.log_step_end("Analyzing fusion results", time.time())
            return analysis
            
        except Exception as e:
            logger.log_error(e, "Error analyzing fusion results")
            raise
            
    @staticmethod
    def save_fusion_artifacts(
        results: List[Dict],
        metrics: Dict[str, Any],
        fusion_info: Dict[str, Any],
        output_dir: str,
        logger: ExperimentLogger
    ) -> None:
        """
        Save fusion experiment artifacts.
        
        Args:
            results: Generation results
            metrics: Performance metrics
            fusion_info: Fusion configuration and statistics
            output_dir: Output directory
            logger: Logger instance
        """
        try:
            logger.log_step_start("Saving fusion artifacts")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results
            results_path = os.path.join(output_dir, 'fusion_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            # Save metrics
            metrics_path = os.path.join(output_dir, 'fusion_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            # Save fusion information
            fusion_path = os.path.join(output_dir, 'fusion_info.json')
            with open(fusion_path, 'w') as f:
                json.dump(fusion_info, f, indent=2)
                
            logger.log_metric('results_path', results_path)
            logger.log_metric('metrics_path', metrics_path)
            logger.log_metric('fusion_path', fusion_path)
            
            logger.log_step_end("Saving fusion artifacts", time.time())
            
        except Exception as e:
            logger.log_error(e, "Error saving fusion artifacts")
            raise
            
    @staticmethod
    def inject_random_documents(
        fused_results: List[Tuple[List[str], List[float]]],
        random_results_path: str,
        ratio: float,
        logger: ExperimentLogger
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Inject random documents into fused results.
        
        Args:
            fused_results: Original fused results
            random_results_path: Path to random documents file
            ratio: Ratio of random documents to inject
            logger: Logger instance
            
        Returns:
            Modified results with random documents
        """
        try:
            logger.log_step_start("Injecting random documents")
            
            # Load random documents
            with open(random_results_path, 'rb') as f:
                random_docs = pickle.load(f)
            
            # Calculate number of random documents to inject
            docs_per_query = len(fused_results[0][0])
            num_random = int(docs_per_query * ratio)
            
            logger.log_metric("num_random_docs", num_random)
            
            # Modify results
            modified_results = []
            for (doc_ids, scores), random_result in zip(fused_results, random_docs):
                # Select random documents
                random_ids = random_result[0][:num_random]
                random_scores = random_result[1][:num_random]
                
                # Combine with original results
                combined_ids = doc_ids[:-num_random] + random_ids
                combined_scores = scores[:-num_random] + random_scores
                
                modified_results.append((combined_ids, combined_scores))
            
            logger.log_step_end("Injecting random documents", time.time())
            return modified_results
            
        except Exception as e:
            logger.log_error(e, "Error injecting random documents")
            raise