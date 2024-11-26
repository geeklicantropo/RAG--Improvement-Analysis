import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from tqdm import tqdm
import logging

class RAGFusionRanker:
    """
    A class for performing RAG-Fusion ranking by combining results from multiple retrievers.
    Implements Reciprocal Rank Fusion and score normalization techniques.
    """
    
    def __init__(
        self,
        k: float = 60.0,
        strategy: str = "rrf",
        normalize_scores: bool = True,
        score_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the RAG-Fusion ranker.
        
        Args:
            k: Constant for RRF calculation. Higher values diminish the impact of rank differences.
            strategy: Fusion strategy ('rrf' for Reciprocal Rank Fusion or 'linear' for linear combination)
            normalize_scores: Whether to normalize scores before fusion
            score_weights: Optional weights for different retrievers (e.g., {'contriever': 0.7, 'bm25': 0.3})
        """
        self.k = k
        self.strategy = strategy
        self.normalize_scores = normalize_scores
        self.score_weights = score_weights or {}
        
        if strategy not in ["rrf", "linear"]:
            raise ValueError("Invalid fusion strategy. Must be 'rrf' or 'linear'.")


    def normalize_score_list(
        self, 
        scores: List[float], 
        min_score: Optional[float] = None, 
        max_score: Optional[float] = None
    ) -> List[float]:
        """
        Normalize scores to [0,1] range using min-max normalization.
        
        Args:
            scores: List of scores to normalize
            min_score: Optional minimum score for normalization
            max_score: Optional maximum score for normalization
            
        Returns:
            List of normalized scores
        """
        if min_score is None:
            min_score = min(scores)
        if max_score is None:
            max_score = max(scores)
            
        if max_score == min_score:
            return [1.0] * len(scores)
            
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def reciprocal_rank_fusion(self, results_list: List[Tuple[List[str], List[float]]]) -> List[Tuple[List[str], List[float]]]:
        """Implement reciprocal rank fusion."""
        fused_results = []
        
        for query_results in zip(*results_list):
            fused_scores = {}
            for rank, (doc_id, score) in enumerate(query_results):
                fused_score = 1 / (rank + self.k)
                if doc_id in fused_scores:
                    fused_scores[doc_id] += fused_score
                else:
                    fused_scores[doc_id] = fused_score
                    
            # Sort by fused scores
            sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            doc_ids, scores = zip(*sorted_results)
            fused_results.append((list(doc_ids), list(scores)))
            
        return fused_results
    
    def _compute_rrf_scores(
        self, 
        doc_ranks: Dict[str, int]
    ) -> float:
        """
        Compute RRF score for a document based on its ranks from different retrievers.
        
        Args:
            doc_ranks: Dictionary mapping retriever names to document ranks
            
        Returns:
            Combined RRF score
        """
        rrf_score = 0.0
        for retriever, rank in doc_ranks.items():
            weight = self.score_weights.get(retriever, 1.0)
            rrf_score += weight * (1.0 / (self.k + rank))
        return rrf_score

    def fuse_search_results(self, retriever_results: Dict[str, List[Tuple[List[str], List[float]]]]) -> List[Tuple[List[str], List[float]]]:
        """
        Safely fuse search results with validation.
        """
        try:
            # Validate input format
            if not isinstance(retriever_results, dict):
                raise ValueError("Retriever results must be a dictionary")
                
            for name, results in retriever_results.items():
                if not isinstance(results, list):
                    raise ValueError(f"Results for {name} must be a list")
                if not results:
                    raise ValueError(f"Empty results for {name}")
                
                # Validate format of each result
                for result in results:
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise ValueError(f"Invalid result format for {name}")
                    doc_ids, scores = result
                    if not isinstance(doc_ids, list) or not isinstance(scores, list):
                        raise ValueError(f"Invalid doc_ids/scores format for {name}")
                    if len(doc_ids) != len(scores):
                        raise ValueError(f"Mismatched lengths for doc_ids/scores in {name}")

            # Process results
            fused_results = []
            num_queries = len(next(iter(retriever_results.values())))
            
            for query_idx in range(num_queries):
                query_results = {
                    name: results[query_idx] 
                    for name, results in retriever_results.items()
                }
                
                # Normalize scores if configured
                if self.normalize_scores:
                    for name in query_results:
                        doc_ids, scores = query_results[name]
                        min_score = min(scores) if scores else 0
                        max_score = max(scores) if scores else 1
                        norm_scores = (
                            [(s - min_score) / (max_score - min_score) if max_score > min_score else 0.5 
                            for s in scores]
                        )
                        query_results[name] = (doc_ids, norm_scores)
                
                # Apply fusion strategy
                if self.strategy == 'rrf':
                    fused_result = self.reciprocal_rank_fusion(query_results)
                else:
                    fused_result = self._linear_combination(query_results)
                    
                fused_results.append(fused_result)
                
            return fused_results
            
        except Exception as e:
            logging.error(f"Error during fusion: {str(e)}")
            raise

    def rescore_with_clusters(
        self,
        fused_results: List[Tuple[List[str], List[float]]],
        cluster_info: Dict[str, int],
        cluster_boost: float = 0.1
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Adjust scores based on cluster information to promote diversity.
        
        Args:
            fused_results: List of (doc_ids, scores) tuples
            cluster_info: Mapping of document IDs to cluster assignments
            cluster_boost: Score boost for documents from underrepresented clusters
            
        Returns:
            Rescored results considering cluster diversity
        """
        rescored_results = []
        
        for doc_ids, scores in fused_results:
            # Track cluster frequencies
            cluster_counts = {}
            rescored_doc_scores = []
            
            # Process documents in order
            for doc_id, score in zip(doc_ids, scores):
                if doc_id in cluster_info:
                    cluster = cluster_info[doc_id]
                    # Boost score if cluster is underrepresented
                    boost = cluster_boost / (cluster_counts.get(cluster, 0) + 1)
                    adjusted_score = score * (1 + boost)
                    cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
                else:
                    adjusted_score = score
                    
                rescored_doc_scores.append((doc_id, adjusted_score))
            
            # Sort by adjusted scores
            rescored_doc_scores.sort(key=lambda x: x[1], reverse=True)
            new_doc_ids, new_scores = zip(*rescored_doc_scores)
            
            rescored_results.append((list(new_doc_ids), list(new_scores)))
            
        return rescored_results