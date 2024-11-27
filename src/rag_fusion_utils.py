import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any
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
    
    def reciprocal_rank_fusion(
        self, 
        results_list: List[Union[Tuple[List[str], List[float]], Dict]]
    ) -> List[Tuple[List[str], List[float]]]:
        """Implement reciprocal rank fusion with better error handling."""
        fused_results = []
        
        try:
            for query_results in zip(*results_list):
                fused_scores = {}
                
                for rank, result in enumerate(query_results):
                    # Handle different result formats
                    if isinstance(result, tuple) and len(result) == 2:
                        doc_ids, scores = result
                    elif isinstance(result, dict) and 'hits' in result and 'scores' in result:
                        doc_ids = [str(hit['docid']) for hit in result['hits']]
                        scores = [float(hit['score']) for hit in result['scores']]
                    else:
                        continue
                        
                    fused_score = 1 / (rank + self.k)
                    for doc_id in doc_ids:
                        if doc_id in fused_scores:
                            fused_scores[doc_id] += fused_score
                        else:
                            fused_scores[doc_id] = fused_score
                        
                # Sort by fused scores
                sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
                doc_ids, scores = zip(*sorted_results) if sorted_results else ([], [])
                fused_results.append((list(doc_ids), list(scores)))
            
            return fused_results
            
        except Exception as e:
            raise ValueError(f"Error during fusion: {str(e)}")
    
    def _linear_combination(
        self, 
        results_list: List[Tuple[List[str], List[float]]]
    ) -> Tuple[List[str], List[float]]:
        """Implement linear score combination."""
        fused_scores = {}
        for retriever, (doc_ids, scores) in enumerate(results_list):
            weight = self.score_weights.get(f"retriever{retriever}", 1.0)
            for doc_id, score in zip(doc_ids, scores):
                if doc_id in fused_scores:
                    fused_scores[doc_id] += score * weight
                else:
                    fused_scores[doc_id] = score * weight
                    
        # Sort by fused scores
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        doc_ids, scores = zip(*sorted_results)
        return list(doc_ids), list(scores)
    
    def fuse_search_results(
        self, 
        retriever_results: Dict[str, List[Union[Dict, Tuple[List[str], List[float]]]]]) -> List[Tuple[List[str], List[float]]]:
        """
        Safely fuse search results with validation and BM25 handling.
        
        Args:
            retriever_results: Dictionary mapping retriever names to their search results.
                              The results can be in the form of a list of (doc_ids, scores) tuples
                              or a list of dicts with 'hits' and 'scores' keys (for BM25 format).
        
        Returns:
            List of (doc_ids, scores) tuples for the fused results.
        """
        try:
            # Validate input format
            if not isinstance(retriever_results, dict):
                raise ValueError("Retriever results must be a dictionary")
                
            normalized_results = {}
            for name, results in retriever_results.items():
                if not isinstance(results, list):
                    raise ValueError(f"Results for {name} must be a list")
                if not results:
                    raise ValueError(f"Empty results for {name}")
                
                # Handle BM25 format
                if name == 'bm25':
                    processed_results = []
                    for result in results:
                        if isinstance(result, dict):
                            # Convert BM25 dict format to tuple format
                            doc_ids = [str(hit['docid']) for hit in result.get('hits', [])]
                            scores = [float(hit['score']) for hit in result.get('scores', [])]
                            processed_results.append((doc_ids, scores))
                        else:
                            processed_results.append(result)
                    normalized_results[name] = processed_results
                else:
                    # Validate format of each non-BM25 result
                    for result in results:
                        if not isinstance(result, tuple) or len(result) != 2:
                            raise ValueError(f"Invalid result format for {name}")
                        doc_ids, scores = result
                        if not isinstance(doc_ids, list) or not isinstance(scores, list):
                            raise ValueError(f"Invalid doc_ids/scores format for {name}")
                        if len(doc_ids) != len(scores):
                            raise ValueError(f"Mismatched lengths for doc_ids/scores in {name}")
                    normalized_results[name] = results
            
            # Process results
            fused_results = []
            num_queries = len(next(iter(normalized_results.values())))
            
            for query_idx in range(num_queries):
                query_results = {
                    name: results[query_idx] 
                    for name, results in normalized_results.items()
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