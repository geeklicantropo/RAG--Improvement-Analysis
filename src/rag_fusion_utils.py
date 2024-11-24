import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from tqdm import tqdm

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

    def fuse_search_results(
        self,
        retriever_results: Dict[str, List[Tuple[List[str], List[float]]]]
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Fuse search results from multiple retrievers into a single ranked list.
        
        Args:
            retriever_results: Dictionary mapping retriever names to their search results
                             Each result is a tuple of (doc_ids, scores)
                             
        Returns:
            Combined and reranked results as (doc_ids, scores) tuples
        """
        fused_results = []
        
        for query_idx in tqdm(range(len(next(iter(retriever_results.values())))), desc="Fusing results"):
            # Gather results for this query from all retrievers
            query_results = {
                name: results[query_idx] 
                for name, results in retriever_results.items()
            }
            
            # Track document ranks and scores
            doc_info: Dict[str, Dict[str, Union[int, float]]] = {}
            
            # Process each retriever's results
            for retriever_name, (doc_ids, scores) in query_results.items():
                # Normalize scores if requested
                if self.normalize_scores:
                    scores = self.normalize_score_list(scores)
                
                # Record rank and score for each document
                for rank, (doc_id, score) in enumerate(zip(doc_ids, scores)):
                    if doc_id not in doc_info:
                        doc_info[doc_id] = {
                            "ranks": {},
                            "scores": {},
                            "retrievers": set()
                        }
                    doc_info[doc_id]["ranks"][retriever_name] = rank
                    doc_info[doc_id]["scores"][retriever_name] = score
                    doc_info[doc_id]["retrievers"].add(retriever_name)
            
            # Compute fused scores
            fused_doc_scores = []
            for doc_id, info in doc_info.items():
                if self.strategy == "rrf":
                    fused_score = self._compute_rrf_scores(info["ranks"])
                else:  # linear combination
                    fused_score = 0.0
                    for retriever in info["retrievers"]:
                        weight = self.score_weights.get(retriever, 1.0)
                        fused_score += weight * info["scores"][retriever]
                
                fused_doc_scores.append((doc_id, fused_score))
            
            # Sort by fused score
            fused_doc_scores.sort(key=lambda x: x[1], reverse=True)
            doc_ids, scores = zip(*fused_doc_scores)
            
            fused_results.append((list(doc_ids), list(scores)))
        
        return fused_results

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