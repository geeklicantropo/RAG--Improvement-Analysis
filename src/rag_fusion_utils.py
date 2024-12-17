import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import logging
from datetime import datetime
import torch
import google.generativeai as genai
import time
from src.utils.rate_limit import rate_limit

class RAGFusionRanker:
    def __init__(
        self,
        api_key: str,
        k: float = 60.0,
        strategy: str = "rrf",
        normalize_scores: bool = True,
        score_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 32,
        memory_threshold: float = 0.9
    ):
        self.k = k
        self.strategy = strategy
        self.normalize_scores = normalize_scores
        self.score_weights = score_weights or {}
        self.batch_size = batch_size
        self.memory_threshold = memory_threshold
        self.logger = self._setup_logger()
        self.stats = {"fusions": 0, "evaluations": 0}

        # Initialize Gemini
        #genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        if strategy not in ["rrf", "linear"]:
            raise ValueError("Invalid fusion strategy. Must be 'rrf' or 'linear'")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("RAGFusionRanker")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def fuse_search_results(
        self,
        retriever_results: Dict[str, List[Tuple[List[str], List[float]]]]
    ) -> List[Tuple[List[str], List[float]]]:
        self.logger.info(f"Starting fusion with strategy: {self.strategy}")
        
        try:
            all_results = []
            num_queries = len(next(iter(retriever_results.values())))
            
            for query_idx in tqdm(range(num_queries), desc="Fusing results"):
                query_results = {
                    name: results[query_idx] 
                    for name, results in retriever_results.items()
                }
                
                if self.normalize_scores:
                    query_results = self._normalize_batch(query_results)
                    
                if self.strategy == "rrf":
                    fused = self.reciprocal_rank_fusion(query_results)
                else:
                    fused = self._linear_combination(query_results)
                    
                all_results.append(fused)
                self.stats["fusions"] += 1
                
            return all_results
            
        except Exception as e:
            self.logger.error(f"Fusion error: {str(e)}")
            raise

    def _normalize_batch(
        self,
        batch_results: Dict[str, Tuple[List[str], List[float]]]
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        normalized = {}
        for retriever, (doc_ids, scores) in batch_results.items():
            if not scores:
                continue
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                norm_scores = [1.0] * len(scores)
            normalized[retriever] = (doc_ids, norm_scores)
        return normalized

    @rate_limit
    def evaluate_retrieval_diversity(
        self,
        fused_results: List[Tuple[List[str], List[float]]],
        documents: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Evaluate diversity of retrieved documents using LLM"""
        try:
            self.logger.info("Evaluating retrieval diversity")
            diversity_scores = []
            
            for doc_ids, scores in tqdm(fused_results, desc="Evaluating diversity"):
                top_docs = [documents[doc_id] for doc_id in doc_ids[:5]]
                doc_texts = "\n".join(d.get('text', '')[:200] for d in top_docs)
                
                prompt = f"""
                Evaluate the diversity of these documents (0-100):
                {doc_texts}
                
                Consider:
                1. Topic diversity
                2. Information complementarity
                3. Redundancy avoidance
                
                Provide only the numerical score (0-100):
                """
                
                response = self.model.generate_content(prompt)
                diversity_scores.append(float(response.text.strip()) / 100)
                time.sleep(0.1)  # Rate limiting
                
            return {
                "avg_diversity": np.mean(diversity_scores),
                "std_diversity": np.std(diversity_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Diversity evaluation failed: {str(e)}")
            return {"error": str(e)}

    def reciprocal_rank_fusion(
        self,
        results: Dict[str, Tuple[List[str], List[float]]]
    ) -> Tuple[List[str], List[float]]:
        try:
            fused_scores = {}
            
            for retriever_name, (doc_ids, scores) in results.items():
                weight = self.score_weights.get(retriever_name, 1.0)
                ranked_pairs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
                
                for rank, (doc_id, _) in enumerate(ranked_pairs):
                    if doc_id not in fused_scores:
                        fused_scores[doc_id] = 0
                    fused_scores[doc_id] += weight / (rank + self.k)
                    
            sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            doc_ids, scores = zip(*sorted_results) if sorted_results else ([], [])
            
            return list(doc_ids), list(scores)
            
        except Exception as e:
            self.logger.error(f"RRF fusion error: {str(e)}")
            raise

    def _linear_combination(
        self,
        results: Dict[str, Tuple[List[str], List[float]]]
    ) -> Tuple[List[str], List[float]]:
        try:
            fused_scores = {}
            
            for retriever_name, (doc_ids, scores) in results.items():
                weight = self.score_weights.get(retriever_name, 1.0)
                for doc_id, score in zip(doc_ids, scores):
                    if doc_id not in fused_scores:
                        fused_scores[doc_id] = 0
                    fused_scores[doc_id] += score * weight
                    
            sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            doc_ids, scores = zip(*sorted_results) if sorted_results else ([], [])
            
            return list(doc_ids), list(scores)
            
        except Exception as e:
            self.logger.error(f"Linear fusion error: {str(e)}")
            raise

    def inject_noise_into_contexts(
        self,
        fused_results: List[Tuple[List[str], List[float]]],
        corpus: Dict[str, Dict],
        noise_ratio: float,
        random_seed: int = 42
    ) -> List[Tuple[List[str], List[float]]]:
        self.logger.info(f"Injecting noise with ratio {noise_ratio}")
        
        try:
            np.random.seed(random_seed)
            noisy_results = []
            
            for doc_ids, scores in tqdm(fused_results, desc="Injecting noise"):
                num_docs = len(doc_ids)
                num_noise = max(1, int(num_docs * noise_ratio))
                
                # Keep original documents
                keep_docs = num_docs - num_noise
                retained_ids = doc_ids[:keep_docs]
                retained_scores = scores[:keep_docs]
                
                # Sample noise documents
                available_ids = list(set(corpus.keys()) - set(retained_ids))
                noise_ids = np.random.choice(available_ids, num_noise, replace=False)
                noise_scores = [0.0] * num_noise
                
                noisy_results.append((
                    retained_ids + list(noise_ids),
                    retained_scores + noise_scores
                ))
                
            return noisy_results
            
        except Exception as e:
            self.logger.error(f"Noise injection error: {str(e)}")
            raise

    def evaluate_batch_relevance(
        self,
        query: str,
        documents: List[Dict],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Evaluate document relevance in batches"""
        if batch_size is None:
            batch_size = self.batch_size
            
        relevance_scores = []
        num_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Evaluating relevance"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]
            
            batch_scores = []
            for doc in batch_docs:
                try:
                    prompt = f"""
                    Query: {query}
                    Document: {doc.get('text', '')[:500]}
                    
                    Rate document relevance to query (0-100):
                    Consider:
                    1. Query relevance
                    2. Information completeness
                    3. Answer presence
                    
                    Provide only the numerical score (0-100):
                    """
                    
                    response = self.model.generate_content(prompt)
                    score = float(response.text.strip()) / 100
                    batch_scores.append(score)
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    self.logger.error(f"Document evaluation failed: {str(e)}")
                    batch_scores.append(0.0)
                    
            relevance_scores.extend(batch_scores)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return relevance_scores

    def get_fusion_info(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "rrf_k": self.k,
            "normalize_scores": self.normalize_scores,
            "weights": self.score_weights,
            "stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }