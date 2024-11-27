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


class CategoriesExperimentUtils:
    """Utilities for category-based RAG experiments."""
    
    @staticmethod
    def load_retrieval_results(
        contriever_path: str,
        bm25_path: Optional[str],
        use_fusion: bool,
        logger: ExperimentLogger
    ) -> Dict[str, List[Tuple[List[str], List[float]]]]:
        """
        Load and prepare retrieval results for categorization.
        
        Args:
            contriever_path: Path to Contriever results
            bm25_path: Optional path to BM25 results
            use_fusion: Whether to use both retrievers with fusion
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
            
            retriever_results = {'contriever': contriever_results}
            
            # Load BM25 results if using fusion
            if use_fusion and bm25_path:
                with open(bm25_path, 'rb') as f:
                    bm25_results = pickle.load(f)
                logger.log_metric("bm25_results_size", len(bm25_results))
                retriever_results['bm25'] = bm25_results
            
            logger.log_step_end("Loading retrieval results", time.time())
            return retriever_results
            
        except Exception as e:
            logger.log_error(e, "Error loading retrieval results")
            raise
            
    @staticmethod
    def categorize_documents(
        documents: List[Dict],
        scores: List[float],
        score_thresholds: Dict[str, float],
        max_docs_per_category: int,
        logger: ExperimentLogger
    ) -> Dict[str, List[Dict]]:
        """
        Categorize documents based on confidence scores.
        
        Args:
            documents: List of retrieved documents
            scores: Document scores
            score_thresholds: Threshold values for categories
            max_docs_per_category: Maximum documents per category
            logger: Logger instance
            
        Returns:
            Dictionary mapping categories to their documents
        """
        try:
            logger.log_step_start("Categorizing documents")
            
            categories = defaultdict(list)
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Sort thresholds from highest to lowest
            sorted_thresholds = sorted(
                score_thresholds.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Categorize documents
            for doc, score in doc_scores:
                # Find appropriate category based on score
                category = "low_confidence"  # default category
                for cat_name, threshold in sorted_thresholds:
                    if score >= threshold:
                        category = cat_name
                        break
                
                # Add to category if not full
                if len(categories[category]) < max_docs_per_category:
                    categories[category].append({
                        "document": doc,
                        "score": score
                    })
            
            # Log categorization statistics
            stats = {
                f"{cat}_count": len(docs)
                for cat, docs in categories.items()
            }
            logger.log_metrics(stats)
            
            logger.log_step_end("Categorizing documents", time.time())
            return dict(categories)
            
        except Exception as e:
            logger.log_error(e, "Error categorizing documents")
            raise

    @staticmethod
    def inject_random_documents(
        categories: Dict[str, List[Dict]],
        random_docs: List[Dict],
        random_category: str,
        max_random_docs: int,
        logger: ExperimentLogger
    ) -> Dict[str, List[Dict]]:
        """
        Inject random documents into categories.
        
        Args:
            categories: Existing document categories
            random_docs: Random documents to inject
            random_category: Category name for random documents
            max_random_docs: Maximum number of random documents
            logger: Logger instance
            
        Returns:
            Updated categories with random documents
        """
        try:
            logger.log_step_start("Injecting random documents")
            
            # Create new category for random documents
            categories[random_category] = []
            
            # Add random documents with artificial scores
            for doc in random_docs[:max_random_docs]:
                categories[random_category].append({
                    "document": doc,
                    "score": 0.0  # Assign zero score to random documents
                })
            
            logger.log_metric("random_docs_added", len(categories[random_category]))
            logger.log_step_end("Injecting random documents", time.time())
            return categories
            
        except Exception as e:
            logger.log_error(e, "Error injecting random documents")
            raise

    @staticmethod
    def analyze_category_impact(
        results: List[Dict],
        logger: ExperimentLogger
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze the impact of categorization on results.
        
        Args:
            results: Generation results
            logger: Logger instance
            
        Returns:
            Dictionary of category-specific metrics
        """
        try:
            logger.log_step_start("Analyzing category impact")
            
            analysis = defaultdict(lambda: {
                "total_queries": 0,
                "correct_answers": 0,
                "avg_response_length": 0,
                "contribution_rate": 0
            })
            
            total_queries = len(results)
            correct_answers = 0
            category_contributions = defaultdict(int)
            
            for result in results:
                # Track overall accuracy
                if result['ans_match_after_norm']:
                    correct_answers += 1
                
                # Analyze category contributions
                if 'category_info' in result:
                    for cat_name, cat_docs in result['category_info'].items():
                        analysis[cat_name]["total_queries"] += 1
                        if result['ans_match_after_norm']:
                            analysis[cat_name]["correct_answers"] += 1
                        if result.get('contributing_category') == cat_name:
                            category_contributions[cat_name] += 1
            
            # Calculate category-specific metrics
            for category, metrics in analysis.items():
                if metrics["total_queries"] > 0:
                    metrics["accuracy"] = metrics["correct_answers"] / metrics["total_queries"]
                    metrics["contribution_rate"] = (
                        category_contributions[category] / total_queries
                    )
            
            # Add overall metrics
            analysis["overall"] = {
                "accuracy": correct_answers / total_queries,
                "total_queries": total_queries,
                "correct_answers": correct_answers
            }
            
            logger.log_metrics(dict(analysis))
            logger.log_step_end("Analyzing category impact", time.time())
            return dict(analysis)
            
        except Exception as e:
            logger.log_error(e, "Error analyzing category impact")
            raise

    @staticmethod
    def save_category_artifacts(
        results: List[Dict],
        metrics: Dict[str, Any],
        category_info: Dict[str, Any],
        output_dir: str,
        logger: ExperimentLogger
    ) -> None:
        """
        Save category experiment artifacts.
        
        Args:
            results: Generation results
            metrics: Performance metrics
            category_info: Category configuration and statistics
            output_dir: Output directory
            logger: Logger instance
        """
        try:
            logger.log_step_start("Saving category artifacts")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results
            results_path = os.path.join(output_dir, 'category_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            # Save metrics
            metrics_path = os.path.join(output_dir, 'category_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            # Save category information
            category_path = os.path.join(output_dir, 'category_info.json')
            with open(category_path, 'w') as f:
                json.dump(category_info, f, indent=2)
                
            # Log saved paths
            logger.log_metric('results_path', results_path)
            logger.log_metric('metrics_path', metrics_path)
            logger.log_metric('category_path', category_path)
            
            logger.log_step_end("Saving category artifacts", time.time())
            
        except Exception as e:
            logger.log_error(e, "Error saving category artifacts")
            raise

    @staticmethod
    def format_category_prompt(
        query: str,
        categories: Dict[str, List[Dict]],
        logger: ExperimentLogger
    ) -> str:
        """
        Format prompt with categorized documents.
        
        Args:
            query: User query
            categories: Categorized documents
            logger: Logger instance
            
        Returns:
            Formatted prompt string
        """
        try:
            logger.log_step_start("Formatting category prompt")
            
            prompt_parts = ["Below are documents organized by confidence level:"]
            
            # Add documents from each category
            for category, docs in categories.items():
                if docs:  # Only include non-empty categories
                    prompt_parts.append(f"\n{category.upper()}:")
                    for doc_info in docs:
                        doc = doc_info["document"]
                        doc_str = f"Document [{doc.get('id', '')}] "
                        if doc.get("title"):
                            doc_str += f"(Title: {doc['title']}) "
                        doc_str += doc.get("text", "")
                        prompt_parts.append(doc_str)
            
            # Add query
            prompt_parts.append("\nQuestion: " + query)
            prompt_parts.append("Answer:")
            
            formatted_prompt = "\n".join(prompt_parts)
            logger.log_step_end("Formatting category prompt", time.time())
            return formatted_prompt
            
        except Exception as e:
            logger.log_error(e, "Error formatting category prompt")
            raise

    @staticmethod
    def compute_category_statistics(
        categories: Dict[str, List[Dict]],
        logger: ExperimentLogger
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute detailed statistics for each category.
        
        Args:
            categories: Categorized documents
            logger: Logger instance
            
        Returns:
            Dictionary of category statistics
        """
        try:
            logger.log_step_start("Computing category statistics")
            
            stats = {}
            for category, docs in categories.items():
                if not docs:
                    continue
                    
                scores = [doc["score"] for doc in docs]
                stats[category] = {
                    "count": len(docs),
                    "avg_score": np.mean(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "std_score": np.std(scores)
                }
            
            logger.log_metrics({"category_statistics": stats})
            logger.log_step_end("Computing category statistics", time.time())
            return stats
            
        except Exception as e:
            logger.log_error(e, "Error computing category statistics")
            raise

    @staticmethod
    def evaluate_retrieval_diversity(
        categories: Dict[str, List[Dict]],
        logger: ExperimentLogger
    ) -> Dict[str, float]:
        """
        Evaluate diversity of retrieved documents across categories.
        
        Args:
            categories: Categorized documents
            logger: Logger instance
            
        Returns:
            Dictionary of diversity metrics
        """
        try:
            logger.log_step_start("Evaluating retrieval diversity")
            
            metrics = {}
            
            # Calculate category distribution
            total_docs = sum(len(docs) for docs in categories.values())
            category_distribution = {
                cat: len(docs) / total_docs 
                for cat, docs in categories.items()
            }
            
            # Calculate entropy of distribution
            entropy = -sum(
                p * np.log2(p) 
                for p in category_distribution.values() 
                if p > 0
            )
            
            metrics.update({
                "category_entropy": entropy,
                "num_categories": len(categories),
                "category_distribution": category_distribution
            })
            
            logger.log_metrics(metrics)
            logger.log_step_end("Evaluating retrieval diversity", time.time())
            return metrics
            
        except Exception as e:
            logger.log_error(e, "Error evaluating retrieval diversity")
            raise