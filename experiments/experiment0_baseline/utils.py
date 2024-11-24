# experiments/experiment0_baseline/utils.py

import random
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from collections import defaultdict

class BaselineExperimentUtils:
    """Utilities for baseline RAG experiments."""
    
    @staticmethod
    def prepare_random_documents(
        corpus: List[Dict],
        num_random_docs: int,
        exclude_indices: Optional[List[int]] = None,
        seed: int = 42
    ) -> Tuple[List[Dict], List[int]]:
        """
        Prepare random documents for injection into the context.
        
        Args:
            corpus: List of corpus documents
            num_random_docs: Number of random documents to select
            exclude_indices: Indices to exclude from random selection
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (selected documents, their indices)
        """
        random.seed(seed)
        
        # Create pool of available indices
        available_indices = list(range(len(corpus)))
        if exclude_indices:
            available_indices = [i for i in available_indices if i not in exclude_indices]
            
        # Select random documents
        selected_indices = random.sample(available_indices, num_random_docs)
        selected_docs = [corpus[i] for i in selected_indices]
        
        return selected_docs, selected_indices

    @staticmethod
    def merge_retrieved_and_random(
        retrieved_docs: List[Dict],
        retrieved_indices: List[int],
        random_docs: List[Dict],
        random_indices: List[int],
        put_retrieved_first: bool = True
    ) -> Tuple[List[Dict], List[int]]:
        """
        Merge retrieved and random documents in specified order.
        
        Args:
            retrieved_docs: List of retrieved documents
            retrieved_indices: Indices of retrieved documents
            random_docs: List of random documents
            random_indices: Indices of random documents
            put_retrieved_first: Whether to put retrieved documents first
            
        Returns:
            Tuple of (merged documents, merged indices)
        """
        if put_retrieved_first:
            merged_docs = retrieved_docs + random_docs
            merged_indices = retrieved_indices + random_indices
        else:
            merged_docs = random_docs + retrieved_docs
            merged_indices = random_indices + retrieved_indices
            
        return merged_docs, merged_indices

    @staticmethod
    def validate_document_context(
        documents: List[Dict],
        max_length: int,
        tokenizer: 'PreTrainedTokenizer'
    ) -> Tuple[List[Dict], List[int]]:
        """
        Validate and filter documents to fit within context window.
        
        Args:
            documents: List of documents to validate
            max_length: Maximum allowed length in tokens
            tokenizer: Tokenizer for length calculation
            
        Returns:
            Tuple of (valid documents, their indices)
        """
        valid_docs = []
        valid_indices = []
        current_length = 0
        
        for idx, doc in enumerate(documents):
            # Calculate document length
            doc_text = doc.get('text', '')
            doc_tokens = tokenizer.encode(doc_text, add_special_tokens=True)
            doc_length = len(doc_tokens)
            
            # Check if adding document exceeds max length
            if current_length + doc_length <= max_length:
                valid_docs.append(doc)
                valid_indices.append(idx)
                current_length += doc_length
            else:
                break
                
        return valid_docs, valid_indices

    @staticmethod
    def analyze_document_distribution(
        results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze distribution and effectiveness of documents in context.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary containing distribution statistics
        """
        stats = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for result in results:
            # Analyze position effectiveness
            doc_positions = result.get('document_indices', [])
            is_correct = result.get('ans_match_after_norm', False)
            
            for pos, doc_idx in enumerate(doc_positions):
                key = f"position_{pos}"
                stats[key]["total"] += 1
                if is_correct:
                    stats[key]["correct"] += 1
                    
            # Analyze gold document effectiveness
            gold_pos = result.get('gold_position')
            if gold_pos is not None:
                key = f"gold_position_{gold_pos}"
                stats[key]["total"] += 1
                if is_correct:
                    stats[key]["correct"] += 1
        
        # Calculate percentages
        analysis = {}
        for key, counts in stats.items():
            if counts["total"] > 0:
                analysis[key] = {
                    "accuracy": counts["correct"] / counts["total"],
                    "total_samples": counts["total"],
                    "correct_samples": counts["correct"]
                }
                
        return analysis

    @staticmethod
    def compute_retrieval_metrics(
        results: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute retrieval-specific metrics.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            "total_examples": len(results),
            "correct_answers": 0,
            "gold_doc_retrieved": 0,
            "gold_doc_helpful": 0
        }
        
        for result in results:
            # Count correct answers
            if result.get('ans_match_after_norm', False):
                metrics["correct_answers"] += 1
            
            # Check gold document presence and usefulness
            gold_in_context = result.get('gold_in_retrieved', False)
            if gold_in_context:
                metrics["gold_doc_retrieved"] += 1
                if result.get('ans_match_after_norm', False):
                    metrics["gold_doc_helpful"] += 1
        
        # Calculate percentages
        metrics["accuracy"] = metrics["correct_answers"] / metrics["total_examples"]
        metrics["gold_retrieval_rate"] = metrics["gold_doc_retrieved"] / metrics["total_examples"]
        metrics["gold_usefulness_rate"] = (metrics["gold_doc_helpful"] / metrics["gold_doc_retrieved"] 
                                         if metrics["gold_doc_retrieved"] > 0 else 0)
        
        return metrics

    @staticmethod
    def analyze_random_document_impact(
        results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze the impact of random document injection.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary containing impact analysis
        """
        impact_analysis = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for result in results:
            num_random = result.get('num_random_docs', 0)
            is_correct = result.get('ans_match_after_norm', False)
            
            # Analyze by number of random documents
            key = f"random_docs_{num_random}"
            impact_analysis[key]["total"] += 1
            if is_correct:
                impact_analysis[key]["correct"] += 1
        
        # Calculate statistics
        analysis = {}
        for key, counts in impact_analysis.items():
            if counts["total"] > 0:
                analysis[key] = {
                    "accuracy": counts["correct"] / counts["total"],
                    "total_samples": counts["total"],
                    "correct_samples": counts["correct"]
                }
                
        return analysis

    @staticmethod
    def format_metrics_report(
        metrics: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Format metrics into a readable report.
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            Formatted report string
        """
        report = ["Experiment Metrics Report", "=" * 50, ""]
        
        for category, values in metrics.items():
            report.append(f"\n{category}:")
            for metric, value in values.items():
                if isinstance(value, float):
                    report.append(f"  {metric}: {value:.4f}")
                else:
                    report.append(f"  {metric}: {value}")
                    
        return "\n".join(report)