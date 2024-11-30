import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from pathlib import Path
import torch
from tqdm import tqdm
from collections import defaultdict
#from torch.utils.data import Dataset, DataLoader
#from sklearn.metrics import silhouette_score, calinski_harabasz_score
from experiment_logger import ExperimentLogger
import pickle
import json
import time

class BaselineExperimentUtils:
    """Utilities for baseline RAG experiments."""
    
    def __init__(self, logger: Optional[ExperimentLogger] = None):
        """Initialize with optional logger."""
        self.logger = logger

    def load_and_preprocess_data(
        self,
        corpus_path: str,
        queries_path: str,
        search_results_path: str,
    ) -> Tuple[List[Dict], List[str], List[Tuple[List[int], List[float]]]]:
        """Load and preprocess experiment data with logging."""
        try:
            if self.logger:
                self.logger.log_step_start("Loading data")
                
            # Load corpus
            with open(corpus_path, 'r') as f:
                corpus = json.load(f)
                
            # Load queries
            with open(queries_path, 'r') as f:
                data = json.load(f)
            queries = [item['question'] for item in data]
                
            # Load search results  
            with open(search_results_path, 'rb') as f:
                search_results = pickle.load(f)

            if self.logger:
                self.logger.log_metric("corpus_size", len(corpus))
                self.logger.log_metric("num_queries", len(queries))
                self.logger.log_metric("num_search_results", len(search_results))
                self.logger.log_step_end("Loading data")
                
            return corpus, queries, search_results
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error loading data")
            raise

    def prepare_random_documents(
        self,
        corpus: List[Dict],
        num_random_docs: int,
        exclude_indices: Optional[List[int]] = None,
        seed: int = 42
    ) -> Tuple[List[Dict], List[int]]:
        """
        Prepare random documents for injection into context.
        
        Args:
            corpus: List of corpus documents
            num_random_docs: Number of random documents to select
            exclude_indices: Indices to exclude from random selection
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (selected documents, their indices)
        """
        try:
            if self.logger:
                self.logger.log_step_start("Preparing random documents")
                
            random.seed(seed)
            available_indices = list(range(len(corpus)))
            if exclude_indices:
                available_indices = [i for i in available_indices if i not in exclude_indices]
                
            selected_indices = random.sample(available_indices, num_random_docs)
            selected_docs = [corpus[i] for i in selected_indices]

            if self.logger:
                self.logger.log_metric("num_random_docs", len(selected_docs))
                self.logger.log_step_end("Preparing random documents")
                
            return selected_docs, selected_indices
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error preparing random documents")
            raise

    def merge_retrieved_and_random(
        self,
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
        try:
            if self.logger:
                self.logger.log_step_start("Merging documents")
                
            if put_retrieved_first:
                merged_docs = retrieved_docs + random_docs
                merged_indices = retrieved_indices + random_indices
            else:
                merged_docs = random_docs + retrieved_docs
                merged_indices = random_indices + retrieved_indices
                
            if self.logger:
                self.logger.log_metric("total_merged_docs", len(merged_docs))
                self.logger.log_step_end("Merging documents")
                
            return merged_docs, merged_indices
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error merging documents")
            raise

    def validate_document_context(
        self,
        documents: List[Dict],
        max_length: int,
        tokenizer: 'PreTrainedTokenizer',
        max_doc_length: Optional[int] = None
    ) -> Tuple[List[Dict], List[int]]:
        """
        Validate and filter documents to fit within context window.
        
        Args:
            documents: List of documents to validate
            max_length: Maximum allowed length in tokens
            tokenizer: Tokenizer for length calculation
            max_doc_length: Optional maximum document length
            
        Returns:
            Tuple of (valid documents, their indices)
        """
        try:
            if self.logger:
                self.logger.log_step_start("Validating document context")
                
            valid_docs = []
            valid_indices = []
            current_length = 0
            
            for idx, doc in enumerate(documents):
                doc_text = doc.get('text', '')
                if max_doc_length:
                    doc_text = doc_text[:max_doc_length]
                    
                doc_tokens = tokenizer.encode(doc_text, add_special_tokens=True)
                doc_length = len(doc_tokens)
                
                if current_length + doc_length <= max_length:
                    valid_docs.append(doc)
                    valid_indices.append(idx)
                    current_length += doc_length
                else:
                    break
                    
            if self.logger:
                self.logger.log_metric("valid_docs", len(valid_docs))
                self.logger.log_metric("total_length", current_length)
                self.logger.log_step_end("Validating document context")
                
            return valid_docs, valid_indices
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error validating document context")
            raise

    def analyze_document_distribution(
        self,
        results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze distribution and effectiveness of documents in context.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary containing distribution statistics
        """
        try:
            if self.logger:
                self.logger.log_step_start("Analyzing document distribution")
                
            stats = defaultdict(lambda: {"total": 0, "correct": 0})
            
            for result in results:
                doc_positions = result.get('document_indices', [])
                is_correct = result.get('ans_match_after_norm', False)
                
                # Track position statistics
                for pos, doc_idx in enumerate(doc_positions):
                    key = f"position_{pos}"
                    stats[key]["total"] += 1
                    if is_correct:
                        stats[key]["correct"] += 1
                        
                # Track gold document statistics if available        
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
                    
            if self.logger:
                self.logger.log_metrics({"distribution_analysis": analysis})
                self.logger.log_step_end("Analyzing document distribution")
                
            return analysis
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error analyzing document distribution")
            raise

    def compute_retrieval_metrics(
        self,
        results: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute retrieval-specific metrics.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary of computed metrics
        """
        try:
            if self.logger:
                self.logger.log_step_start("Computing retrieval metrics")
                
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
            
            if self.logger:
                self.logger.log_metrics(metrics)
                self.logger.log_step_end("Computing retrieval metrics")
                
            return metrics
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error computing retrieval metrics")
            raise

    def analyze_random_document_impact(
        self,
        results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze the impact of random document injection.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary containing impact analysis
        """
        try:
            if self.logger:
                self.logger.log_step_start("Analyzing random document impact")
                
            impact_analysis = defaultdict(lambda: {"total": 0, "correct": 0})
            
            for result in results:
                num_random = result.get('num_random_docs', 0)
                is_correct = result.get('ans_match_after_norm', False)
                
                # Track impact by number of random documents
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
                    
            if self.logger:
                self.logger.log_metrics({"random_impact_analysis": analysis})
                self.logger.log_step_end("Analyzing random document impact")
                
            return analysis
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error analyzing random document impact")
            raise

    def format_metrics_report(
        self,
        metrics: Dict[str, Dict[str, float]]
    ) -> str:
        """Format metrics into a readable report."""
        report = ["Experiment Metrics Report", "=" * 50, ""]
        
        for category, values in metrics.items():
            report.append(f"\n{category}:")
            for metric, value in values.items():
                if isinstance(value, float):
                    report.append(f"  {metric}: {value:.4f}")
                else:
                    report.append(f"  {metric}: {value}")
                    
        return "\n".join(report)

    def save_experiment_artifacts(
        self,
        results: List[Dict],
        metrics: Dict[str, float],
        output_dir: str
    ) -> None:
        """Save all experiment artifacts with proper organization."""
        try:
            if self.logger:
                self.logger.log_step_start("Saving experiment artifacts")
                
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results
            results_path = output_path / 'experiment_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            # Save metrics
            metrics_path = output_path / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            # Save detailed report
            report_path = output_path / 'experiment_report.txt'
            with open(report_path, 'w') as f:
                f.write(self.format_metrics_report({"metrics": metrics}))
                
            if self.logger:
                self.logger.log_metric("artifacts_saved", {
                    "results": str(results_path),
                    "metrics": str(metrics_path),
                    "report": str(report_path)
                })
                self.logger.log_step_end("Saving experiment artifacts")
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error saving experiment artifacts")
            raise