import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json
import pickle
import torch
from src.experiment_logger import ExperimentLogger


class BaselineExperimentUtils:
    """Utilities for baseline RAG experiments."""

    def __init__(self, logger: Optional[ExperimentLogger] = None):
        self.logger = logger

    def load_and_preprocess_data(
        self,
        corpus_path: str,
        queries_path: str,
        search_results_path: str,
    ) -> Tuple[List[Dict], List[str], List[Tuple[List[int], List[float]]]]:
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

    def inject_noise_into_contexts(
        self,
        corpus: List[Dict],
        search_results: List[Tuple[List[int], List[float]]],
        noise_ratio: float,
        seed: int = 42
    ) -> List[Tuple[List[int], List[float]]]:
        """Inject random distractor documents into retrieval results."""
        try:
            if self.logger:
                self.logger.log_step_start("Injecting noise into contexts")

            random.seed(seed)
            noisy_results = []

            for doc_indices, scores in tqdm(search_results, desc="Injecting Noise"):
                num_noise_docs = max(1, int(len(doc_indices) * noise_ratio))
                noise_indices = random.sample(
                    range(len(corpus)), num_noise_docs
                )
                doc_indices.extend(noise_indices)
                scores.extend([0.0] * len(noise_indices))  # Assign minimal scores for noise docs
                noisy_results.append((doc_indices, scores))

            if self.logger:
                self.logger.log_metric("num_noisy_results", len(noisy_results))
                self.logger.log_step_end("Injecting noise into contexts")

            return noisy_results

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error injecting noise into contexts")
            raise

    def position_gold_document(
        self,
        doc_indices: List[int],
        gold_doc_index: int,
        position: Optional[int] = None
    ) -> List[int]:
        """Place the gold document at a specified position in the document list."""
        try:
            if self.logger:
                self.logger.log_step_start("Positioning gold document")

            if gold_doc_index not in doc_indices:
                doc_indices.append(gold_doc_index)

            doc_indices = [
                doc for doc in doc_indices if doc != gold_doc_index
            ]  # Remove duplicates
            if position is None or position >= len(doc_indices):
                doc_indices.append(gold_doc_index)
            else:
                doc_indices.insert(position, gold_doc_index)

            if self.logger:
                self.logger.log_metric("gold_doc_position", position if position is not None else -1)
                self.logger.log_step_end("Positioning gold document")

            return doc_indices

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error positioning gold document")
            raise

    def prepare_random_documents(
        self,
        corpus: List[Dict],
        num_random_docs: int,
        exclude_indices: Optional[List[int]] = None,
        seed: int = 42
    ) -> Tuple[List[Dict], List[int]]:
        """Select random documents from the corpus."""
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
        """Merge retrieved and random documents into a single list."""
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
        """Validate and trim document contexts based on tokenizer limits."""
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