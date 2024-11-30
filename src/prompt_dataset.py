import os
import json
import random
import hashlib
from typing import List, Tuple, Dict, Any, Optional, Set
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging
import warnings
import gc

import normalize_text
from normalize_answers import *
from cluster_utils import DocumentClusterer
from src.experiment_logger import ExperimentLogger

class PromptDataset(Dataset):
    """Dataset class for managing and organizing document data into structured prompts."""
    
    def __init__(
        self,
        corpus: List[Dict],
        data_path: str,
        tokenizer: AutoTokenizer,
        max_tokenized_length: int,
        search_results: List[Tuple[List[str], List[float]]],
        full_to_subset_idx_map: Optional[Dict[int, int]] = None,
        do_normalize_query: bool = False,
        num_documents_in_context: int = 5,
        gold_position: Optional[int] = None,
        randomize_gold_position: bool = False,
        get_documents_without_answer: bool = False,
        max_doc_length: Optional[int] = None,
        use_clustering: bool = False,
        num_clusters: Optional[int] = None,
        cluster_seed: int = 42,
        category_info: Optional[Dict[str, Any]] = None,
        noise_config: Optional[Dict[str, Any]] = None,
        experiment_type: str = "baseline",
        output_dir: Optional[str] = None,
        logger: Optional[ExperimentLogger] = None
    ):
        super().__init__()
        self.corpus = corpus
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_tokenized_length = max_tokenized_length
        self.search_results = search_results
        self.full_to_subset_idx_map = full_to_subset_idx_map
        self.do_normalize_query = do_normalize_query
        self.num_documents_in_context = num_documents_in_context
        self.gold_position = gold_position
        self.randomize_gold_position = randomize_gold_position
        self.get_documents_without_answer = get_documents_without_answer
        self.max_doc_length = max_doc_length
        self.use_clustering = use_clustering
        self.clusterer = (
            DocumentClusterer(num_clusters, cluster_seed) if use_clustering else None
        )
        self.category_info = category_info
        self.noise_config = noise_config or {}
        self.experiment_type = experiment_type
        self.output_dir = Path(output_dir) if output_dir else None
        self.logger = logger or ExperimentLogger("prompt_dataset", "logs")

        self.document_info = []  # Store document metadata for logging
        self._validate_initialization_parameters()
        self.preprocess_search_results()
        self._load_data()

    def _validate_initialization_parameters(self):
        """Validates initialization parameters with logging."""
        self.logger.log_step_start("Validating parameters")
        try:
            if self.num_documents_in_context <= 0:
                raise ValueError("num_documents_in_context must be positive")
            
            if self.gold_position is not None:
                if self.gold_position < 0 or self.gold_position >= self.num_documents_in_context:
                    raise ValueError(f"Invalid gold position: {self.gold_position}")
            
            if self.gold_position is not None and self.randomize_gold_position:
                raise ValueError("Cannot set both gold_position and randomize_gold_position")

            self.logger.log_step_end("Parameter validation successful")
        except Exception as e:
            self.logger.log_error(e, "Parameter validation failed")
            raise

    def preprocess_search_results(self):
        """Preprocess and validate search results with logging."""
        self.logger.log_step_start("Preprocessing search results")
        try:
            if not self.search_results:
                raise ValueError("No search results available")

            processed_results = []
            for result in self.search_results:
                if isinstance(result, tuple) and len(result) == 2:
                    doc_ids, scores = result
                    processed_results.append((
                        [str(doc_id) for doc_id in doc_ids],
                        [float(score) for score in scores]
                    ))

            self.search_results = processed_results
            self.logger.log_metric("processed_results_count", len(processed_results))
            self.logger.log_step_end("Search results preprocessing")
        except Exception as e:
            self.logger.log_error(e, "Search results preprocessing failed")
            raise

    def save_prompts(self):
        """Save prompts in both JSON and text formats with logging."""
        if not self.output_dir:
            self.logger.experiment_logger.warning("No output directory specified")
            return

        self.logger.log_step_start("Saving prompts")
        try:
            prompt_dir = self.output_dir / "prompts"
            prompt_dir.mkdir(parents=True, exist_ok=True)

            # Save as JSON
            prompts_data = {
                'example_id': self.example_ids,
                'prompts': self.prompts,
                'document_metadata': [
                    {
                        'indices': indices,
                        'gold_position': gold_pos,
                        'cluster_info': self.cluster_assignments.get(str(doc_idx)) if self.use_clustering else None,
                    }
                    for indices, gold_pos in self.document_info
                ]
            }
            json_path = prompt_dir / f"{self.experiment_type}_prompts.json"
            with open(json_path, 'w') as f:
                json.dump(prompts_data, f, indent=2)

            # Save as text for debugging
            text_path = prompt_dir / f"{self.experiment_type}_prompts.txt"
            with open(text_path, 'w') as f:
                for idx, prompt in enumerate(self.prompts):
                    f.write(f"Example {self.example_ids[idx]}\n")
                    f.write("-" * 80 + "\n")
                    f.write(prompt + "\n\n")

            self.logger.log_metric("saved_prompts_count", len(self.prompts))
            self.logger.log_step_end("Prompts saved successfully")
        except Exception as e:
            self.logger.log_error(e, "Error saving prompts")
            raise

    def _inject_noise_documents(
        self,
        formatted_docs: List[str],
        doc_indices: List[int]
    ) -> Tuple[List[str], List[int]]:
        """Inject noise documents based on configuration."""
        if not self.noise_config:
            return formatted_docs, doc_indices

        self.logger.log_step_start("Injecting noise documents")
        try:
            num_noise = self.noise_config.get('num_noise_docs', 0)
            noise_type = self.noise_config.get('noise_type', 'random')
            
            if noise_type == 'random':
                noise_docs, noise_indices = self._get_random_documents(num_noise)
            else:
                noise_docs, noise_indices = self._get_distractor_documents(num_noise)
            
            # Mix documents based on configuration
            insert_positions = self.noise_config.get('insert_positions', None)
            if insert_positions:
                for pos, (doc, idx) in zip(insert_positions, zip(noise_docs, noise_indices)):
                    formatted_docs.insert(pos, doc)
                    doc_indices.insert(pos, idx)
            else:
                formatted_docs.extend(noise_docs)
                doc_indices.extend(noise_indices)

            self.logger.log_metric(f"noise_{noise_type}_count", num_noise)
            self.logger.log_step_end("Noise injection complete")
            return formatted_docs[:self.num_documents_in_context], doc_indices[:self.num_documents_in_context]
        except Exception as e:
            self.logger.log_error(e, "Error injecting noise documents")
            return formatted_docs, doc_indices

    def _get_random_documents(self, num_docs: int) -> Tuple[List[str], List[int]]:
        """Get random documents from corpus."""
        available_indices = list(range(len(self.corpus)))
        selected_indices = random.sample(available_indices, min(num_docs, len(available_indices)))
        
        docs = []
        indices = []
        for idx in selected_indices:
            doc_info = self.corpus[idx]
            text = doc_info.get("text", "")
            if self.max_doc_length:
                text = text[:self.max_doc_length]
            doc_str = f"Document [{idx}](Title: {doc_info.get('title', '')}) {text}"
            docs.append(doc_str)
            indices.append(idx)
        
        return docs, indices

    def build_experiment_prompt(
        self,
        query: str,
        docs: List[str],
        experiment_type: str
    ) -> str:
        """Build prompt based on experiment type."""
        self.logger.log_step_start(f"Building {experiment_type} prompt")
        try:
            if experiment_type == 'baseline':
                prompt = self.build_qa_prompt(query, '\n'.join(docs))
            elif experiment_type == 'clustering':
                prompt = self.build_clustered_prompt(query, docs)
            elif experiment_type == 'noise':
                prompt = self.build_noise_aware_prompt(query, docs)
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")
                
            self.logger.log_step_end("Prompt building complete")
            return prompt
        except Exception as e:
            self.logger.log_error(e, "Error building prompt")
            raise

    def build_clustered_prompt(self, query: str, docs: List[str]) -> str:
        """Build prompt for clustered documents."""
        clusters = {}
        for doc, cluster_id in zip(docs, self.cluster_assignments.values()):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(doc)
        
        context = []
        for cluster_id, cluster_docs in sorted(clusters.items()):
            context.append(f"\nCluster {cluster_id}:")
            context.extend(cluster_docs)
        
        return self.build_qa_prompt(query, '\n'.join(context))

    def build_noise_aware_prompt(self, query: str, docs: List[str]) -> str:
        """Build prompt with awareness of noise documents."""
        context = []
        for idx, doc in enumerate(docs):
            prefix = "[NOISE] " if idx in self.noise_config.get('noise_indices', []) else ""
            context.append(prefix + doc)
        
        return self.build_qa_prompt(query, '\n'.join(context))

    def _load_data(self):
        """Loads and processes data from file with logging."""
        self.logger.log_step_start("Loading data")
        try:
            with open(self.data_path, "r") as fin:
                data = json.load(fin)
            self.process_file_data(data)
            self.logger.log_step_end("Data loading complete")
        except Exception as e:
            self.logger.log_error(e, f"Error loading data from {self.data_path}")
            raise

    def process_file_data(self, data: List[Dict]):
        """Processes dataset examples with logging."""
        self.logger.log_step_start("Processing file data")
        self.example_ids = []
        self.queries = []
        self.prompts = []
        self.gold_document_idxs = []
        self.excluded_samples_ids = []
        self.preprocessed_data = []
        self.prompt_tokens_lengths = []
        self.cluster_assignments = {} if self.use_clustering else None

        for idx, example in enumerate(data):
            try:
                example_id = str(example['example_id'])
                gold_document_idx = str(example['idx_gold_in_corpus'])
                answers = example['answers']

                formatted_documents, document_indices = self.prepare_documents_for_prompt(
                    idx, gold_document_idx, answers
                )

                if len(formatted_documents) != self.num_documents_in_context:
                    self.logger.log_metric("skipped_insufficient_docs", 1)
                    continue

                documents_str = self._format_documents_for_prompt(formatted_documents, document_indices)
                query = example['question']
                if self.do_normalize_query:
                    query = normalize_text.normalize(query)

                prompt = self.build_experiment_prompt(query, formatted_documents, self.experiment_type)

                # Check token length
                tokens = self.tokenizer.tokenize(prompt)
                tokens_len = len(tokens)
                if tokens_len >= self.max_tokenized_length:
                    self.excluded_samples_ids.append((idx, example_id))
                    self.logger.log_metric("excluded_length_exceeded", 1)
                    continue

                self.preprocessed_data.append((formatted_documents, document_indices))
                self.example_ids.append(example_id)
                self.queries.append(query)
                self.prompts.append(prompt)
                self.gold_document_idxs.append(gold_document_idx)
                self.prompt_tokens_lengths.append(tokens_len)
                
                # Save document metadata for logging
                self.document_info.append((document_indices, self.gold_position))

            except Exception as e:
                self.logger.log_error(e, f"Error processing example {idx}")
                continue

        self.logger.log_metric("total_processed", len(self.example_ids))
        self.logger.log_metric("total_excluded", len(self.excluded_samples_ids))
        self.logger.log_step_end("File data processing complete")

    def _get_documents(self, indices: List[int], answers: List[str], gold_document_idx: Optional[int], gold_position: Optional[int]) -> Tuple[List[str], List[int]]:
        """Get documents based on configuration."""
        if self.get_documents_without_answer:
            return self._get_answerless_documents_from_indices(
                indices, answers, gold_document_idx, gold_position
            )
        else:
            return self._get_documents_from_indices(indices)

    def _get_documents_from_indices(self, indices: List[int]) -> Tuple[List[str], List[int]]:
        """Get documents from corpus by indices."""
        formatted_documents = []
        document_indices = []
        seen_hashes = set()
        
        for idx in indices:
            if self.full_to_subset_idx_map:
                if idx not in self.full_to_subset_idx_map:
                    continue
                idx = self.full_to_subset_idx_map[idx]
                
            if 0 <= idx < len(self.corpus):
                doc_info = self.corpus[idx]
                text = doc_info.get("text", "")
                if self.max_doc_length:
                    text = text[:self.max_doc_length]
                    
                # Avoid duplicates using hash
                doc_hash = hashlib.sha256(text.encode()).hexdigest()
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)
                
                doc_str = f"Document [{idx}](Title: {doc_info.get('title', '')}) {text}"
                formatted_documents.append(doc_str)
                document_indices.append(idx)

                if len(formatted_documents) == self.num_documents_in_context:
                    break

        return formatted_documents, document_indices

    def _format_documents_for_prompt(self, formatted_documents: List[str], document_indices: List[int]) -> str:
        """Format documents for prompt based on configuration."""
        if self.use_clustering and self.cluster_assignments:
            return self._format_clustered_documents(formatted_documents, document_indices)
        elif self.category_info:
            return self._format_categorized_documents(formatted_documents, document_indices)
        else:
            return '\n'.join(formatted_documents)
        
    def prepare_documents_for_prompt(
        self, 
        example_idx: int, 
        gold_document_idx: int, 
        answers: List[str]
    ) -> Tuple[List[str], List[int]]:
        """Prepares documents for prompt creation with logging."""
        self.logger.log_step_start(f"Preparing documents for example {example_idx}")
        try:
            indices = self._get_indices(example_idx)
            updated_indices, gold_position = self._insert_gold_document_idx(
                indices, gold_document_idx
            )

            formatted_documents, document_indices = self._get_documents(
                updated_indices, answers, gold_document_idx, gold_position
            )

            if self.use_clustering and formatted_documents:
                self._update_cluster_assignments(formatted_documents, document_indices)

            # Add noise documents if configured
            if self.noise_config:
                formatted_documents, document_indices = self._inject_noise_documents(
                    formatted_documents, document_indices
                )

            self.logger.log_metric("documents_prepared", len(formatted_documents))
            self.logger.log_step_end("Document preparation complete")
            return formatted_documents, document_indices
        except Exception as e:
            self.logger.log_error(e, "Error preparing documents")
            raise

    def _get_indices(self, example_idx: int) -> List[int]:
        """Get document indices with validation and logging."""
        self.logger.log_step_start(f"Getting indices for example {example_idx}")
        try:
            if not self.search_results:
                raise ValueError("No search results available")
                
            if example_idx >= len(self.search_results):
                raise ValueError(f"Example index {example_idx} out of range")

            indices, scores = self.search_results[example_idx]
            valid_indices = []
            max_corpus_idx = len(self.corpus) - 1
                
            for idx in indices:
                corpus_idx = int(idx)
                    
                if self.full_to_subset_idx_map is not None:
                    if corpus_idx not in self.full_to_subset_idx_map:
                        continue
                    corpus_idx = self.full_to_subset_idx_map[corpus_idx]
                        
                if 0 <= corpus_idx <= max_corpus_idx:
                    valid_indices.append(corpus_idx)
                    
            if len(valid_indices) < self.num_documents_in_context:
                self._fill_missing_indices(valid_indices, max_corpus_idx)

            self.logger.log_metric("valid_indices_found", len(valid_indices))
            self.logger.log_step_end("Index retrieval complete")
            return valid_indices[:self.num_documents_in_context]
        except Exception as e:
            self.logger.log_error(e, "Error getting indices")
            raise

    def _fill_missing_indices(self, valid_indices: List[int], max_idx: int):
        """Fill missing indices to reach required context size."""
        current_indices = set(valid_indices)
        additional_needed = self.num_documents_in_context - len(valid_indices)
        
        additional_indices = []
        for i in range(max_idx + 1):
            if len(additional_indices) >= additional_needed:
                break
            if i not in current_indices:
                additional_indices.append(i)
                
        valid_indices.extend(additional_indices[:additional_needed])

    def _insert_gold_document_idx(
        self,
        indices: List[int],
        gold_document_idx: int
    ) -> Tuple[List[int], int]:
        """Insert gold document at specified or random position."""
        gold_position = None
        
        if self.gold_position is not None:
            gold_position = self.gold_position
        elif self.randomize_gold_position:
            gold_position = random.randint(0, self.num_documents_in_context - 1)
            
        if gold_position is not None:
            indices = indices[:gold_position] + [gold_document_idx] + indices[gold_position:]
            
        return indices, gold_position

    def _update_cluster_assignments(
        self,
        formatted_documents: List[str],
        document_indices: List[int]
    ):
        """Update cluster assignments with logging."""
        self.logger.log_step_start("Updating cluster assignments")
        try:
            texts = [doc.split(') ', 1)[1] if ') ' in doc else doc for doc in formatted_documents]
            assignments = self.clusterer.fit_predict(texts, document_indices)
            self.cluster_assignments.update(assignments)
            
            self.logger.log_metric("clusters_updated", len(assignments))
            self.logger.log_step_end("Cluster assignments updated")
        except Exception as e:
            self.logger.log_error(e, "Error updating cluster assignments")
            raise

    def _get_distractor_documents(self, num_docs: int) -> Tuple[List[str], List[int]]:
        """Get distractor documents that don't contain the answer."""
        available_docs = []
        available_indices = []
        
        for idx, doc in enumerate(self.corpus):
            if not any(is_answer_in_text(doc['text'], ans) for ans in self.answers):
                available_docs.append(doc)
                available_indices.append(idx)
                
        selected_indices = random.sample(
            range(len(available_docs)),
            min(num_docs, len(available_docs))
        )
        
        docs = []
        indices = []
        for idx in selected_indices:
            doc_info = available_docs[idx]
            text = doc_info.get("text", "")
            if self.max_doc_length:
                text = text[:self.max_doc_length]
            doc_str = f"Document [{available_indices[idx]}](Title: {doc_info.get('title', '')}) {text}"
            docs.append(doc_str)
            indices.append(available_indices[idx])
            
        return docs, indices

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item with complete metadata."""
        formatted_docs, document_indices = self.preprocessed_data[idx]
        
        item = {
            "example_id": self.example_ids[idx],
            "query": self.queries[idx],
            "prompt": self.prompts[idx],
            "document_indices": document_indices,
            "gold_document_idx": self.gold_document_idxs[idx],
            "prompt_tokens_len": self.prompt_tokens_lengths[idx]
        }
        
        if self.use_clustering and self.cluster_assignments:
            item["cluster_assignments"] = {
                str(doc_idx): self.cluster_assignments.get(str(doc_idx))
                for doc_idx in document_indices
            }
            
        if self.noise_config:
            item["noise_info"] = {
                "num_noise_docs": self.noise_config.get('num_noise_docs', 0),
                "noise_type": self.noise_config.get('noise_type', 'random'),
                "noise_indices": [i for i, idx in enumerate(document_indices) 
                                if idx in self.noise_config.get('noise_indices', [])]
            }
            
        return item

    def __len__(self) -> int:
        return len(self.example_ids)

class MixedDocumentsDataset(PromptDataset):
    """Dataset for handling mixed document sources with enhanced logging."""
    
    def __init__(
        self, 
        corpus: List[Dict],
        data_path: str,  
        tokenizer: AutoTokenizer,
        max_tokenized_length: int,
        retriever_search_results: List[Tuple[List[int], List[float]]],
        random_search_results: List[Tuple[List[int], List[float]]],
        documents_disposition_info: Dict[str, Any],
        full_to_subset_idx_map: Optional[Dict[int, int]] = None,
        do_normalize_query: bool = False,
        gold_position: Optional[int] = None,
        randomize_gold_position: bool = False,
        get_documents_without_answer: bool = False,
        logger: Optional[ExperimentLogger] = None
    ):
        self.retriever_search_results = retriever_search_results
        self.random_search_results = random_search_results
        self.documents_disposition_info = documents_disposition_info
        self.logger = logger or ExperimentLogger("mixed_documents_dataset", "logs")

        required_keys = ['num_retrieved_documents', 'num_random_documents', 'put_retrieved_first']
        if not all(key in documents_disposition_info for key in required_keys):
            raise ValueError("Missing required keys in documents_disposition_info")

        num_documents_in_context = (
            documents_disposition_info['num_retrieved_documents'] +
            documents_disposition_info['num_random_documents']
        )

        super().__init__(
            corpus=corpus,
            data_path=data_path,
            tokenizer=tokenizer,
            max_tokenized_length=max_tokenized_length,
            search_results=None,
            full_to_subset_idx_map=full_to_subset_idx_map,
            do_normalize_query=do_normalize_query,
            num_documents_in_context=num_documents_in_context,
            gold_position=gold_position,
            randomize_gold_position=randomize_gold_position,
            get_documents_without_answer=get_documents_without_answer,
            logger=logger
        )

    def _mix_documents(
        self, 
        retrieved_indices: List[int], 
        random_indices: List[int]
    ) -> List[int]:
        """Mix retrieved and random documents with logging."""
        self.logger.log_step_start("Mixing documents")
        try:
            num_retrieved = self.documents_disposition_info['num_retrieved_documents']    
            num_random = self.documents_disposition_info['num_random_documents']
            put_retrieved_first = self.documents_disposition_info['put_retrieved_first']

            if put_retrieved_first:
                indices = retrieved_indices[:num_retrieved] + random_indices[:num_random]
            else:
                indices = random_indices[:num_random] + retrieved_indices[:num_retrieved][::-1]

            self.logger.log_metric("mixed_documents_count", len(indices))
            self.logger.log_step_end("Document mixing complete")
            return indices
        except Exception as e:
            self.logger.log_error(e, "Error mixing documents")
            raise

    def _get_indices(self, example_idx: int) -> List[int]:
        """Get indices from both retrieved and random results."""
        self.logger.log_step_start(f"Getting mixed indices for example {example_idx}")
        try:
            retrieved_indices, _ = self.retriever_search_results[example_idx]
            random_indices, _ = self.random_search_results[example_idx]
            mixed_indices = self._mix_documents(retrieved_indices, random_indices)
            
            self.logger.log_metric("total_indices", len(mixed_indices))
            self.logger.log_step_end("Mixed indices retrieval complete")
            return mixed_indices
        except Exception as e:
            self.logger.log_error(e, "Error getting mixed indices")
            raise


class MultiCorpusDataset(PromptDataset):
    """Dataset for handling multiple document corpora with enhanced logging."""
    
    def __init__(
        self,
        corpus: List[Dict],
        data_path: str,
        tokenizer: AutoTokenizer,
        max_tokenized_length: int,
        search_results: List[Tuple[List[str], List[float]]],
        documents_other_corpus: List[str],
        search_results_other_corpus: List[Tuple[List[int], List[float]]],
        documents_disposition_info: Dict[str, Any],
        full_to_subset_idx_map: Optional[Dict[int, int]] = None,
        do_normalize_query: bool = False,
        gold_position: Optional[int] = None,
        randomize_gold_position: bool = False,
        get_documents_without_answer: bool = False,
        logger: Optional[ExperimentLogger] = None
    ):
        self.documents_other_corpus = documents_other_corpus
        self.search_results_other_corpus = search_results_other_corpus
        self.documents_disposition_info = documents_disposition_info
        self.logger = logger or ExperimentLogger("multi_corpus_dataset", "logs")
        required_keys = ['num_main_documents', 'num_other_documents', 'put_main_first']
        if not all(key in documents_disposition_info for key in required_keys):
            raise ValueError("Missing keys in documents_disposition_info")

        num_documents_in_context = (
            documents_disposition_info['num_main_documents'] + 
            documents_disposition_info['num_other_documents']
        )

        super().__init__(
            corpus=corpus,
            data_path=data_path,
            tokenizer=tokenizer,
            max_tokenized_length=max_tokenized_length,
            search_results=search_results,
            full_to_subset_idx_map=full_to_subset_idx_map,
            do_normalize_query=do_normalize_query,
            num_documents_in_context=num_documents_in_context,
            gold_position=gold_position,
            randomize_gold_position=randomize_gold_position,
            get_documents_without_answer=get_documents_without_answer,
            logger=logger
        )

    def prepare_documents_for_prompt(
        self, 
        example_idx: int, 
        gold_document_idx: int, 
        answers: List[str]
    ) -> Tuple[List[str], List[int]]:
        """Prepare documents from multiple corpora."""
        indices_main_corpus = self._get_indices(example_idx)
        indices_main_corpus, gold_position = self._insert_gold_document_idx(
            indices_main_corpus, gold_document_idx
        )
        documents_main_corpus, document_indices_main_corpus = self._get_documents(
            indices_main_corpus, answers, gold_document_idx, gold_position
        )

        indices_other_corpus, _ = self.search_results_other_corpus[example_idx]
        documents_other_corpus, document_indices_other_corpus = self._get_documents_from_indices_other_corpus(
            indices_other_corpus
        )

        merged_documents, merged_document_indices = self._merge_documents(
            documents_main_corpus, documents_other_corpus,
            document_indices_main_corpus, document_indices_other_corpus
        )
        return merged_documents, merged_document_indices

    def _merge_documents(
        self, 
        documents_main: List[str], 
        documents_other: List[str], 
        indices_main: List[int], 
        indices_other: List[int]
    ) -> Tuple[List[str], List[int]]:
        """Merge documents from both corpora."""
        num_main = self.documents_disposition_info.get('num_main_documents', len(documents_main))
        num_other = self.documents_disposition_info.get('num_other_documents', len(documents_other))
        put_main_first = self.documents_disposition_info.get('put_main_first', True)

        if put_main_first:
            merged_documents = documents_main[:num_main] + documents_other[:num_other]
            merged_document_indices = indices_main[:num_main] + indices_other[:num_other]
        else:
            merged_documents = documents_other[:num_other] + documents_main[:num_main][::-1]
            merged_document_indices = indices_other[:num_other] + indices_main[:num_main][::-1]

        return merged_documents, merged_document_indices

    def _get_documents_from_indices_other_corpus(
        self,
        indices: List[int]
    ) -> Tuple[List[str], List[int]]:
        """Get documents from secondary corpus."""
        try:
            formatted_documents = []
            document_indices = []
            
            for idx in indices:
                if 0 <= idx < len(self.documents_other_corpus):
                    doc_info = self.documents_other_corpus[idx]
                    text = doc_info.get("text", "")
                    if self.max_doc_length:
                        text = text[:self.max_doc_length]
                    doc_str = f"Document [{idx}](Title: {doc_info.get('title', '')}) {text}"
                    formatted_documents.append(doc_str)
                    document_indices.append(idx)
                    
                if len(formatted_documents) == self.documents_disposition_info.get('num_other_documents', 0):
                    break

            return formatted_documents, document_indices
            
        except Exception as e:
            self.logger.log_error(e, "Error accessing multi-corpus documents")
            return [], []