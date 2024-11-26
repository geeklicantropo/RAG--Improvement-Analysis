import json
import random
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging
import warnings

import normalize_text
from normalize_answers import *
from cluster_utils import DocumentClusterer

class QueryDataset(Dataset):
    """
    A dataset class for managing queries data into structured prompts suitable for input to LLMS.

    Attributes:
        data_path (str): Path to the dataset file containing the query and related information.
        model_name (str): The name of the language model used for generating answers.
        do_normalize_query (bool): Flag to determine if text normalization is applied to the query.
    """
    def __init__(
        self, 
        data_path: str, 
        model_name: str,
        do_normalize_query: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.model_name = model_name
        self.do_normalize_query = do_normalize_query
        self._load_data()

    def _load_data(self):
        """
        Loads data from the specified path and processes it.
        """
        try:
            with open(self.data_path, "r") as fin:
                data = json.load(fin)
            self.process_file_data(data)
        except IOError as e:
            # Replace print with logging.error
            logging.error(f"Error reading file {self.data_path}: {e}")

    def process_file_data(self, data: List[Dict]):
        """
        Processes each example in the dataset to prepare prompts for the LLM.
        Silent processing with logging at appropriate levels.
        """
        self.questions = []
        self.example_ids = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for idx, example in enumerate(data):
                try:
                    self.example_ids.append(example['example_id'])

                    if 'query' in example:
                        question = example['query']
                    elif 'question' in example:
                        question = example['question']
                    else:
                        logging.error(f"Example {idx}: Missing query and question keys")
                        raise ValueError("No 'query' or 'question' key in example")
                    
                    if self.do_normalize_query:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            question = normalize_text.normalize(question)
                            
                    self.questions.append(question)

                    # Log progress only occasionally for large datasets
                    if idx % 1000 == 0 and idx > 0:
                        logging.debug(f"Processed {idx} examples")

                except Exception as e:
                    logging.error(f"Error processing example {idx}: {str(e)}")
                    continue

            # Log final statistics at debug level
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Processing complete: {len(self.example_ids)} examples processed")

    def build_qa_prompt(
        self,
        query: str,
        documents_str: str,
        organization_type: str = "standard"
    ) -> str:
        """
        Build a prompt with appropriate formatting based on document organization type.
        
        Args:
            query: The question to be answered
            documents_str: The context documents
            organization_type: How documents are organized ('standard', 'clustered', 'categorized', or 'fused')
            
        Returns:
            Formatted prompt string
        """
        # Base task instruction
        base_instruction = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."
        
        # Organization-specific instructions
        organization_instructions = {
            "standard": "",
            "clustered": "The documents are grouped into clusters based on their semantic similarity. Consider the relationships between documents within each cluster when searching for the answer.",
            "categorized": "The documents are organized into categories. Pay attention to the category labels as they indicate the type of information in each document.",
            "fused": "The documents are ranked by multiple retrieval methods. Higher-ranked documents are more likely to contain relevant information."
        }
        
        # Combine instructions
        task_instruction = f"{base_instruction} {organization_instructions.get(organization_type, '')}"
        
        # Build full prompt
        prompt = f"{task_instruction}\nDocuments:\n{documents_str}\nQuestion: {query}\nAnswer:"
        
        # Custom prompt format for MPT models
        if 'mpt' in self.tokenizer.name_or_path:
            INSTRUCTION_KEY = "### Instruction:"
            RESPONSE_KEY = "### Response:"
            INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            PROMPT_FOR_GENERATION_FORMAT = """{intro}\n{instruction_key}\n{instruction}\n{response_key}""".format(
                intro=INTRO_BLURB,
                instruction_key=INSTRUCTION_KEY,
                instruction="{instruction}",
                response_key=RESPONSE_KEY,
            )
            prompt = PROMPT_FOR_GENERATION_FORMAT.format(
                instruction=prompt[:-8]
            )
            
        return prompt

    def __getitem__(self, idx: int):   
        prompt = self.build_qa_prompt(self.questions[idx])

        return {
            "example_id": self.example_ids[idx],
            "query": self.questions[idx],
            "prompt": prompt,
        }

    def __len__(self):
        return len(self.example_ids)

def hash_document(text: str) -> str:
    """
    Generate a SHA-256 hash for a given text.
    """
    return hashlib.sha256(text.encode()).hexdigest()

class PromptDataset(Dataset):
    """
    A dataset class for managing, preprocessing, and organizing document data into structured prompts suitable for input to LLMs.

    Attributes:
        corpus (List[Dict]): The list containing the document corpus.
        data_path (str): Path to the dataset file containing the query and related information.
        tokenizer (AutoTokenizer): The tokenizer used to tokenize the prompt.
        max_tokenized_length (int): The maximum length of tokenized prompt.
        search_results (List[Tuple[List[str], List[float]]]): A list of tuples containing document indices and their scores.
        full_to_subset_idx_map (Dict[int, int]): Dictionary that maps the indices in the full corpus to the given subset.
        do_normalize_query (bool): Flag to determine if text normalization is applied to the query.
        num_documents_in_context (int): The total number of documents to consider in the context.
        gold_position (int): The specific position (0-indexed) of the gold document in the context.
        randomize_gold_position (bool): Flag to determine if the gold document position should be random.
        get_documents_without_answer (bool): Flag to determine if documents without the answer should be included.
        use_clustering (bool): Whether to use document clustering.
        num_clusters (Optional[int]): Number of clusters when clustering is enabled.
        cluster_seed (int): Random seed for clustering.
        category_info (Optional[Dict[str, Any]]): Information for category-based organization.
    """
    def __init__(
        self, 
        corpus: List[Dict],
        data_path: str,  
        tokenizer: AutoTokenizer,
        max_tokenized_length: int,
        search_results: List[Tuple[List[int], List[float]]],
        full_to_subset_idx_map: Dict[int, int] = None,
        do_normalize_query: bool = False,
        num_documents_in_context: int = 5,
        gold_position: int = None,
        randomize_gold_position: bool = False,
        get_documents_without_answer: bool = False,
        use_clustering: bool = False,
        num_clusters: Optional[int] = None,
        cluster_seed: int = 42,
        category_info: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        # Core parameters
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

        # Clustering configuration
        self.use_clustering = use_clustering
        if use_clustering and num_clusters is None:
            raise ValueError("num_clusters must be specified when use_clustering is True")
        self.clusterer = DocumentClusterer(num_clusters, cluster_seed) if use_clustering else None
        
        # Category organization
        self.category_info = category_info

        # Add preprocessing step before validation and data loading
        self.preprocess_search_results()
        
        self._validate_initialization_parameters()
        self._load_data()

    def _validate_initialization_parameters(self):
        """Validates initialization parameters for logical consistency and correctness."""
        if self.num_documents_in_context <= 0:
            raise ValueError("num_documents_in_context must be positive.")
        
        if self.max_tokenized_length <= 0:
            raise ValueError("max_tokenized_length must be positive.")

        if self.gold_position is not None:
            if self.gold_position < 0 or (self.gold_position >= self.num_documents_in_context):
                raise ValueError(f"Invalid gold position: {self.gold_position}")
        
        if self.gold_position is not None and self.randomize_gold_position:
            raise ValueError("Both 'gold_position' and 'randomize_gold_position' cannot be set at the same time.")

    def _load_data(self):
        """
        Loads data from the specified path and processes it.
        """
        try:
            with open(self.data_path, "r") as fin:
                data = json.load(fin)
            self.process_file_data(data)
        except IOError as e:
            print(f"Error reading file {self.data_path}: {e}")

    def process_file_data(self, data: List[Dict]):  
        """
        Processes each example in the dataset to prepare prompts for the LLM.
        """
        self.example_ids = []
        self.queries = []
        self.prompts = []
        self.gold_document_idxs = []
        self.excluded_samples_ids = []
        self.preprocessed_data = []
        self.prompt_tokens_lengths = []
        self.cluster_assignments = {} if self.use_clustering else None

        for idx, example in enumerate(data):
            example_id = str(example['example_id'])
            gold_document_idx = str(example['idx_gold_in_corpus'])
            answers = example['answers']

            formatted_documents, document_indices = self.prepare_documents_for_prompt(
                idx, gold_document_idx, answers
            )

            if len(formatted_documents) != self.num_documents_in_context:
                print(f"Warning: Not enough documents for example {idx}.")
                continue

            # Build the prompt with appropriate structure
            documents_str = self._format_documents_for_prompt(formatted_documents, document_indices)
            
            query = example['question']
            if self.do_normalize_query:
                query = normalize_text.normalize(query)
            prompt = self.build_qa_prompt(query, documents_str)

            # Check token length
            tokens = self.tokenizer.tokenize(prompt)
            tokens_len = len(tokens)
            if tokens_len >= self.max_tokenized_length:
                self.excluded_samples_ids.append((idx, example_id))
                print(f"Skipping example {idx} due to prompt length.")
                continue

            self.preprocessed_data.append((formatted_documents, document_indices))
            self.example_ids.append(example_id)
            self.queries.append(query)
            self.prompts.append(prompt)
            self.gold_document_idxs.append(gold_document_idx)
            self.prompt_tokens_lengths.append(tokens_len)

    def _format_documents_for_prompt(
        self,
        formatted_documents: List[str],
        document_indices: List[int]
    ) -> str:
        """Format documents based on clustering or categories if enabled."""
        if self.use_clustering and self.cluster_assignments:
            return self._format_clustered_documents(formatted_documents, document_indices)
        elif self.category_info:
            return self._format_categorized_documents(formatted_documents, document_indices)
        else:
            return '\n'.join(formatted_documents)

    def _format_clustered_documents(
        self,
        formatted_documents: List[str],
        document_indices: List[int]
    ) -> str:
        """Format documents grouped by their clusters."""
        clusters = {}
        for doc, idx in zip(formatted_documents, document_indices):
            cluster_id = self.cluster_assignments.get(str(idx), 0)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(doc)
        
        result = []
        for cluster_id in sorted(clusters.keys()):
            result.append(f"\nCluster {cluster_id}:")
            result.extend(clusters[cluster_id])
        
        return '\n'.join(result)

    def _format_categorized_documents(
        self,
        formatted_documents: List[str],
        document_indices: List[int]
    ) -> str:
        """Format documents grouped by their categories."""
        if not self.category_info:
            return '\n'.join(formatted_documents)
            
        categories = {}
        for doc, idx in zip(formatted_documents, document_indices):
            category = self.category_info.get(str(idx), "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(doc)
        
        result = []
        for category in sorted(categories.keys()):
            result.append(f"\nCategory: {category}")
            result.extend(categories[category])
        
        return '\n'.join(result)
    
    def _format_fused_documents(
        self,
        formatted_documents: List[str],
        document_indices: List[int],
        fusion_scores: Dict[str, float]
    ) -> str:
        """Format documents with fusion information."""
        documents_with_scores = []
        for doc, idx in zip(formatted_documents, document_indices):
            score = fusion_scores.get(str(idx), 0.0)
            documents_with_scores.append((doc, score))
        
        # Sort by fusion score
        documents_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for doc, score in documents_with_scores:
            confidence_level = "High" if score > 0.7 else "Medium" if score > 0.3 else "Low"
            result.append(f"[Confidence: {confidence_level}]\n{doc}")
        
        return '\n'.join(result)

    def prepare_documents_for_prompt(
        self, 
        example_idx: int, 
        gold_document_idx: int, 
        answers: List[str]
    ) -> Tuple[List[str], List[int]]:
        """
        Prepares and formats a set of documents for inclusion in a prompt, including the insertion
        of a gold document at the appropriate position.
        """
        indices = self._get_indices(example_idx)
        updated_indices, gold_position = self._insert_gold_document_idx(
            indices, gold_document_idx
        )

        # Get the documents and their indices
        formatted_documents, document_indices = self._get_documents(
            updated_indices, answers, gold_document_idx, gold_position
        )

        # Apply clustering if enabled
        if self.use_clustering and formatted_documents:
            self._update_cluster_assignments(formatted_documents, document_indices)

        return formatted_documents, document_indices

    def _update_cluster_assignments(
        self,
        formatted_documents: List[str],
        document_indices: List[int]
    ):
        """Update cluster assignments for the current batch of documents."""
        # Extract text content for clustering
        texts = [doc.split(') ', 1)[1] if ') ' in doc else doc for doc in formatted_documents]
        
        # Compute clusters for documents
        new_assignments = self.clusterer.fit_predict(texts, document_indices)
        self.cluster_assignments.update(new_assignments)

    def _get_indices(self, example_idx: int) -> List[int]:
        """
        Get indices in the corpus of the documents retrieved by a retriever with validation.
        
        Args:
            example_idx: Index of the example in the dataset
            
        Returns:
            List of valid corpus indices
            
        Raises:
            ValueError: If there is an index mismatch or invalid indices
        """
        # Validate example index against search results
        if not self.search_results:
            raise ValueError("No search results available")
            
        if example_idx >= len(self.search_results):
            raise ValueError(f"Example index {example_idx} out of range for search results of length {len(self.search_results)}")

        # Get search result indices and validate
        indices, scores = self.search_results[example_idx]
        
        # Convert to integers and validate corpus bounds
        try:
            valid_indices = []
            max_corpus_idx = len(self.corpus) - 1
            
            for idx in indices:
                corpus_idx = int(idx)  # Convert to int
                
                # Handle index mapping if using subset
                if self.full_to_subset_idx_map is not None:
                    if corpus_idx not in self.full_to_subset_idx_map:
                        continue
                    corpus_idx = self.full_to_subset_idx_map[corpus_idx]
                    
                # Validate corpus bounds
                if 0 <= corpus_idx <= max_corpus_idx:
                    valid_indices.append(corpus_idx)
                
            # Ensure we have enough valid indices
            if len(valid_indices) < self.num_documents_in_context:
                # If not enough valid indices, find additional valid indices
                current_indices = set(valid_indices)
                additional_needed = self.num_documents_in_context - len(valid_indices)
                
                # Find additional valid indices from corpus
                additional_indices = []
                for i in range(max_corpus_idx + 1):
                    if len(additional_indices) >= additional_needed:
                        break
                    if i not in current_indices:
                        additional_indices.append(i)
                        
                valid_indices.extend(additional_indices[:additional_needed])
                
            return valid_indices[:self.num_documents_in_context]
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid index in search results: {str(e)}")
        
    def preprocess_search_results(self) -> bool:
        """
        Preprocess and validate search results before dataset creation.
        
        Returns:
            bool: True if preprocessing successful, False otherwise
            
        Raises:
            ValueError: If search results are invalid
        """
        if not self.search_results:
            raise ValueError("No search results available")
            
        # Validate format
        if not all(isinstance(result, tuple) and len(result) == 2 for result in self.search_results):
            raise ValueError("Invalid search results format")
            
        processed_results = []
        max_corpus_idx = len(self.corpus) - 1
        
        for indices, scores in self.search_results:
            # Validate and convert indices
            valid_indices = []
            valid_scores = []
            
            for idx, score in zip(indices, scores):
                try:
                    corpus_idx = int(idx)
                    if self.full_to_subset_idx_map is not None:
                        if corpus_idx not in self.full_to_subset_idx_map:
                            continue
                        corpus_idx = self.full_to_subset_idx_map[corpus_idx]
                            
                    if 0 <= corpus_idx <= max_corpus_idx:
                        valid_indices.append(corpus_idx)
                        valid_scores.append(score)
                        
                except (ValueError, TypeError):
                    continue
                    
            # Ensure minimum required indices
            while len(valid_indices) < self.num_documents_in_context:
                # Find unused valid index
                for i in range(max_corpus_idx + 1):
                    if i not in valid_indices:
                        valid_indices.append(i)
                        valid_scores.append(0.0)  # Default score for added indices
                        break
                        
            processed_results.append((valid_indices[:self.num_documents_in_context], 
                                    valid_scores[:self.num_documents_in_context]))
                                    
        self.search_results = processed_results
        return True

    def _insert_gold_document_idx(
        self, 
        indices: List[int], 
        gold_document_idx: int
    ) -> Tuple[List[int], int]:
        """
        Inserts the index of a gold document into the provided list of indices
        at a specified or random position.
        """
        gold_position = None
        
        if self.gold_position is not None:
            # Direct insertion
            gold_position = self.gold_position
            indices = indices[:gold_position] + [gold_document_idx] + indices[gold_position:]
        elif self.randomize_gold_position:
            # Insert at a random position
            gold_position = random.randint(0, self.num_documents_in_context - 1)
            indices = indices[:gold_position] + [gold_document_idx] + indices[gold_position:]
        return indices, gold_position

    def _get_documents(    
        self,
        indices: List[int],
        answers: List[str],
        gold_document_idx: Optional[int],
        gold_position: Optional[int]
    ) -> Tuple[List[str], List[int]]:
        """ Choose the appropriate method based on the flag """
        if self.get_documents_without_answer:
            return self._get_answerless_documents_from_indices(
                indices, answers, gold_document_idx, gold_position
            )
        else:
            return self._get_documents_from_indices(indices)

    def _get_documents_from_indices(self, indices: List[int]) -> Tuple[List[str], List[int]]:
        """
        Selects documents from the corpus based on provided indices with robust index handling.
        """
        try:
            formatted_documents = []
            document_indices = []
            seen_hashes = set()
            
            # Validate indices before mapping
            valid_indices = []
            max_idx = len(self.corpus) - 1
            
            for i in map(int, indices):
                # Handle index mapping if using subset
                if self.full_to_subset_idx_map is not None:
                    # Only proceed if index exists in mapping
                    if i in self.full_to_subset_idx_map:
                        mapped_idx = self.full_to_subset_idx_map[i]
                        if 0 <= mapped_idx <= max_idx:
                            valid_indices.append(mapped_idx)
                else:
                    # Direct index validation for full corpus
                    if 0 <= i <= max_idx:
                        valid_indices.append(i)
                        
            # Process valid indices
            for idx in valid_indices:
                doc_info = self.corpus[idx]
                doc_idx = doc_info.get('full_corpus_idx', idx)
                title = doc_info.get('title', '')
                text = doc_info.get('text', '')
                
                doc_hash = hash_document(text)
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)
                
                doc_str = f"Document [{doc_idx}](Title: {title}) {text}"
                formatted_documents.append(doc_str)
                document_indices.append(doc_idx)
                
                if len(formatted_documents) == self.num_documents_in_context:
                    break
                    
            return formatted_documents, document_indices
            
        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")
            return [], []

    def _get_answerless_documents_from_indices(
        self,
        indices: List[int],
        answers: List[str],
        gold_document_idx: Optional[int],
        gold_position: Optional[int]
    ) -> Tuple[List[str], List[int]]:
        """
        Selects documents from the corpus that do not contain any of the given answers,
        optionally including a specific 'gold' document at a designated position.
        """
        # Full corpus
        if self.full_to_subset_idx_map is None:
            documents_info = [self.corpus[i] for i in map(int, indices)]
        else: 
            documents_info: List[Dict] = []
            # 'indices' are from the full corpus, so we need to map them to the subset
            for i in map(int, indices):
                documents_info.append(self.corpus[self.full_to_subset_idx_map[i]])

        answerless_documents = []
        gold_document = None
        seen_hashes = set()
        # List to store the indices of documents actually added
        document_indices = [] 

        for doc_info in documents_info:
            doc_idx = doc_info['full_corpus_idx']
            title = doc_info['title']
            text = doc_info['text']

            doc_hash = hash_document(text)
            # Skip the document if it's a duplicate
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)

            if str(doc_idx) == gold_document_idx:
                gold_document = f"Document [{doc_idx}](Title: {title}) {text}"
                continue
            
            if not is_answer_in_text(text, answers):
                answerless_doc = f"Document [{doc_idx}](Title: {title}) {text}"
                answerless_documents.append(answerless_doc)
                document_indices.append(doc_idx)

        # Insert gold document at the specified/random position
        if gold_position is not None and gold_document is not None:
            gold_position = min(gold_position, len(answerless_documents))
            answerless_documents.insert(gold_position, gold_document)
            document_indices.insert(gold_position, gold_document_idx)

        # Limit the number of documents to the specified context size
        docs = answerless_documents[:self.num_documents_in_context]
        indices = document_indices[:self.num_documents_in_context]
        return docs, indices

    def build_qa_prompt(self, query: str, documents_str: str) -> str:
        task_instruction = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."
        prompt = f"""{task_instruction}\nDocuments:\n{documents_str}\nQuestion: {query}\nAnswer:"""

        # Custom prompt format for mpt models
        if 'mpt' in self.tokenizer.name_or_path:
            INSTRUCTION_KEY = "### Instruction:"
            RESPONSE_KEY = "### Response:"
            INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            PROMPT_FOR_GENERATION_FORMAT = """{intro}\n{instruction_key}\n{instruction}\n{response_key}""".format(
                intro=INTRO_BLURB,
                instruction_key=INSTRUCTION_KEY,
                instruction="{instruction}",
                response_key=RESPONSE_KEY,
            )
            prompt = PROMPT_FOR_GENERATION_FORMAT.format(
                instruction=prompt[:-8]
            )

        return prompt

    def __getitem__(self, idx: int):
        """Get a specific item from the dataset."""
        _, document_indices = self.preprocessed_data[idx]
        
        item = {
            "example_id": self.example_ids[idx],
            "query": self.queries[idx],
            "prompt": self.prompts[idx],
            "document_indices": document_indices,
            "gold_document_idx": self.gold_document_idxs[idx],
            "prompt_tokens_len": self.prompt_tokens_lengths[idx]
        }
        
        # Add clustering information if available
        if self.use_clustering and self.cluster_assignments:
            item["cluster_assignments"] = {
                str(doc_idx): self.cluster_assignments.get(str(doc_idx))
                for doc_idx in document_indices
            }
            
        return item

    def __len__(self):
        return len(self.example_ids)


class MixedDocumentsDataset(PromptDataset):
    """
    Extends the prompt dataset for creating prompts containing a mixed of retrieved and randomly selected documents.
    """
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
    ):
        self.retriever_search_results = retriever_search_results
        self.random_search_results = random_search_results
        self.documents_disposition_info = documents_disposition_info

        # Validate 'documents_disposition_info' contains all necessary keys
        required_keys = ['num_retrieved_documents', 'num_random_documents', 'put_retrieved_first']
        if not all(key in documents_disposition_info for key in required_keys):
            raise ValueError("Missing keys in 'documents_disposition_info'.")

        num_documents_in_context = documents_disposition_info['num_retrieved_documents'] + \
                                   documents_disposition_info['num_random_documents']

        super().__init__(
            corpus=corpus,
            data_path=data_path,
            tokenizer=tokenizer,
            max_tokenized_length=max_tokenized_length,
            search_results=None, # Handled separately in this subclass.
            full_to_subset_idx_map=full_to_subset_idx_map,
            do_normalize_query=do_normalize_query,
            num_documents_in_context=num_documents_in_context,
            gold_position=gold_position,
            randomize_gold_position=randomize_gold_position,
            get_documents_without_answer=get_documents_without_answer,
        )

    def _get_indices(self, example_idx: int) -> List[int]:
        """ Overridden method that selects and properly mixes the indices from the retrieved and random search results. """
        retrieved_indices, _ = self.retriever_search_results[example_idx]
        random_indices, _ = self.random_search_results[example_idx]
        indices = self._mix_documents(retrieved_indices, random_indices)
        return indices

    def _mix_documents(
        self, 
        retrieved_indices: List[int], 
        random_indices: List[int]
    ) -> List[int]:
        """ Mixes retrieved and random document indices according to the documents disposition configuration. """
        num_retrieved_documents = self.documents_disposition_info['num_retrieved_documents']    
        num_random_documents = self.documents_disposition_info['num_random_documents']
        put_retrieved_first = self.documents_disposition_info['put_retrieved_first']

        indices = []
        if put_retrieved_first:
            indices = retrieved_indices[:num_retrieved_documents] + random_indices[:num_random_documents]
        else:
            # Retrieved documents are reversed ([::-1]), so that the documents with higher scores are at the end
            indices = random_indices[:num_random_documents] + retrieved_indices[:num_retrieved_documents][::-1]
        return indices



class MultiCorpusDataset(PromptDataset):
    """
    Extends PromptDataset to handle multiple corpora, merging documents from the main and another corpus
    based on specified disposition info to create prompts for LLMs.
    """
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
    ):
        self.documents_other_corpus = documents_other_corpus
        self.search_results_other_corpus = search_results_other_corpus
        self.documents_disposition_info = documents_disposition_info

        # Validate 'documents_disposition_info' contains all necessary keys
        required_keys = ['num_main_documents', 'num_other_documents', 'put_main_first']
        if not all(key in documents_disposition_info for key in required_keys):
            raise ValueError("Missing keys in 'documents_disposition_info'.")

        num_documents_in_context = documents_disposition_info['num_main_documents'] + \
                                   documents_disposition_info['num_other_documents']

        # Initialize inherited attributes from the PromptDataset class
        super().__init__(
            corpus=corpus,
            data_path=data_path,
            search_results=search_results,
            tokenizer=tokenizer,
            max_tokenized_length=max_tokenized_length,
            full_to_subset_idx_map=full_to_subset_idx_map,
            do_normalize_query=do_normalize_query,
            num_documents_in_context=num_documents_in_context,
            gold_position=gold_position,
            randomize_gold_position=randomize_gold_position,
            get_documents_without_answer=get_documents_without_answer,
        )

    def prepare_documents_for_prompt(
        self, 
        example_idx: int, 
        gold_document_idx: int, 
        answers: List[str]
    ) -> Tuple[List[str], List[int]]:
        """ Overridden method to prepare and merge documents from both the main and additional corpora for prompt creation. """
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
        """
        Merges documents from the main and additional corpora based on the criteria specified in documents_disposition_info.
        """
        num_main_documents = self.documents_disposition_info['num_main_documents']    
        num_other_documents = self.documents_disposition_info['num_other_documents']
        put_main_first = self.documents_disposition_info['put_main_first']

        if put_main_first:
            merged_documents = documents_main[:num_main_documents] + documents_other[:num_other_documents]
            merged_document_indices = indices_main[:num_main_documents] + indices_other[:num_other_documents]
        else:
            # Retrieved documents are reversed ([::-1]), so that the documents with higher scores are at the end
            merged_documents = documents_other[:num_other_documents] + documents_main[:num_main_documents][::-1]
            merged_document_indices = indices_other[:num_other_documents] + indices_main[:num_main_documents][::-1]
        return merged_documents, merged_document_indices

    def _get_documents_from_indices(self, indices: List[int]) -> Tuple[List[str], List[int]]:
        """
        Safely get documents from corpus using indices, with validation.
        """
        formatted_documents = []
        document_indices = []
        
        # Full corpus
        try:
            if self.full_to_subset_idx_map is None:
                # Validate indices are within corpus bounds
                max_idx = len(self.corpus) - 1
                valid_indices = [i for i in map(int, indices) if 0 <= i <= max_idx]
                documents_info = [self.corpus[i] for i in valid_indices]
            else:
                documents_info = []
                # Map indices and validate
                for i in map(int, indices):
                    if i in self.full_to_subset_idx_map:
                        subset_idx = self.full_to_subset_idx_map[i]
                        if 0 <= subset_idx < len(self.corpus):
                            documents_info.append(self.corpus[subset_idx])
            
            # Process valid documents
            seen_hashes = set()
            for doc_info in documents_info:
                if len(formatted_documents) == self.num_documents_in_context:
                    break
                
                doc_idx = doc_info['full_corpus_idx']
                title = doc_info.get('title', '')
                text = doc_info.get('text', '')

                doc_hash = hash_document(text)
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)
                
                doc_str = f"Document [{doc_idx}](Title: {title}) {text}"
                formatted_documents.append(doc_str)
                document_indices.append(doc_idx)

            return formatted_documents, document_indices
            
        except Exception as e:
            logging.error(f"Error accessing corpus documents: {str(e)}")
            # Return empty results rather than failing
            return [], []