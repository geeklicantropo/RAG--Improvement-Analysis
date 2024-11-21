from torch.utils.data import Dataset, IterableDataset
from typing import List, Dict, Optional, Tuple, Iterator, Union
import logging
from pathlib import Path
import json
import ijson
from tqdm import tqdm
import gc
import numpy as np
from src.config import paths, settings

class MemoryEfficientIterator:
    """Memory efficient iterator for JSON files."""
    def __init__(self, file_path: str, batch_size: int = 32):
        self.file_path = file_path
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Dict]:
        with open(self.file_path, 'rb') as f:
            parser = ijson.parse(f)
            current_item = {}
            batch = []
            
            for prefix, event, value in parser:
                if event == 'end_map':
                    if current_item:
                        batch.append(current_item)
                        current_item = {}
                        
                        if len(batch) >= self.batch_size:
                            yield batch
                            batch = []
                            gc.collect()
                elif prefix.endswith('.question'):
                    current_item['question'] = value
                elif prefix.endswith('.example_id'):
                    current_item['example_id'] = value
                elif prefix.endswith('.answers'):
                    current_item['answers'] = value
                elif prefix.endswith('.idx_gold_in_corpus'):
                    current_item['idx_gold_in_corpus'] = value
            
            if batch:
                yield batch

class BaseStreamingDataset(IterableDataset):
    """Base streaming dataset class with memory-efficient loading."""
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_tokenized_length: int,
        batch_size: int = 32,
        do_normalize_query: bool = True,
        experiment_mode: str = 'baseline'
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_tokenized_length = max_tokenized_length
        self.batch_size = batch_size
        self.do_normalize_query = do_normalize_query
        self.experiment_mode = experiment_mode
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def __iter__(self):
        iterator = MemoryEfficientIterator(self.data_path, self.batch_size)
        for batch in iterator:
            for item in batch:
                yield self.process_item(item)

    def normalize_query(self, query: str) -> str:
        return query.strip().lower() if self.do_normalize_query else query

    def process_item(self, item: Dict) -> Dict:
        raise NotImplementedError

class MixedDocumentsDataset(BaseStreamingDataset):
    """Memory-efficient dataset for mixed document experiments."""
    def __init__(
        self,
        corpus: List[Dict],
        data_path: str,
        tokenizer,
        max_tokenized_length: int,
        retriever_search_results: List[Tuple[List[int], List[float]]],
        random_search_results: List[Tuple[List[int], List[float]]],
        documents_disposition_info: Dict,
        full_to_subset_idx_map: Optional[Dict[int, int]] = None,
        batch_size: int = 32,
        do_normalize_query: bool = True,
        experiment_mode: str = 'baseline'
    ):
        super().__init__(
            data_path,
            tokenizer,
            max_tokenized_length,
            batch_size,
            do_normalize_query,
            experiment_mode
        )
        
        # Store search results references (not loading full content)
        self.retriever_search_results = retriever_search_results
        self.random_search_results = random_search_results
        self.documents_disposition_info = documents_disposition_info
        self.full_to_subset_idx_map = full_to_subset_idx_map

        # Create memory-mapped corpus access
        self._setup_corpus_access(corpus)

        self.logger.info(
            f"Initialized MixedDocumentsDataset with streaming access to "
            f"{len(corpus)} documents"
        )

    def _setup_corpus_access(self, corpus: List[Dict]):
        """Setup memory-efficient corpus access."""
        self.corpus_path = paths.CORPUS_DIR / "corpus_mapped.npy"
        if not self.corpus_path.exists():
            # Create memory-mapped array for corpus
            corpus_array = np.array([(doc['text'], doc.get('title', '')) for doc in corpus],
                                 dtype=[('text', 'U1000'), ('title', 'U100')])
            np.save(self.corpus_path, corpus_array)
        self.corpus = np.load(self.corpus_path, mmap_mode='r')

    def retrieve_documents(self, idx: int, doc_type: str) -> List[str]:
        """Memory-efficient document retrieval."""
        if doc_type == 'retriever':
            doc_indices = self.retriever_search_results[idx][0]
        else:  # random
            doc_indices = self.random_search_results[idx][0]
            
        # Access documents through memory mapping
        return [self.corpus[i]['text'] for i in doc_indices]

    def process_item(self, item: Dict) -> Dict:
        """Process a single item with memory efficiency."""
        query = self.normalize_query(item['question'])
        
        # Get required documents based on experiment mode
        if self.experiment_mode == 'baseline':
            retrieved_docs = self.retrieve_documents(item['example_id'], 'retriever')
            random_docs = self.retrieve_documents(item['example_id'], 'random')
        else:
            retrieved_docs = self.retrieve_documents(item['example_id'], 'retriever')
            random_docs = self.retrieve_documents(item['example_id'], 'random')

        # Build context efficiently
        if self.documents_disposition_info['put_retrieved_first']:
            context_parts = (
                retrieved_docs[:self.documents_disposition_info['num_retrieved_documents']] +
                random_docs[:self.documents_disposition_info['num_random_documents']]
            )
        else:
            context_parts = (
                random_docs[:self.documents_disposition_info['num_random_documents']] +
                retrieved_docs[:self.documents_disposition_info['num_retrieved_documents']]
            )
        
        context = " ".join(context_parts)

        # Tokenize efficiently
        inputs = self.tokenizer(
            query,
            context,
            max_length=self.max_tokenized_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'example_id': item.get('example_id', -1),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'query': query,
            'prompt': f"Question: {query}\nContext: {context}\nAnswer:",
            'document_indices': list(range(len(context_parts))),
            'gold_document_idx': item.get('idx_gold_in_corpus', -1),
            'prompt_tokens_len': len(inputs['input_ids'].squeeze())
        }

class QueryDataset(BaseStreamingDataset):
    """Memory-efficient dataset for query-only generation."""
    def __init__(
        self,
        data_path: str,
        tokenizer,
        model_name: str,
        batch_size: int = 32,
        do_normalize_query: bool = True
    ):
        super().__init__(
            data_path,
            tokenizer,
            tokenizer.model_max_length - 2,
            batch_size,
            do_normalize_query,
            'query_only'
        )
        self.model_name = model_name
        self.logger.info("Initialized streaming QueryDataset")

    def process_item(self, item: Dict) -> Dict:
        query = self.normalize_query(item['question'])
        prompt = f"Question: {query}\nAnswer:"

        inputs = self.tokenizer(
            prompt,
            max_length=self.max_tokenized_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'example_id': item.get('example_id', -1),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'query': query,
            'prompt': prompt,
            'prompt_tokens_len': len(inputs['input_ids'].squeeze())
        }