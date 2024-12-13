from torch.utils.data import Dataset, IterableDataset
from typing import List, Dict, Tuple, Optional, Any, Iterator
from collections import defaultdict
import torch
import logging
from tqdm import tqdm
import json
import gc
from src.llm import LLM 
from pathlib import Path
import ijson
from torch.utils.data import DataLoader

class BaseDataset(Dataset):
    def __init__(
        self,
        corpus_path: str,
        tokenizer: 'PreTrainedTokenizer',
        llm_evaluator: LLM,
        cache_dir: Optional[str] = "cache/dataset",
        batch_size: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.llm_evaluator = llm_evaluator
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.corpus_stream = self._create_corpus_stream(corpus_path)
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def _create_corpus_stream(self, corpus_path: str) -> Iterator:
        with open(corpus_path, 'rb') as f:
            return ijson.items(f, 'item')

class DocumentCategorizationMixin:
    def _determine_category(
        self,
        doc: Dict,
        gold_doc: Dict,
        threshold: float
    ) -> str:
        if doc['id'] == gold_doc['id']:
            return 'gold'
            
        try:
            similarity = self.llm_evaluator.compute_semantic_similarity(
                doc['text'],
                gold_doc['text']
            )
            return 'distracting' if similarity >= threshold else 'random'
        except Exception as e:
            self.logger.warning(f"Similarity computation failed: {str(e)}")
            return 'random'

    def _determine_position(self, idx: int, total_docs: int) -> str:
        """Determine document position category (near/mid/far)."""
        position_ratio = idx / total_docs
        if position_ratio < 0.33:
            return 'near'
        elif position_ratio < 0.66:
            return 'mid'
        return 'far'

class PromptDataset(BaseDataset, DocumentCategorizationMixin):
    def __init__(
        self,
        corpus: List[Dict],
        data_path: str,
        tokenizer: 'PreTrainedTokenizer',
        llm_evaluator: LLM,
        max_tokenized_length: int,
        search_results: List[Tuple[List[int], List[float]]],
        num_documents_in_context: int,
        semantic_threshold: float = 0.7,
        get_documents_without_answer: bool = False,
        gold_position: Optional[int] = None,
        max_doc_length: Optional[int] = None
    ):
        super().__init__(data_path, tokenizer, llm_evaluator)
        self.corpus = corpus
        self.max_tokenized_length = max_tokenized_length
        self.search_results = search_results
        self.num_documents_in_context = num_documents_in_context
        self.semantic_threshold = semantic_threshold
        self.get_documents_without_answer = get_documents_without_answer
        self.gold_position = gold_position
        self.max_doc_length = max_doc_length
        self.examples = self._load_examples_stream(data_path)
        self.prompts = self._create_prompts_batched()

    def _load_examples_stream(self, data_path: str) -> Iterator:
        with open(data_path, 'rb') as f:
            return ijson.items(f, 'item')

    def _create_prompts_batched(self) -> List[Dict]:
        prompts = []
        batch = []
        
        for example in tqdm(self.examples, desc="Creating prompts"):
            batch.append(example)
            
            if len(batch) >= self.batch_size:
                prompts.extend(self._process_batch(batch))
                batch = []
                gc.collect()
                
        if batch:
            prompts.extend(self._process_batch(batch))
            
        return prompts

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        processed = []
        for example in batch:
            doc_indices, scores = self.search_results[example['id']]
            gold_doc = next(self.corpus_stream)
            
            documents = [
                {'id': idx, 'text': next(self.corpus_stream)['text'], 'score': score}
                for idx, score in zip(doc_indices[:self.num_documents_in_context], scores)
            ]
            
            categorized_docs = self._categorize_documents(
                documents,
                gold_doc,
                self.semantic_threshold
            )
            
            prompt = self._format_prompt(example['question'], categorized_docs)
            if self._validate_prompt(prompt):
                processed.append({
                    'prompt': prompt,
                    'query': example['question'],
                    'document_indices': [d['id'] for d in categorized_docs],
                    'document_categories': [d['category'] for d in categorized_docs],
                    'document_positions': [d['position'] for d in categorized_docs],
                    'example_id': example['id']
                })
                
        return processed

    def _categorize_documents(
        self,
        documents: List[Dict],
        gold_doc: Dict,
        threshold: float
    ) -> List[Dict]:
        categorized = []
        total_docs = len(documents)
        
        for idx, doc in enumerate(documents):
            category = self._determine_category(doc, gold_doc, threshold)
            position = self._determine_position(idx, total_docs)
            
            doc['category'] = category
            doc['position'] = position
            categorized.append(doc)
            
        # Handle gold document positioning if specified
        if self.gold_position is not None:
            gold_idx = next((i for i, d in enumerate(categorized) if d['category'] == 'gold'), None)
            if gold_idx is not None and gold_idx != self.gold_position:
                categorized.insert(self.gold_position, categorized.pop(gold_idx))

        return categorized

    def _format_prompt(self, question: str, documents: List[Dict]) -> str:
        context_parts = [f"Question: {question}\n\nContext:"]
        for idx, doc in enumerate(documents, 1):
            context_parts.append(
                f"\nDocument [{idx}] (Category: {doc['category']}, Position: {doc['position']}):\n{doc['text']}"
            )
        context_parts.append("\nAnswer:")
        return "\n".join(context_parts)

    def _validate_prompt(self, prompt: str) -> bool:
        return len(self.tokenizer.encode(prompt)) <= self.max_tokenized_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.prompts[idx]


class StreamingPromptDataset(IterableDataset, DocumentCategorizationMixin):
    def __init__(
        self,
        corpus_path: str,
        data_path: str,
        tokenizer: 'PreTrainedTokenizer',
        llm_evaluator: LLM,
        max_tokenized_length: int,
        search_results: List[Tuple[List[int], List[float]]],
        num_documents_in_context: int,
        semantic_threshold: float = 0.7,
        batch_size: int = 1000
    ):
        self.corpus_path = corpus_path
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.llm_evaluator = llm_evaluator
        self.max_tokenized_length = max_tokenized_length
        self.search_results = search_results
        self.num_documents_in_context = num_documents_in_context
        self.semantic_threshold = semantic_threshold
        self.batch_size = batch_size
        self._setup_logger()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        
        with open(self.data_path, 'rb') as f:
            examples = ijson.items(f, 'item')
            
            if worker_info is not None:
                examples = self._split_for_worker(examples, worker_info)
                
            batch = []
            for example in examples:
                batch.append(example)
                
                if len(batch) >= self.batch_size:
                    yield from self._process_batch(batch)
                    batch = []
                    gc.collect()
                    
            if batch:
                yield from self._process_batch(batch)

    def _split_for_worker(self, items: Iterator, worker_info: Any) -> Iterator:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        for i, item in enumerate(items):
            if i % num_workers == worker_id:
                yield item

    def _process_batch(self, batch: List[Dict]) -> Iterator[Dict[str, Any]]:
        for prompt in super()._process_batch(batch):
            yield prompt