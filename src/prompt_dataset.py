import os
import gc
import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from llm_evaluator import LLMEvaluator 
import hashlib
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(
        self,
        corpus_path: str,
        tokenizer: PreTrainedTokenizer,
        llm_evaluator: LLMEvaluator,
        cache_dir: Optional[str] = "cache/dataset",
        batch_size: int = 1000,
        max_memory_usage: float = 0.9
    ):
        self.tokenizer = tokenizer
        self.llm_evaluator = llm_evaluator
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_memory_usage = max_memory_usage
        self.logger = self._setup_logger()
        self.prompt_cache = {}
        self.corpus_chunks = self._initialize_corpus_chunks(corpus_path)
        self.loaded_chunks = {}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _initialize_corpus_chunks(self, corpus_path: str) -> Dict[int, str]:
        chunk_mapping = {}
        chunk_size = 1000  # Documents per chunk

        with open(corpus_path) as f:
            corpus = json.load(f)
            
        for i in range(0, len(corpus), chunk_size):
            chunk_id = i // chunk_size
            chunk_path = self.cache_dir / f"corpus_chunk_{chunk_id}.json"
            
            with open(chunk_path, 'w') as f:
                json.dump(corpus[i:i + chunk_size], f)
            
            chunk_mapping[chunk_id] = str(chunk_path)

        return chunk_mapping

    def _load_chunk(self, chunk_id: int) -> List[Dict]:
        if chunk_id in self.loaded_chunks:
            return self.loaded_chunks[chunk_id]

        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if current_memory > self.max_memory_usage:
                self._cleanup_memory()

        try:
            chunk_path = self.corpus_chunks[chunk_id]
            with open(chunk_path) as f:
                chunk_data = json.load(f)
            
            self.loaded_chunks[chunk_id] = chunk_data
            return chunk_data

        except Exception as e:
            self.logger.error(f"Error loading chunk {chunk_id}: {str(e)}")
            return []

    def _cleanup_memory(self):
        if len(self.loaded_chunks) > 3:  # Keep only 3 most recent chunks
            oldest_chunk = min(self.loaded_chunks.keys())
            del self.loaded_chunks[oldest_chunk]
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class PromptDataset(BaseDataset):
    def __init__(
        self,
        corpus: List[Dict],
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        llm_evaluator: LLMEvaluator,
        max_tokenized_length: int,
        num_documents_in_context: int,
        get_documents_without_answer: bool = False,
        gold_position: Optional[int] = None,
        max_doc_length: Optional[int] = None,
        cache_dir: Optional[str] = "cache/prompts"
    ):
        super().__init__(data_path, tokenizer, llm_evaluator, cache_dir)
        self.max_tokenized_length = max_tokenized_length
        self.num_documents_in_context = num_documents_in_context
        self.get_documents_without_answer = get_documents_without_answer
        self.gold_position = gold_position
        self.max_doc_length = max_doc_length
        self.prompts = self._create_prompts_batched()

    def _create_prompts_batched(self) -> List[Dict]:
        prompts = []
        examples = self._load_examples()
        
        for i in tqdm(range(0, len(examples), self.batch_size), desc="Creating prompts"):
            batch = examples[i:i + self.batch_size]
            prompts.extend(self._process_batch(batch))
            self._cleanup_memory()
            
        return prompts

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        processed = []
        
        for example in batch:
            try:
                cache_key = self._get_cache_key(example)
                cached_prompt = self._check_cache(cache_key)
                
                if cached_prompt:
                    processed.append(cached_prompt)
                    continue

                question = example['question']
                gold_answer = example['answers'][0]

                chunk_id = example.get('chunk_id', 0)
                documents = self._load_chunk(chunk_id)
                selected_docs = documents[:self.num_documents_in_context]

                if self.max_doc_length:
                    selected_docs = [
                        {**doc, 'text': doc['text'][:self.max_doc_length]}
                        for doc in selected_docs
                    ]

                categories = self.llm_evaluator.classify_documents(selected_docs, question, gold_answer)
                all_docs = categories['gold'] + categories['distracting'] + categories['random']

                if self.get_documents_without_answer:
                    all_docs = [d for d in all_docs if d['category'] != 'gold']

                prompt = self._format_prompt(question, all_docs)
                if self._validate_prompt(prompt):
                    result = {
                        'prompt': prompt,
                        'query': question,
                        'document_indices': [d['id'] for d in all_docs],
                        'document_categories': [d['category'] for d in all_docs],
                        'document_positions': [d.get('position', i) for i, d in enumerate(all_docs)],
                        'example_id': example['id'],
                        'gold_answer': gold_answer
                    }
                    processed.append(result)
                    self._save_to_cache(cache_key, result)

            except Exception as e:
                self.logger.error(f"Error processing example {example.get('id', 'unknown')}: {str(e)}")
                continue

        return processed

    def _format_prompt(self, question: str, documents: List[Dict]) -> str:
        context_parts = [f"Question: {question}\n\nContext:"]
        for idx, doc in enumerate(documents, 1):
            context_parts.append(
                f"\nDocument [{idx}] (Category: {doc['category']}, Position: {doc.get('position', idx)}):\n{doc['text']}"
            )
        context_parts.append("\nAnswer:")
        return "\n".join(context_parts)

    def _validate_prompt(self, prompt: str) -> bool:
        """Validate prompt length against model constraints"""
        tokens = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_tokenized_length,
            return_tensors="pt"
        )
        return len(tokens['input_ids'][0]) <= self.max_tokenized_length

    def _load_examples(self) -> List[Dict]:
        """Load and validate dataset examples"""
        try:
            with open(self.data_path) as f:
                examples = json.load(f)
            return examples
        except Exception as e:
            self.logger.error(f"Error loading examples: {str(e)}")
            return []

    def _get_cache_key(self, example: Dict) -> str:
        """Generate cache key for prompt"""
        content = f"{example['question']}:{example['answers'][0]}:{self.num_documents_in_context}"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if prompt exists in cache"""
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Cache read error: {str(e)}")
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save processed prompt to cache"""
        self.prompt_cache[cache_key] = data
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Cache write error: {str(e)}")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.prompts[idx]