import os
import gc
import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.llm_evaluator import LLMEvaluator
from src.document_classifier import DocumentClassifier
from src.experiment_logger import ExperimentLogger

class BaseDataset(Dataset):
    def __init__(
        self,
        corpus_path: str,
        tokenizer: PreTrainedTokenizer,
        llm_evaluator: LLMEvaluator,
        cache_dir: Optional[str] = "cache/dataset",
        batch_size: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.llm_evaluator = llm_evaluator
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.examples = self._load_examples(corpus_path)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _load_examples(self, data_path: str) -> List[Dict]:
        self.logger.info(f"Loading examples from {data_path}")
        with open(data_path) as f:
            return json.load(f)

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
        max_doc_length: Optional[int] = None
    ):
        super().__init__(data_path, tokenizer, llm_evaluator)
        self.corpus = corpus
        self.max_tokenized_length = max_tokenized_length
        self.num_documents_in_context = num_documents_in_context
        self.get_documents_without_answer = get_documents_without_answer
        self.gold_position = gold_position
        self.max_doc_length = max_doc_length

        self.document_classifier = DocumentClassifier(self.llm_evaluator)
        self.prompts = self._create_prompts_batched()

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
            try:
                question = example['question']
                gold_answer = example['answers'][0]

                selected_docs = self.corpus[:self.num_documents_in_context]

                if self.max_doc_length:
                    for doc in selected_docs:
                        doc['text'] = doc['text'][:self.max_doc_length]

                categories = self.document_classifier.classify_documents(selected_docs, question, gold_answer)
                all_docs = categories['gold'] + categories['distracting'] + categories['random']

                if self.get_documents_without_answer:
                    all_docs = [d for d in all_docs if d['category'] != 'gold']

                prompt = self._format_prompt(question, all_docs)
                if self._validate_prompt(prompt):
                    processed.append({
                        'prompt': prompt,
                        'query': question,
                        'document_indices': [d['id'] for d in all_docs],
                        'document_categories': [d['category'] for d in all_docs],
                        'document_positions': [d['position'] for d in all_docs],
                        'example_id': example['id'],
                        'gold_answer': gold_answer
                    })
            except Exception as e:
                self.logger.warning(f"Error processing example {example.get('id', 'unknown')}: {str(e)}")
                continue
                
        return processed

    def _format_prompt(self, question: str, documents: List[Dict]) -> str:
        context_parts = [f"Question: {question}\n\nContext:"]
        for idx, doc in enumerate(documents, 1):
            context_parts.append(
                f"\nDocument [{idx}] (Category: {doc['category']}, Position: {doc['position']}):\n{doc['text']}"
            )
        context_parts.append("\nAnswer:")
        return "\n".join(context_parts)

    def _validate_prompt(self, prompt: str) -> bool:
        tokens = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_tokenized_length,
            return_tensors="pt"
        )
        return len(tokens['input_ids'][0]) <= self.max_tokenized_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.prompts[idx]
