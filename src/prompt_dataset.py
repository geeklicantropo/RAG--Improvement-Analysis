# Dataset Management
from torch.utils.data import Dataset

# Typing and Utilities
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

# File and Path Handling
from pathlib import Path

# Core Libraries
import torch
import logging
import json
import gc
import numpy as np

# Progress Tracking
from tqdm import tqdm

# Project-Specific Utilities
from normalize_answers import is_answer_in_text
from src.experiment_logger import ExperimentLogger
from src.cluster_utils import DocumentClusterer

class PromptDataset(Dataset):
    def __init__(
        self,
        corpus: List[Dict],
        data_path: str,
        tokenizer: 'PreTrainedTokenizer',
        max_tokenized_length: int,
        search_results: List[Any],
        num_documents_in_context: int,
        gold_position: Optional[int] = None,
        get_documents_without_answer: bool = False,
        max_doc_length: Optional[int] = None,
        do_normalize_query: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_tokenized_length = max_tokenized_length
        self.search_results = search_results
        self.num_documents_in_context = num_documents_in_context
        self.gold_position = gold_position
        self.get_documents_without_answer = get_documents_without_answer
        self.max_doc_length = max_doc_length
        self.do_normalize_query = do_normalize_query
        self.logger = logger or logging.getLogger(__name__)
        
        self.prompts = []
        self._load_examples(data_path)
        self._create_prompts()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.prompts):
            raise IndexError("Index out of range")
            
        item = self.prompts[idx]
        doc_indices = item['document_indices']
        
        # Handle empty document indices
        if not doc_indices:
            doc_indices = [0] * self.num_documents_in_context
        # Handle padding if fewer documents than expected
        elif len(doc_indices) < self.num_documents_in_context:
            # Use first document index for padding if list not empty
            pad_idx = doc_indices[0] if doc_indices else 0
            doc_indices.extend([pad_idx] * (self.num_documents_in_context - len(doc_indices)))
        # Truncate if more documents than expected
        elif len(doc_indices) > self.num_documents_in_context:
            doc_indices = doc_indices[:self.num_documents_in_context]
            
        return {
            'prompt': item['prompt'],
            'query': item['query'],
            'document_indices': torch.tensor(doc_indices, dtype=torch.long),
            'prompt_tokens_len': len(self.tokenizer.encode(item['prompt']))
        }

    def _load_examples(self, data_path: str):
        try:
            with open(data_path) as f:
                self.examples = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading examples from {data_path}: {str(e)}")
            raise

    def _create_prompts(self):
        for idx, example in enumerate(self.examples):
            try:
                doc_indices, scores = self.search_results[idx]
                if not isinstance(doc_indices, (list, tuple)):
                    doc_indices = [doc_indices]
                if not isinstance(scores, (list, tuple)):
                    scores = [scores]

                selected_docs = []
                for doc_idx, score in zip(doc_indices, scores):
                    if len(selected_docs) >= self.num_documents_in_context:
                        break

                    if doc_idx >= len(self.corpus):
                        continue

                    doc = self.corpus[doc_idx]
                    if self.max_doc_length:
                        doc['text'] = doc['text'][:self.max_doc_length]

                    if not self.get_documents_without_answer or not self._contains_answer(doc['text'], example['answers']):
                        selected_docs.append({
                            'text': doc['text'],
                            'score': score,
                            'idx': doc_idx
                        })

                if self.gold_position is not None:
                    selected_docs = self._insert_gold_document(selected_docs, example['idx_gold_in_corpus'])

                prompt = self._format_prompt(example['question'], selected_docs)
                if self._validate_prompt(prompt):
                    self.prompts.append({
                        'prompt': prompt,
                        'document_indices': [d['idx'] for d in selected_docs],
                        'scores': [d['score'] for d in selected_docs],
                        'query': example['question'],
                        'example_id': example.get('id', idx),
                        'answers': example['answers']
                    })

            except Exception as e:
                self.logger.error(f"Error processing example {idx}: {str(e)}")
                continue

    def _format_prompt(self, query: str, docs: List[Dict]) -> str:
        prompt = [f"Question: {query}\n\nContext:"]
        for idx, doc in enumerate(docs, 1):
            prompt.append(f"\nDocument [{idx}]: {doc['text']}")
        prompt.append("\nAnswer:")
        return "\n".join(prompt)

    def _validate_prompt(self, prompt: str) -> bool:
        tokens = self.tokenizer.encode(prompt)
        return len(tokens) <= self.max_tokenized_length

    def _contains_answer(self, text: str, answers: List[str]) -> bool:
        return any(answer.lower() in text.lower() for answer in answers)

    def _insert_gold_document(self, docs: List[Dict], gold_idx: int) -> List[Dict]:
        gold_doc = {
            'text': self.corpus[gold_idx]['text'],
            'score': 1.0,
            'idx': gold_idx
        }
        docs.insert(self.gold_position, gold_doc)
        return docs
    
class BaseDataset(Dataset):
    def __init__(
        self,
        corpus: List[Dict],
        data_path: str,
        tokenizer: 'PreTrainedTokenizer',
        max_tokenized_length: int,
        debug_output_dir: Optional[str] = None,
        logger: Optional[ExperimentLogger] = None
    ):
        self.corpus = corpus
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_tokenized_length = max_tokenized_length
        self.debug_dir = Path(debug_output_dir) if debug_output_dir else None
        self.logger = logger or ExperimentLogger("dataset", "logs")
        self.examples = self._load_examples()
        self.prompts = []
        
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _load_examples(self) -> List[Dict]:
        self.logger.log_step_start("Loading examples")
        try:
            with open(self.data_path) as f:
                examples = json.load(f)
            self.logger.log_metric("num_examples", len(examples))
            self.logger.log_step_end("Examples loaded")
            return examples
        except Exception as e:
            self.logger.log_error(e, "Error loading examples")
            raise

    def save_debug_info(self, example_id: str, info: Dict):
        if not self.debug_dir:
            return
        
        debug_file = self.debug_dir / f"example_{example_id}.json"
        with open(debug_file, 'w') as f:
            json.dump(info, f, indent=2)
            
        text_file = self.debug_dir / f"example_{example_id}.txt"
        with open(text_file, 'w') as f:
            f.write(f"Query: {info['query']}\n\n")
            f.write("Documents:\n")
            for idx, doc in enumerate(info.get('documents', []), 1):
                f.write(f"\nDocument [{idx}]")
                if 'type' in doc:
                    f.write(f" ({doc['type']})")
                if 'cluster' in doc:
                    f.write(f" [Cluster {doc['cluster']}]")
                f.write(f":\n{doc['text']}\n")
            f.write(f"\nPrompt:\n{info['prompt']}\n")

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.prompts[idx]

class MixedDocumentsDataset(BaseDataset):
    def __init__(
        self,
        corpus: List[Dict],
        data_path: str,
        tokenizer: 'PreTrainedTokenizer',
        max_tokenized_length: int,
        retriever_search_results: List[Tuple[List[int], List[float]]],
        random_search_results: List[Tuple[List[int], List[float]]],
        documents_disposition_info: Dict[str, Any],
        full_to_subset_idx_map: Optional[Dict[int, int]] = None,
        do_normalize_query: bool = True,
        gold_position: Optional[int] = None,
        get_documents_without_answer: bool = False,
        use_clustering: bool = False,
        num_clusters: Optional[int] = None,
        noise_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(corpus, data_path, tokenizer, max_tokenized_length, **kwargs)
        self.retriever_search_results = retriever_search_results
        self.random_search_results = random_search_results
        self.documents_disposition_info = documents_disposition_info
        self.full_to_subset_idx_map = full_to_subset_idx_map
        self.do_normalize_query = do_normalize_query
        self.gold_position = gold_position
        self.get_documents_without_answer = get_documents_without_answer
        self.use_clustering = use_clustering
        self.noise_config = noise_config or {}
        
        self.clusterer = DocumentClusterer(num_clusters) if use_clustering else None
        self._create_prompts()

    def _cluster_documents(self, docs: List[Dict]) -> List[Dict]:
        if not docs:
            return docs
            
        texts = [doc["text"] for doc in docs]
        cluster_ids = self.clusterer.fit_predict(texts)
        
        for doc, cluster_id in zip(docs, cluster_ids):
            doc["cluster_id"] = int(cluster_id)
            doc["type"] = "retrieved_clustered"
            
        return docs

    def _inject_noise(self, docs: List[Dict]) -> List[Dict]:
        noise_ratio = self.noise_config.get("ratio", 0.2)
        noise_type = self.noise_config.get("type", "random")
        
        num_noise = int(len(docs) * noise_ratio)
        if num_noise == 0:
            return docs
            
        noise_indices = np.random.choice(len(docs), num_noise, replace=False)
        
        for idx in noise_indices:
            if noise_type == "random":
                random_doc = np.random.choice(self.corpus)
                docs[idx] = {
                    "idx": random_doc.get("idx", -1),
                    "text": random_doc["text"],
                    "score": 0.0,
                    "type": "noise_random"
                }
            elif noise_type == "adversarial":
                original_text = docs[idx]["text"]
                noisy_text = self._add_adversarial_noise(original_text)
                docs[idx]["text"] = noisy_text
                docs[idx]["type"] = "noise_adversarial"
                
        return docs

    def _add_adversarial_noise(self, text: str) -> str:
        words = text.split()
        num_words = len(words)
        num_changes = max(1, int(num_words * 0.1))
        change_indices = np.random.choice(num_words, num_changes, replace=False)
        
        for idx in change_indices:
            if np.random.random() < 0.5:
                words[idx] = np.random.choice(self.noise_config.get("word_pool", ["the", "a", "an"]))
            else:
                words[idx] = words[idx] + "'"
                
        return " ".join(words)

    def _randomize_document_order(self, docs: List[Dict]) -> List[Dict]:
        if self.gold_position is not None:
            gold_doc = docs[self.gold_position]
            other_docs = docs[:self.gold_position] + docs[self.gold_position+1:]
            np.random.shuffle(other_docs)
            other_docs.insert(self.gold_position, gold_doc)
            return other_docs
        
        shuffled = docs.copy()
        np.random.shuffle(shuffled)
        return shuffled


    def _create_prompts(self):
        self.logger.log_step_start("Creating mixed document prompts")
        try:
            for example_idx, example in enumerate(tqdm(self.examples, desc="Processing examples")):
                try:
                    # Get and validate retriever results
                    doc_indices, scores = self.retriever_search_results[example_idx]
                    if isinstance(doc_indices, int):
                        doc_indices = [doc_indices] 
                        scores = [scores]
                    
                    # Process retrieved documents
                    retrieved_docs = []
                    for idx, score in zip(doc_indices, scores):
                        if len(retrieved_docs) >= self.documents_disposition_info["num_retrieved_documents"]:
                            break
                            
                        if self.full_to_subset_idx_map:
                            idx = self.full_to_subset_idx_map.get(idx, idx)
                            
                        if idx >= len(self.corpus):
                            continue
                            
                        doc = self.corpus[idx].copy()  
                        if not self.get_documents_without_answer or not is_answer_in_text(doc['text'], example['answers']):
                            retrieved_docs.append({
                                "text": doc['text'],
                                "score": float(score),
                                "idx": idx,
                                "type": "retrieved"
                            })

                    # Process random documents
                    random_docs = self._get_random_documents(example_idx)
                    
                    # Apply clustering if enabled
                    if self.use_clustering:
                        retrieved_docs = self._cluster_documents(retrieved_docs)
                    
                    # Inject noise if configured
                    if self.noise_config:
                        retrieved_docs = self._inject_noise(retrieved_docs)
                    
                    # Mix documents according to disposition
                    docs = self._mix_documents(retrieved_docs, random_docs)
                    
                    # Handle gold document if specified
                    if self.gold_position is not None:
                        docs = self._insert_gold_document(docs, example["idx_gold_in_corpus"])
                            
                    # Randomize order if requested
                    if self.documents_disposition_info.get("randomize_order", False):
                        docs = self._randomize_document_order(docs)
                    
                    prompt = self._format_prompt(example["question"], docs)
                    if self._validate_prompt(prompt):
                        prompt_info = {
                            "prompt": prompt,
                            "query": example["question"], 
                            "example_id": example.get("id", example_idx),
                            "document_indices": [d["idx"] for d in docs],
                            "document_types": [d.get("type", "unknown") for d in docs],
                            "cluster_ids": [d.get("cluster_id") for d in docs if "cluster_id" in d],
                            "scores": [d["score"] for d in docs]
                        }
                        self.prompts.append(prompt_info)
                        
                        debug_info = {
                            "query": example["question"],
                            "documents": [{
                                "text": d["text"],
                                "type": d.get("type", "unknown"),
                                "cluster": d.get("cluster_id"),
                                "score": d["score"]
                            } for d in docs],
                            "prompt": prompt
                        }
                        self.save_debug_info(example.get("id", example_idx), debug_info)
                    
                    if len(self.prompts) % 1000 == 0:
                        self.cleanup()
                            
                except Exception as e:
                    self.logger.error(f"Error processing example {example_idx}: {str(e)}")
                    continue
                    
            self.logger.log_step_end("Prompts created")
            
        except Exception as e:
            self.logger.log_error(e, "Error creating prompts")
            raise

    def _get_retrieved_documents(self, example_idx: int, answers: List[str]) -> List[Dict]:
        doc_indices, scores = self.retriever_search_results[example_idx]
        retrieved_docs = []
        
        for idx, score in zip(doc_indices, scores):
            if self.full_to_subset_idx_map:
                idx = self.full_to_subset_idx_map.get(idx, idx)
            
            if idx >= len(self.corpus):
                continue
                
            doc = self.corpus[idx]
            if self.get_documents_without_answer:
                if not is_answer_in_text(doc["text"], answers):
                    retrieved_docs.append({
                        "idx": idx,
                        "text": doc["text"],
                        "score": score
                    })
            else:
                retrieved_docs.append({
                    "idx": idx,
                    "text": doc["text"],
                    "score": score
                })
                
        return retrieved_docs[:self.documents_disposition_info["num_retrieved_documents"]]

    def _get_random_documents(self, example_idx: int) -> List[Dict]:
        doc_indices, scores = self.random_search_results[example_idx]
        random_docs = []
        
        for idx, score in zip(doc_indices, scores):
            if self.full_to_subset_idx_map:
                idx = self.full_to_subset_idx_map.get(idx, idx)
            
            if idx >= len(self.corpus):
                continue
                
            doc = self.corpus[idx]
            random_docs.append({
                "idx": idx,
                "text": doc["text"],
                "score": score
            })
            
        return random_docs[:self.documents_disposition_info["num_random_documents"]]

    def _mix_documents(
        self,
        retrieved_docs: List[Dict],
        random_docs: List[Dict]
    ) -> List[Dict]:
        if self.documents_disposition_info["put_retrieved_first"]:
            return retrieved_docs + random_docs
        return random_docs + retrieved_docs

    def _insert_gold_document(self, docs: List[Dict], gold_idx: int) -> List[Dict]:
        gold_doc = {
            "idx": gold_idx,
            "text": self.corpus[gold_idx]["text"],
            "score": 1.0
        }
        docs.insert(self.gold_position, gold_doc)
        return docs

    def _format_prompt(self, query: str, docs: List[Dict]) -> str:
        prompt = [f"Query: {query}\n\nContext:"]
        for idx, doc in enumerate(docs, 1):
            prompt.append(f"Document [{idx}]: {doc['text']}")
        prompt.append("\nAnswer:")
        return "\n".join(prompt)

    def _validate_prompt(self, prompt: str) -> bool:
        tokens = self.tokenizer.tokenize(prompt)
        return len(tokens) <= self.max_tokenized_length
