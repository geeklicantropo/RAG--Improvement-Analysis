import logging
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Set, Any, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import torch
import json
import pickle
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
from datasets import Dataset, load_from_disk, IterableDataset
from tqdm import tqdm
import mmap
import gc

@dataclass
class RetrievalResult:
    """Stores document retrieval results with index-based identification."""
    corpus_idx: int  # Primary identifier as corpus index
    text: str
    score: float
    metadata: Optional[Dict] = None

class DocumentStore:
    """Memory-efficient document store using memory mapping."""
    def __init__(self, dataset_path: str):
        self.logger = logging.getLogger(f"{__name__}.DocumentStore")
        self.dataset: Optional[Dataset] = None
        self.index_metadata: Dict[int, Dict] = {}
        self._mmap_file = None
        self._initialize_store(dataset_path)

    def _initialize_store(self, dataset_path: str):
        """Initialize document store with memory mapping."""
        try:
            # Load dataset using Arrow format's memory mapping
            self.dataset = load_from_disk(
                dataset_path,
                keep_in_memory=False
            )
            self.logger.info(f"Loaded dataset with {len(self.dataset)} documents")

            # Initialize metadata with minimal memory footprint
            for idx in range(len(self.dataset)):
                self.index_metadata[idx] = {
                    'accessed_count': 0,
                    'last_accessed': None
                }
            
            self.logger.info("Document store initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing document store: {e}")
            raise

    def get_document(self, corpus_idx: int) -> Optional[Dict]:
        """Get document by corpus index using memory mapping."""
        try:
            if 0 <= corpus_idx < len(self.dataset):
                # Update access metadata
                self.index_metadata[corpus_idx]['accessed_count'] += 1
                self.index_metadata[corpus_idx]['last_accessed'] = datetime.now()
                
                # Get document using Arrow's memory mapping
                doc = self.dataset[corpus_idx]
                return {
                    'text': doc['text'],
                    'metadata': {
                        'corpus_idx': corpus_idx,
                        'access_count': self.index_metadata[corpus_idx]['accessed_count']
                    }
                }
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving document at index {corpus_idx}: {e}")
            return None

    def get_documents_batch(
        self, 
        corpus_indices: List[int], 
        batch_size: Optional[int] = None
    ) -> List[Optional[Dict]]:
        """Get multiple documents by corpus indices in batches."""
        results = []
        batch_size = batch_size or 100  # Default batch size
        
        for i in range(0, len(corpus_indices), batch_size):
            batch_indices = corpus_indices[i:i + batch_size]
            batch_docs = [self.get_document(idx) for idx in batch_indices]
            results.extend(batch_docs)
            
            # Clear memory after each batch
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        return results

    def __del__(self):
        """Cleanup resources."""
        if self._mmap_file:
            self._mmap_file.close()

class Retriever:
    """Enhanced retriever with memory-efficient processing."""
    def __init__(
        self,
        retriever_type: str = "contriever",
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_embeddings: bool = True,
        dataset_path: Optional[str] = None,
        embeddings_path: Optional[str] = None
    ):
        self.logger = logging.getLogger(f"{__name__}.Retriever")
        
        self.retriever_type = retriever_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings
        
        # Initialize document store with memory mapping
        self.doc_store = DocumentStore(dataset_path) if dataset_path else None
        
        # Load embeddings if provided
        self.embeddings_path = embeddings_path
        self.embeddings_mmap = None
        
        # Initialize appropriate retriever
        if retriever_type == "contriever":
            self._init_contriever()
        elif retriever_type == "bm25":
            self._init_bm25()
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")
        
        self.logger.info(
            f"Initialized {retriever_type} retriever on {self.device} "
            f"with batch_size={batch_size}"
        )

    def _init_contriever(self):
        """Initialize Contriever with memory optimization."""
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                'facebook/contriever',
                use_fast=True,  # Use faster tokenizer
                model_max_length=512  # Limit max length
            )
            
            # Load model with memory optimizations
            model_config = {
                'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
                'low_cpu_mem_usage': True
            }
            self.model = AutoModel.from_pretrained('facebook/contriever', **model_config)
            self.model.to(self.device)
            
            # Initialize FAISS index if embeddings path provided
            if self.embeddings_path:
                self._build_contriever_index()
            else:
                self.index = None
            
            self.logger.info("Contriever initialization complete")
        except Exception as e:
            self.logger.error(f"Error initializing Contriever: {e}")
            raise

    def _init_bm25(self):
        """Initialize BM25 with memory-efficient processing."""
        try:
            if not self.doc_store or not self.doc_store.dataset:
                raise ValueError("Document store not initialized")

            # Process documents in batches to build BM25 index
            tokenized_docs = []
            self.corpus_texts = []
            
            batch_size = 1000
            for i in range(0, len(self.doc_store.dataset), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(self.doc_store.dataset))))
                batch_docs = self.doc_store.get_documents_batch(batch_indices)
                
                for doc in batch_docs:
                    if doc:
                        text = doc['text']
                        self.corpus_texts.append(text)
                        tokens = text.lower().split()
                        tokenized_docs.append(tokens)
                
                # Clear memory after each batch
                if i % (batch_size * 10) == 0:
                    gc.collect()
            
            # Initialize BM25
            self.bm25 = BM25Okapi(tokenized_docs)
            self.logger.info("BM25 initialization complete")
        except Exception as e:
            self.logger.error(f"Error initializing BM25: {e}")
            raise

    def _build_contriever_index(self):
        """Build FAISS index with memory mapping for embeddings."""
        try:
            # Open embeddings file with memory mapping
            embeddings = np.load(self.embeddings_path, mmap_mode='r')
            dimension = embeddings.shape[1]
            
            # Build index in batches
            self.index = faiss.IndexFlatIP(dimension)
            batch_size = 1000
            
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                self.index.add(np.array(batch))
                
                # Clear memory after each batch
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            self.logger.info(f"Built index with {len(embeddings)} documents")
        except Exception as e:
            self.logger.error(f"Error building index: {e}")
            raise

    def retrieve(
        self, 
        query: str,
        top_k: int = 5,
        include_metadata: bool = False,
        filter_fn: Optional[callable] = None
    ) -> List[RetrievalResult]:
        """Memory-efficient document retrieval."""
        try:
            if self.retriever_type == "contriever":
                results = self._retrieve_contriever(query, top_k)
            else:  # BM25
                results = self._retrieve_bm25(query, top_k)
            
            # Apply custom filtering if provided
            if filter_fn:
                results = [r for r in results if filter_fn(r)]
            
            # Add document content and metadata in batches
            processed_results = []
            batch_size = 100
            
            for i in range(0, len(results), batch_size):
                batch_results = results[i:i + batch_size]
                batch_docs = self.doc_store.get_documents_batch(
                    [r.corpus_idx for r in batch_results]
                )
                
                for result, doc in zip(batch_results, batch_docs):
                    if doc:
                        result.text = doc['text']
                        if include_metadata:
                            result.metadata = doc.get('metadata', {})
                        processed_results.append(result)
                
                # Clear memory after each batch
                if i % (batch_size * 10) == 0:
                    gc.collect()
            
            self.logger.info(f"Retrieved {len(processed_results)} documents")
            return processed_results
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            return []

    def _retrieve_contriever(
        self,
        query: str,
        top_k: int,
        batch_size: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Memory-efficient Contriever retrieval."""
        try:
            batch_size = batch_size or self.batch_size
            
            # Encode query
            inputs = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                query_embedding = self.model(**inputs).last_hidden_state[:, 0].cpu().numpy()
            
            # Search index if available
            if self.cache_embeddings and self.index:
                scores, indices = self.index.search(query_embedding, top_k)
                results = [
                    RetrievalResult(
                        corpus_idx=int(idx),
                        text="",  # Will be filled later
                        score=float(score)
                    )
                    for score, idx in zip(scores[0], indices[0])
                ]
            else:
                # Compute similarities in batches
                all_scores = []
                all_indices = []
                
                for i in range(0, len(self.doc_store.dataset), batch_size):
                    batch_docs = self.doc_store.get_documents_batch(
                        list(range(i, min(i + batch_size, len(self.doc_store.dataset))))
                    )
                    
                    batch_embeddings = self._get_embeddings_batch(
                        [doc['text'] for doc in batch_docs if doc]
                    )
                    
                    similarities = np.dot(query_embedding, batch_embeddings.T)
                    
                    all_scores.extend(similarities[0])
                    all_indices.extend(range(i, i + len(batch_docs)))
                    
                    # Clear memory after batch
                    del batch_embeddings
                    if i % (batch_size * 10) == 0:
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Get top-k results
                top_indices = np.argsort(all_scores)[-top_k:][::-1]
                results = [
                    RetrievalResult(
                        corpus_idx=all_indices[idx],
                        text="",  # Will be filled later
                        score=float(all_scores[idx])
                    )
                    for idx in top_indices
                ]
            
            return results
        except Exception as e:
            self.logger.error(f"Error in Contriever retrieval: {e}")
            return []

    def _retrieve_bm25(
        self,
        query: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """Memory-efficient BM25 retrieval."""
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append(RetrievalResult(
                    corpus_idx=idx,
                    text=self.corpus_texts[idx],
                    score=float(scores[idx])
                ))
            
            return results
        except Exception as e:
            self.logger.error(f"Error in BM25 retrieval: {e}")
            return []

    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a batch with memory optimization."""
        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state[:, 0].cpu().numpy()
            
            return embeddings
        except Exception as e:
            self.logger.error(f"Error getting embeddings batch: {e}")
            raise

    def get_random_documents(
        self,
        k: int = 3,
        seed: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Get random documents with memory efficiency."""
        try:
            if not self.doc_store or not self.doc_store.dataset:
                raise ValueError("Document store not initialized")
            
            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)
            
            # Get random indices
            total_docs = len(self.doc_store.dataset)
            random_indices = np.random.choice(total_docs, size=k, replace=False)
            
            # Get documents in batches
            results = []
            batch_size = 100
            
            for i in range(0, len(random_indices), batch_size):
                batch_indices = random_indices[i:i + batch_size]
                batch_docs = self.doc_store.get_documents_batch(batch_indices)
                
                for idx, doc in zip(batch_indices, batch_docs):
                    if doc:
                        results.append(RetrievalResult(
                            corpus_idx=idx,
                            text=doc['text'],
                            score=0.0,
                            metadata={'random_selection': True}
                        ))
                
                # Clear memory after batch
                if i % (batch_size * 10) == 0:
                    gc.collect()
            
            return results
        except Exception as e:
            self.logger.error(f"Error getting random documents: {e}")
            return []

    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Memory-efficient text encoding."""
        try:
            batch_size = batch_size or self.batch_size
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                embeddings = self._get_embeddings_batch(batch_texts)
                
                if normalize:
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                all_embeddings.append(embeddings)
                
                # Clear memory after batch
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Combine all embeddings
            final_embeddings = np.vstack(all_embeddings)
            return final_embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            raise

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding with memory optimization."""
        try:
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(**inputs).last_hidden_state[:, 0].cpu().numpy()
                
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error computing query embedding: {e}")
            raise

    def get_doc_embeddings(
        self,
        docs: List[RetrievalResult],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Get document embeddings in memory-efficient batches."""
        try:
            batch_size = batch_size or self.batch_size
            all_embeddings = []
            
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                batch_texts = [doc.text for doc in batch_docs]
                
                embeddings = self._get_embeddings_batch(batch_texts)
                all_embeddings.append(embeddings)
                
                # Clear memory after batch
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Combine all embeddings
            final_embeddings = np.vstack(all_embeddings)
            return final_embeddings
            
        except Exception as e:
            self.logger.error(f"Error getting document embeddings: {e}")
            raise

    def save_state(self, save_dir: Union[str, Path]):
        """Save retriever state with memory optimization."""
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            if self.retriever_type == "contriever":
                self.model.save_pretrained(save_path / "model")
                self.tokenizer.save_pretrained(save_path / "tokenizer")
            
            # Save index if exists
            if hasattr(self, 'index') and self.index is not None:
                faiss.write_index(self.index, str(save_path / "faiss_index.bin"))
            
            # Save BM25 state if applicable
            if hasattr(self, 'bm25'):
                with open(save_path / "bm25_state.pkl", 'wb') as f:
                    pickle.dump({
                        'tokenized_docs': self.bm25.tokenized_docs,
                        'corpus_texts': self.corpus_texts
                    }, f)
            
            # Save metadata
            metadata = {
                'retriever_type': self.retriever_type,
                'device': str(self.device),
                'batch_size': self.batch_size,
                'cache_embeddings': self.cache_embeddings,
                'embeddings_path': str(self.embeddings_path) if self.embeddings_path else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(save_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved retriever state to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving retriever state: {e}")
            raise

    def load_state(self, load_dir: Union[str, Path]):
        """Load retriever state with memory optimization."""
        try:
            load_path = Path(load_dir)
            
            # Load metadata first
            with open(load_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Verify retriever type
            if metadata['retriever_type'] != self.retriever_type:
                raise ValueError(f"Mismatched retriever type: {metadata['retriever_type']}")
            
            # Load model and tokenizer if applicable
            if self.retriever_type == "contriever":
                self.model = AutoModel.from_pretrained(
                    load_path / "model",
                    low_cpu_mem_usage=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(load_path / "tokenizer")
                self.model.to(self.device)
            
            # Load index if exists
            index_path = load_path / "faiss_index.bin"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            
            # Load BM25 state if applicable
            bm25_path = load_path / "bm25_state.pkl"
            if bm25_path.exists():
                with open(bm25_path, 'rb') as f:
                    bm25_state = pickle.load(f)
                    self.bm25 = BM25Okapi(bm25_state['tokenized_docs'])
                    self.corpus_texts = bm25_state['corpus_texts']
            
            # Update configuration
            self.batch_size = metadata['batch_size']
            self.cache_embeddings = metadata['cache_embeddings']
            if metadata['embeddings_path']:
                self.embeddings_path = Path(metadata['embeddings_path'])
            
            self.logger.info(f"Loaded retriever state from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading retriever state: {e}")
            raise

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'embeddings_mmap') and self.embeddings_mmap is not None:
                self.embeddings_mmap.close()
        except Exception:
            pass