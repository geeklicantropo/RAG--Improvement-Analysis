import os
from typing import List, Optional, Dict
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F
import normalize_text
import tqdm
from typing import List, Optional, Dict, Any, Union
from src.utils.file_utils import clear_memory
from src.llm import LLM



class Encoder(PreTrainedModel):
    """
    A wrapper class for encoding text using pre-trained transformer models with specified pooling strategy.
    """
    def __init__(self, config: AutoConfig, pooling: str = "average"):
        super().__init__(config)
        self.config = config
        if not hasattr(self.config, "pooling"):
            self.config.pooling = pooling

        self.model = AutoModel.from_pretrained(
            config.name_or_path, config=self.config
        )


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    
    def encode(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        normalize: bool = False
    ) -> torch.Tensor:
        model_output = self.forward(
            input_ids, 
            attention_mask,
            token_type_ids,
        )
        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = F.normalize(emb, dim=-1)

        return emb


class Retriever:
    """
    A class for retrieving document embeddings using a specified encoder, using a bi-encoder approach.
    """
    def __init__(
        self,
        device: torch.device,
        tokenizer: AutoTokenizer,
        query_encoder: Union[Encoder, LLM],
        doc_encoder: Optional[Union[Encoder, LLM]] = None,
        max_length: int = 512,
        add_special_tokens: bool = True,
        norm_query_emb: bool = False,
        norm_doc_emb: bool = False,
        lower_case: bool = False,
        do_normalize_text: bool = False,
    ):
        
        self.device = device
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query_emb = norm_query_emb
        self.norm_doc_emb = norm_doc_emb
        self.lower_case = lower_case
        self.do_normalize_text = do_normalize_text

        if isinstance(query_encoder, LLM):
            self.query_encoder = query_encoder
        else:
            self.query_encoder = query_encoder.to(device)

        if doc_encoder is None:
            self.doc_encoder = self.query_encoder
        else:
            if isinstance(doc_encoder, LLM):
                self.doc_encoder = doc_encoder
            else:
                self.doc_encoder = doc_encoder.to(device)

    def encode_queries(self, queries: List[str], batch_size: int) -> np.ndarray:
        if self.do_normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        all_embeddings = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                q_inputs = self.tokenizer(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                ).to(self.device)

                if isinstance(self.query_encoder, LLM):
                    emb = self.query_encoder.generate(queries[start_idx:end_idx], max_new_tokens=1, return_tensors=True)
                    emb = emb[:, -1, :]  # Use the last token embedding as the query embedding
                else:
                    emb = self.query_encoder.encode(**q_inputs, normalize=self.norm_query_emb)
                all_embeddings.append(emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    def encode_corpus(
        self, 
        corpus_info: List[Dict[str, str]], 
        batch_size: int, 
        output_dir: str, 
        prefix_name: str,
        save_every: int = 500,
        logger: Optional[Any] = None  # Add logger parameter
    ) -> None:
        """Encode corpus documents with progress tracking and logging."""
        os.makedirs(output_dir, exist_ok=True)
        
        all_embeddings = []
        num_steps = 0

        # Use logger's progress bar if available, otherwise use standard tqdm
        progress_func = logger.log_progress if logger else tqdm
        nbatch = (len(corpus_info) - 1) // batch_size + 1
        
        with torch.no_grad():
            for k in progress_func(range(nbatch), desc="Encoding corpus", total=nbatch):
                try:
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, len(corpus_info))

                    # Process batch
                    corpus = [
                        c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] 
                        for c in corpus_info[start_idx: end_idx]
                    ]
                    if self.do_normalize_text:
                        corpus = [normalize_text.normalize(c) for c in corpus]
                    if self.lower_case:
                        corpus = [c.lower() for c in corpus]

                    # Encode batch
                    if isinstance(self.doc_encoder, LLM):
                        emb = self.doc_encoder.generate(corpus, max_new_tokens=1, return_tensors=True)
                        emb = emb[:, -1, :]  # Use the last token embedding as the document embedding
                    else:
                        doc_inputs = self.tokenizer(
                            corpus,
                            max_length=self.max_length,
                            padding=True,
                            truncation=True,
                            add_special_tokens=self.add_special_tokens,
                            return_tensors="pt",
                        ).to(self.device)

                        emb = self.doc_encoder.encode(**doc_inputs, normalize=self.norm_doc_emb)
                    all_embeddings.append(emb)

                    num_steps += 1

                    # Save periodically
                    if num_steps == save_every or k == nbatch - 1:
                        embeddings = torch.cat(all_embeddings, dim=0)
                        file_index = end_idx - 1
                        file_path = os.path.join(output_dir, f'{prefix_name}_{file_index}_embeddings.npy')
                        np.save(file_path, embeddings.cpu().numpy())
                        
                        if logger:
                            logger.experiment_logger.info(f"Saved embeddings for {file_index} passages to {file_path}")
                        
                        num_steps = 0
                        all_embeddings = []
                        
                except Exception as e:
                    if logger:
                        logger.log_error(e, f"Error processing batch {k}")
                    raise

    def load_embeddings_batch(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Load batch of embeddings from disk."""
        file_path = os.path.join(
            self.embeddings_dir,
            f'{self.prefix_name}_{end_idx}_embeddings.npy'
        )
        return np.load(file_path, mmap_mode='r')

    def retrieve_documents(
        self,
        query: str, 
        corpus_size: int,
        batch_size: int = 1000,
        top_k: int = 100
    ) -> List[int]:
        """Retrieve relevant documents for a query."""
        # Encode query
        query_inputs = self.tokenizer(
            query,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)
        
        query_embeddings = self.query_encoder.encode(**query_inputs, normalize=True)
        doc_scores = []
        
        for start_idx in range(0, corpus_size, batch_size):
            batch_embeddings = self.load_embeddings_batch(start_idx, start_idx + batch_size)
            batch_scores = np.matmul(query_embeddings.cpu().numpy(), batch_embeddings.T)
            doc_scores.extend(batch_scores[0].tolist())
            
            clear_memory()
            
        top_indices = np.argsort(doc_scores)[-top_k:][::-1].tolist()
        top_scores = [doc_scores[i] for i in top_indices]
            
        return [top_indices, top_scores]