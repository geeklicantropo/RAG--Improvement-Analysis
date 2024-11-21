import os 
import faiss
import numpy as np
from typing import List, Tuple, Optional
import logging
from utils import read_pickle, write_pickle

# Configure logging
logger = logging.getLogger(__name__)

# Indexer class adapted from Contriever file https://github.com/facebookresearch/contriever/blob/main/src/index.py

class Indexer(object):
    """
    Initializes an indexer with a specified vector size and index type.
    """
    def __init__(self, vector_sz: int, idx_type: str = 'IP'):
        self.idx_type = idx_type

        if idx_type == 'IP':
            quantizer = faiss.IndexFlatIP(vector_sz)
        elif idx_type == 'L2':
            quantizer = faiss.IndexFlatL2(vector_sz)
        else:
            raise NotImplementedError('Only L2 norm and Inner Product metrics are supported')
        
        self.index = quantizer
        self.index_id_to_db_id = []


    def index_data(self, ids: List[int], embeddings: np.array):
        """
        Adds data to the index.
        """
        try:
            self._update_id_mapping(ids)
            if not self.index.is_trained:
                self.index.train(embeddings)
            self.index.add(embeddings)
            logger.info(f'Total data indexed: {len(self.index_id_to_db_id)}')
        except Exception as e:
            logger.error(f"Error indexing data: {e}")
            raise e

    def search_knn(
        self, 
        query_vectors: np.array, 
        top_docs: int, 
        index_batch_size: int = 2048
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Performs a k-nearest neighbor search for the given query vectors.
        """
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in range(nbatch):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(
        self, 
        dir_path: str, 
        index_file_name: Optional[str] = None, 
        meta_file_name: Optional[str] = None
    ):
        """
        Serializes the index and its metadata to disk.
        """
        if index_file_name is None:
            index_file_name = f'{self.idx_type}_index.faiss'
        if meta_file_name is None:
            meta_file_name = f'{self.idx_type}_index_meta.faiss'

        index_file = os.path.join(dir_path, index_file_name)
        meta_file = os.path.join(dir_path, meta_file_name)
        try:
            faiss.write_index(self.index, index_file)
            write_pickle(self.index_id_to_db_id, meta_file)
            logger.info(f"Index serialized to {index_file}, metadata to {meta_file}.")
        except Exception as e:
            logger.error(f"Error serializing index: {e}")
            raise e


    def deserialize_from(
        self, 
        dir_path: str, 
        index_file_name: Optional[str] = None, 
        meta_file_name: Optional[str] = None,
        gpu_id: Optional[int] = None
    ):
        """
        Loads the index and its metadata from disk.
        """
        if index_file_name is None:
            index_file_name = f'{self.idx_type}_index.faiss'
        if meta_file_name is None:
            meta_file_name = f'{self.idx_type}_index_meta.faiss'

        index_file = os.path.join(dir_path, index_file_name)
        meta_file = os.path.join(dir_path, meta_file_name)
        try:
            self.index = faiss.read_index(index_file)
            logger.info(f'Loaded index of type {type(self.index)} and size {self.index.ntotal}')

            self.index_id_to_db_id = read_pickle(meta_file)
            assert len(
                self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'
            
            # Move index to GPU if specified
            if gpu_id is not None and torch.cuda.is_available():
                res = faiss.StandardGpuResources()  
                self.index = faiss.index_cpu_to_gpu(res, gpu_id , self.index)
                logger.info(f'Moved index to GPU {gpu_id}')
        except Exception as e:
            logger.error(f"Error deserializing index: {e}")
            raise e
        

    def _update_id_mapping(self, db_ids: List[int]):
        self.index_id_to_db_id.extend(db_ids)

    def get_index_name(self):
        return f"{self.idx_type}_index"
