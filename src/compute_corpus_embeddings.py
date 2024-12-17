import os
import argparse
import warnings
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gc
import google.generativeai as genai
import time
from typing import Optional, List, Dict
from experiment_logger import ExperimentLogger
from src.utils.file_utils import read_corpus_json, str2bool, read_json
from src.utils.rate_limit import rate_limit
warnings.filterwarnings('ignore')

def parse_arguments() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description="Script for computing corpus embeddings.")
   parser.add_argument('--corpus_path', type=str, required=True)
   parser.add_argument('--output_dir', type=str, default='data/corpus/embeddings/')
   parser.add_argument('--prefix_name', type=str, default='gemini')
   parser.add_argument('--batch_size', type=int, default=32)
   parser.add_argument('--save_every', type=int, default=100)
   parser.add_argument('--max_memory_usage', type=float, default=0.8)
   parser.add_argument('--subset_indices', type=str, default=None)
   parser.add_argument('--embedding_dim', type=int, default=768)
   parser.add_argument('--min_batch_size', type=int, default=1)
   return parser.parse_args()

def load_subset_indices(subset_indices_path: Optional[str], corpus_size: int, logger: ExperimentLogger) -> List[int]:
   if not subset_indices_path:
       return list(range(corpus_size))

   try:
       logger.log_step_start("Loading subset indices")
       subset_indices = read_json(subset_indices_path)
       if not isinstance(subset_indices, list) or not all(isinstance(idx, int) for idx in subset_indices):
           raise ValueError("Invalid format for subset indices file. Expected a list of integers.")
       logger.log_step_end("Subset indices loaded")
       return subset_indices
   except Exception as e:
       logger.log_error(e, "Failed to load subset indices")
       raise

class EmbeddingsComputer:
    def __init__(self, args: argparse.Namespace, logger: ExperimentLogger):
       self.args = args
       self.logger = logger
       self.cache_dir = Path(args.output_dir) / "cache"
       self.cache_dir.mkdir(parents=True, exist_ok=True)
       self.memory_tracker = MemoryTracker(args.max_memory_usage)
       
       #genai.configure(api_key=os.getenv("GEMINI_TOKEN"))
       self.model = genai.GenerativeModel('gemini-pro')

    def compute_embeddings(self, corpus: List[Dict], subset_indices: Optional[List[int]] = None) -> None:
       if subset_indices is None:
           subset_indices = list(range(len(corpus)))
           
       total_batches = (len(subset_indices) + self.args.batch_size - 1) // self.args.batch_size
       
       for batch_idx in tqdm(range(total_batches)):
           try:
               # Adjust batch size based on memory
               self.args.batch_size = self.memory_tracker.get_optimal_batch_size(
                   self.args.batch_size,
                   self.args.min_batch_size
               )

               # Process batch
               start_idx = batch_idx * self.args.batch_size
               end_idx = min((batch_idx + 1) * self.args.batch_size, len(subset_indices))
               batch_indices = subset_indices[start_idx:end_idx]
               batch_corpus = [corpus[idx] for idx in batch_indices]

               # Compute embeddings
               batch_embeddings = self._compute_batch_embeddings(batch_corpus)

               # Cache results
               self._cache_embeddings(
                   batch_embeddings,
                   start_idx,
                   end_idx
               )

               time.sleep(0.1)  # Rate limiting for Gemini API
               self.memory_tracker.cleanup()

           except Exception as e:
               self.logger.log_error(e, f"Error processing batch {batch_idx}")
               continue

    @rate_limit
    def _compute_batch_embeddings(self, batch_corpus: List[Dict]) -> np.ndarray:
        embeddings = []
        for doc in batch_corpus:
            prompt = f"Generate a {self.args.embedding_dim}-dimensional dense vector representation for this text: {doc['text'][:1000]}"
            
            try:
                response = self.model.generate_content(prompt)
                vector = self._parse_vector_response(response.text)
                embeddings.append(vector)
            except Exception as e:
                self.logger.log_error(e, "Error generating embedding")
                embeddings.append(np.zeros(self.args.embedding_dim))
                
            #time.sleep(0.1)
            
        return np.array(embeddings)

    def _parse_vector_response(self, response: str) -> np.ndarray:
       try:
           values = [float(x) for x in response.strip().split()]
           if len(values) != self.args.embedding_dim:
               values = np.zeros(self.args.embedding_dim)
           return np.array(values)
       except:
           return np.zeros(self.args.embedding_dim)

    def _cache_embeddings(self, embeddings: np.ndarray, start_idx: int, end_idx: int) -> None:
       cache_file = self.cache_dir / f"embeddings_{start_idx}_{end_idx}.npy"
       np.save(cache_file, embeddings)

class MemoryTracker:
   def __init__(self, max_memory_usage: float):
       self.max_memory_usage = max_memory_usage
       
   def get_optimal_batch_size(self, current_batch_size: int, min_batch_size: int) -> int:
       if gc.collect() > 0:
           return max(min_batch_size, current_batch_size // 2)
       return current_batch_size

   def cleanup(self):
       gc.collect()

def compute_embeddings_with_memory_tracking(
   corpus: List[Dict],
   subset_indices: List[int],
   args: argparse.Namespace,
   logger: ExperimentLogger
) -> None:
   try:
       embeddings_computer = EmbeddingsComputer(args, logger)
       embeddings_computer.compute_embeddings(corpus, subset_indices)
   except Exception as e:
       logger.log_error(e, "Error computing embeddings")
       raise

def main():
   args = parse_arguments()
   logger = ExperimentLogger("corpus_embeddings", "logs")

   try:
       with logger:
           corpus = read_corpus_json(args.corpus_path)
           subset_indices = load_subset_indices(args.subset_indices, len(corpus), logger)
           compute_embeddings_with_memory_tracking(corpus, subset_indices, args, logger)

   except Exception as e:
       logger.log_error(e, "Main execution failed")
       raise

if __name__ == '__main__':
   main()