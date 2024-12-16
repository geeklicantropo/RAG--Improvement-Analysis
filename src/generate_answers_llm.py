import os 
import gc
import psutil
import argparse
import warnings
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import json
from torch.utils.data import Dataset, DataLoader

from llm import LLM
from utils import *
from src.prompt_dataset import PromptDataset
from src.utils.file_utils import str2bool, read_pickle, read_json, write_pickle, read_corpus_json, seed_everything

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
   "train": {
       "data_path": 'data/10k_train_dataset.json',
   },
   "test": {
       "data_path": 'data/test_dataset.json',
   }
}

def parse_arguments():
   parser = argparse.ArgumentParser(description="Run LLM Closed-Book Generation.")
   parser.add_argument('--output_dir', type=str, default='data/gen_res')
   parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf')
   parser.add_argument('--model_max_length', type=int, default=4096)
   parser.add_argument('--subset_path', type=str, default='data/processed/corpus_with_contriever_at150.json')
   parser.add_argument('--use_random', type=str2bool, default=False)
   parser.add_argument('--use_adore', type=str2bool, default=False)
   parser.add_argument('--use_test', type=str2bool, default=False)
   parser.add_argument('--max_new_tokens', type=int, default=15)
   parser.add_argument('--batch_size', type=int)
   parser.add_argument('--save_every', type=int, default=250)
   return parser.parse_args()

def load_corpus(args: argparse.Namespace) -> Dict:
   corpus_path = args.subset_path
   if args.use_random:
       corpus_path = "data/processed/corpus_with_random_at60.json"
   elif args.use_adore:
       corpus_path = "data/processed/corpus_with_adore_at200.json"
   
   mapping_path = "data/mappings/subset_to_full_contriever_at150_in_corpus.pkl"
   if args.use_random:
       mapping_path = "data/mappings/subset_to_full_random_at60_in_corpus.pkl"
   elif args.use_adore:
       mapping_path = "data/mappings/subset_to_full_adore_at200_in_corpus.pkl"
       
   corpus = read_corpus_json(corpus_path)
   mapping = read_pickle(mapping_path)
   
   print(f"Loaded corpus from {corpus_path}")
   print(f"Using mapping file: {mapping_path}")
   
   return corpus, mapping

def _format_prompt(query: str, documents: List[Dict], gold_answer: str) -> str:
    context_parts = [f"Question: {query}\n\nContext:"]
    
    doc_idx = 1
    for doc in documents:
        # Check if document contains answer
        is_gold = _contains_answer(doc['text'], gold_answer)
        prefix = "[GOLD] " if is_gold else ""
        
        context_parts.append(
            f"\nDocument [{doc_idx}] {prefix}:\n{doc['text']}"
        )
        if is_gold:
            print(f"Found gold document containing answer: {gold_answer}")
        doc_idx += 1
        
    context_parts.append("\nAnswer:")
    return "\n".join(context_parts)

def initialize_dataset_and_loader(args: argparse.Namespace):
    data_path = info[args.split]['data_path']
    print(f"Using dataset: {data_path}")
    
    prompt_ds = PromptDataset(
        data_path=data_path,
        model_name=args.llm_id, 
        do_normalize_query=True,
    )
    prompt_dataloader = DataLoader(
        prompt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return prompt_dataloader

def _cleanup_memory():
   gc.collect()
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
       torch.cuda.synchronize()

def _get_memory_usage() -> Dict[str, float]:
   memory_stats = {
       'ram_used_gb': psutil.Process().memory_info().rss / (1024 ** 3),
       'ram_percent': psutil.virtual_memory().percent
   }
   
   if torch.cuda.is_available():
       memory_stats.update({
           f'gpu_{i}_used_gb': torch.cuda.memory_allocated(i) / (1024 ** 3)
           for i in range(torch.cuda.device_count())
       })
   
   return memory_stats

def _make_serializable(obj: Any) -> Any:
   if torch.is_tensor(obj):
       return obj.tolist()
   elif isinstance(obj, dict):
       return {k: _make_serializable(v) for k, v in obj.items()}
   elif isinstance(obj, list):
       return [_make_serializable(item) for item in obj]
   elif hasattr(obj, '__dict__'):
       return _make_serializable(obj.__dict__)
   return obj

def save_checkpoint(results: List[Dict], batch_idx: int, output_dir: Path) -> None:
   checkpoint_dir = output_dir / "checkpoints"
   checkpoint_dir.mkdir(exist_ok=True)
   
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   checkpoint_path = checkpoint_dir / f"checkpoint_batch_{batch_idx}_{timestamp}.json"
   
   with tqdm(total=1, desc=f"Saving checkpoint {batch_idx}") as pbar:
       serializable_results = _make_serializable(results)
       with open(checkpoint_path, 'w') as f:
           json.dump(serializable_results, f, indent=2)
       pbar.update(1)

def _contains_answer(text: str, answer: str) -> bool:
    if not text or not answer:
        return False
    text = text.lower().strip()
    answer = answer.lower().strip()
    answer_words = set(answer.split())
    text_words = set(text.split())
    overlap = len(answer_words & text_words)
    return overlap >= len(answer_words) * 0.8

def generate_and_save(args: argparse.Namespace, llm: LLM, prompt_dataloader: DataLoader, corpus: List[Dict]):
   llm_id = args.llm_id
   save_every = args.save_every
   mapping = read_pickle("data/mappings/subset_to_full_contriever_at150_in_corpus.pkl")

   llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
   saving_dir = Path(f"{args.output_dir}/{llm_folder}/{args.split}/only_query")
   saving_dir.mkdir(parents=True, exist_ok=True)

   answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"

   all_info = []  
   for idx, prompt_batch in enumerate(tqdm(prompt_dataloader, desc="Generating")):
       if _get_memory_usage()['ram_percent'] > 90:
           _cleanup_memory()
           
       prompts = []
       for item in prompt_batch:
           doc_ids = [mapping.get(doc_id, doc_id) for doc_id in item['document_indices']]
           docs = [corpus[doc_id] for doc_id in doc_ids]
           
           # Check for gold documents
           for doc in docs:
               if _contains_answer(doc['text'], item['gold_answer']):
                   print(f"Found gold document {doc['id']} containing answer: {item['gold_answer']}")
           
           formatted_prompt = _format_prompt(item['query'], docs, item['gold_answer'])
           prompts.append(formatted_prompt)
       
       generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
       
       generated_answers = []
       for output in generated_output:
           start = output.find(answer_string_in_prompt) + len(answer_string_in_prompt)
           response = output[start:].strip()
           generated_answers.append(response)

       prompt_batch['generated_answer'] = generated_answers
       all_info.append(prompt_batch)
       
       if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
           save_checkpoint(all_info, idx + 1, saving_dir)
           all_info = []
           _cleanup_memory()

def main():
   args = parse_arguments()

   print("Loading LLM...")
   llm = LLM(
       llm_id=args.llm_id,
       device=device,
       quantization_bits=4, 
       model_max_length=args.model_max_length
   )
   print("LLM loaded")

   print("Loading corpus...")
   corpus, mapping = load_corpus(args)
   print("Corpus loaded")

   print("Loading prompt dataset...")
   prompt_dataloader = initialize_dataset_and_loader(args)
   print("Prompt dataset loaded")

   generate_and_save(args, llm, prompt_dataloader, corpus)

if __name__ == "__main__":
   seed_everything(SEED)
   main()