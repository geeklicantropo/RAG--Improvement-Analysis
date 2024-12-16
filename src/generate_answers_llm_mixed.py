import os 
import argparse
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from llm import LLM
from utils import *
#from prompt_dataset import PromptDataset
from src.utils.file_utils import seed_everything, write_pickle, read_pickle, read_corpus_json, str2bool

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

class MixedDocumentsDataset(Dataset):
    def __init__(self, corpus, data_path, tokenizer, max_tokenized_length,
                 retriever_search_results, random_search_results, 
                 documents_disposition_info, full_to_subset_idx_map=None,
                 do_normalize_query=True, gold_position=None,
                 get_documents_without_answer=False):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_tokenized_length = max_tokenized_length
        self.retriever_search_results = retriever_search_results
        self.random_search_results = random_search_results
        self.documents_disposition_info = documents_disposition_info
        self.full_to_subset_idx_map = full_to_subset_idx_map
        self.do_normalize_query = do_normalize_query
        self.gold_position = gold_position
        self.get_documents_without_answer = get_documents_without_answer
        
        with open(data_path) as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['question']
        gold_answer = item['answers'][0]
        
        retrieved_docs = []
        if self.retriever_search_results:
            doc_indices = self.retriever_search_results[idx][0]
            if self.full_to_subset_idx_map:
                doc_indices = [self.full_to_subset_idx_map[i] for i in doc_indices]
            retrieved_docs = [self.corpus[i] for i in doc_indices]
            
        random_docs = []
        if self.random_search_results:
            rand_indices = self.random_search_results[idx][0]
            if self.full_to_subset_idx_map:
                rand_indices = [self.full_to_subset_idx_map[i] for i in rand_indices]
            random_docs = [self.corpus[i] for i in rand_indices]

        documents = []
        if self.documents_disposition_info['put_retrieved_first']:
            documents = retrieved_docs[:self.documents_disposition_info['num_retrieved_documents']]
            documents.extend(random_docs[:self.documents_disposition_info['num_random_documents']])
        else:
            documents = random_docs[:self.documents_disposition_info['num_random_documents']]
            documents.extend(retrieved_docs[:self.documents_disposition_info['num_retrieved_documents']])

        prompt = self._format_prompt(query, documents)
        tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_tokenized_length)

        return {
            'prompt': prompt,
            'example_id': item['id'],
            'query': query,
            'gold_answer': gold_answer,
            'document_indices': [d['id'] for d in documents],
            'tokenized': tokenized
        }

    def _format_prompt(self, query: str, documents: List[Dict]) -> str:
        context_parts = [f"Question: {query}\n\nContext:"]
        
        for idx, doc in enumerate(documents, 1):
            prefix = "[GOLD] " if self._contains_answer(doc['text'], doc.get('answer', '')) else ""
            context_parts.append(
                f"\nDocument [{idx}] {prefix}:\n{doc['text']}"
            )
        
        context_parts.append("\nAnswer:")
        return "\n".join(context_parts)

    def _contains_answer(self, text: str, answer: str) -> bool:
        if not text or not answer:
            return False
        text = text.lower().strip()
        answer = answer.lower().strip()
        answer_words = set(answer.split())
        text_words = set(text.split())
        overlap = len(answer_words & text_words)
        return overlap >= len(answer_words) * 0.8

info = {
    "train": {
        "data_path": 'data/10k_train_dataset.json',
        "random_results_path": "data/10k_random_results_at60.pkl",
        "contriever_search_results_path": "data/contriever_search_results_at150.pkl",
    },
    "test": {
        "data_path": 'data/test_dataset.json',
        "random_results_path": "data/10k_other_random_results_at60.pkl",
        "contriever_search_results_path": "data/contriever_test_search_results_at150.pkl",
        "bm25_search_results_path": "data/bm25_test_search_results_at250.pkl",
    }
}

def parse_arguments():
   parser = argparse.ArgumentParser(description="Run LLM Generation with mixed documents.")
   parser.add_argument('--output_dir', type=str, default='data/gen_res')
   parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf')
   parser.add_argument('--model_max_length', type=int, default=4096)
   parser.add_argument('--use_bm25', type=str2bool, default=False)
   parser.add_argument('--gold_position', type=int, default=None)
   parser.add_argument('--num_retrieved_documents', type=int)
   parser.add_argument('--num_random_documents', type=int)
   parser.add_argument('--put_retrieved_first', type=str2bool, default=False)
   parser.add_argument('--get_documents_without_answer', type=str2bool, default=False)
   parser.add_argument('--use_test', type=str2bool, default=True)
   parser.add_argument('--max_new_tokens', type=int, default=15)
   parser.add_argument('--batch_size', type=int)
   parser.add_argument('--save_every', type=int, default=250)

   args = parser.parse_args()
   args.split = "test" if args.use_test else "train"
   args.num_documents_in_context = args.num_retrieved_documents + args.num_random_documents

   if args.num_retrieved_documents is None or args.num_random_documents is None:
       parser.error("'num_retrieved_documents' and 'num_random_documents' must be specified")
   return args

def load_corpus(args: argparse.Namespace) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
   if args.use_test and args.use_bm25:
       corpus_path = "data/processed/test_corpus_with_random_bm25.json" 
       mapping_path = "data/mappings/subset_to_full_test_random_bm25_in_corpus.pkl"
   elif args.use_bm25:
       corpus_path = "data/processed/corpus_with_random_at60.json"
       mapping_path = "data/mappings/subset_to_full_random_at60_in_corpus.pkl"
   else:
       corpus_path = "data/processed/corpus_with_contriever_at150.json"
       mapping_path = "data/mappings/subset_to_full_contriever_at150_in_corpus.pkl"

   corpus = read_corpus_json(corpus_path)
   mapping = read_pickle(mapping_path)
   
   return corpus, mapping

def load_search_results(args: argparse.Namespace) -> List[Tuple[List[int], List[float]]]:
   random_results_path = info[args.split]['random_results_path']
   random_search_results = read_pickle(random_results_path)

   if args.use_bm25:
       search_results_path = info[args.split]['bm25_search_results_path']
   else:
       search_results_path = info[args.split]['contriever_search_results_path']
   retriever_search_results = read_pickle(search_results_path)

   return retriever_search_results, random_search_results

def initialize_dataset_and_loader(
   args: argparse.Namespace, 
   corpus: List[Dict], 
   mapping: Optional[Dict[int, int]], 
   retriever_search_results: List[Tuple[List[int], List[float]]], 
   random_search_results: List[Tuple[List[int], List[float]]], 
   tokenizer: PreTrainedTokenizer
) -> DataLoader:
   documents_disposition_info = {
       "num_retrieved_documents": args.num_retrieved_documents,
       "num_random_documents": args.num_random_documents,
       "put_retrieved_first": args.put_retrieved_first,
   }
   
   data_path = info[args.split]['data_path']
   
   prompt_ds = MixedDocumentsDataset(
       corpus=corpus, 
       data_path=data_path,
       tokenizer=tokenizer,
       max_tokenized_length=args.model_max_length - 2,
       retriever_search_results=retriever_search_results,
       random_search_results=random_search_results,
       documents_disposition_info=documents_disposition_info,
       full_to_subset_idx_map=mapping,
       do_normalize_query=True,
       gold_position=args.gold_position,
       get_documents_without_answer=args.get_documents_without_answer,
   )
       
   prompt_dataloader = DataLoader(
       prompt_ds,
       batch_size=args.batch_size,
       shuffle=False,
       num_workers=8,
       pin_memory=True,
   )
   return prompt_dataloader

def print_info(args: argparse.Namespace):
   print("INFO:")    
   print("MIXED")
   print(f"DATA: {info[args.split]['data_path']}")
   print(f"USE TEST: {args.use_test}")
   print(f"MODEL: {args.llm_id}")
   print(f"USE BM25: {args.use_bm25}")
   print(f"NUM RETRIEVED DOCS: {args.num_retrieved_documents}")
   print(f"NUM RANDOM DOCS: {args.num_random_documents}")
   print(f"PUT RETRIEVED DOCS FIRST: {args.put_retrieved_first}")
   print(f"GOLD POSITION: {args.gold_position}")
   print(f"NUM DOCUMENTS IN CONTEXT: {args.num_documents_in_context}")
   print(f"DOCUMENTS WITHOUT ANSWER: {args.get_documents_without_answer}")
   print(f"BATCH SIZE: {args.batch_size}")
   print(f"SAVE EVERY: {args.save_every}")

def generate_and_save(
   args: argparse.Namespace, 
   llm: LLM, 
   prompt_dataloader: DataLoader
):
   llm_id = args.llm_id
   num_doc = args.num_documents_in_context
   save_every = args.save_every
   answerless_str = "_answerless" if args.get_documents_without_answer else ""
   retriever_str = "bm25" if args.use_bm25 else "contriever"

   if args.put_retrieved_first:
       first_type_str = f"_retr{args.num_retrieved_documents}"
       second_type_str = f"_rand{args.num_random_documents}"
   else:
       first_type_str = f"_rand{args.num_random_documents}"
       second_type_str = f"_retr{args.num_retrieved_documents}"

   llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
   saving_dir = f"{args.output_dir}/{llm_folder}/{args.split}/mixed/{retriever_str}/{num_doc}_doc"
   if not os.path.exists(saving_dir):
       os.makedirs(saving_dir)
   
   answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"

   all_info = []  
   for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
       prompts = prompt_batch['prompt']
       generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
       
       generated_answers = []
       for output in generated_output:
           start = output.find(answer_string_in_prompt) + len(answer_string_in_prompt)
           response = output[start:].strip()
           print(f"Found gold document in response: {response}")
           generated_answers.append(response)

       prompt_batch['generated_answer'] = generated_answers
       all_info.append(prompt_batch)
       
       if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
           print(f"Saving at {idx + 1}...")
           file_name = f"{saving_dir}/numdoc{num_doc}{first_type_str}{second_type_str}{answerless_str}_info_{idx+1}.pkl"
           write_pickle(all_info, file_name)
           all_info = []

def main():
   args = parse_arguments()

   print("Loading LLM...")
   llm = LLM(
       llm_id=args.llm_id,
       device=device, 
       quantization_bits=4,
       model_max_length=args.model_max_length
   )
   tokenizer = llm.tokenizer
   print("LLM loaded")

   print("Loading corpus and search results...")
   corpus, mapping = load_corpus(args)
   retriever_search_results, random_search_results = load_search_results(args)
   print("Corpus and search results loaded")

   print("Loading prompt dataset...") 
   prompt_dataloader = initialize_dataset_and_loader(
       args, corpus, mapping, retriever_search_results,
       random_search_results, tokenizer
   )
   print("Prompt dataset loaded")

   print_info(args)
   generate_and_save(args, llm, prompt_dataloader)

if __name__ == "__main__":
   seed_everything(SEED)
   main()