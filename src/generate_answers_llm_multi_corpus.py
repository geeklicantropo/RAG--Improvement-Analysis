import os
import json
import argparse
import warnings
from tqdm import tqdm
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
import torch

from src.llm import LLM
from src.utils.file_utils import seed_everything, read_pickle, read_corpus_json, str2bool

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED = 10

class MultiCorpusDataset(Dataset):
    def __init__(
        self,
        corpus: List[Dict],
        data_path: str,
        tokenizer,
        max_tokenized_length: int,
        documents_other_corpus: List[Dict],
        full_to_subset_idx_map: Optional[Dict[int, int]] = None,
        do_normalize_query: bool = True
    ):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_tokenized_length = max_tokenized_length
        self.documents_other_corpus = documents_other_corpus
        self.full_to_subset_idx_map = full_to_subset_idx_map
        self.do_normalize_query = do_normalize_query
        
        with open(data_path) as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['question']
        gold_answer = item['answers'][0]

        # Get gold document and other documents
        gold_doc = None
        for doc in self.corpus:
            if self._contains_answer(doc['text'], gold_answer):
                gold_doc = doc
                break

        other_docs = []
        for doc in self.documents_other_corpus[:5]:  # Get 5 other documents
            if not self._contains_answer(doc['text'], gold_answer):
                other_docs.append(doc)

        # Format prompt with gold doc first
        prompt = self._format_prompt(query, gold_doc, other_docs, gold_answer)
        
        return {
            'prompt': prompt,
            'example_id': item['id'],
            'query': query,
            'gold_answer': gold_answer,
            'tokenized': self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_tokenized_length
            )
        }

    def _contains_answer(self, text: str, answer: str) -> bool:
        if not text or not answer:
            return False
        text = text.lower().strip()
        answer = answer.lower().strip()
        answer_words = set(answer.split())
        text_words = set(text.split())
        overlap = len(answer_words & text_words)
        return overlap >= len(answer_words) * 0.8

    def _format_prompt(self, query: str, gold_doc: Optional[Dict], other_docs: List[Dict], gold_answer: str) -> str:
        context_parts = [f"Question: {query}\n\nContext:"]
        
        # Add gold document first with [GOLD] marker
        if gold_doc:
            context_parts.append(f"\nDocument [1] [GOLD]:\n{gold_doc['text']}")
            print(f"Found gold document containing answer: {gold_answer}")
        
        # Add other documents
        for idx, doc in enumerate(other_docs, start=2):
            context_parts.append(f"\nDocument [{idx}]:\n{doc['text']}")
        
        context_parts.append("\nAnswer:")
        return "\n".join(context_parts)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Generation with multi corpus documents.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res')
    parser.add_argument('--llm_id', type=str, default='gemini-pro')
    parser.add_argument('--model_max_length', type=int, default=4096)
    parser.add_argument('--subset_path', type=str, 
                       default='data/processed/corpus_with_contriever_at150.json')
    parser.add_argument('--use_corpus_nonsense', type=str2bool, default=False)
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--use_test', type=str2bool, default=False)
    parser.add_argument('--save_every', type=int, default=250)
    return parser.parse_args()

def load_corpus(args):
    if args.use_corpus_nonsense:
        corpus = read_pickle('data/processed/corpus_with_random_50_words.pkl')
        mapping = None
    else:
        corpus = read_corpus_json(args.subset_path)
        mapping = read_pickle('data/mappings/subset_to_full_contriever_at150_in_corpus.pkl')
    
    other_corpus = read_pickle('data/processed/reddit_corpus.pkl')
    return corpus, mapping, other_corpus

def generate_and_save(args: argparse.Namespace, llm: LLM, prompt_dataloader: DataLoader):
    llm_id = args.llm_id
    save_every = args.save_every
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{llm_folder}/{'test' if args.use_test else 'train'}/multi_corpus"
    os.makedirs(saving_dir, exist_ok=True)

    all_info = []
    answer_string = "### Response:" if 'mpt' in llm_id else "Answer:"

    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader, desc="Generating")):
        prompts = prompt_batch['prompt']
        generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
        
        generated_answers = []
        for output in generated_output:
            start = output.find(answer_string) + len(answer_string)
            response = output[start:].strip()
            generated_answers.append(response)

        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)

        if (idx + 1) % save_every == 0:
            file_name = f"{saving_dir}/batch_{idx+1}_results.pkl"
            read_pickle(all_info, file_name)
            all_info = []

    if all_info:
        file_name = f"{saving_dir}/final_batch_results.pkl"
        read_pickle(all_info, file_name)

def main():
    args = parse_arguments()
    
    print("Loading LLM...")
    #llm = LLM(args.llm_id, device, quantization_bits=4, model_max_length=args.model_max_length)
    
    llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
    print("Loading corpus...")
    corpus, mapping, other_corpus = load_corpus(args)
    
    print("Initializing dataset...")
    dataset_path = 'data/test_dataset.json' if args.use_test else 'data/10k_train_dataset.json'
    
    prompt_ds = MultiCorpusDataset(
        corpus=corpus,
        data_path=dataset_path,
        tokenizer=llm.tokenizer,
        max_tokenized_length=args.model_max_length - 2,
        documents_other_corpus=other_corpus,
        full_to_subset_idx_map=mapping,
        do_normalize_query=True
    )
    
    prompt_dataloader = DataLoader(
        prompt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    generate_and_save(args, llm, prompt_dataloader)

if __name__ == "__main__":
    seed_everything(SEED)
    main()