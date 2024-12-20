import os
import gc
import psutil
import argparse
import warnings
import random
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from src.utils.rate_limit import rate_limit

import torch
import json
from torch.utils.data import DataLoader
import logging
import pickle
from llm import LLM
from src.prompt_dataset import PromptDataset
from src.utils.file_utils import str2bool, read_pickle, read_json, seed_everything, read_corpus_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
SEED = 10

info = {
   "train": {"data_path": 'data/10k_train_dataset.json'},
   "test": {"data_path": 'data/test_dataset.json'}
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Closed-Book Generation.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res')
    parser.add_argument('--llm_id', type=str, default='gemini-1.5-flash')
    parser.add_argument('--model_max_length', type=int, default=4096)
    parser.add_argument('--subset_path', type=str, default='data/processed/corpus_with_contriever_at150.json')
    parser.add_argument('--use_random', type=str2bool, default=False)
    parser.add_argument('--use_adore', type=str2bool, default=False)
    parser.add_argument('--use_test', type=str2bool, default=False)
    parser.add_argument('--random_docs_path', type=str, default='data/processed/corpus_with_random_50_words.pkl')
    parser.add_argument('--adv_docs_path', type=str, default='data/processed/reddit_corpus.pkl')
    parser.add_argument('--gold_position', type=int, default=0)
    parser.add_argument('--num_documents_in_context', type=int, default=7)
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)
    return parser.parse_args()

def load_corpus_and_documents(args: argparse.Namespace) -> Dict:
    corpus_path = args.subset_path
    if args.use_random:
        corpus_path = args.random_docs_path
    elif args.use_adore:
        corpus_path = args.adv_docs_path

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

def prepare_context(
    question: str,
    gold_doc: Optional[Dict],
    random_docs: Optional[List[Dict]] = None,
    adv_docs: Optional[List[Dict]] = None,
    cluster_docs: Optional[List[Dict]] = None,
    gold_position: int = 0,
    num_docs: int = 7
) -> List[Dict]:
    """
    Prepares the context by combining gold, random, adversarial, and cluster documents.

    Args:
        question (str): The question to be answered.
        gold_doc (Optional[Dict]): The gold document, if available.
        random_docs (Optional[List[Dict]]): Random documents to include in the context.
        adv_docs (Optional[List[Dict]]): Adversarial documents to include in the context.
        cluster_docs (Optional[List[Dict]]): Cluster-specific documents to include.
        gold_position (int): The position to insert the gold document.
        num_docs (int): Maximum number of documents in the context.

    Returns:
        List[Dict]: The prepared context of documents.
    """
    context = []

    if cluster_docs:
        context.extend(random.sample(cluster_docs, min(len(cluster_docs), num_docs - 1)))

    if random_docs:
        context.extend(random.sample(random_docs, min(len(random_docs), num_docs - len(context) - 1)))

    if adv_docs:
        context.extend(random.sample(adv_docs, min(len(adv_docs), num_docs - len(context) - 1)))

    if gold_doc:
        context.insert(gold_position, gold_doc)

    return context[:num_docs]


def _format_prompt(question: str, documents: List[Dict], gold_answer: Optional[str] = None, cluster_id: Optional[int] = None) -> str:
    """
    Formats the prompt with the question, documents, and optional clustering context.

    Args:
        question (str): The question to be answered.
        documents (List[Dict]): List of documents to include in the context.
        gold_answer (Optional[str]): The gold answer, if available.
        cluster_id (Optional[int]): ID of the cluster the documents belong to (optional).

    Returns:
        str: The formatted prompt for the LLM.
    """
    context_parts = [f"Question: {question}\n\nContext:"]
    if cluster_id is not None:
        context_parts.append(f"\nCluster ID: {cluster_id}")

    for idx, doc in enumerate(documents, 1):
        prefix = "[GOLD] " if gold_answer and _contains_answer(doc.get('text', ''), gold_answer) else ""
        context_parts.append(f"\nDocument [{idx}] {prefix}:\n{doc.get('text', '')}")

    context_parts.append("\nAnswer:")
    return "\n".join(context_parts)


def initialize_dataset_and_loader(args: argparse.Namespace):
    """
    Initializes the dataset and DataLoader for prompts.
    """
    data_path = info['test' if args.use_test else 'train']['data_path']
    print(f"Using dataset: {data_path}")

    prompt_ds = PromptDataset(
        data_path=data_path,
        model_name=args.llm_id,
        do_normalize_query=True
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
    """
    Cleans up GPU and RAM memory to prevent memory leaks.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def _contains_answer(text: str, answer: str) -> bool:
    """
    Checks if the text contains the answer.
    """
    if not text or not answer:
        return False
    text = text.lower().strip()
    answer = answer.lower().strip()
    answer_words = set(answer.split())
    text_words = set(text.split())
    overlap = len(answer_words & text_words)
    return overlap >= len(answer_words) * 0.8

def _get_memory_usage() -> Dict[str, float]:
    """
    Returns the memory usage of the current process and GPU(s), if available.
    """
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
    """
    Converts unsupported data types to serializable formats.
    """
    if torch.is_tensor(obj):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    return obj

def save_checkpoint(results: List[Dict], batch_idx: int, output_dir: Path, llm: LLM) -> None:
    """
    Saves a checkpoint of the generated results.
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"checkpoint_batch_{batch_idx}_{timestamp}.json"

    processed_results = [_make_serializable(result) for result in results]

    with open(checkpoint_path, 'w') as f:
        json.dump(processed_results, f, indent=2)


@rate_limit
def generate_and_save(
    args: argparse.Namespace,
    llm: LLM,
    prompt_dataloader: DataLoader,
    corpus: List[Dict],
    random_docs: Optional[List[Dict]] = None,
    adv_docs: Optional[List[Dict]] = None,
    cluster_docs: Optional[Dict[int, List[Dict]]] = None
):
    """
    Generate responses and save results, incorporating clustering-specific context.

    Args:
        args (argparse.Namespace): Parsed arguments.
        llm (LLM): LLM instance for generating answers.
        prompt_dataloader (DataLoader): DataLoader for question prompts.
        corpus (List[Dict]): Full document corpus.
        random_docs (Optional[List[Dict]]): Random documents for context.
        adv_docs (Optional[List[Dict]]): Adversarial documents for context.
        cluster_docs (Optional[Dict[int, List[Dict]]]): Documents grouped by cluster ID.
    """
    saving_dir = Path(f"{args.output_dir}/{args.llm_id}/{args.split}/clustered_rag")
    saving_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        for item in prompt_batch:
            gold_doc = next((doc for doc in corpus if _contains_answer(doc['text'], item['gold_answer'])), None)

            context = prepare_context(
                item['query'],
                gold_doc,
                random_docs=random_docs,
                adv_docs=adv_docs,
                cluster_docs=cluster_docs.get(item.get('cluster_id')) if cluster_docs else None,
                gold_position=args.gold_position,
                num_docs=args.num_documents_in_context
            )

            formatted_prompt = _format_prompt(
                item['query'], context, item['gold_answer'], cluster_id=item.get('cluster_id')
            )
            response = llm.generate([formatted_prompt], max_new_tokens=args.max_new_tokens)[0]

            result = {
                'query': item['query'],
                'example_id': item['example_id'],
                'generated_answer': response.strip(),
                'gold_answer': item['gold_answer'],
                'context': context,
                'experiment_type': 'clustered_rag'
            }
            all_results.append(result)

        if (idx + 1) % args.save_every == 0:
            save_checkpoint(all_results, idx + 1, saving_dir, llm)
            all_results = []

    save_checkpoint(all_results, "final", saving_dir, llm)


'''
def _load_json_data(path: str) -> Any:
    with open(path) as f:
        return json.load(f)

def _load_pickle_data(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
'''

def main():
    args = parse_arguments()

    print("Loading LLM...")
    llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
    print("LLM loaded")

    print("Loading corpus and additional documents...")
    corpus, _ = load_corpus_and_documents(args)
    random_docs = read_pickle(args.random_docs_path) if args.use_random else None
    adv_docs = read_pickle(args.adv_docs_path) if args.use_adore else None
    print("Corpus and documents loaded")

    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(args)
    print("Prompt dataset loaded")

    generate_and_save(args, llm, prompt_dataloader, corpus, random_docs, adv_docs)

if __name__ == "__main__":
    seed_everything(SEED)
    main()
