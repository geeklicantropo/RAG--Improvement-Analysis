import os 
import argparse
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from llm import LLM
from utils import *
from prompt_dataset import PromptDataset
from experiment_logger import ExperimentLogger
from cluster_utils import DocumentClusterer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED = 10

info = {
    "data_path": 'data/10k_train_dataset.json',
    "random_results_path": "data/10k_random_results_at60.pkl",
    "adore_search_results_path": "data/adore_search_results_at200.pkl",
    "contriever_search_results_path": "data/contriever_search_results_at150.pkl",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Generation with clustering support.")
    
    # Basic configuration
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--load_full_corpus', type=str2bool, help='Load the full corpus', default=True)    
    
    # Retrieval configuration
    parser.add_argument('--use_random', type=str2bool, help='Use random irrelevant documents')
    parser.add_argument('--use_adore', type=str2bool, help="Use the retrieved documents from ADORE", default=False)
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context')
    parser.add_argument('--num_documents_in_context', type=int, help='Total number of documents in the context')
    parser.add_argument('--get_documents_without_answer', type=str2bool, help='Select only documents without the answer', default=True)
    
    # Generation configuration
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=15)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)
    
    # Clustering configuration
    parser.add_argument('--use_clustering', type=str2bool, default=False, help='Whether to use document clustering')
    parser.add_argument('--num_clusters', type=int, default=None, help='Number of clusters for K-means')
    parser.add_argument('--cluster_seed', type=int, default=42, help='Random seed for clustering')
    
    args = parser.parse_args()

    if args.num_documents_in_context is None:
        parser.error("'num_documents_in_context' must be specified.")
    if args.num_documents_in_context <= 0:
        parser.error("'num_documents_in_context' must be a positive integer.")
    if args.gold_position is not None and (args.gold_position < 0 or args.gold_position >= args.num_documents_in_context):
        parser.error("'gold_position' must be within the range of 'num_documents_in_context'.")
    if args.use_clustering and args.num_clusters is None:
        parser.error("'num_clusters' must be specified when using clustering.")

    return args

def load_corpus(
    args: argparse.Namespace,
    logger: ExperimentLogger
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    """Load corpus with logging."""
    logger.log_step_start("Loading corpus")
    try:
        if args.load_full_corpus:
            corpus = read_corpus_json('data/corpus.json')
            full_to_subset_idx_map = None
        else:
            if args.use_random:
                corpus, full_to_subset_idx_map = read_corpus_with_random()
            elif args.use_adore:
                corpus, full_to_subset_idx_map = read_corpus_with_adore()
            else:
                corpus, full_to_subset_idx_map = read_corpus_with_contriever()
        
        logger.log_metric("corpus_size", len(corpus))
        logger.log_step_end("Loading corpus", time.time())
        return corpus, full_to_subset_idx_map
    
    except Exception as e:
        logger.log_error(e, "Error loading corpus")
        raise

def load_search_results(
    args: argparse.Namespace,
    logger: ExperimentLogger
) -> List[Tuple[List[int], List[float]]]:
    """Load search results with logging."""
    logger.log_step_start("Loading search results")
    try:
        if args.use_random:
            search_results_path = info['random_results_path']
        elif args.use_adore:
            search_results_path = info['adore_search_results_path']
        else:
            search_results_path = info['contriever_search_results_path']
        
        search_results = read_pickle(search_results_path)
        logger.log_metric("num_search_results", len(search_results))
        logger.log_step_end("Loading search results", time.time())
        return search_results
    
    except Exception as e:
        logger.log_error(e, "Error loading search results")
        raise

def initialize_dataset_and_loader(
    args: argparse.Namespace,
    corpus: List[Dict],
    full_to_subset_idx_map: Optional[Dict[int, int]],
    search_results: List[Tuple[List[int], List[float]]],
    tokenizer: PreTrainedTokenizer,
    logger: ExperimentLogger
) -> DataLoader:
    """Initialize dataset and dataloader with clustering support."""
    logger.log_step_start("Initializing dataset")
    try:
        prompt_ds = PromptDataset(
            corpus=corpus,
            data_path=info['data_path'],
            tokenizer=tokenizer,
            max_tokenized_length=args.model_max_length - 2,
            search_results=search_results,
            full_to_subset_idx_map=full_to_subset_idx_map,
            do_normalize_query=True,
            num_documents_in_context=args.num_documents_in_context,
            gold_position=args.gold_position,
            get_documents_without_answer=args.get_documents_without_answer,
            use_clustering=args.use_clustering,
            num_clusters=args.num_clusters,
            cluster_seed=args.cluster_seed
        )
        
        prompt_dataloader = DataLoader(
            prompt_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        
        logger.log_metric("dataset_size", len(prompt_ds))
        logger.log_step_end("Initializing dataset", time.time())
        return prompt_dataloader
    
    except Exception as e:
        logger.log_error(e, "Error initializing dataset")
        raise

def print_experiment_info(args: argparse.Namespace, logger: ExperimentLogger):
    """Log experiment information."""
    info_dict = {
        "DATA_PATH": info['data_path'],
        "MODEL": args.llm_id,
        "USE_RANDOM": args.use_random,
        "USE_ADORE": args.use_adore,
        "USE_CLUSTERING": args.use_clustering,
        "NUM_CLUSTERS": args.num_clusters if args.use_clustering else "N/A",
        "GOLD_POSITION": args.gold_position,
        "NUM_DOCUMENTS": args.num_documents_in_context,
        "DOCUMENTS_WITHOUT_ANSWER": args.get_documents_without_answer,
        "BATCH_SIZE": args.batch_size,
        "SAVE_EVERY": args.save_every
    }
    
    logger.experiment_logger.info("Experiment Configuration:")
    for key, value in info_dict.items():
        logger.experiment_logger.info(f"{key}: {value}")

def generate_and_save(
    args: argparse.Namespace,
    llm: LLM,
    prompt_dataloader: DataLoader,
    logger: ExperimentLogger
):
    """Generate and save responses with comprehensive logging."""
    # Setup paths and configurations
    llm_id = args.llm_id
    num_doc = args.num_documents_in_context
    save_every = args.save_every
    gold_pos = args.gold_position
    
    retriever_str = "adore" if args.use_adore else "contriever"
    rand_str = "_rand" if args.use_random else ""
    answerless_str = "_answerless" if args.get_documents_without_answer else ""
    cluster_str = f"_cluster{args.num_clusters}" if args.use_clustering else ""

    # Create saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{llm_folder}/train/classic/{retriever_str}/{num_doc}_doc"
    os.makedirs(saving_dir, exist_ok=True)
    
    answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"
    
    logger.log_step_start("Generation phase")
    all_info = []
    try:
        for idx, prompt_batch in enumerate(tqdm(prompt_dataloader, desc="Generating responses")):
            # Generate responses
            prompts = prompt_batch['prompt']
            generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
            
            # Extract answers
            generated_answers = []
            for output in generated_output:
                start = output.find(answer_string_in_prompt) + len(answer_string_in_prompt)
                response = output[start:].strip()
                generated_answers.append(response)
            
            prompt_batch['generated_answer'] = generated_answers
            all_info.append(prompt_batch)
            
            # Save periodically
            if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
                logger.experiment_logger.info(f"Saving at batch {idx + 1}...")
                file_name = f"{saving_dir}/numdoc{num_doc}_gold_at{gold_pos}{rand_str}{answerless_str}{cluster_str}_info_{idx+1}.pkl"
                write_pickle(all_info, file_name)
                logger.log_metric(f"saved_batch_{idx+1}", file_name)
                all_info = []
        
        logger.log_step_end("Generation phase", time.time())
        
    except Exception as e:
        logger.log_error(e, "Error during generation")
        raise

def main():
    args = parse_arguments()
    
    # Initialize logger
    logger = ExperimentLogger(
        experiment_name="rag_generation",
        base_log_dir="logs"
    )
    
    try:
        with logger:
            logger.log_experiment_params(vars(args))
            print_experiment_info(args, logger)
            
            # Load LLM
            logger.log_step_start("Loading LLM")
            llm = LLM(
                args.llm_id,
                device,
                quantization_bits=4,
                model_max_length=args.model_max_length
            )
            tokenizer = llm.tokenizer
            logger.log_step_end("Loading LLM", time.time())
            
            # Load data
            corpus, full_to_subset_idx_map = load_corpus(args, logger)
            search_results = load_search_results(args, logger)
            
            # Initialize dataset and dataloader
            prompt_dataloader = initialize_dataset_and_loader(
                args, corpus, full_to_subset_idx_map,
                search_results, tokenizer, logger
            )
            
            # Generate and save results
            generate_and_save(args, llm, prompt_dataloader, logger)
            
    except Exception as e:
        logger.log_error(e, "Error in main execution")
        raise

if __name__ == "__main__":
    seed_everything(SEED)
    main()