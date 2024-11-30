import os
import argparse
import warnings
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple

import torch
from torch.utils.data import DataLoader

from llm import LLM
from utils import seed_everything, read_pickle, read_corpus_json
from prompt_dataset import PromptDataset
from experiment_logger import ExperimentLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED = 10

info = {
    "train": {
        "data_path": 'data/10k_train_dataset.json',
        "search_results_path": "data/contriever_search_results_at150.pkl",
    },
    "test": {
        "data_path": 'data/test_dataset.json',
        "search_results_path": "data/contriever_test_search_results_at150.pkl",
    }
}

class AnswerGenerationConfig:
    def __init__(self, args: argparse.Namespace):
        self.output_dir = args.output_dir
        self.llm_id = args.llm_id
        self.model_max_length = args.model_max_length
        self.use_clustering = args.use_clustering
        self.num_clusters = args.num_clusters if hasattr(args, 'num_clusters') else None
        self.gold_position = args.gold_position
        self.num_documents = args.num_documents_in_context
        self.get_documents_without_answer = args.get_documents_without_answer
        self.noise_ratio = args.noise_ratio if hasattr(args, 'noise_ratio') else 0.0
        self.use_test = args.use_test
        self.max_new_tokens = args.max_new_tokens
        self.batch_size = args.batch_size
        self.save_every = args.save_every

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Generation with clustering and noise support.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_max_length', type=int, default=4096)
    parser.add_argument('--use_clustering', type=bool, default=False)
    parser.add_argument('--num_clusters', type=int)
    parser.add_argument('--gold_position', type=int)
    parser.add_argument('--num_documents_in_context', type=int, required=True)
    parser.add_argument('--get_documents_without_answer', type=bool, default=False)
    parser.add_argument('--noise_ratio', type=float, default=0.0)
    parser.add_argument('--use_test', type=bool, default=False)
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--save_every', type=int, default=250)
    return parser.parse_args()

class AnswerGenerator:
    def __init__(self, config: AnswerGenerationConfig, logger: ExperimentLogger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_llm()

    def _initialize_llm(self):
        self.logger.log_step_start("Initializing LLM")
        self.llm = LLM(
            self.config.llm_id,
            self.device,
            quantization_bits=4,
            model_max_length=self.config.model_max_length
        )
        self.logger.log_step_end("LLM initialization")

    def _load_data(self) -> Tuple[List[Dict], List[Tuple[List[int], List[float]]]]:
        self.logger.log_step_start("Loading data")
        split = "test" if self.config.use_test else "train"
        
        corpus = read_corpus_json('data/corpus.json')
        search_results = read_pickle(info[split]["search_results_path"])
        
        self.logger.log_metric("corpus_size", len(corpus))
        self.logger.log_metric("num_search_results", len(search_results))
        self.logger.log_step_end("Data loading")
        
        return corpus, search_results

    def _create_dataset(self, corpus: List[Dict], search_results: List[Tuple[List[int], List[float]]]) -> DataLoader:
        self.logger.log_step_start("Creating dataset")
        split = "test" if self.config.use_test else "train"
        
        dataset = PromptDataset(
            corpus=corpus,
            data_path=info[split]["data_path"],
            tokenizer=self.llm.tokenizer,
            max_tokenized_length=self.config.model_max_length - 2,
            search_results=search_results,
            use_clustering=self.config.use_clustering,
            num_clusters=self.config.num_clusters,
            num_documents_in_context=self.config.num_documents,
            gold_position=self.config.gold_position,
            get_documents_without_answer=self.config.get_documents_without_answer,
            noise_ratio=self.config.noise_ratio
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.log_metric("dataset_size", len(dataset))
        self.logger.log_step_end("Dataset creation")
        
        return dataloader

    def generate(self) -> List[Dict]:
        corpus, search_results = self._load_data()
        dataloader = self._create_dataset(corpus, search_results)
        
        results = []
        answer_string = "### Response:" if 'mpt' in self.config.llm_id else "Answer:"
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            try:
                # Monitor memory
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    if current_memory > 0.95:
                        self.logger.log_metric("memory_warning", current_memory)
                        torch.cuda.empty_cache()

                prompts = batch['prompt']
                outputs = self.llm.generate(prompts, max_new_tokens=self.config.max_new_tokens)
                
                batch_results = self._process_batch(batch, outputs, answer_string)
                results.extend(batch_results)
                
                if (batch_idx + 1) % self.config.save_every == 0:
                    self._save_checkpoint(results, batch_idx + 1)
                
                # Clean up batch tensors
                del batch
                torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.log_error(e, f"Error processing batch {batch_idx}")
                continue
                
        return results

    def _process_batch(self, batch: Dict[str, Any], outputs: List[str], answer_string: str) -> List[Dict]:
        batch_results = []
        
        for idx, output in enumerate(outputs):
            try:
                start_idx = output.find(answer_string) + len(answer_string)
                
                result = {
                    'query': batch['query'][idx],
                    'generated_answer': output[start_idx:].strip(),
                    'document_indices': [int(i) for i in batch['document_indices'][idx]],
                    'prompt_tokens_len': int(batch['prompt_tokens_len'][idx])
                }
                
                if self.config.use_clustering:
                    result['cluster_assignments'] = batch['cluster_assignments'][idx]
                    
                batch_results.append(result)
                
            except Exception as e:
                self.logger.log_error(e, f"Error processing batch item {idx}")
                continue
                
        return batch_results

    def _save_checkpoint(self, results: List[Dict], batch_idx: int):
        output_dir = Path(self.config.output_dir)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"results_checkpoint_batch_{batch_idx}.json"
        write_json(results, str(checkpoint_path))
        self.logger.log_metric(f"checkpoint_saved", str(checkpoint_path))

def main():
    args = parse_arguments()
    config = AnswerGenerationConfig(args)
    
    logger = ExperimentLogger(
        experiment_name="answer_generation",
        base_log_dir="logs"
    )
    
    try:
        with logger:
            generator = AnswerGenerator(config, logger)
            results = generator.generate()
            return results
    except Exception as e:
        logger.log_error(e, "Error in main execution")
        raise

if __name__ == "__main__":
    seed_everything(SEED)
    main()