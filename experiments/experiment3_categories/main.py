import os
import sys
from pathlib import Path  
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from tqdm import tqdm
import logging
import gc
import json

from src.utils.file_utils import seed_everything, clear_memory 
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.document_classifier import DocumentClassifier
from src.llm_evaluator import LLMEvaluator
from src.batch_processor import BatchProcessor
from src.memory_manager import MemoryManager  
from src.generation_cache_manager import GenerationCacheManager
from src.document_combination_validator import DocumentCombinationValidator
from src.utils.corpus_manager import CorpusManager
from .config import CategoriesConfig

class CategoriesExperiment:
    def __init__(
        self,
        config: CategoriesConfig,
        corpus_manager: CorpusManager,
        llm_evaluator: LLMEvaluator,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.corpus_manager = corpus_manager
        self.llm_evaluator = llm_evaluator
        self.logger = logger or ExperimentLogger(
            experiment_name="categories_experiment",
            base_log_dir=str(Path(__file__).parent.parent.parent / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.doc_classifier = DocumentClassifier(self.llm_evaluator)  
        self.batch_processor = BatchProcessor(
            api_key=os.getenv("GEMINI_TOKEN"),
            initial_batch_size=config.batch_size,
            min_batch_size=config.min_batch_size,
            max_batch_size=config.max_batch_size,
            gpu_memory_threshold=config.gpu_memory_threshold,
            cpu_memory_threshold=config.cpu_memory_threshold
        )
        self.memory_manager = MemoryManager(
            gpu_memory_threshold=config.gpu_memory_threshold,
            cpu_memory_threshold=config.cpu_memory_threshold,  
            min_batch_size=config.min_batch_size,
            max_batch_size=config.max_batch_size
        )
        self.cache_manager = GenerationCacheManager(
            api_key=os.getenv("GEMINI_TOKEN"),
            cache_dir=config.cache_dir
        )
        self.doc_validator = DocumentCombinationValidator(
            api_key=os.getenv("GEMINI_TOKEN"),
            logger=self.logger
        )

    def _prepare_evaluation_prompt(self, documents: List[Dict]) -> str:
        # Prepare a prompt for LLM evaluation from categorized documents  
        prompt_parts = ["Categorized Documents:"]
        for category, cat_docs in documents.items():
            prompt_parts.append(f"Category: {category}")
            for i, doc in enumerate(cat_docs, start=1):
                prompt_parts.append(f"Document {i}:")
                prompt_parts.append(doc['text'])
                prompt_parts.append("\n")

        prompt_parts.append("Question: How well do the document categories represent distinct and meaningful groups relevant to the original query? Provide a score from 1 to 5 and a brief explanation.")
        return "\n".join(prompt_parts)

    def run_categories_phase(self, documents: List[Dict], gold_docs: List[Dict]):
        self.logger.log_step_start("Categories Phase")

        try:  
            # Assign categories
            categorized_docs = self.doc_classifier.classify_documents(documents)

            # Validate category assignments
            validation_results = self.doc_validator.validate_combinations(
                documents, gold_docs, self.config
            )
            
            # Evaluate with LLM  
            eval_prompt = self._prepare_evaluation_prompt(categorized_docs)
            eval_result = self.batch_processor.process_batches(
                [{"prompt": eval_prompt}],
                processing_fn=lambda x: x, 
                evaluate=True
            )[0]

            category_metrics = {
                "validation": validation_results,
                "score": eval_result['evaluation']['score'],
                "explanation": eval_result['evaluation']['explanation']
            }

            # Save metrics 
            self.logger.log_metric("category_metrics", category_metrics)

        except Exception as e:
            self.logger.log_error(str(e), "Error in categories phase")
            raise
        finally:
            self.logger.log_step_end("Categories Phase")

        return category_metrics   

    def run(self):
        try:
            # Load documents
            documents = self.corpus_manager.get_corpus()
            gold_docs = self.corpus_manager.get_gold_documents()

            with self.memory_manager:
                # Run categorization and evaluation
                category_metrics = self.run_categories_phase(documents, gold_docs)
                self._save_results(category_metrics)

        except Exception as e:
            self.logger.log_error(str(e), "Error in categories experiment")
            raise
        finally:
            self.logger.log_step_end("Categories Experiment")

        return category_metrics

    def _save_results(self, category_metrics: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = self.config.output_dir / f"category_metrics_{timestamp}.json"
        
        with open(output_file, 'w') as f:  
            json.dump(category_metrics, f, indent=2)
        self.logger.info(f"Category metrics saved to {output_file}")  

def main():
    try:  
        config = CategoriesConfig()
        corpus_manager = CorpusManager(config.corpus_path)
        llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
        
        experiment = CategoriesExperiment(config, corpus_manager, llm_evaluator)
        metrics = experiment.run()
        return metrics

    except Exception as e:  
        logging.error(f"Error in categories experiment: {str(e)}", exc_info=True)
        raise
    finally:
        clear_memory()

if __name__ == "__main__":  
    seed_everything(10)
    main()