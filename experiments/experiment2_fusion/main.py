import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional  
from datetime import datetime
import torch
from tqdm import tqdm
import logging
import gc
import json

from src.utils.file_utils import seed_everything, clear_memory
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.llm_evaluator import LLMEvaluator  
from src.batch_processor import BatchProcessor
from src.memory_manager import MemoryManager
from src.generation_cache_manager import GenerationCacheManager
from .config import FusionConfig

class FusionExperiment:
    def __init__(
        self,
        config: FusionConfig, 
        llm_evaluator: LLMEvaluator,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.llm_evaluator = llm_evaluator
        self.logger = logger or ExperimentLogger(
            experiment_name="fusion_experiment",
            base_log_dir=str(Path(__file__).parent.parent.parent / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def _prepare_evaluation_prompt(self, fused_docs: List[Dict], query: str) -> str:
        prompt_parts = [f"Query: {query}\n"]
        prompt_parts.append("Fused Documents:")
        
        for i, doc in enumerate(fused_docs, start=1):
            prompt_parts.append(f"Document {i}:")
            prompt_parts.append(doc['text'])
            prompt_parts.append(f"Retriever Score: {doc['retriever_score']:.3f}")
            prompt_parts.append(f"Fusion Score: {doc['fusion_score']:.3f}")  
            prompt_parts.append("\n")
        
        prompt_parts.append("Question: How well do the fused documents come together to provide a comprehensive and relevant response to the original query? Provide a score from 1 to 5 and a brief explanation.")
        return "\n".join(prompt_parts)

    def run_fusion_phase(self, fusion_results: Dict[str, List[Dict]]):
        self.logger.log_step_start("Fusion Phase")
        results = []
        
        try:
            for query_id, query_results in tqdm(fusion_results.items(), desc="Processing queries"):
                query = query_results['query']
                fused_docs = query_results['fused_docs']
                
                # Prepare evaluation prompt  
                eval_prompt = self._prepare_evaluation_prompt(fused_docs, query)

                # Check cache
                cached_result = self.cache_manager.get_response(eval_prompt)
                if cached_result:
                    self.logger.info(f"Cache hit for query {query_id}")
                    eval_result = cached_result
                else:
                    # Process with LLM  
                    eval_result = self.batch_processor.process_batches(
                        [{"prompt": eval_prompt}],
                        processing_fn=lambda x: x,
                        evaluate=True  
                    )[0]
                    self.cache_manager.save_response(eval_prompt, eval_result)
                
                query_result = {
                    "query_id": query_id, 
                    "query": query,
                    "fusion_docs": fused_docs,
                    "llm_evaluation": eval_result
                }
                results.append(query_result)
                
                clear_memory()

        except Exception as e:  
            self.logger.log_error(str(e), "Error in fusion phase")
            raise
        finally:
            self.logger.log_step_end("Fusion Phase")
        
        return results

    def run(self):
        try:
            # Load fusion results
            fusion_results = self._load_fusion_results()
            
            with self.memory_manager:  
                fusion_eval_results = self.run_fusion_phase(fusion_results)
                self._save_results(fusion_eval_results)
                
        except Exception as e:
            self.logger.log_error(str(e), "Error in fusion experiment")  
            raise
        finally:
            self.logger.log_step_end("Fusion Experiment")
        
        return fusion_eval_results

    def _load_fusion_results(self) -> Dict[str, List[Dict]]:  
        fusion_results = {}
        
        for file_path in self.config.fusion_results_files:
            with open(file_path) as f:
                file_results = json.load(f)
                fusion_results.update(file_results)

        self.logger.info(f"Loaded {len(fusion_results)} fusion results")         
        return fusion_results

    def _save_results(self, results: List[Dict]):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = self.config.output_dir / f"fusion_eval_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {output_file}")

def main():
    try:
        config = FusionConfig()
        llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
        
        experiment = FusionExperiment(config, llm_evaluator)
        results = experiment.run()  
        return results
        
    except Exception as e:
        logging.error(f"Error in fusion experiment: {str(e)}", exc_info=True)
        raise
    finally:
        clear_memory()

if __name__ == "__main__":  
    seed_everything(10)
    main()