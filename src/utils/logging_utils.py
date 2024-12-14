import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
from tqdm import tqdm

class ExperimentLogger:
    def __init__(self, output_dir: Path, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.log_dir = output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.llm_logger = self._setup_llm_logger()
        self.eval_logger = self._setup_eval_logger()
        self.stats = {"llm_calls": 0, "cache_hits": 0, "evaluations": 0}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(self.log_dir / f"{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(ch)

        return logger

    def _setup_llm_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.experiment_name}_llm")
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(self.log_dir / f"llm_{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - LLM: %(message)s'))
        logger.addHandler(fh)
        
        return logger

    def _setup_eval_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.experiment_name}_eval")
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(self.log_dir / f"eval_{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - Evaluation: %(message)s'))
        logger.addHandler(fh)
        
        return logger

    def log_llm_call(self, prompt: str, response: str, metadata: Dict[str, Any]):
        self.stats["llm_calls"] += 1
        self.llm_logger.info(
            f"Call {self.stats['llm_calls']} - "
            f"Prompt: {prompt[:100]}... Response: {response[:100]}... "
            f"Metadata: {json.dumps(metadata)}"
        )

    def log_evaluation(self, query: str, generated_answer: str, gold_answer: str, score: float):
        self.stats["evaluations"] += 1
        self.eval_logger.info(
            f"Evaluation {self.stats['evaluations']} - "
            f"Query: {query} | Generated: {generated_answer} | "
            f"Gold: {gold_answer} | Score: {score:.2f}"
        )

    def log_cache_hit(self, cache_key: str):
        self.stats["cache_hits"] += 1
        self.llm_logger.debug(f"Cache hit for key: {cache_key}")

    def log_progress(self, current: int, total: int, desc: str = "", llm_stats: Optional[Dict] = None):
        progress_msg = f"{desc} - Progress: {(current/total)*100:.1f}% ({current}/{total})"
        if llm_stats:
            progress_msg += f" | LLM calls: {llm_stats.get('calls', 0)} | Cache hits: {llm_stats.get('hits', 0)}"
        self.logger.info(progress_msg)

    def log_error(self, error: Exception, context: str = "", query: Optional[str] = None):
        error_msg = f"{context}: {str(error)}"
        if query:
            error_msg = f"Query: {query} | {error_msg}"
        self.logger.error(error_msg, exc_info=True)
        self.llm_logger.error(error_msg)

    def log_experiment_start(self, config: dict):
        self.logger.info("Experiment started")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    def log_experiment_end(self, metrics: dict):
        final_stats = {
            **metrics,
            "llm_stats": {
                "total_calls": self.stats["llm_calls"],
                "cache_hits": self.stats["cache_hits"],
                "total_evaluations": self.stats["evaluations"]
            }
        }
        self.logger.info("Experiment completed")
        self.logger.info(f"Final metrics: {json.dumps(final_stats, indent=2)}")
        
        results_file = self.output_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_file, 'w') as f:
            json.dump(final_stats, f, indent=2)