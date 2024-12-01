import os
import torch
import warnings
import gc
import logging
from tqdm import tqdm
from typing import Union, List, Dict, Tuple, Optional
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria
)
from experiment_logger import ExperimentLogger


class LLM:
    """
    Language Model wrapper for generating text using transformers with memory optimization.
    """

    def __init__(
        self,
        model_id: str,
        device: str = 'cuda',
        quantization_bits: Optional[int] = None,
        stop_list: Optional[List[str]] = None,
        model_max_length: int = 4096,
        logger: Optional[ExperimentLogger] = None
    ):
        self.device = device
        self.model_max_length = model_max_length
        self.logger = logger

        self.stop_list = stop_list or ['\nHuman:', '\n```\n', '\nQuestion:', '<|endoftext|>', '\n']
        self.bnb_config = self._set_quantization(quantization_bits)
        self.model, self.tokenizer = self._initialize_model_tokenizer(model_id)
        self.stopping_criteria = self._define_stopping_criteria()

        if logger:
            logger.log_step_start(f"Initialized LLM with model {model_id}")

    def _set_quantization(self, quantization_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
        if quantization_bits in [4, 8]:
            config = BitsAndBytesConfig()
            if quantization_bits == 4:
                config.load_in_4bit = True
                config.bnb_4bit_quant_type = 'nf4'
                config.bnb_4bit_use_double_quant = True
                config.bnb_4bit_compute_dtype = torch.bfloat16
            elif quantization_bits == 8:
                config.load_in_8bit = True
            return config
        return None

    def _initialize_model_tokenizer(self, model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_config.max_seq_len = self.model_max_length

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        model_kwargs = {
            "trust_remote_code": True,
            "config": model_config,
            "torch_dtype": torch.float16,
            "device_map": device_map,
            "low_cpu_mem_usage": True
        }

        if self.bnb_config:
            model_kwargs["quantization_config"] = self.bnb_config

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            truncation_side="left",
            model_max_length=self.model_max_length
        )
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> Dict[str, float]:
        """Retrieve current GPU memory usage stats."""
        memory_stats = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                memory_stats[f'gpu_{i}'] = {
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'available_gb': reserved - allocated
                }
        return memory_stats

    def optimize_batch_size(self, initial_batch_size: int) -> int:
        """Dynamically optimize batch size based on available memory."""
        if not torch.cuda.is_available():
            return initial_batch_size

        memory_stats = self.get_memory_usage()
        min_available_gb = min(stat['available_gb'] for stat in memory_stats.values())

        if min_available_gb < 2:  # Low memory, reduce batch size
            return max(1, initial_batch_size // 2)
        elif min_available_gb > 8:  # High memory, increase batch size
            return min(initial_batch_size * 2, 64)

        return initial_batch_size

    def generate(self, prompts: Union[str, List[str]], max_new_tokens: int = 15) -> List[str]:
        """
        Generates text based on the given prompt(s).
        """
        try:
            if isinstance(prompts, str):
                prompts = [prompts]

            results = []
            batch_size = len(prompts)
            optimal_batch_size = self.optimize_batch_size(batch_size)

            for i in tqdm(range(0, batch_size, optimal_batch_size), desc="Generating text", unit="batch"):
                batch_prompts = prompts[i:i + optimal_batch_size]
                results.extend(self._generate_batch(batch_prompts, max_new_tokens))
                self.cleanup_memory()

            return results
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Error during generation")
            raise
        finally:
            self.cleanup_memory()

    def _generate_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.model_max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.1,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def _define_stopping_criteria(self) -> StoppingCriteriaList:
        if not self.stop_list:
            return StoppingCriteriaList()

        class StopOnTokens(StoppingCriteria):
            def __init__(self, tokenizer, stop_list):
                super().__init__()
                self.tokenizer = tokenizer
                self.stop_token_ids = [
                    tokenizer(stop_word, add_special_tokens=False)['input_ids']
                    for stop_word in stop_list
                ]

            def __call__(self, input_ids, scores, **kwargs):
                for stop_ids in self.stop_token_ids:
                    if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                        return True
                return False

        return StoppingCriteriaList([StopOnTokens(self.tokenizer, self.stop_list)])

    def log_generation_details(self, prompts: List[str], responses: List[str]):
        if self.logger:
            for prompt, response in zip(prompts, responses):
                self.logger.log_metric(
                    "generation_detail",
                    {"prompt": prompt, "response": response}
                )
