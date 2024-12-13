import os
import gc
import warnings
from typing import Union, List, Optional
from datetime import datetime
import logging
from tqdm import tqdm

import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria
)

class LLM:
    def __init__(
        self,
        model_id: str,
        device: str = 'cuda',
        quantization_bits: Optional[int] = None,
        stop_list: Optional[List[str]] = None,
        model_max_length: int = 4096,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        memory_threshold: float = 0.9,
        checkpoint_dir: Optional[str] = None
    ):
        self.device = device
        self.model_max_length = model_max_length
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.checkpoint_dir = checkpoint_dir
        
        # Setup logging
        self.logger = logging.getLogger("LLM")
        self.logger.setLevel(logging.INFO)
        
        self.stop_list = stop_list or ['\nHuman:', '\n```\n', '\nQuestion:', '<|endoftext|>', '\n']
        self.bnb_config = self._set_quantization(quantization_bits)
        
        self._initialize_model_tokenizer(model_id)
        self.stopping_criteria = self._define_stopping_criteria()

    def _set_quantization(self, quantization_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
        if quantization_bits in [4, 8]:
            bnb_config = BitsAndBytesConfig()
            if quantization_bits == 4:
                bnb_config.load_in_4bit = True
                bnb_config.bnb_4bit_quant_type = 'nf4'
                bnb_config.bnb_4bit_use_double_quant = True
                bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
            else:
                bnb_config.load_in_8bit = True
            return bnb_config
        return None

    def _initialize_model_tokenizer(self, model_id: str):
        try:
            self.logger.info(f"Initializing model: {model_id}")
            
            model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model_config.max_seq_len = self.model_max_length
            
            device_map = self._get_optimal_device_map()
            
            model_kwargs = {
                "trust_remote_code": True,
                "config": model_config,
                "torch_dtype": torch.float16,
                "device_map": device_map,
                "low_cpu_mem_usage": True
            }
            
            if self.bnb_config:
                model_kwargs["quantization_config"] = self.bnb_config

            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
                truncation_side="left",
                model_max_length=self.model_max_length
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _get_optimal_device_map(self) -> Union[str, dict]:
        if not torch.cuda.is_available():
            return "cpu"
            
        total_gpus = torch.cuda.device_count()
        if total_gpus == 0:
            return "cpu"

        available_memory = []
        for i in range(total_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            used_mem = torch.cuda.memory_allocated(i)
            available_mem = (total_mem - used_mem) / 1024**3  # Convert to GB
            available_memory.append(available_mem)

        if max(available_memory) < 2:  # Less than 2GB available
            return "cpu"
            
        return "auto"

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 15,
        batch_size: Optional[int] = None
    ) -> List[str]:
        try:
            if isinstance(prompts, str):
                prompts = [prompts]

            current_batch_size = batch_size or len(prompts)
            results = []
            
            for i in tqdm(range(0, len(prompts), current_batch_size), desc="Generating"):
                try:
                    batch_prompts = prompts[i:i + current_batch_size]
                    batch_results = self._generate_batch(batch_prompts, max_new_tokens)
                    results.extend(batch_results)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.warning("OOM error - reducing batch size")
                        self._cleanup_memory()
                        current_batch_size = max(self.min_batch_size, current_batch_size // 2)
                        
                        # Retry with smaller batch size
                        batch_results = self._generate_batch(batch_prompts, max_new_tokens)
                        results.extend(batch_results)
                        
                if self._should_checkpoint():
                    self._save_checkpoint(results, i + len(batch_prompts))
                
                self._cleanup_memory()
                
            return results

        except Exception as e:
            self.logger.error(f"Generation error: {str(e)}")
            raise
        finally:
            self._cleanup_memory()

    def _generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int
    ) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.model_max_length,
            return_tensors="pt"
        ).to(self.device)
        
        current_memory = self._get_memory_usage()
        if current_memory > self.memory_threshold:
            self._cleanup_memory()
            
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

    def _get_memory_usage(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0.0

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _should_checkpoint(self) -> bool:
        return (
            self.checkpoint_dir is not None and
            self._get_memory_usage() > self.memory_threshold * 0.8
        )

    def _save_checkpoint(self, results: List[str], current_idx: int):
        if not self.checkpoint_dir:
            return
            
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"generation_checkpoint_{current_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        
        torch.save({
            'results': results,
            'current_idx': current_idx
        }, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint at index {current_idx}")

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