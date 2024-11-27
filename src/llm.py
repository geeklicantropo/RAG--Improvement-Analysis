import torch
import warnings
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria
)
from typing import List, Tuple, Optional

class LLM:
    """
    A class for loading and generating text using a Language Model (LM) with support for quantization
    and custom stopping criteria.
    
    Attributes:
        model_id (str): Identifier for the model to load.
        device (str): Device to run the model on, e.g. 'cuda'.
        quantization_bits (Optional[int]): Number of bits for quantization, supports 4 or 8 bits.
        stop_list (Optional[List[str]]): List of tokens where generation should stop.
        model_max_length (int): Maximum length of the model inputs.
    """
    def __init__(
        self, 
        model_id: str, 
        device: str = 'cuda', 
        quantization_bits: Optional[int] = None, 
        stop_list: Optional[List[str]] = None, 
        model_max_length: int = 4096
    ):
        self.device = device
        self.model_max_length = model_max_length

        self.stop_list = stop_list
        if stop_list is None:
            self.stop_list = ['\nHuman:', '\n```\n', '\nQuestion:', '<|endoftext|>', '\n']
        
        self.bnb_config = self._set_quantization(quantization_bits)
        self.model, self.tokenizer = self._initialize_model_tokenizer(model_id)
        self.stopping_criteria = self._define_stopping_criteria()
        

    def _set_quantization(self, quantization_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
        """
        Configure quantization settings based on the specified number of bits.
        """
        if quantization_bits in [4, 8]:
            bnb_config = BitsAndBytesConfig()
            if quantization_bits == 4:
                bnb_config.load_in_4bit = True
                bnb_config.bnb_4bit_quant_type = 'nf4'
                bnb_config.bnb_4bit_use_double_quant = True
                bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
            elif quantization_bits == 8:
                bnb_config.load_in_8bit = True
            return bnb_config
        return None
 

    def _initialize_model_tokenizer(self, model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_config.max_seq_len = self.model_max_length

        # Split model across available GPU memory and CPU
        device_map = self._get_optimal_device_map()
        
        model_kwargs = {
            "trust_remote_code": True,
            "config": model_config,
            "torch_dtype": torch.float16,  # Use float16 instead of bfloat16 for better memory efficiency
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        }

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

    def _get_optimal_device_map(self) -> dict:
        """Calculate optimal device map based on available GPU memory."""
        if not torch.cuda.is_available():
            return "cpu"
            
        total_gpus = torch.cuda.device_count()
        if total_gpus == 0:
            return "cpu"

        # Get available memory on each GPU
        available_memory = []
        for i in range(total_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            used_mem = torch.cuda.memory_allocated(i)
            available_mem = (total_mem - used_mem) / 1024**3  # Convert to GB
            available_memory.append(available_mem)

        # If very limited GPU memory, use CPU
        if max(available_memory) < 2:  # Less than 2GB available
            return "cpu"
            
        # Create balanced device map
        device_map = "balanced"
        return device_map


    def _define_stopping_criteria(self) -> StoppingCriteriaList:
        """
        Defines stopping criteria for text generation based on the provided stop_list.
        """
        stop_token_ids = [self.tokenizer(x)['input_ids'] for x in self.stop_list]
        stop_token_ids = [torch.LongTensor(x).to(self.device) for x in stop_token_ids]

        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_ids in stop_token_ids:
                    if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                        return True
                return False

        return StoppingCriteriaList([StopOnTokens()])
    
    
    def generate(self, prompt: str, max_new_tokens: int = 15) -> List[str]:
        """
        Generates text based on the given prompt.
        
        Args:
            prompt (str): Input text prompt for generation.
        
        Returns:
            List[str]: The generated text responses.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            inputs = self.tokenizer(
                prompt, 
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

