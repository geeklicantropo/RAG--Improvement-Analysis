import logging
import torch
import os
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
from threading import Thread
import time
import json
from queue import Queue
import numpy as np

@dataclass
class GenerationResult:
    """Stores generation results with enhanced metadata and document tracking."""
    text: str
    tokens_used: int
    generation_time: float
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    doc_ids: List[str]  # Track which documents were used
    metadata: Optional[Dict] = None
    attention_weights: Optional[np.ndarray] = None

class LLM:
    """Enhanced LLM class with document tracking and attention analysis."""
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        quantization_bits: Optional[int] = None,
        model_max_length: Optional[int] = None,
        experiment_name: Optional[str] = None,
        attention_tracking: bool = False
    ):
        # Set up basic attributes
        self.model_id = model_id or "meta-llama/Llama-2-7b-chat-hf"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization_bits = quantization_bits
        self.model_max_length = model_max_length or 4096
        self.experiment_name = experiment_name or f"llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.attention_tracking = attention_tracking
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.experiment_name}")
        self.setup_logging()
        
        # Initialize LLM
        self.initialize_model()
        
        # Setup attention tracking if enabled
        if self.attention_tracking:
            self.attention_weights = {}
        
        self.logger.info(
            f"Initialized LLM with model: {self.model_id}, "
            f"device: {self.device}, "
            f"quantization: {self.quantization_bits}, "
            f"attention_tracking: {attention_tracking}"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs/llm")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        fh = logging.FileHandler(
            log_dir / f"{self.experiment_name}.log"
        )
        fh.setLevel(logging.DEBUG)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def initialize_model(self):
        """Initialize the language model and tokenizer."""
        try:
            self.logger.info(f"Initializing model: {self.model_id}")
            
            if "gpt" in self.model_id.lower():
                self._init_openai()
            else:
                self._init_huggingface()
                
            self.logger.info("Model initialization complete")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def _init_openai(self):
        """Initialize OpenAI API client."""
        try:
            from openai import OpenAI
            
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self.client = OpenAI(api_key=self.api_key)
            self.is_openai = True
            self.logger.info("OpenAI API client initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {e}")
            raise

    def _init_huggingface(self):
        """Initialize Hugging Face model with enhanced configuration."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization
            if self.quantization_bits:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=(self.quantization_bits == 4),
                    load_in_8bit=(self.quantization_bits == 8),
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
            
            # Load model with attention tracking if enabled
            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "output_attentions": self.attention_tracking
            }
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            self.is_openai = False
            self.logger.info(
                f"Hugging Face model loaded with quantization_bits={self.quantization_bits}, "
                f"attention_tracking={self.attention_tracking}"
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing Hugging Face model: {e}")
            raise

    def generate(
        self, 
        prompts: Union[str, List[str]], 
        doc_ids: Optional[Union[List[str], List[List[str]]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        num_beams: int = 1,
        stream: bool = False
    ) -> Union[List[GenerationResult], GenerationResult]:
        """Enhanced generation with document tracking and attention analysis."""
        single_prompt = isinstance(prompts, str)
        if single_prompt:
            prompts = [prompts]
            doc_ids = [doc_ids] if doc_ids else None
        
        self.logger.info(f"Generating responses for {len(prompts)} prompts")
        
        try:
            if self.is_openai:
                results = self._generate_openai(
                    prompts, 
                    doc_ids,
                    max_new_tokens,
                    temperature,
                    stream
                )
            else:
                results = self._generate_huggingface(
                    prompts,
                    doc_ids,
                    max_new_tokens,
                    temperature,
                    num_beams,
                    stream
                )
            
            return results[0] if single_prompt else results
            
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise

    def _generate_openai(
        self,
        prompts: List[str],
        doc_ids: Optional[List[List[str]]],
        max_tokens: Optional[int],
        temperature: float,
        stream: bool
    ) -> List[GenerationResult]:
        """Generate responses using OpenAI API with enhanced tracking."""
        results = []
        
        try:
            for idx, prompt in enumerate(tqdm(prompts, desc="Generating responses")):
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream
                )
                
                if stream:
                    collected_messages = []
                    for chunk in response:
                        chunk_message = chunk.choices[0].delta.content or ""
                        collected_messages.append(chunk_message)
                        if chunk_message:
                            print(chunk_message, end="", flush=True)
                    full_response_text = "".join(collected_messages)
                else:
                    full_response_text = response.choices[0].message.content
                
                generation_time = time.time() - start_time
                
                results.append(GenerationResult(
                    text=full_response_text,
                    tokens_used=response.usage.total_tokens,
                    generation_time=generation_time,
                    model_name=self.model_id,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    doc_ids=doc_ids[idx] if doc_ids else [],
                    metadata={
                        'model': self.model_id,
                        'temperature': temperature,
                        'max_tokens': max_tokens
                    }
                ))
                
                self.logger.info(
                    f"Generated response {idx + 1}/{len(prompts)} in {generation_time:.2f}s, "
                    f"tokens used: {response.usage.total_tokens}"
                )
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in OpenAI generation: {e}")
            raise

    def _generate_huggingface(
        self,
        prompts: List[str],
        doc_ids: Optional[List[List[str]]],
        max_new_tokens: Optional[int],
        temperature: float,
        num_beams: int,
        stream: bool
    ) -> List[GenerationResult]:
        """Generate responses using Hugging Face model with attention tracking."""
        results = []
        max_new_tokens = max_new_tokens or self.model_max_length
        
        try:
            for idx, prompt in enumerate(tqdm(prompts, desc="Generating responses")):
                start_time = time.time()
                
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_max_length
                ).to(self.device)
                
                prompt_tokens = len(inputs["input_ids"][0])
                
                # Generate with attention tracking if enabled
                if stream:
                    streamer = TextIteratorStreamer(self.tokenizer)
                    generation_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "num_beams": num_beams,
                        "streamer": streamer,
                        "output_attentions": self.attention_tracking
                    }
                    
                    thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    generated_text = ""
                    for new_text in streamer:
                        generated_text += new_text
                        print(new_text, end="", flush=True)
                        
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        num_beams=num_beams,
                        pad_token_id=self.tokenizer.pad_token_id,
                        output_attentions=self.attention_tracking,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    generated_text = self.tokenizer.decode(
                        outputs.sequences[0],
                        skip_special_tokens=True
                    )
                    
                    # Process attention weights if available
                    attention_weights = None
                    if self.attention_tracking and hasattr(outputs, 'attentions'):
                        attention_weights = self._process_attention_weights(outputs.attentions)
                
                generation_time = time.time() - start_time
                completion_tokens = len(self.tokenizer.encode(generated_text)) - prompt_tokens
                
                results.append(GenerationResult(
                    text=generated_text,
                    tokens_used=prompt_tokens + completion_tokens,
                    generation_time=generation_time,
                    model_name=self.model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    doc_ids=doc_ids[idx] if doc_ids else [],
                    attention_weights=attention_weights,
                    metadata={
                        'model': self.model_id,
                        'temperature': temperature,
                        'max_new_tokens': max_new_tokens,
                        'num_beams': num_beams
                    }
                ))
                
                self.logger.info(
                    f"Generated response {idx + 1}/{len(prompts)} in {generation_time:.2f}s, "
                    f"tokens used: {prompt_tokens + completion_tokens}"
                )
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Hugging Face generation: {e}")
            raise

    def _process_attention_weights(self, attention_outputs: Tuple) -> np.ndarray:
        """Process and aggregate attention weights."""
        try:
            # Convert attention outputs to numpy arrays
            attention_weights = []
            for layer_attention in attention_outputs:
                # Take mean across heads
                layer_weights = layer_attention.mean(dim=1).cpu().numpy()
                attention_weights.append(layer_weights)
            
            # Average across layers
            return np.mean(attention_weights, axis=0)
            
        except Exception as e:
            self.logger.error(f"Error processing attention weights: {e}")
            return None

    def save_generation_results(
        self,
        results: List[GenerationResult],
        output_path: str,
        save_attention_weights: bool = True
    ):
        """Save generation results with attention weights if available."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare results for saving
            results_data = []
            for result in results:
                result_dict = {
                    "text": result.text,
                    "tokens_used": result.tokens_used,
                    "generation_time": result.generation_time,
                    "model_name": result.model_name,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "doc_ids": result.doc_ids,
                    "metadata": result.metadata
                }
                
                # Save attention weights separately if present
                if save_attention_weights and result.attention_weights is not None:
                        attention_path = output_path.parent / "attention_weights"
                        attention_path.mkdir(exist_ok=True)
                        attention_file = attention_path / f"attention_{len(results_data)}.npy"
                        np.save(attention_file, result.attention_weights)
                        result_dict["attention_file"] = str(attention_file)
                    
                results_data.append(result_dict)
            
            # Save main results
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            self.logger.info(f"Saved generation results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving generation results: {e}")
            raise

    def load_generation_results(
        self,
        input_path: str,
        load_attention_weights: bool = False
    ) -> List[GenerationResult]:
        """Load generation results with optional attention weights."""
        try:
            with open(input_path, 'r') as f:
                results_data = json.load(f)
                
            results = []
            for data in results_data:
                # Load attention weights if requested
                attention_weights = None
                if load_attention_weights and "attention_file" in data:
                    try:
                        attention_weights = np.load(data["attention_file"])
                    except Exception as e:
                        self.logger.warning(f"Could not load attention weights: {e}")
                
                result = GenerationResult(
                    text=data["text"],
                    tokens_used=data["tokens_used"],
                    generation_time=data["generation_time"],
                    model_name=data["model_name"],
                    prompt_tokens=data["prompt_tokens"],
                    completion_tokens=data["completion_tokens"],
                    doc_ids=data["doc_ids"],
                    metadata=data.get("metadata", {}),
                    attention_weights=attention_weights
                )
                results.append(result)
            
            self.logger.info(f"Loaded {len(results)} generation results from {input_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading generation results: {e}")
            raise

    def analyze_attention_patterns(
        self,
        result: GenerationResult,
        doc_lengths: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Analyze attention patterns for document usage."""
        if not result.attention_weights is not None:
            return {}
            
        try:
            attention_weights = result.attention_weights
            
            # If doc_lengths provided, analyze per-document attention
            if doc_lengths:
                doc_attentions = []
                current_pos = 0
                
                for length in doc_lengths:
                    doc_slice = slice(current_pos, current_pos + length)
                    doc_attention = attention_weights[:, doc_slice].mean()
                    doc_attentions.append(float(doc_attention))
                    current_pos += length
                    
                # Normalize attention scores
                doc_attentions = np.array(doc_attentions)
                doc_attentions = doc_attentions / doc_attentions.sum()
            else:
                doc_attentions = None
            
            # Calculate overall attention statistics
            analysis = {
                'mean_attention': float(attention_weights.mean()),
                'max_attention': float(attention_weights.max()),
                'attention_std': float(attention_weights.std()),
                'doc_attentions': doc_attentions
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing attention patterns: {e}")
            return {}

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss / (1024 * 1024),  # MB
                'vms': memory_info.vms / (1024 * 1024),  # MB
                'percent': process.memory_percent()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {}

    def clear_cache(self):
        """Clear model cache and CUDA memory."""
        try:
            if hasattr(self, 'model'):
                self.model.cpu()
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")