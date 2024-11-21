import logging
import logging.config
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List
from datetime import datetime

import torch
import argparse
import warnings

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from rich.console import Console
from rich.progress import track
import yaml

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import modules from your project
from src.utils.llm import LLM
from src.utils import *
from src.prompt_dataset import QueryDataset
from src.core.env.env_loader import load_environment
from src.core.utils.system_check import perform_system_checks
from src.config import paths, settings

# Initialize Console for Rich
console = Console()

# Initialize Logging
def configure_logging():
    with open(paths.LOGGING_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

def seed_everything(seed: int):
    """Set seed for reproducibility."""
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@timing_decorator
def initialize_dataset_and_loader(
    args: argparse.Namespace, 
) -> DataLoader:
    """Initialize query-only dataset and dataloader."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing query-only dataset")
    try:
        prompt_ds = QueryDataset(
            data_path=info[args.split]['data_path'],
            model_name=args.llm_id, 
            do_normalize_query=True, 
        )
        prompt_dataloader = DataLoader(
            prompt_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        logger.info(f"Initialized dataloader with {len(prompt_ds)} examples")
        return prompt_dataloader
    except Exception as e:
        logger.error(f"Error initializing dataset: {e}")
        raise

def print_info(args: argparse.Namespace):
    """Print experiment configuration."""
    console = Console()
    console.print("\n[bold cyan]Only Query Generation Experiment Configuration:[/bold cyan]")
    console.print(f"Data: {info[args.split]['data_path']}")
    console.print(f"Use Test: {args.use_test}")
    console.print(f"Model: {args.llm_id}")
    console.print(f"Batch Size: {args.batch_size}")
    console.print(f"Save Every: {args.save_every}\n")

@timing_decorator
def generate_and_save(
    args: argparse.Namespace, 
    llm: LLM, 
    prompt_dataloader: DataLoader
):
    """Generate answers and save results."""
    logger = logging.getLogger(__name__)
    try:
        # Info from arguments
        llm_id = args.llm_id
        save_every = args.save_every

        # Create the saving directory
        llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
        saving_dir = Path(args.output_dir) / llm_folder / args.split / "only_query"
        saving_dir.mkdir(parents=True, exist_ok=True)

        # MPT has a different answer string in the prompt
        answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"

        all_info = []  
        for idx, prompt_batch in enumerate(track(prompt_dataloader, description="Generating answers")):
            batch_start_time = datetime.now()
            
            prompts = prompt_batch['prompt']
            generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
            
            generated_answers = []
            for output in generated_output:
                start = output.find(answer_string_in_prompt) + len(answer_string_in_prompt)
                response = output[start:].strip()
                generated_answers.append(response)

            prompt_batch['generated_answer'] = generated_answers
            all_info.append(prompt_batch)
            
            if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
                logger.info(f"Saving checkpoint at batch {idx + 1}")
                file_name = saving_dir / f"only_query_info_{idx+1}.pkl"
                write_pickle(all_info, file_name)
                all_info = []
    
    except Exception as e:
        logger.error(f"Error in generation: {e}")
        raise

def main():
    """Main execution for only query generation."""
    start_time = datetime.now()
    try:
        # Configure logging
        configure_logging()

        # Load environment variables
        load_environment()

        # Perform system checks
        perform_system_checks()

        # Parse arguments
        args = parse_arguments()
        
        # Initialize experiment tracking (optional)
        # If you want to track metrics similar to other experiments, implement a similar Experiment class

        # Create necessary directories
        Path("logs/generation").mkdir(parents=True, exist_ok=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize LLM
        console.print("\n[bold cyan]Initializing LLM...[/bold cyan]")
        llm = LLM(
            model_id=args.llm_id,
            device=device,
            quantization_bits=4, 
            model_max_length=args.model_max_length
        )
        tokenizer = llm.tokenizer
        logging.getLogger(__name__).info("LLM initialized successfully")

        # Initialize dataset
        console.print("\n[bold cyan]Initializing dataset...[/bold cyan]")
        prompt_dataloader = initialize_dataset_and_loader(args)

        # Print configuration
        print_info(args)

        # Generate answers
        console.print("\n[bold cyan]Generating answers...[/bold cyan]")
        generate_and_save(args, llm, prompt_dataloader)

        # Log completion
        duration = datetime.now() - start_time
        console.print(f"\n[bold green]✓ Only query generation completed in {duration}[/bold green]")

    except Exception as e:
        logging.getLogger(__name__).error(f"Error in main execution: {e}")
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise

if __name__ == "__main__":
    seed_everything(settings.SEED)
    main()
