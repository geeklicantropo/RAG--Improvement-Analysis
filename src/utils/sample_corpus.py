import os
import json
import pickle
import random
from pathlib import Path
import logging
import sys

def sample_corpus(
    corpus_path: str = "data/processed/corpus_with_contriever_at150.json", 
    sample_size: int = 10
):
    """
    Create a small sample from a JSON or pickle file.
    
    Args:
        corpus_path (str): Path to the input file, defaulting to the Contriever corpus
        sample_size (int): Number of items to sample
    
    Returns:
        str: Path to the created sample file
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Validate file exists
    input_path = Path(corpus_path)
    if not input_path.exists():
        logger.error(f"File not found: {corpus_path}")
        return None

    # Determine file type
    file_ext = input_path.suffix.lower()

    try:
        # Read the full file
        if file_ext == '.json':
            with open(input_path, 'r') as f:
                data = json.load(f)
        elif file_ext == '.pkl':
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
        else:
            logger.error(f"Unsupported file type: {file_ext}")
            return None

        # Validate data and sample size
        if not isinstance(data, (list, dict)):
            logger.error("File content must be a list or dictionary")
            return None

        # Ensure sample size doesn't exceed total data
        if isinstance(data, list):
            sample_size = min(sample_size, len(data))
            sampled_data = random.sample(data, sample_size)
        else:
            # For dictionaries, take first few items
            sample_size = min(sample_size, len(data))
            sampled_data = dict(list(data.items())[:sample_size])

        # Create output path (always .json, indicating original source)
        output_path = input_path.parent / f"small_{input_path.stem}_from_{input_path.suffix[1:]}.json"

        # Always save as JSON
        with open(output_path, 'w') as f:
            json.dump(sampled_data, f, indent=2)

        logger.info(f"Created sample file: {output_path}")
        logger.info(f"Sample size: {sample_size}")
        logger.info(f"Total original size: {len(data)}")
        logger.info(f"Original file type: {file_ext[1:]}")

        return str(output_path)

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return None

def main():
    # Default usage
    if len(sys.argv) == 1:
        print("Usage:")
        print("python sample_corpus.py <file_path> [sample_size]")
        print("\nExample:")
        print("python sample_corpus.py data/processed/corpus_with_contriever_at150.json 10")
        print("\nOr to use default path and sample size:")
        print("python sample_corpus.py")
        sys.exit(1)

    # User-specified path and sample size
    if len(sys.argv) == 2:
        corpus_path = sys.argv[1]
        sample_corpus(corpus_path)
    elif len(sys.argv) == 3:
        corpus_path = sys.argv[1]
        sample_size = int(sys.argv[2])
        sample_corpus(corpus_path, sample_size)
    else:
        print("Too many arguments. Usage: python sample_corpus.py <file_path> [sample_size]")
        sys.exit(1)

if __name__ == "__main__":
    main()

# To use, run something like this python3 src/utils/sample_corpus.py data/processed/reddit_corpus.pkl 30, with the desired file.