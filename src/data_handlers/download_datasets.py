import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Now import paths after setting the correct path
from src.config import paths

import logging
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import torch

from src.config import paths
from src.core.utils.system_check import perform_system_checks

class DatasetDownloader:
    """Downloads and prepares datasets following the paper's methodology."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

    def download_all(self):
        """Download all required datasets."""
        try:
            # 1. Download corpus
            self.logger.info("Downloading corpus from HuggingFace...")
            corpus = load_dataset('florin-hf/wiki_dump2018_nq_open')
            corpus.save_to_disk(paths.CORPUS_DIR)
            self.logger.info(f"Corpus saved to {paths.CORPUS_DIR}")

            # 2. Download NQ dataset
            self.logger.info("Downloading NQ dataset from HuggingFace...")
            dataset = load_dataset('florin-hf/nq_open_gold')
            dataset.save_to_disk(paths.DATASETS_DIR)
            self.logger.info(f"NQ dataset saved to {paths.DATASETS_DIR}")

            # 3. Download precomputed search results if needed
            if not os.path.exists(paths.CONTRIEVER_RESULTS_PATH):
                self.logger.info("Note: Contriever search results need to be computed.")
            if not os.path.exists(paths.BM25_RESULTS_PATH):
                self.logger.info("Note: BM25 search results need to be computed.")

        except Exception as e:
            self.logger.error(f"Error downloading datasets: {e}")
            raise

    def verify_downloads(self) -> bool:
        """Verify all required data is available."""
        try:
            # Check corpus
            corpus = load_dataset('florin-hf/wiki_dump2018_nq_open', keep_in_memory=False)
            self.logger.info(f"Corpus size: {len(corpus['train'])} documents")

            # Check NQ dataset
            dataset = load_dataset('florin-hf/nq_open_gold', keep_in_memory=False)
            self.logger.info(f"Train examples: {len(dataset['train'])}")
            self.logger.info(f"Test examples: {len(dataset['test'])}")

            return True

        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return False

def main():
    """Main execution function."""
    try:
        # Perform system checks
        perform_system_checks()
        
        # Initialize and run downloader
        downloader = DatasetDownloader()
        downloader.download_all()
        
        # Verify downloads
        if downloader.verify_downloads():
            logging.info("All data downloaded and verified successfully")
        else:
            logging.error("Download verification failed")
            
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()