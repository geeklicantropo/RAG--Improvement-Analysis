import sys
from pathlib import Path
import logging
from rich.console import Console
from rich.logging import RichHandler
from datasets import load_dataset
import requests
from tqdm.auto import tqdm
import json
import shutil
import os
import gdown
import pickle
from filelock import FileLock

class DatasetDownloader:
    """Downloads and organizes all required datasets and files."""
    
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        
        # Setup directories
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.mappings_dir = self.data_dir / "mappings"
        self.download_dir = self.data_dir / "downloaded_files"
        
        # Create directories
        for directory in [self.data_dir, self.processed_dir, self.mappings_dir, self.download_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # File organization map
        self.file_organization = {
            # Files that should go to processed directory
            self.processed_dir: {
                "test_corpus_with_random_contriever.json",
                "test_corpus_with_random_bm25.json",
                "reddit_corpus.pkl",
                "corpus_with_random_contriever.json",
                "corpus_with_random_at60.json",
                "corpus_with_random_50_words.pkl",
                "corpus_with_contriever_at150.json",
                "corpus_with_adore_at200.json"
            },
            
            # Files that should go to main data directory
            self.data_dir: {
                "contriever_search_results_at150.pkl",
                "bm25_test_search_results_at250.pkl",
                "10k_random_results_at60.pkl",
                "10k_other_random_results_at60.pkl",
                "nonsense_random_results.pkl"
            },
            
            # Files that should go to mappings directory
            self.mappings_dir: {
                "full_to_subset_adore_at200_in_corpus.pkl",
                "full_to_subset_contriever_at150_in_corpus.pkl",
                "full_to_subset_random_at60_in_corpus.pkl",
                "full_to_subset_random_contriever_in_corpus.pkl",
                "full_to_subset_test_random_bm25_in_corpus.pkl",
                "full_to_subset_test_random_contriever_in_corpus.pkl",
                "subset_to_full_adore_at200_in_corpus.pkl",
                "subset_to_full_contriever_at150_in_corpus.pkl",
                "subset_to_full_random_at60_in_corpus.pkl",
                "subset_to_full_random_contriever_in_corpus.pkl",
                "subset_to_full_test_random_bm25_in_corpus.pkl",
                "subset_to_full_test_random_contriever_in_corpus.pkl"
            }
        }

    def download_from_gdrive(self):
        """Download all files from Google Drive folder."""
        try:
            folder_id = "1MfR7mJ76tyVpjbMwUkMVbEjOQKpdL-Lq"  # Main folder ID
            
            self.console.print("\n[cyan]Downloading all files from Google Drive...[/cyan]")
            gdown.download_folder(
                id=folder_id,
                output=str(self.download_dir),
                quiet=False,
                use_cookies=False
            )
            
            return self.organize_downloaded_files()
            
        except Exception as e:
            self.logger.error(f"Error downloading from Google Drive: {e}")
            raise

    def organize_downloaded_files(self):
        """Organize downloaded files into their proper directories with progress tracking."""
        try:
            self.console.print("\n[cyan]Organizing downloaded files...[/cyan]")
            
            # Get list of all downloaded files
            all_downloaded_files = list(self.download_dir.glob("**/*"))
            
            # Create progress bar for file organization
            with tqdm(total=len(all_downloaded_files), desc="Organizing files") as pbar:
                for file_path in all_downloaded_files:
                    if file_path.is_file():  # Skip directories
                        filename = file_path.name
                        
                        # Find the correct destination directory
                        destination_dir = None
                        for target_dir, files in self.file_organization.items():
                            if filename in files:
                                destination_dir = target_dir
                                break
                        
                        if destination_dir:
                            # Create destination directory if it doesn't exist
                            destination_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Move file to its proper location
                            destination_path = destination_dir / filename
                            if not destination_path.exists():
                                shutil.move(str(file_path), str(destination_path))
                                self.console.print(f"[green]Moved {filename} to {destination_dir}[/green]")
                            else:
                                self.console.print(f"[yellow]Skipping {filename} - already exists in destination[/yellow]")
                    
                    pbar.update(1)
            
            # Clean up download directory
            if self.download_dir.exists():
                shutil.rmtree(self.download_dir)
                self.download_dir.mkdir(parents=True, exist_ok=True)
            
            return self.verify_file_organization()
            
        except Exception as e:
            self.logger.error(f"Error organizing files: {e}")
            raise

    def verify_file_organization(self):
        """Verify all required files are in their correct locations."""
        self.console.print("\n[cyan]Verifying file organization...[/cyan]")
        
        missing_files = []
        with tqdm(total=sum(len(files) for files in self.file_organization.values()),
                 desc="Verifying files") as pbar:
            for target_dir, files in self.file_organization.items():
                for filename in files:
                    file_path = target_dir / filename
                    if not file_path.exists():
                        missing_files.append(f"{filename} (should be in {target_dir})")
                    pbar.update(1)
        
        if missing_files:
            self.console.print("[red]Missing files:[/red]")
            for file in missing_files:
                self.console.print(f"[red]  - {file}[/red]")
            return False
        
        self.console.print("[green]All files are in their correct locations![/green]")
        return True

    def download_huggingface_dataset(self):
        """Download Natural Questions dataset from HuggingFace."""
        train_path = self.data_dir / "10k_train_dataset.json"
        test_path = self.data_dir / "test_dataset.json"
        
        if train_path.exists() and test_path.exists():
            self.console.print("[yellow]HuggingFace datasets already downloaded![/yellow]")
            return True
            
        try:
            self.console.print("\n[cyan]Downloading NQ dataset from HuggingFace...[/cyan]")
            dataset = load_dataset('florin-hf/nq_open_gold')
            
            # Save train split (10k samples)
            if not train_path.exists():
                train_data = dataset['train'].select(range(10000))
                self._save_split(train_data, train_path, "train")
            
            # Save test split
            if not test_path.exists():
                self._save_split(dataset['test'], test_path, "test")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading from HuggingFace: {e}")
            return False

    def _save_split(self, data, output_path: Path, split_name: str):
        """Save dataset split with progress tracking."""
        self.console.print(f"[cyan]Saving {split_name} split...[/cyan]")
        
        with FileLock(str(output_path) + ".lock"):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('[\n')
                
                for idx, item in enumerate(tqdm(data, desc=f"Saving {split_name} split")):
                    json_str = json.dumps(item, ensure_ascii=False)
                    if idx < len(data) - 1:
                        f.write(json_str + ',\n')
                    else:
                        f.write(json_str + '\n')
                        
                f.write(']\n')

    def download_all(self):
        """Download and organize all required files."""
        try:
            # Download NQ dataset from HuggingFace
            hf_success = self.download_huggingface_dataset()
            
            # Download all files from Google Drive and organize them
            gdrive_success = self.download_from_gdrive()
            
            if hf_success and gdrive_success:
                self.console.print("\n[bold green]✓ All files downloaded and organized successfully![/bold green]")
            else:
                self.console.print("\n[bold red]Warning: Some files could not be downloaded or organized![/bold red]")
            
        except Exception as e:
            self.logger.error(f"Error downloading files: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        # Initialize and run downloader
        downloader = DatasetDownloader()
        downloader.download_all()
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()