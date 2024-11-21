# src/data_handlers/analysis/analyze_arrow_datasets.py

import sys
from pathlib import Path
import logging
import pickle
from typing import Dict, Any, Tuple
from datetime import datetime
from rich import print
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from rich.console import Console
import numpy as np

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from datasets import load_from_disk
from src.config import paths

class DatasetAnalyzer:
    """Analyzes Arrow dataset structures and their relationships."""
    
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[
                RichHandler(rich_tracebacks=True),
                logging.FileHandler(
                    paths.LOGS_DIR / f"dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.console = Console()

    def analyze_corpus_structure(self) -> Dict[str, Any]:
        """Analyze corpus Arrow files structure."""
        self.logger.info("Analyzing corpus structure...")
        
        try:
            corpus = load_from_disk(str(paths.CORPUS_DIR))
            
            # Analyze sharding
            arrow_files = list(Path(paths.CORPUS_DIR / 'train').glob('*.arrow'))
            shard_sizes = [Path(f).stat().st_size / (1024 * 1024) for f in arrow_files]
            
            corpus_info = {
                'num_shards': len(arrow_files),
                'features': corpus['train'].features,
                'total_documents': len(corpus['train']),
                'total_size_mb': sum(shard_sizes),
                'avg_shard_size_mb': np.mean(shard_sizes),
                'sample_document': corpus['train'][0]
            }
            
            # Display corpus information
            self._display_corpus_info(corpus_info)
            
            return corpus_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing corpus: {e}")
            raise

    def analyze_nq_dataset(self) -> Dict[str, Any]:
        """Analyze NQ dataset structure."""
        self.logger.info("\nAnalyzing NQ dataset structure...")
        
        try:
            dataset = load_from_disk(str(paths.DATASETS_DIR))
            
            dataset_info = {
                'splits': list(dataset.keys()),
                'train_size': len(dataset['train']),
                'test_size': len(dataset['test']),
                'features': dataset['train'].features,
                'sample_train': dataset['train'][0],
                'sample_test': dataset['test'][0]
            }
            
            # Display dataset information
            self._display_dataset_info(dataset_info)
            
            return dataset_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing NQ dataset: {e}")
            raise

    def analyze_search_results(self) -> Dict[str, Any]:
        """Analyze search results format and structure."""
        self.logger.info("\nAnalyzing search results...")
        
        try:
            results_info = {}
            
            # Load and analyze each type of search results
            result_files = {
                'contriever': paths.CONTRIEVER_RESULTS_PATH,
                'bm25': paths.BM25_RESULTS_PATH,
                'adore': paths.ADORE_RESULTS_PATH
            }
            
            for name, path in result_files.items():
                if path.exists():
                    with open(path, 'rb') as f:
                        results = pickle.load(f)
                        results_info[name] = {
                            'length': len(results),
                            'sample': results[0],
                            'doc_id_type': type(results[0][0][0]).__name__,
                            'score_type': type(results[0][1][0]).__name__
                        }
            
            # Display search results information
            self._display_search_results_info(results_info)
            
            return results_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing search results: {e}")
            raise

    def analyze_id_mapping(self) -> Dict[str, Any]:
        """Analyze document ID relationships between datasets."""
        self.logger.info("\nAnalyzing document ID mapping...")
        
        try:
            # Load small samples from each dataset
            corpus = load_from_disk(str(paths.CORPUS_DIR))
            dataset = load_from_disk(str(paths.DATASETS_DIR))
            
            with open(paths.CONTRIEVER_RESULTS_PATH, 'rb') as f:
                contriever_results = pickle.load(f)
            
            # Analyze ID formats and relationships
            mapping_info = {
                # For corpus, we'll use index as ID since there's no explicit ID field
                'corpus_structure': corpus['train'].features,
                'dataset_id_format': type(dataset['train'][0]['example_id']).__name__,
                'gold_corpus_idx_format': type(dataset['train'][0]['idx_gold_in_corpus']).__name__,
                'search_result_id_format': type(contriever_results[0][0][0]).__name__,
                'id_samples': {
                    'dataset_example_ids': [dataset['train'][i]['example_id'] for i in range(5)],
                    'dataset_gold_corpus_idx': [dataset['train'][i]['idx_gold_in_corpus'] for i in range(5)],
                    'search_results': [contriever_results[0][0][i] for i in range(5)]
                }
            }
            
            # Display ID mapping information
            self._display_id_mapping_info(mapping_info)
            
            return mapping_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing ID mapping: {e}")
            raise

    def _display_corpus_info(self, info: Dict[str, Any]):
        """Display corpus information in a formatted table."""
        table = Table(title="Corpus Information")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Number of Shards", str(info['num_shards']))
        table.add_row("Total Documents", str(info['total_documents']))
        table.add_row("Total Size (MB)", f"{info['total_size_mb']:.2f}")
        table.add_row("Avg Shard Size (MB)", f"{info['avg_shard_size_mb']:.2f}")
        table.add_row("Features", str(info['features']))
        
        self.console.print(table)

    def _display_dataset_info(self, info: Dict[str, Any]):
        """Display dataset information in a formatted table."""
        table = Table(title="NQ Dataset Information")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Available Splits", ", ".join(info['splits']))
        table.add_row("Train Examples", str(info['train_size']))
        table.add_row("Test Examples", str(info['test_size']))
        table.add_row("Features", str(info['features']))
        
        self.console.print(table)

    def _display_search_results_info(self, info: Dict[str, Any]):
        """Display search results information in a formatted table."""
        table = Table(title="Search Results Information")
        table.add_column("Retriever", style="cyan")
        table.add_column("Metric", style="green")
        table.add_column("Value", style="magenta")
        
        for retriever, data in info.items():
            table.add_row(retriever, "Number of Results", str(data['length']))
            table.add_row("", "Document ID Type", data['doc_id_type'])
            table.add_row("", "Score Type", data['score_type'])
        
        self.console.print(table)

    def _display_id_mapping_info(self, info: Dict[str, Any]):
        """Display ID mapping information in a formatted table."""
        table = Table(title="ID Mapping Information")
        table.add_column("Component", style="cyan")
        table.add_column("ID Format", style="green")
        table.add_column("Sample IDs", style="magenta")
        
        # Corpus structure
        table.add_row("Corpus Features", str(info['corpus_structure']), "N/A")
        # Dataset IDs
        table.add_row("Dataset Example IDs", info['dataset_id_format'], 
                    str(info['id_samples']['dataset_example_ids']))
        table.add_row("Gold Corpus Indices", info['gold_corpus_idx_format'],
                    str(info['id_samples']['dataset_gold_corpus_idx']))
        # Search Results
        table.add_row("Search Results", info['search_result_id_format'],
                    str(info['id_samples']['search_results']))
        
        self.console.print(table)

def main():
    """Main execution function."""
    try:
        analyzer = DatasetAnalyzer()
        
        # Run all analyses
        corpus_info = analyzer.analyze_corpus_structure()
        dataset_info = analyzer.analyze_nq_dataset()
        search_results_info = analyzer.analyze_search_results()
        mapping_info = analyzer.analyze_id_mapping()
        
        # Save analysis results
        analysis_path = paths.RESULTS_DIR / 'analysis'
        analysis_path.mkdir(exist_ok=True)
        
        full_analysis = {
            'corpus': corpus_info,
            'dataset': dataset_info,
            'search_results': search_results_info,
            'id_mapping': mapping_info
        }
        
        with open(analysis_path / f'dataset_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
            pickle.dump(full_analysis, f)
            
        analyzer.logger.info(f"\nAnalysis results saved to {analysis_path}")
        
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        raise

if __name__ == "__main__":
    main()