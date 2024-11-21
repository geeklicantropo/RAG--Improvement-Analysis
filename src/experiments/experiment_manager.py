import logging
import logging.config
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Iterator
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict
from rich.console import Console
from rich.progress import track, Progress
import json
import gc
import mmap
import tempfile
import os
from contextlib import contextmanager

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.baseline_rag import BaselineRAG
from src.experiments.kmeans_rag import KMeansRAG
from src.experiments.fusion_kmeans import FusionKMeansRAG
from src.experiments.fusion_categories import FusionCategoriesRAG
from src.utils.id_mapping import IDMapper
from src.core.env.env_loader import load_environment
from src.core.utils.system_check import perform_system_checks
from src.config import paths, settings

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run with index tracking."""
    name: str
    experiment_type: str
    retriever_type: str
    dataset_path: Path
    num_samples: int
    batch_size: int
    inject_random: bool
    random_ratio: float
    use_diverse: bool
    metrics_to_track: List[str]
    enable_gc: bool = True
    additional_params: Optional[Dict[str, Any]] = None
    index_tracking: Optional[Dict[str, Any]] = None

@dataclass
class ExperimentResult:
    """Stores experiment results with enhanced metrics."""
    config: ExperimentConfig
    results: List[Any]
    metrics: Dict[str, Any]
    index_metrics: Dict[str, Any]
    run_time: float
    timestamp: str
    resources: Dict[str, Any]

class ExperimentManager:
    """
    Memory-efficient experiment manager with comprehensive tracking and resource management.
    """
    def __init__(
        self,
        experiment_name: str = "rag_experiments",
        enable_gc: bool = True,
        batch_size: int = 8,
        cache_dir: Optional[str] = None
    ):
        self.setup_logging()
        self.console = Console()
        self.logger = logging.getLogger(f"{__name__}.{experiment_name}")
        
        self.experiment_name = experiment_name
        self.enable_gc = enable_gc
        self.batch_size = batch_size
        self.start_time = datetime.now()
        
        # Setup cache directory
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="experiment_manager_")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Setup metrics tracking with memory mapping
        self.metrics_file = Path(self.cache_dir) / "metrics.mmap"
        self._init_metrics()
        
        # Initialize ID mapper
        self.id_mapper = IDMapper(
            experiment_name=experiment_name,
            batch_size=batch_size,
            enable_gc=enable_gc
        )
        
        # Create experiment directories
        self.setup_directories()
        
        self.logger.info(
            f"Initialized ExperimentManager: {experiment_name} "
            f"(batch_size={batch_size})"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        with open(paths.LOGGING_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)

    def setup_directories(self):
        """Create necessary directories with proper structure."""
        try:
            self.results_dir = paths.RESULTS_DIR / self.experiment_name / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.models_dir = self.results_dir / "models"
            self.metrics_dir = self.results_dir / "metrics"
            self.plots_dir = self.results_dir / "plots"
            self.index_dir = self.results_dir / "indices"
            self.temp_dir = self.results_dir / "temp"
            
            for dir_path in [
                self.results_dir,
                self.models_dir,
                self.metrics_dir,
                self.plots_dir,
                self.index_dir,
                self.temp_dir
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Created experiment directories under {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            raise

    def _init_metrics(self):
        """Initialize metrics with memory mapping."""
        base_metrics = {
            'experiments': [],
            'start_time': self.start_time.isoformat(),
            'metrics': defaultdict(dict)
        }
        
        with open(self.metrics_file, 'wb') as f:
            f.write(json.dumps(base_metrics).encode())

    @contextmanager
    def _update_metrics(self):
        """Context manager for thread-safe metrics updates."""
        try:
            with open(self.metrics_file, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                metrics = json.loads(mm.read().decode())
                yield metrics
                
                mm.seek(0)
                mm.write(json.dumps(metrics).encode())
                mm.flush()
        finally:
            if 'mm' in locals():
                mm.close()

    def create_experiment(
        self,
        name: str,
        experiment_type: str,
        retriever_type: str = "contriever",
        num_samples: int = -1,
        batch_size: Optional[int] = None,
        inject_random: bool = False,
        random_ratio: float = 0.3,
        use_diverse: bool = True,
        metrics_to_track: Optional[List[str]] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """Create a new experiment configuration with index tracking support."""
        try:
            # Use default metrics if none provided
            if metrics_to_track is None:
                metrics_to_track = [
                    'accuracy',
                    'retrieval_time',
                    'generation_time',
                    'index_consistency'
                ]
            
            # Use class batch size if none provided
            batch_size = batch_size or self.batch_size
            
            # Determine dataset path
            dataset_path = paths.TEST_DATASET_PATH if settings.get('USE_TEST', True) else paths.TRAIN_DATASET_PATH
            
            # Create index tracking configuration
            index_tracking = {
                'enabled': True,
                'mappings_dir': self.index_dir / name,
                'validate_indices': True,
                'track_unused_indices': False
            }
            
            config = ExperimentConfig(
                name=name,
                experiment_type=experiment_type,
                retriever_type=retriever_type,
                dataset_path=dataset_path,
                num_samples=num_samples,
                batch_size=batch_size,
                inject_random=inject_random,
                random_ratio=random_ratio,
                use_diverse=use_diverse,
                metrics_to_track=metrics_to_track,
                enable_gc=self.enable_gc,
                additional_params=additional_params or {},
                index_tracking=index_tracking
            )
            
            self.logger.info(f"Created experiment configuration: {name}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error creating experiment config: {e}")
            raise

    def _initialize_experiment(self, config: ExperimentConfig) -> Any:
        """Initialize appropriate experiment class based on type."""
        try:
            # Common parameters for all experiment types
            common_params = {
                'experiment_name': config.name,
                'batch_size': config.batch_size,
                'id_mapper': self.id_mapper,
                'enable_gc': config.enable_gc,
                'cache_dir': os.path.join(self.cache_dir, config.name)
            }
            
            if config.experiment_type == "baseline":
                return BaselineRAG(
                    retriever_type=config.retriever_type,
                    **common_params
                )
            elif config.experiment_type == "kmeans":
                return KMeansRAG(
                    retriever_type=config.retriever_type,
                    n_clusters=config.additional_params.get('n_clusters', 5),
                    **common_params
                )
            elif config.experiment_type == "fusion_kmeans":
                return FusionKMeansRAG(
                    retriever_type=config.retriever_type,
                    n_clusters=config.additional_params.get('n_clusters', 5),
                    fusion_weights=config.additional_params.get('fusion_weights'),
                    **common_params
                )
            elif config.experiment_type == "fusion_categories":
                return FusionCategoriesRAG(
                    retriever_type=config.retriever_type,
                    num_categories=config.additional_params.get('num_categories', 5),
                    min_docs_per_category=config.additional_params.get('min_docs_per_category', 2),
                    fusion_weights=config.additional_params.get('fusion_weights'),
                    **common_params
                )
            else:
                raise ValueError(f"Unknown experiment type: {config.experiment_type}")
                
        except Exception as e:
            self.logger.error(f"Error initializing experiment: {e}")
            raise

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with enhanced monitoring and index tracking."""
        try:
            start_time = datetime.now()
            self.logger.info(f"Starting experiment: {config.name}")
            
            with Progress() as progress:
                # Initialize experiment
                task1 = progress.add_task(
                    "[cyan]Initializing experiment...",
                    total=1
                )
                experiment = self._initialize_experiment(config)
                progress.update(task1, advance=1)
                
                # Initialize indices if needed
                if config.index_tracking and config.index_tracking['enabled']:
                    task2 = progress.add_task(
                        "[yellow]Initializing indices...",
                        total=1
                    )
                    self._initialize_indices(config)
                    progress.update(task2, advance=1)
                
                # Run experiment with resource cleanup
                task3 = progress.add_task(
                    "[green]Running experiment...",
                    total=1
                )
                with experiment:  # Use context manager for cleanup
                    results = experiment.run_experiment(
                        dataset_path=str(config.dataset_path),
                        num_samples=config.num_samples,
                        batch_size=config.batch_size,
                        inject_random=config.inject_random,
                        random_ratio=config.random_ratio if config.inject_random else None
                    )
                progress.update(task3, advance=1)
            
            # Create result object
            result = ExperimentResult(
                config=config,
                results=results,
                metrics=experiment.metrics,
                index_metrics=self.id_mapper.get_mapping_stats(),
                run_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat(),
                resources=self._get_resource_usage()
            )
            
            # Update metrics
            with self._update_metrics() as metrics:
                metrics['experiments'].append(config.name)
                metrics['metrics'][config.name] = result.metrics
            
            # Save results
            self._save_experiment_result(result)
            
            # Memory cleanup if enabled
            if self.enable_gc:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            self.logger.info(
                f"Completed experiment {config.name} "
                f"in {result.run_time:.2f} seconds"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running experiment {config.name}: {e}")
            raise

    def _initialize_indices(self, config: ExperimentConfig):
        """Initialize indices for experiment."""
        try:
            # Create index directory
            index_dir = config.index_tracking['mappings_dir']
            index_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize indices with batching
            self.id_mapper.initialize_indices(
                index_dir=index_dir,
                experiment_type=config.experiment_type,
                batch_size=self.batch_size
            )
            
            self.logger.info(f"Initialized indices in {index_dir}")
            
        except Exception as e:
            self.logger.error(f"Error initializing indices: {e}")
            raise

    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        import psutil
        process = psutil.Process()
        
        resources = {
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(),
            'memory_info': dict(process.memory_info()._asdict()),
            'num_threads': process.num_threads()
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            resources['gpu'] = {
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved(),
                'max_memory_allocated': torch.cuda.max_memory_allocated()
            }
        
        return resources

    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment results with memory efficiency."""
        try:
            # Create experiment directory
            exp_dir = self.results_dir / result.config.name
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_path = exp_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(vars(result.config), f, indent=2)
            
            # Save metrics in chunks
            metrics_path = exp_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    'experiment_metrics': result.metrics,
                    'index_metrics': result.index_metrics,
                    'resource_metrics': result.resources
                }, f, indent=2)
            
            # Save detailed results in chunks
            results_path = exp_dir / "results.json"
            with open(results_path, 'w') as f:
                # Write header
                f.write('[\n')
                
                # Write results in chunks
                for idx, item in enumerate(result.results):
                    f.write(json.dumps(item))
                    if idx < len(result.results) - 1:
                        f.write(',\n')
                
                # Write footer
                f.write('\n]')
            
            self.logger.info(f"Saved experiment results to {exp_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving experiment result: {e}")
            raise

    def run_all_experiments(self) -> Dict[str, ExperimentResult]:
        """Run all experiments with comprehensive tracking."""
        try:
            results = {}
            
            # Create experiment configurations
            experiment_configs = self._create_all_experiment_configs()
            
            # Run experiments with progress tracking
            with Progress() as progress:
                task = progress.add_task(
                    "[cyan]Running experiments...",
                    total=len(experiment_configs)
                )
                
                for config in experiment_configs:
                    # Update progress description
                    progress.update(
                        task,
                        description=f"[cyan]Running {config.name}",
                        advance=0
                    )
                    
                    # Run experiment
                    results[config.name] = self.run_experiment(config)
                    
                    # Update progress
                    progress.update(task, advance=1)
                    
                    # Memory cleanup
                    if self.enable_gc:
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Generate comparative analysis
            self._save_comparative_analysis(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running experiments: {e}")
            raise

    def _create_all_experiment_configs(self) -> List[ExperimentConfig]:
        """Create configurations for all experiments."""
        configs = []
        
        # Add baseline configs
        configs.extend([
            self.create_experiment(
                name="baseline_normal",
                experiment_type="baseline",
                inject_random=False
            ),
            self.create_experiment(
                name="baseline_random",
                experiment_type="baseline",
                inject_random=True,
                random_ratio=settings.get('RANDOM_INJECTION', {}).get('injection_ratio', 0.3)
            )
        ])
        
        # Add kmeans configs
        configs.extend([
            self.create_experiment(
                name="kmeans_normal",
                experiment_type="kmeans",
                inject_random=False,
                additional_params={'n_clusters': settings.KMEANS_CLUSTERING['n_clusters']}
            ),
            self.create_experiment(
                name="kmeans_random",
                experiment_type="kmeans",
                inject_random=True,
                random_ratio=settings.get('RANDOM_INJECTION', {}).get('injection_ratio', 0.3),
                additional_params={'n_clusters': settings.KMEANS_CLUSTERING['n_clusters']}
            )
        ])
        
        # Add fusion configs
        configs.extend([
            self.create_experiment(
                name="fusion_normal",
                experiment_type="fusion_kmeans",
                inject_random=False,
                additional_params={
                    'n_clusters': settings.KMEANS_CLUSTERING['n_clusters'],
                    'fusion_weights': settings.RAG_FUSION['fusion_weights']
                }
            ),
            self.create_experiment(
                name="fusion_random",
                experiment_type="fusion_kmeans",
                inject_random=True,
                random_ratio=settings.get('RANDOM_INJECTION', {}).get('injection_ratio', 0.3),
                additional_params={
                    'n_clusters': settings.KMEANS_CLUSTERING['n_clusters'],
                    'fusion_weights': settings.RAG_FUSION['fusion_weights']
                }
            )
        ])
        
        # Add categories configs
        configs.extend([
            self.create_experiment(
                name="categories_normal",
                experiment_type="fusion_categories",
                inject_random=False,
                additional_params={
                    'num_categories': settings.FUSION_CATEGORIES['num_categories'],
                    'min_docs_per_category': settings.FUSION_CATEGORIES['min_docs_per_category'],
                    'fusion_weights': settings.RAG_FUSION['fusion_weights']
                }
            ),
            self.create_experiment(
                name="categories_random",
                experiment_type="fusion_categories",
                inject_random=True,
                random_ratio=settings.get('RANDOM_INJECTION', {}).get('injection_ratio', 0.3),
                additional_params={
                    'num_categories': settings.FUSION_CATEGORIES['num_categories'],
                    'min_docs_per_category': settings.FUSION_CATEGORIES['min_docs_per_category'],
                    'fusion_weights': settings.RAG_FUSION['fusion_weights']
                }
            )
        ])
        
        return configs

    def _save_comparative_analysis(self, results: Dict[str, ExperimentResult]):
        """Save comparative analysis of all experiments."""
        try:
            analysis_dir = self.results_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Prepare analysis data
            analysis = {
                'experiment_comparison': {},
                'performance_metrics': {},
                'resource_usage': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate comparative metrics
            for name, result in results.items():
                analysis['experiment_comparison'][name] = {
                    'duration': result.run_time,
                    'samples_processed': len(result.results),
                    'completion_rate': result.metrics.get('completion_rate', 0)
                }
                
                analysis['performance_metrics'][name] = {
                    k: v for k, v in result.metrics.items()
                    if isinstance(v, (int, float))
                }
                
                analysis['resource_usage'][name] = result.resources
            
            # Save analysis in chunks
            analysis_path = analysis_dir / "comparative_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Generate summary tables
            summary_dir = analysis_dir / "summaries"
            summary_dir.mkdir(exist_ok=True)
            
            self._save_summary_tables(results, summary_dir)
            
            self.logger.info(f"Saved comparative analysis to {analysis_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving comparative analysis: {e}")
            raise

    def _save_summary_tables(
        self,
        results: Dict[str, ExperimentResult],
        output_dir: Path
    ):
        """Save summary tables with memory efficiency."""
        try:
            # Performance summary
            performance = {
                name: {
                    'Duration': f"{result.run_time:.2f}s",
                    'Samples': len(result.results),
                    'Success Rate': f"{result.metrics.get('success_rate', 0):.2%}",
                    'Memory Usage': f"{result.resources['memory_percent']:.1f}%"
                }
                for name, result in results.items()
            }
            
            perf_path = output_dir / "performance_summary.json"
            with open(perf_path, 'w') as f:
                json.dump(performance, f, indent=2)
            
            # Resource usage summary
            resources = {
                name: {
                    'CPU': f"{result.resources['cpu_percent']:.1f}%",
                    'Memory': f"{result.resources['memory_percent']:.1f}%",
                    'GPU Memory': f"{result.resources.get('gpu', {}).get('memory_allocated', 0) / 1e9:.2f}GB"
                    if torch.cuda.is_available() else "N/A"
                }
                for name, result in results.items()
            }
            
            resource_path = output_dir / "resource_summary.json"
            with open(resource_path, 'w') as f:
                json.dump(resources, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving summary tables: {e}")
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            # Remove temp directory
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                for file in self.temp_dir.iterdir():
                    file.unlink()
                self.temp_dir.rmdir()
            
            # Remove cache directory
            if hasattr(self, 'cache_dir') and os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    os.unlink(os.path.join(self.cache_dir, file))
                os.rmdir(self.cache_dir)
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass

def main():
    """Main execution function with comprehensive experiment management."""
    try:
        # Configure argument parser
        parser = argparse.ArgumentParser(
            description="Run RAG experiments with various strategies"
        )
        parser.add_argument(
            '--experiments',
            nargs='+',
            choices=['baseline', 'kmeans', 'fusion', 'categories', 'all'],
            default=['all'],
            help='Specify which experiments to run'
        )
        parser.add_argument(
            '--random',
            action='store_true',
            help='Include random document injection variants'
        )
        parser.add_argument(
            '--num_samples',
            type=int,
            default=-1,
            help='Number of samples to process (-1 for all)'
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            default=8,
            help='Batch size for processing'
        )
        parser.add_argument(
            '--output_dir',
            type=str,
            default=None,
            help='Custom output directory for results'
        )
        parser.add_argument(
            '--enable_gc',
            type=bool,
            default=True,
            help='Enable garbage collection'
        )
        
        args = parser.parse_args()
        
        # Load environment
        load_environment()
        
        # Perform system checks
        perform_system_checks()
        
        # Update settings based on arguments
        if args.num_samples > 0:
            settings.NUM_SAMPLES = args.num_samples
        if args.output_dir:
            paths.RESULTS_DIR = Path(args.output_dir)
        
        # Create console for rich output
        console = Console()
        
        # Initialize experiment manager
        with ExperimentManager(enable_gc=args.enable_gc, batch_size=args.batch_size) as manager:
            # Run experiments
            if 'all' in args.experiments:
                results = manager.run_all_experiments()
            else:
                # Create appropriate configurations
                configs = []
                for exp_type in args.experiments:
                    if exp_type == 'baseline':
                        configs.extend([
                            manager.create_experiment(
                                name="baseline_normal",
                                experiment_type="baseline",
                                inject_random=False
                            )
                        ])
                        if args.random:
                            configs.append(
                                manager.create_experiment(
                                    name="baseline_random",
                                    experiment_type="baseline",
                                    inject_random=True
                                )
                            )
                    elif exp_type == 'kmeans':
                        configs.extend([
                            manager.create_experiment(
                                name="kmeans_normal",
                                experiment_type="kmeans",
                                inject_random=False
                            )
                        ])
                        if args.random:
                            configs.append(
                                manager.create_experiment(
                                    name="kmeans_random",
                                    experiment_type="kmeans",
                                    inject_random=True
                                )
                            )
                    elif exp_type == 'fusion':
                        configs.extend([
                            manager.create_experiment(
                                name="fusion_normal",
                                experiment_type="fusion_kmeans",
                                inject_random=False
                            )
                        ])
                        if args.random:
                            configs.append(
                                manager.create_experiment(
                                    name="fusion_random",
                                    experiment_type="fusion_kmeans",
                                    inject_random=True
                                )
                            )
                    elif exp_type == 'categories':
                        configs.extend([
                            manager.create_experiment(
                                name="categories_normal",
                                experiment_type="fusion_categories",
                                inject_random=False
                            )
                        ])
                        if args.random:
                            configs.append(
                                manager.create_experiment(
                                    name="categories_random",
                                    experiment_type="fusion_categories",
                                    inject_random=True
                                )
                            )
                
                # Run configured experiments
                results = {}
                for config in configs:
                    results[config.name] = manager.run_experiment(config)
            
            # Display final results
            console.print("\n[bold cyan]Experiment Results Summary:")
            for name, result in results.items():
                accuracy = result.metrics.get('accuracy', 0)
                runtime = result.run_time
                console.print(
                    f"[green]{name}:[/green] "
                    f"Accuracy = {accuracy:.4f}, "
                    f"Runtime = {runtime:.2f}s"
                )
            
            console.print("\n[bold green]✓ All experiments completed successfully!")
            
    except KeyboardInterrupt:
        console.print("\n[bold red]Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error in experiment execution: {str(e)}")
        logging.error("Error in experiment execution", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()