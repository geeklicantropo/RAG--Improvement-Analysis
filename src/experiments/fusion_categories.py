import logging
import logging.config
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.llm import LLM, GenerationResult
from src.utils.retriever import Retriever, RetrievalResult
from src.fusion.fusion import RAGFusion, Document, FusionResult
from src.utils.id_mapping import IDMapper
from src.core.env.env_loader import load_environment
from src.core.utils.system_check import perform_system_checks
from src.config import paths, settings

@dataclass
class CategoryInfo:
    """Stores category information with index tracking."""
    name: str
    keywords: Set[str]
    docs: List[Document]
    centroid: np.ndarray
    similarity_threshold: float
    doc_indices: List[int]  # Original corpus indices
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class FusionCategoryResult:
    """Stores fusion-categories experiment results with index tracking."""
    query: str
    query_idx: int  # Dataset index
    query_category: str
    initial_docs: List[RetrievalResult]
    category_docs: List[Document]
    fused_docs: List[Document]
    final_docs: List[Document]
    generation: GenerationResult
    fusion: FusionResult
    categories: Dict[str, CategoryInfo]
    original_doc_indices: List[int]  # Original corpus indices
    gold_doc_idx: Optional[int] = None
    metrics: Dict[str, Any] = None

class FusionCategoriesRAG:
    """
    Enhanced Fusion-Categories RAG implementation with index tracking, diversity,
    and random document injection.
    """
    def __init__(
        self,
        retriever_type: str = "contriever",
        num_categories: int = 5,
        min_docs_per_category: int = 2,
        fusion_weights: Optional[List[float]] = None,
        embedding_model: str = 'sentence-transformers/all-mpnet-base-v2',
        experiment_name: Optional[str] = None,
        batch_size: int = 8,
        id_mapper: Optional[IDMapper] = None,
        enable_gc: bool = True
    ):
        self.setup_logging()
        self.logger = logging.getLogger(f"{__name__}.{experiment_name}")
        
        # Initialize parameters
        self.retriever_type = retriever_type
        self.num_categories = num_categories
        self.min_docs_per_category = min_docs_per_category
        self.fusion_weights = fusion_weights or [1.0, 0.8, 0.6, 0.4, 0.2]
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.enable_gc = enable_gc
        self.experiment_name = experiment_name or f"fusion_categories_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup metrics tracking
        self.metrics = {
            'experiment_type': 'fusion_categories_rag',
            'retriever_type': retriever_type,
            'num_categories': num_categories,
            'min_docs_per_category': min_docs_per_category,
            'start_time': datetime.now().isoformat(),
            'batch_size': batch_size
        }
        
        # Initialize components
        self.id_mapper = id_mapper or IDMapper(
            experiment_name=self.experiment_name,
            batch_size=batch_size,
            enable_gc=enable_gc
        )
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
        # Create experiment directories
        self.setup_directories()
        
        self.logger.info(
            f"Initialized FusionCategoriesRAG with {retriever_type} retriever, "
            f"{num_categories} categories"
        )

    def setup_logging(self):
        """Setup logging configuration."""
        with open(paths.LOGGING_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)

    def setup_directories(self):
        """Create necessary directories."""
        self.results_dir = paths.FUSION_RESULTS_DIR / "categories" / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.models_dir = self.results_dir / "models"
        self.fusion_dir = self.results_dir / "fusion"
        self.categories_dir = self.results_dir / "categories"
        
        for dir_path in [self.results_dir, self.models_dir, self.fusion_dir, self.categories_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def initialize_components(self, index_dir: Optional[str] = None):
        """Initialize all required components."""
        try:
            self.logger.info("Initializing components...")
            
            # Initialize device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize retriever
            self.retriever = Retriever(
                retriever_type=self.retriever_type,
                device=device,
                batch_size=self.batch_size,
                cache_embeddings=True
            )
            
            # Initialize LLM
            self.generator = LLM(
                model_id=settings.LLM_MODEL_ID,
                device=device,
                quantization_bits=4,
                model_max_length=settings.MODEL_MAX_LENGTH,
                experiment_name=self.experiment_name
            )
            
            # Initialize sentence transformer for embeddings
            self.embed_model = SentenceTransformer(self.embedding_model)
            self.embed_model.to(device)
            
            # Initialize RAG Fusion
            self.fusion = RAGFusion(
                embedding_model=self.embedding_model,
                n_clusters=self.num_categories,
                fusion_weights=self.fusion_weights,
                experiment_name=self.experiment_name
            )
            
            # Load index mappings if provided
            if index_dir:
                index_dir = Path(index_dir)
                if index_dir.exists():
                    self.id_mapper.load_mappings(index_dir)
                    self.logger.info(f"Loaded index mappings from {index_dir}")
            
            self.metrics.update({
                'model_id': settings.LLM_MODEL_ID,
                'model_max_length': settings.MODEL_MAX_LENGTH,
                'device': device
            })
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise

    def _extract_categories(self, documents: List[Document]) -> Dict[str, CategoryInfo]:
        """Extract document categories using TF-IDF and clustering."""
        try:
            # Get document texts and compute TF-IDF
            texts = [doc.text for doc in documents]
            doc_vectors = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get embeddings for similarity computation
            embeddings = self.embed_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Initialize categories
            categories = {}
            doc_assignments = defaultdict(list)
            
            # Process documents
            for idx, doc in enumerate(documents):
                # Get top terms from TF-IDF
                vector = doc_vectors[idx].toarray()[0]
                top_term_indices = vector.argsort()[-5:][::-1]
                top_terms = {feature_names[i] for i in top_term_indices}
                
                # Create category name
                category_name = "_".join(sorted(list(top_terms)[:3]))
                doc_assignments[category_name].append((doc, vector, embeddings[idx]))
            
            # Create final categories
            for category_name, doc_info in doc_assignments.items():
                if len(doc_info) >= self.min_docs_per_category:
                    docs, vectors, doc_embeddings = zip(*doc_info)
                    
                    # Get corpus indices
                    doc_indices = [
                        self.id_mapper.get_index_mapping(doc.id, 'search_to_corpus_idx')
                        or doc.id for doc in docs
                    ]
                    
                    # Compute centroid and threshold
                    centroid = np.mean(doc_embeddings, axis=0)
                    similarities = cosine_similarity(doc_embeddings, centroid.reshape(1, -1))
                    similarity_threshold = np.percentile(similarities, 25)
                    
                    # Extract keywords
                    keywords = set()
                    for vector in vectors:
                        top_indices = vector.argsort()[-5:][::-1]
                        keywords.update(feature_names[i] for i in top_indices)
                    
                    categories[category_name] = CategoryInfo(
                        name=category_name,
                        keywords=keywords,
                        docs=list(docs),
                        centroid=centroid,
                        similarity_threshold=float(similarity_threshold),
                        doc_indices=doc_indices,
                        metadata={
                            'size': len(docs),
                            'avg_score': float(np.mean([d.score for d in docs])),
                            'keywords': sorted(list(keywords))[:10],
                            'similarity_stats': {
                                'mean': float(np.mean(similarities)),
                                'std': float(np.std(similarities))
                            }
                        }
                    )
            
            # Limit number of categories
            if len(categories) > self.num_categories:
                sorted_categories = sorted(
                    categories.items(),
                    key=lambda x: len(x[1].docs),
                    reverse=True
                )
                categories = dict(sorted_categories[:self.num_categories])
            
            return categories
            
        except Exception as e:
            self.logger.error(f"Error extracting categories: {e}")
            raise

    def _assign_query_category(
        self,
        query: str,
        categories: Dict[str, CategoryInfo],
        use_embeddings: bool = True
    ) -> Tuple[str, float]:
        """Assign query to most relevant category."""
        try:
            if use_embeddings:
                # Get query embedding
                query_embedding = self.embed_model.encode(
                    query,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Calculate similarities with category centroids
                similarities = {
                    name: float(cosine_similarity(
                        query_embedding.reshape(1, -1),
                        info.centroid.reshape(1, -1)
                    )[0][0])
                    for name, info in categories.items()
                }
                
                # Get best category
                best_category = max(similarities.items(), key=lambda x: x[1])
                return best_category[0], best_category[1]
            else:
                # Use TF-IDF for category assignment
                query_vector = self.vectorizer.transform([query]).toarray()[0]
                
                best_score = -1
                best_category = None
                
                for name, info in categories.items():
                    score = sum(query_vector[self.vectorizer.vocabulary_[word]]
                              for word in info.keywords
                              if word in self.vectorizer.vocabulary_)
                    if score > best_score:
                        best_score = score
                        best_category = name
                
                return best_category, float(best_score)
                
        except Exception as e:
            self.logger.error(f"Error assigning query category: {e}")
            raise

    def run_experiment(
        self,
        dataset_path: str,
        num_samples: int = -1,
        batch_size: Optional[int] = None,
        inject_random: bool = False,
        random_ratio: float = 0.3,
        index_dir: Optional[str] = None
    ) -> List[FusionCategoryResult]:
        """
        Run the fusion-categories experiment with random document injection.
        """
        try:
            # Initialize components
            self.initialize_components(index_dir)
            
            # Load dataset
            with open(dataset_path, 'r') as f:
                dataset = yaml.safe_load(f)
            
            if num_samples > 0:
                dataset = dataset[:num_samples]
            
            self.logger.info(f"Processing {len(dataset)} queries")
            
            # Update metrics
            self.metrics.update({
                'num_samples': len(dataset),
                'inject_random': inject_random,
                'random_ratio': random_ratio if inject_random else None
            })
            
            results = []
            total_metrics = defaultdict(float)
            
            # Process in batches
            batch_size = batch_size or self.batch_size
            for i in range(0, len(dataset), batch_size):
                batch_data = dataset[i:i + batch_size]
                
                # Process batch
                batch_results = self._process_batch(
                    batch_data,
                    inject_random=inject_random,
                    random_ratio=random_ratio
                )
                
                if batch_results:
                    results.extend(batch_results)
                    
                    # Update metrics
                    for result in batch_results:
                        for key, value in result.metrics.items():
                            if isinstance(value, (int, float)):
                                total_metrics[key] += value
                    
                    # Memory optimization
                    if self.enable_gc and i % (batch_size * 5) == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Update final metrics
            if results:
                avg_metrics = {
                    key: value / len(results)
                    for key, value in total_metrics.items()
                }
                
                self.metrics.update({
                    'total_samples': len(results),
                    'completion_rate': len(results) / len(dataset) * 100,
                    'average_metrics': avg_metrics,
                    'end_time': datetime.now().isoformat(),
                    'total_duration': (datetime.now() - self.start_time).total_seconds()
                })
                
                # Save results
                self._save_final_results(results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error running experiment: {e}")
            raise

    def _process_batch(
        self,
        batch_data: List[Dict],
        inject_random: bool = False,
        random_ratio: float = 0.3
    ) -> List[FusionCategoryResult]:
        """Process a batch of queries with category-based fusion."""
        batch_results = []
        
        for item in batch_data:
            try:
                # Extract query information
                query = item['question']
                query_idx = int(item.get('example_id', -1))
                gold_doc_idx = int(item.get('idx_gold_in_corpus', -1))
                start_time = datetime.now()
                
                # Track original corpus index
                if gold_doc_idx >= 0:
                    gold_doc_idx = self.id_mapper.get_index_mapping(
                        gold_doc_idx,
                        'dataset_idx_to_corpus'
                    ) or gold_doc_idx
                
                # Initial retrieval
                retrieval_start = datetime.now()
                initial_docs = self.retriever.retrieve(
                    query,
                    top_k=self.num_categories * 3,
                    include_metadata=True
                )
                retrieval_time = (datetime.now() - retrieval_start).total_seconds()
                
                # Track corpus indices
                original_indices = []
                for doc in initial_docs:
                    corpus_idx = self.id_mapper.get_index_mapping(
                        doc.corpus_idx,
                        'search_to_corpus_idx'
                    ) or doc.corpus_idx
                    original_indices.append(corpus_idx)
                
                # Convert to fusion Document format
                documents = [
                    Document(
                        id=doc.corpus_idx,
                        text=doc.text,
                        score=doc.score,
                        metadata={
                            **doc.metadata,
                            'corpus_idx': self.id_mapper.get_index_mapping(
                                doc.corpus_idx,
                                'search_to_corpus_idx'
                            ) or doc.corpus_idx
                        }
                    )
                    for doc in initial_docs
                ]
                
                # Apply fusion
                fusion_start = datetime.now()
                fusion_result = self.fusion.fuse_results(
                    query,
                    documents,
                    fusion_method="reciprocal_rank"
                )
                fusion_time = (datetime.now() - fusion_start).total_seconds()
                
                # Extract categories
                categorization_start = datetime.now()
                categories = self._extract_categories(fusion_result.documents)
                
                # Get query category
                query_category, similarity = self._assign_query_category(
                    query,
                    categories,
                    use_embeddings=True
                )
                
                # Get documents from query category
                category_docs = categories[query_category].docs
                category_docs = sorted(category_docs, key=lambda x: x.score, reverse=True)
                
                categorization_time = (datetime.now() - categorization_start).total_seconds()
                
                # Final document set
                final_docs = category_docs.copy()
                
                # Add random documents if requested
                num_random = 0
                if inject_random and random_ratio > 0:
                    num_random = int(len(category_docs) * random_ratio)
                    if num_random > 0:
                        random_docs = self.retriever.get_random_documents(num_random)
                        
                        # Convert to Document format
                        random_fusion_docs = [
                            Document(
                                id=doc.corpus_idx,
                                text=doc.text,
                                score=0.0,
                                category="random",
                                metadata={
                                    'corpus_idx': self.id_mapper.get_index_mapping(
                                        doc.corpus_idx,
                                        'random_to_corpus_idx'
                                    ) or doc.corpus_idx,
                                    'is_random': True
                                }
                            )
                            for doc in random_docs
                        ]
                        
                        # Track random document indices
                        for doc in random_docs:
                            corpus_idx = self.id_mapper.get_index_mapping(
                                doc.corpus_idx,
                                'random_to_corpus_idx'
                            ) or doc.corpus_idx
                            original_indices.append(corpus_idx)
                            
                        # Add random documents to final set
                        final_docs.extend(random_fusion_docs)
                
                # Generate response
                generation_start = datetime.now()
                context = self._prepare_context(final_docs, query_category)
                generation_result = self.generator.generate(
                    f"Question: {query}\n\nContext: {context}\n\nAnswer:",
                    doc_ids=[doc.id for doc in final_docs]
                )
                generation_time = (datetime.now() - generation_start).total_seconds()
                
                # Record metrics
                metrics = {
                    'query_idx': query_idx,
                    'query_category': query_category,
                    'category_similarity': float(similarity),
                    'retrieval_time': retrieval_time,
                    'fusion_time': fusion_time,
                    'categorization_time': categorization_time,
                    'generation_time': generation_time,
                    'total_time': (datetime.now() - start_time).total_seconds(),
                    'num_initial_docs': len(initial_docs),
                    'num_fused_docs': len(fusion_result.documents),
                    'num_category_docs': len(category_docs),
                    'num_final_docs': len(final_docs),
                    'num_random_docs': num_random if inject_random else 0,
                    'initial_doc_scores': [doc.score for doc in initial_docs],
                    'fused_doc_scores': fusion_result.scores,
                    'category_doc_scores': [doc.score for doc in category_docs],
                    'gold_doc_retrieved': gold_doc_idx in original_indices,
                    'gold_doc_category': next(
                        (cat_name for cat_name, info in categories.items()
                         if gold_doc_idx in info.doc_indices),
                        None
                    ),
                    'category_stats': {
                        cat_name: {
                            'size': len(info.docs),
                            'avg_score': float(np.mean([d.score for d in info.docs])),
                            'keywords': list(info.keywords)[:10]
                        }
                        for cat_name, info in categories.items()
                    }
                }
                
                # Create result
                result = FusionCategoryResult(
                    query=query,
                    query_idx=query_idx,
                    query_category=query_category,
                    initial_docs=initial_docs,
                    category_docs=category_docs,
                    fused_docs=fusion_result.documents,
                    final_docs=final_docs,
                    generation=generation_result,
                    fusion=fusion_result,
                    categories=categories,
                    original_doc_indices=original_indices,
                    gold_doc_idx=gold_doc_idx,
                    metrics=metrics
                )
                
                batch_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing query '{query}': {e}")
                continue
        
        return batch_results

    def _prepare_context(self, documents: List[Document], query_category: str) -> str:
        """Prepare context string with category information."""
        context_parts = [f"[Category {query_category}] Retrieved and categorized documents:"]
        
        for idx, doc in enumerate(documents, 1):
            # Get corpus index mapping
            corpus_idx = doc.metadata.get('corpus_idx', doc.id)
            
            # Add category and document information
            category_info = f"Category {getattr(doc, 'category', query_category)}"
            context_parts.append(
                f"Document [ID: {corpus_idx}] "
                f"(Score: {doc.score:.3f}, {category_info}): {doc.text}"
            )
        
        return "\n\n".join(context_parts)

    def _save_intermediate_results(self, results: List[FusionCategoryResult], current_idx: int):
        """Save intermediate results during experiment."""
        try:
            checkpoint_path = self.results_dir / f"checkpoint_{current_idx}.yaml"
            
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                result_dict = {
                    'query': result.query,
                    'query_idx': result.query_idx,
                    'query_category': result.query_category,
                    'gold_doc_idx': result.gold_doc_idx,
                    'initial_docs': [
                        {
                            'id': doc.corpus_idx,
                            'text': doc.text,
                            'score': doc.score,
                            'metadata': doc.metadata
                        }
                        for doc in result.initial_docs
                    ],
                    'category_docs': [
                        {
                            'id': doc.id,
                            'text': doc.text,
                            'score': doc.score,
                            'category': getattr(doc, 'category', result.query_category),
                            'metadata': doc.metadata
                        }
                        for doc in result.category_docs
                    ],
                    'fused_docs': [
                        {
                            'id': doc.id,
                            'text': doc.text,
                            'score': doc.score,
                            'category': getattr(doc, 'category', result.query_category),
                            'metadata': doc.metadata
                        }
                        for doc in result.fused_docs
                    ],
                    'final_docs': [
                        {
                            'id': doc.id,
                            'text': doc.text,
                            'score': doc.score,
                            'category': getattr(doc, 'category', result.query_category),
                            'metadata': doc.metadata
                        }
                        for doc in result.final_docs
                    ],
                    'generation': {
                        'text': result.generation.text,
                        'tokens_used': result.generation.tokens_used,
                        'generation_time': result.generation.generation_time
                    },
                    'categories': {
                        name: {
                            'size': len(info.docs),
                            'keywords': list(info.keywords),
                            'similarity_threshold': info.similarity_threshold,
                            'metadata': info.metadata,
                            'doc_indices': info.doc_indices
                        }
                        for name, info in result.categories.items()
                    },
                    'fusion': {
                        'method': result.fusion.fusion_method,
                        'scores': result.fusion.scores,
                        'original_ranking': result.fusion.original_ranking,
                        'metadata': result.fusion.metadata
                    },
                    'original_doc_indices': result.original_doc_indices,
                    'metrics': result.metrics
                }
                serializable_results.append(result_dict)
            
            # Save results and current metrics
            checkpoint_data = {
                'results': serializable_results,
                'current_metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(checkpoint_path, 'w') as f:
                yaml.dump(checkpoint_data, f)
            
            self.logger.info(f"Saved checkpoint at {current_idx} queries")
            
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {e}")
            # Continue execution even if save fails
            pass

    def _save_final_results(self, results: List[FusionCategoryResult]):
        """Save final results and comprehensive metrics."""
        try:
            # Save detailed results
            results_path = self.results_dir / "final_results.yaml"
            self._save_intermediate_results(results, len(results))  # Reuse intermediate save logic
            
            # Save metrics
            metrics_path = self.results_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                yaml.dump(self.metrics, f, default_flow_style=False)
            
            # Save index mapping stats
            mapping_stats = self.id_mapper.get_mapping_stats()
            mapping_path = self.results_dir / "index_mapping_stats.json"
            with open(mapping_path, 'w') as f:
                yaml.dump(mapping_stats, f, default_flow_style=False)
            
            # Save analysis
            analysis = self._analyze_categories_performance(results)
            analysis_path = self.results_dir / "categories_analysis.json"
            with open(analysis_path, 'w') as f:
                yaml.dump(analysis, f, default_flow_style=False)
            
            # Save models
            if hasattr(self, 'embed_model'):
                self.embed_model.save(str(self.models_dir / "embed_model"))
            if hasattr(self.fusion, 'save_fusion_state'):
                self.fusion.save_fusion_state(self.models_dir / "fusion_state.pkl")
            
            # Save experiment configuration
            config = {
                'retriever_type': self.retriever_type,
                'num_categories': self.num_categories,
                'min_docs_per_category': self.min_docs_per_category,
                'fusion_weights': self.fusion_weights,
                'batch_size': self.batch_size,
                'model_config': {
                    'model_id': settings.LLM_MODEL_ID,
                    'model_max_length': settings.MODEL_MAX_LENGTH
                },
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat()
            }
            
            config_path = self.results_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            self.logger.info(f"Saved final results to {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving final results: {e}")
            raise

    def _analyze_categories_performance(
        self,
        results: List[FusionCategoryResult]
    ) -> Dict[str, Any]:
        """Analyze category-based performance metrics."""
        try:
            analysis = {
                'category_performance': defaultdict(lambda: defaultdict(list)),
                'query_distribution': defaultdict(int),
                'category_overlap': defaultdict(int),
                'random_impact': {
                    'with_random': defaultdict(list),
                    'without_random': defaultdict(list)
                }
            }
            
            for result in results:
                category = result.query_category
                
                # Track query distribution
                analysis['query_distribution'][category] += 1
                
                # Track performance metrics
                metrics = result.metrics
                performance = analysis['category_performance'][category]
                performance['retrieval_time'].append(metrics['retrieval_time'])
                performance['generation_time'].append(metrics['generation_time'])
                performance['doc_scores'].extend(metrics['category_doc_scores'])
                
                # Track random document impact
                if metrics['num_random_docs'] > 0:
                    analysis['random_impact']['with_random'][category].append(metrics)
                else:
                    analysis['random_impact']['without_random'][category].append(metrics)
                
                # Track category overlap
                if metrics['gold_doc_category']:
                    if metrics['gold_doc_category'] != category:
                        analysis['category_overlap'][
                            f"{category}->{metrics['gold_doc_category']}"
                        ] += 1
                
                # Compute average metrics
                for category, performance in analysis['category_performance'].items():
                    for metric, values in performance.items():
                        if values:
                            performance[f'avg_{metric}'] = float(np.mean(values))
                            performance[f'std_{metric}'] = float(np.std(values))
                
                return dict(analysis)
                
        except Exception as e:
            self.logger.error(f"Error analyzing category performance: {e}")
            return {}

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run Fusion-Categories RAG experiment")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/gen_res',
        help='Output directory'
    )
    parser.add_argument(
        '--experiment_mode',
        type=str,
        default='fusion_categories',
        help='Experiment mode'
    )
    parser.add_argument(
        '--retriever_type',
        type=str,
        choices=['contriever', 'bm25', 'adore'],
        default='contriever',
        help='Type of retriever to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--index_dir',
        type=str,
        help='Directory containing index mapping files'
    )
    parser.add_argument(
        '--enable_gc',
        type=bool,
        default=True,
        help='Enable garbage collection for memory optimization'
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    try:
        # Load environment variables
        load_environment()
        
        # Perform system checks
        perform_system_checks()
        
        # Initialize experiment
        experiment = FusionCategoriesRAG(
            retriever_type=args.retriever_type,
            num_categories=settings.FUSION_CATEGORIES['num_categories'],
            min_docs_per_category=settings.FUSION_CATEGORIES['min_docs_per_category'],
            fusion_weights=settings.RAG_FUSION['fusion_weights'],
            embedding_model=settings.KMEANS_CLUSTERING['embedding_model'],
            batch_size=args.batch_size,
            enable_gc=args.enable_gc
        )
        
        # Run experiment
        results = experiment.run_experiment(
            dataset_path=str(paths.TEST_DATASET_PATH),
            num_samples=settings.get('NUM_SAMPLES', -1),
            inject_random=settings.get('RANDOM_INJECTION', {}).get('enabled', False),
            random_ratio=settings.get('RANDOM_INJECTION', {}).get('injection_ratio', 0.3),
            index_dir=args.index_dir
        )
        
        # Log completion
        duration = datetime.now() - start_time
        experiment.logger.info(f"Fusion-Categories experiment completed in {duration}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()