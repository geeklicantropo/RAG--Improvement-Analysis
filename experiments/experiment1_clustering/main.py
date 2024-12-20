import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional  
from datetime import datetime
import torch
from tqdm import tqdm
import logging
import gc
import json
import random
import numpy as np
from collections import defaultdict

from src.utils.file_utils import seed_everything, clear_memory
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.llm_evaluator import LLMEvaluator  
from src.batch_processor import BatchProcessor
from src.memory_manager import MemoryManager
from src.generation_cache_manager import GenerationCacheManager
from src.utils.corpus_manager import CorpusManager
from .config import ClusteringConfig
import time
from src.utils.rate_limit import rate_limit

class ClusteringExperiment:
    def __init__(
        self,
        config: ClusteringConfig, 
        corpus_manager: CorpusManager,
        llm_evaluator: LLMEvaluator,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.corpus_manager = corpus_manager
        self.llm_evaluator = llm_evaluator
        self.logger = logger or ExperimentLogger(
            experiment_name="clustering_experiment",
            base_log_dir=str(Path(__file__).parent.parent.parent / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        self.output_dir = Path("experiments/experiment1_clustering/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load corpus and documents
        self.base_corpus = self.corpus_manager.get_corpus()
        self.gold_docs = self.corpus_manager.get_gold_documents()
        self.random_corpus = self.corpus_manager.get_random_subset(
            corpus_type='random', 
            num_docs=config.num_random_docs,
            seed=config.random_seed
        )
        self.adversarial_corpus = self.corpus_manager.get_random_subset(
            corpus_type='adversarial',
            num_docs=config.num_adversarial_docs,
            seed=config.adversarial_seed
        )


    def run_cluster_experiment(self, mode: str, cluster_id: int, cluster_docs: List[Dict]) -> List[Dict]:
        """Run experiments for a specific cluster"""
        results = []
        gold_docs = [doc for doc in cluster_docs if self._contains_answer(doc)]
        
        if not gold_docs:
            return results

        for gold_doc in gold_docs:
            context_docs = [gold_doc]
            
            if mode == "gold_random":
                random_samples = random.sample(self.random_corpus, min(len(self.random_corpus), 
                                            self.config.num_documents_in_context - 1))
                context_docs.extend(random_samples)
            elif mode == "gold_adversarial":
                adv_samples = random.sample(self.adversarial_corpus, min(len(self.adversarial_corpus),
                                         self.config.num_documents_in_context - 1))
                context_docs.extend(adv_samples)

            prompt = self._format_prompt(gold_doc["query"], context_docs)
            response = self.llm.generate(prompt, max_new_tokens=self.config.max_new_tokens)

            result = {
                "cluster_id": cluster_id,
                "mode": mode,
                "query": gold_doc["query"],
                "generated_answer": response,
                "gold_answer": gold_doc["gold_answer"],
                "context": [doc["text"] for doc in context_docs],
                "num_context_docs": len(context_docs)
            }

            evaluation = self.llm_evaluator.evaluate_answer(
                question=result["query"],
                generated_answer=result["generated_answer"],
                gold_answer=result["gold_answer"]
            )
            result["llm_evaluation"] = evaluation
            results.append(result)

        return results

    def run(self) -> Dict[str, List[Dict]]:
        """Run clustering experiment comparing performance across different document combinations"""
        # Generate document embeddings
        embeddings = self._generate_embeddings(self.base_corpus)
        
        # Cluster documents
        clusters = self._cluster_documents(embeddings)
        
        results = {}
        for mode in self.modes:
            mode_results = []
            for cluster_id, cluster_docs in clusters.items():
                cluster_results = self.run_cluster_experiment(mode, cluster_id, cluster_docs)
                mode_results.extend(cluster_results)
                
            results[mode] = mode_results
            self._save_results(mode_results, mode)
            
        return results
    
    @rate_limit
    def _generate_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """Generate embeddings for documents using Gemini"""
        embeddings = []
        for doc in tqdm(documents, desc="Generating embeddings"):
            try:
                prompt = f"""Generate a 768-dimensional vector for this text. Output ONLY space-separated numbers between -1 and 1, with NO other text:

                Text: {doc['text'][:500]}

                Vector:"""

                response = self.llm.generate(prompt, max_new_tokens=768)
                
                # Clean and validate response
                text = response[0].strip()
                numbers = []
                for token in text.split():
                    try:
                        num = float(token)
                        if -1 <= num <= 1:
                            numbers.append(num)
                    except ValueError:
                        continue
                        
                # Pad or truncate to exact dimension
                if len(numbers) < 768:
                    numbers.extend([0.0] * (768 - len(numbers)))
                else:
                    numbers = numbers[:768]
                    
                embeddings.append(numbers)
                
            except Exception as e:
                self.logger.error(f"Error generating embedding: {str(e)}")
                # Fallback to zero vector
                embeddings.append([0.0] * 768)
                
            time.sleep(0.1)  # Rate limiting

        return np.array(embeddings)

    def _cluster_documents(self, embeddings: np.ndarray) -> Dict[int, List[Dict]]:
        """Cluster documents using KMeans"""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.config.num_clusters, random_state=self.config.random_seed)
        labels = kmeans.fit_predict(embeddings)
        
        clusters = defaultdict(list)
        for doc, label in zip(self.base_corpus, labels):
            clusters[label].append(doc)
            
        return dict(clusters)

    def _contains_answer(self, document: Dict) -> bool:
        """Check if document contains gold answer"""
        return 'gold_answer' in document

    def _format_prompt(self, question: str, documents: List[Dict]) -> str:
        """Format prompt for LLM"""
        context_parts = [f"Question: {question}\n\nContext:"]
        for idx, doc in enumerate(documents, 1):
            context_parts.append(f"\nDocument [{idx}]:\n{doc['text']}")
        context_parts.append("\nAnswer:")
        return "\n".join(context_parts)

    def _save_results(self, results: List[Dict], mode: str):
        """Save results for a specific mode"""
        mode_dir = self.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = mode_dir / f"results_{mode}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    try:
        config = ClusteringConfig()
        corpus_manager = CorpusManager(str(config.corpus_path))
        llm_evaluator = LLMEvaluator(api_key=os.getenv("GEMINI_TOKEN"))
        
        experiment = ClusteringExperiment(config, corpus_manager, llm_evaluator)
        results = experiment.run()
        return results
        
    except Exception as e:
        logging.error(f"Error in clustering experiment: {str(e)}", exc_info=True)
        raise
    finally:
        clear_memory()

if __name__ == "__main__":
    seed_everything(10)
    main()