import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from tqdm import tqdm
import logging
from datetime import datetime

from src.utils import seed_everything, clear_memory
from src.experiment_logger import ExperimentLogger
from src.llm import LLM
from src.document_classifier import DocumentClassifier
from src.llm_evaluator import LLMEvaluator
from .config import ClusteringConfig
from experiments.evaluation.noise_impact import NoiseImpactEvaluator

class ClusteringExperiment:
    def __init__(
        self,
        config: ClusteringConfig,
        experiment_name: str,
        logger: Optional[ExperimentLogger] = None
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.logger = logger or ExperimentLogger(
            experiment_name=experiment_name,
            base_log_dir=str(Path(__file__).parent.parent.parent / "logs")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(api_key=os.getenv("GEMINI_TOKEN"))
        self.evaluator = LLMEvaluator(self.llm)
        self.doc_classifier = DocumentClassifier(self.llm)
        self.noise_evaluator = NoiseImpactEvaluator(self.evaluator)

    def run_clustering_phase(self):
        """Run clustering with systematic testing of configurations."""
        self.logger.log_step_start("Clustering Phase")
        try:
            test_configs = self._generate_test_configs()
            results = []
            
            for config in tqdm(test_configs, desc="Testing configurations"):
                cluster_results = self._run_cluster_test(config)
                position_results = self._test_positions(cluster_results)
                noise_results = self._test_noise_injection(cluster_results)
                combination_results = self._test_document_combinations(cluster_results)
                
                # Evaluate using Gemini
                evaluation = self.evaluator.evaluate_clusters(
                    cluster_results, 
                    position_results,
                    noise_results,
                    combination_results
                )
                
                results.append({
                    "config": config,
                    "clustering": cluster_results,
                    "positions": position_results,
                    "noise": noise_results,
                    "combinations": combination_results,
                    "evaluation": evaluation
                })
                
                clear_memory()
                
            self._save_results(results)
            self.logger.log_step_end("Clustering Phase")
            return results
            
        except Exception as e:
            self.logger.log_error(e, "Error in clustering phase")
            raise

    def _generate_test_configs(self) -> List[Dict]:
        """Generate systematic test configurations."""
        configs = []
        
        # Test different cluster sizes
        for n_clusters in [3, 5, 7, 10]:
            # Test different document positions
            for doc_positions in ['near', 'mid', 'far']:
                # Test different noise ratios
                for noise_ratio in [0.0, 0.1, 0.2]:
                    configs.append({
                        "n_clusters": n_clusters,
                        "doc_positions": doc_positions,
                        "noise_ratio": noise_ratio
                    })
                    
        return configs

    def _run_cluster_test(self, config: Dict) -> Dict:
        """Run clustering test with given configuration."""
        self.logger.log_step_start(f"Testing clusters={config['n_clusters']}")
        
        clusters = self.clusterer.fit_clusters(
            self.embeddings,
            n_clusters=config['n_clusters']
        )
        
        # Categorize documents in clusters
        categorized = self.doc_classifier.categorize_documents(clusters)
        
        self.logger.log_step_end(f"Clusters={config['n_clusters']}")
        return {
            "clusters": clusters,
            "categories": categorized
        }

    def _test_positions(self, cluster_results: Dict) -> Dict:
        """Test impact of document positions."""
        self.logger.log_step_start("Testing positions")
        
        positions = ['near', 'mid', 'far']
        position_results = {}
        
        for pos in positions:
            # Rearrange documents based on position
            docs = self._position_documents(
                cluster_results['clusters'],
                position=pos
            )
            
            # Evaluate with LLM
            eval_results = self.evaluator.evaluate_positioned_docs(docs)
            position_results[pos] = eval_results
            
        self.logger.log_step_end("Position testing")
        return position_results

    def _test_noise_injection(self, cluster_results: Dict) -> Dict:
        """Test impact of noise injection."""
        self.logger.log_step_start("Testing noise")
        
        noise_ratios = [0.1, 0.2, 0.3]
        noise_results = {}
        
        for ratio in noise_ratios:
            # Inject noise
            noisy_clusters = self._inject_noise(
                cluster_results['clusters'],
                ratio=ratio
            )
            
            # Evaluate with LLM
            eval_results = self.evaluator.evaluate_noisy_clusters(noisy_clusters)
            noise_results[ratio] = eval_results
            
        self.logger.log_step_end("Noise testing")
        return noise_results

    def _test_document_combinations(self, cluster_results: Dict) -> Dict:
        """Test impact of different document combinations."""
        self.logger.log_step_start("Testing document combinations")
        
        # Gold + Random
        gold_random = self._combine_docs(
            cluster_results['clusters'], 
            cluster_results['clusters'],
            mode='gold_random'
        )
        gold_random_eval = self.evaluator.evaluate_doc_combination(gold_random)
        
        # Gold + Distracting  
        gold_distracting = self._combine_docs(
            cluster_results['clusters'],
            cluster_results['clusters'], 
            mode='gold_distracting'
        )
        gold_distracting_eval = self.evaluator.evaluate_doc_combination(gold_distracting)
        
        # Random + Distracting
        random_distracting = self._combine_docs(
            cluster_results['clusters'],
            cluster_results['clusters'],
            mode='random_distracting' 
        )
        random_distracting_eval = self.evaluator.evaluate_doc_combination(random_distracting)
        
        combination_results = {
            'gold_random': gold_random_eval,
            'gold_distracting': gold_distracting_eval,
            'random_distracting': random_distracting_eval
        }
        
        self.logger.log_step_end("Document combination testing")
        return combination_results

    def _save_results(self, results: List[Dict]):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for phase in ['clustering', 'positions', 'noise', 'combinations']:
            phase_results = {
                config['n_clusters']: result[phase]
                for result in results
                for config in [result['config']]
            }
            
            output_path = output_dir / f"{phase}_results.json"
            self.utils.save_json(phase_results, output_path)

    def _position_documents(
        self, 
        clusters: Dict[int, List[int]], 
        position: str
    ) -> Dict[int, List[int]]:
        """Rearrange documents based on specified position."""
        positioned_clusters = {}
        for cluster_id, doc_ids in clusters.items():
            gold_idx = self._find_gold_document(doc_ids)
            if gold_idx is None:
                positioned_clusters[cluster_id] = doc_ids
                continue
                
            new_doc_ids = doc_ids.copy()
            gold_doc = new_doc_ids.pop(gold_idx)
            
            if position == 'near':
                new_doc_ids.insert(0, gold_doc)
            elif position == 'mid':
                mid = len(new_doc_ids) // 2
                new_doc_ids.insert(mid, gold_doc)
            else:  # 'far'
                new_doc_ids.append(gold_doc)
                
            positioned_clusters[cluster_id] = new_doc_ids
            
        return positioned_clusters

    def _find_gold_document(self, doc_ids: List[int]) -> Optional[int]:
        """Find index of document containing answer."""
        for idx, doc_id in enumerate(doc_ids):
            doc = self.corpus[doc_id]
            if any(answer in doc['text'] for answer in self.gold_answers):
                return idx
        return None

    def _inject_noise(
        self, 
        clusters: Dict[int, List[int]], 
        ratio: float
    ) -> Dict[int, List[int]]:
        """Inject random documents as noise."""
        noisy_clusters = {}
        for cluster_id, doc_ids in clusters.items():
            num_noise = max(1, int(len(doc_ids) * ratio))
            noise_ids = self.utils.sample_noise_docs(num_noise, exclude=doc_ids)
            noisy_clusters[cluster_id] = doc_ids + noise_ids
        return noisy_clusters

    def _combine_docs(
        self,
        clusters1: Dict[int, List[int]],
        clusters2: Dict[int, List[int]],
        mode: str
    ) -> Dict[int, List[int]]:
        """Combine documents from two clusterings."""
        combined_clusters = {}
        for c1, docs1 in clusters1.items():
            for c2, docs2 in clusters2.items():
                cluster_id = f"{c1}_{c2}"
                
                if mode == 'gold_random':
                    gold_docs = [d for d in docs1 if self._is_gold_doc(d)]
                    random_docs = self.utils.sample_docs(len(docs2), exclude=docs1)
                    combined_clusters[cluster_id] = gold_docs + random_docs
                elif mode == 'gold_distracting':
                    gold_docs = [d for d in docs1 if self._is_gold_doc(d)]
                    distracting_docs = [d for d in docs2 if not self._is_gold_doc(d)]
                    combined_clusters[cluster_id] = gold_docs + distracting_docs
                else:  #'random_distracting'
                    random_docs = self.utils.sample_docs(len(docs1), exclude=docs1)
                    distracting_docs = [d for d in docs2 if not self._is_gold_doc(d)]
                    combined_clusters[cluster_id] = random_docs + distracting_docs
                        
        return combined_clusters
                     
    def _is_gold_doc(self, doc_id: int) -> bool:
        """Check if document contains answer."""
        return any(answer in self.corpus[doc_id]['text'] for answer in self.gold_answers)

def main():
    config = ClusteringConfig()
    experiment = ClusteringExperiment(config, "clustering_experiment")
    results = experiment.run_clustering_phase()
    return results

if __name__ == "__main__":
    seed_everything(10)
    main()