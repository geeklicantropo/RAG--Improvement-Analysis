import logging
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import google.generativeai as genai
import os
import time

class DocumentCombinationValidator:
    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()
        
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DocumentCombinationValidator")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def validate_combinations(
        self,
        documents: List[Dict],
        gold_docs: List[Dict],
        combination_config: Dict
    ) -> Dict[str, bool]:
        self.logger.info("Starting document combination validation")
        
        validations = {}
        progress_bar = tqdm(total=4, desc="Validation Steps")
        
        # Track document positions and categories
        validations["position_valid"] = self._validate_positions(documents, gold_docs)
        progress_bar.update(1)
        
        validations["category_distribution"] = self._validate_category_distribution(
            documents, combination_config
        )
        progress_bar.update(1)
        
        validations["statistical_significance"] = self._validate_significance(documents)
        progress_bar.update(1)
        
        validations["experimental_conditions"] = self._validate_conditions(
            documents, combination_config
        )
        progress_bar.update(1)
        
        progress_bar.close()
        self.logger.info("Document combination validation completed")
        return validations
        
    def _validate_positions(self, documents: List[Dict], gold_docs: List[Dict]) -> bool:
        self.logger.info("Validating document positions")
        with tqdm(total=len(documents), desc="Position validation") as pbar:
            try:
                gold_positions = []
                for doc in documents:
                    if doc["id"] in [g["id"] for g in gold_docs]:
                        gold_positions.append(doc["position"])
                    pbar.update(1)
                    
                position_stats = {
                    "near": len([p for p in gold_positions if p < len(documents) // 3]),
                    "mid": len([p for p in gold_positions if len(documents) // 3 <= p < 2 * len(documents) // 3]),
                    "far": len([p for p in gold_positions if p >= 2 * len(documents) // 3])
                }
                
                self.logger.info(f"Position distribution: {position_stats}")
                return all(count > 0 for count in position_stats.values())
                
            except Exception as e:
                self.logger.error(f"Position validation failed: {str(e)}")
                return False

    def _validate_category_distribution(
        self,
        documents: List[Dict],
        config: Dict
    ) -> bool:
        self.logger.info("Validating category distribution")
        with tqdm(total=len(documents), desc="Category validation") as pbar:
            try:
                categories = {"gold": 0, "distracting": 0, "random": 0}
                
                for doc in documents:
                    # Use LLM to validate category assignments
                    if doc["category"] == "distracting":
                        relevance = self.evaluate_document_relevance(doc, config["gold_document"])
                        if relevance < 0.3:  # Low relevance suggests wrong categorization
                            self.logger.warning(f"Document {doc['id']} may be miscategorized as distracting")
                            
                    categories[doc["category"]] += 1
                    pbar.update(1)
                    
                expected_ratios = config.get("category_ratios", {
                    "gold": 0.2,
                    "distracting": 0.4,
                    "random": 0.4
                })
                
                actual_ratios = {k: v/len(documents) for k, v in categories.items()}
                self.logger.info(f"Category distribution: {actual_ratios}")
                
                return all(
                    abs(actual_ratios[k] - expected_ratios[k]) < 0.1 
                    for k in categories
                )
                
            except Exception as e:
                self.logger.error(f"Category distribution validation failed: {str(e)}")
                return False

    def _validate_significance(self, documents: List[Dict]) -> bool:
        self.logger.info("Validating statistical significance")
        try:
            # Minimum sample size check
            if len(documents) < 100:
                return False
                
            # Category balance check
            categories = [doc["category"] for doc in documents]
            unique_categories = set(categories)
            
            category_counts = {
                cat: categories.count(cat) for cat in unique_categories
            }
            
            min_category_size = min(category_counts.values())
            return min_category_size >= 30  # Minimum for statistical significance
            
        except Exception as e:
            self.logger.error(f"Significance validation failed: {str(e)}")
            return False

    def _validate_conditions(self, documents: List[Dict], config: Dict) -> bool:
        self.logger.info("Validating experimental conditions")
        try:
            required_conditions = [
                self._validate_document_length(documents, config),
                self._validate_position_balance(documents),
                self._validate_category_exclusivity(documents)
            ]
            
            return all(required_conditions)
            
        except Exception as e:
            self.logger.error(f"Conditions validation failed: {str(e)}")
            return False
            
    def _validate_document_length(self, documents: List[Dict], config: Dict) -> bool:
        max_length = config.get("max_document_length", 512)
        return all(len(doc["text"]) <= max_length for doc in documents)
        
    def _validate_position_balance(self, documents: List[Dict]) -> bool:
        positions = [doc["position"] for doc in documents]
        position_counts = {p: positions.count(p) for p in set(positions)}
        return max(position_counts.values()) <= 2 * min(position_counts.values())
        
    def _validate_category_exclusivity(self, documents: List[Dict]) -> bool:
        doc_ids = [doc["id"] for doc in documents]
        return len(doc_ids) == len(set(doc_ids))