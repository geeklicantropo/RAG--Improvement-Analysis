import os
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from pathlib import Path

from src.llm import LLM
from src.experiment_logger import ExperimentLogger

class DocumentClassifier:
    """Classifies and validates documents for RAG experiments."""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DocumentClassifier")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def classify_documents(
        self,
        documents: List[Dict],
        query: str,
        gold_answer: str,
    ) -> Dict[str, List[Dict]]:
        """
        Classify documents into gold, distracting, and random categories.
        
        Args:
            documents: List of documents to classify
            query: Original query
            gold_answer: Gold standard answer
            
        Returns:
            Dictionary mapping categories to document lists
        """
        self.logger.info("Starting document classification")
        categories = {
            "gold": [],
            "distracting": [],
            "random": []
        }

        for doc in tqdm(documents, desc="Classifying documents"):
            # Evaluate document relevance
            eval_prompt = f"""
            Question: {query}
            Gold Answer: {gold_answer}
            Document: {doc['text']}
            
            Classify this document as either:
            1. "gold" if it contains the answer
            2. "distracting" if semantically similar but doesn't contain answer
            3. "random" if unrelated
            
            Classification:"""
            
            category = self.llm.generate(eval_prompt).strip().lower()
            
            if category in categories:
                doc["category"] = category
                doc["position"] = len(categories[category])
                categories[category].append(doc)
            else:
                self.logger.warning(f"Unknown category {category}, defaulting to random")
                doc["category"] = "random"
                doc["position"] = len(categories["random"])
                categories["random"].append(doc)

        self.logger.info(f"Classified {len(documents)} documents")
        for cat, docs in categories.items():
            self.logger.info(f"{cat}: {len(docs)} documents")
            
        return categories

    def validate_gold_documents(
        self,
        documents: List[Dict],
        query: str,
        gold_answer: str
    ) -> List[Dict]:
        """Validate documents classified as gold."""
        self.logger.info("Validating gold documents")
        validated_docs = []
        
        for doc in tqdm(documents, desc="Validating gold documents"):
            validation_prompt = f"""
            Question: {query}
            Expected Answer: {gold_answer}
            Document: {doc['text']}
            
            Does this document contain enough information to answer the question correctly?
            Answer yes/no and explain why:"""
            
            validation = self.llm.generate(validation_prompt).strip().lower()
            
            if validation.startswith("yes"):
                validated_docs.append(doc)
                
        self.logger.info(f"Validated {len(validated_docs)} gold documents")
        return validated_docs

    def compute_semantic_similarity(
        self,
        doc1: Dict,
        doc2: Dict
    ) -> float:
        """Compute semantic similarity between documents using LLM."""
        similarity_prompt = f"""
        Document 1: {doc1['text']}
        Document 2: {doc2['text']}
        
        Rate the semantic similarity of these documents on a scale of 0-100
        where 0 is completely different and 100 is nearly identical.
        
        Similarity score:"""
        
        try:
            score = float(self.llm.generate(similarity_prompt).strip())
            return score / 100
        except ValueError:
            self.logger.warning("Could not parse similarity score")
            return 0.0