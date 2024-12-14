import logging
from typing import Dict, List, Any
from tqdm import tqdm

from src.llm_evaluator import LLMEvaluator

class DocumentClassifier:
    def __init__(self, llm_evaluator: LLMEvaluator):
        """
        Initialize the DocumentClassifier with an LLMEvaluator instance.
        The LLMEvaluator handles all LLM-based judgments for categorization,
        gold validation, and semantic similarity.

        Args:
            llm_evaluator (LLMEvaluator): The evaluator instance used for all LLM-based checks.
        """
        self.llm_evaluator = llm_evaluator
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
        Classify documents into gold, distracting, and random categories using LLM-based evaluation.

        For each document, the LLMEvaluator determines if it's 'gold', 'distracting', or 'random'.

        Args:
            documents (List[Dict]): A list of documents, each a dict with at least 'text' key.
            query (str): The user question.
            gold_answer (str): The known correct answer to the question.

        Returns:
            Dict[str, List[Dict]]: A dictionary with keys 'gold', 'distracting', and 'random',
            each containing the corresponding documents with added 'category' and 'position' fields.
        """
        self.logger.info("Starting document classification")
        categories = {
            "gold": [],
            "distracting": [],
            "random": []
        }

        for doc in tqdm(documents, desc="Classifying documents"):
            doc_text = doc.get('text', '')
            category = self.llm_evaluator.classify_document(query, gold_answer, doc_text)

            if category in categories:
                doc["category"] = category
                doc["position"] = len(categories[category])
                categories[category].append(doc)
            else:
                # In case the LLM returns something unexpected, default to 'random'
                self.logger.warning(f"Unknown category '{category}' returned by LLM, defaulting to 'random'")
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
        """
        Validate documents classified as gold using LLM-based checks.

        This method uses the LLMEvaluator to confirm if a gold document truly supports the gold answer.

        Args:
            documents (List[Dict]): A list of gold-labeled documents.
            query (str): The user question.
            gold_answer (str): The known correct answer.

        Returns:
            List[Dict]: The subset of documents that the LLM confirms as truly supportive of the gold answer.
        """
        self.logger.info("Validating gold documents")
        validated_docs = []

        for doc in tqdm(documents, desc="Validating gold documents"):
            doc_text = doc.get('text', '')
            is_valid = self.llm_evaluator.validate_gold_document(query, gold_answer, doc_text)
            if is_valid:
                validated_docs.append(doc)

        self.logger.info(f"Validated {len(validated_docs)} gold documents")
        return validated_docs

    def compute_semantic_similarity(
        self,
        doc1: Dict,
        doc2: Dict
    ) -> float:
        """
        Compute semantic similarity between two documents using LLM-based evaluation.

        Args:
            doc1 (Dict): The first document, must have 'text' key.
            doc2 (Dict): The second document, must have 'text' key.

        Returns:
            float: A similarity score between 0.0 and 1.0.
        """
        doc_text1 = doc1.get('text', '')
        doc_text2 = doc2.get('text', '')

        similarity = self.llm_evaluator.compute_semantic_similarity(doc_text1, doc_text2)
        return similarity
    
    def evaluate_generation(self, prompt: str, response: str, gold_answer: str) -> Dict[str, Any]:
        try:
            eval_prompt = f"""
            Question: {prompt}
            Generated Answer: {response}
            Gold Answer: {gold_answer}
            
            Classify as exactly one of: gold, distracting, random
            """
            
            response = self.model.generate_content(eval_prompt)
            category = response.text.strip().lower()
            
            # Clean up category
            if 'gold' in category:
                return 'gold'
            elif 'distracting' in category:
                return 'distracting'
            return 'random'
                
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return 'random'
