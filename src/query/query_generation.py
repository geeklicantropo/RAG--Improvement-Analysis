import logging
from tqdm import tqdm

class QueryGenerator:
    def __init__(self, model):
        self.model = model
        logging.info("QueryGenerator initialized.")

    def generate_query(self, input_text):
        logging.info("Generating query...")
        # Implement query generation logic using the model
        query = self.model.generate_query(input_text)
        logging.info("Query generation complete.")
        return query

    def batch_generate(self, inputs):
        logging.info("Generating queries in batch...")
        queries = []
        for text in tqdm(inputs, desc="Generating Queries"):
            queries.append(self.generate_query(text))
        logging.info("Batch query generation complete.")
        return queries
