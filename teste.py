import os
import json
from pathlib import Path
import logging
import pickle

data_folder = "data"
# Function to check for example_ids 
def validate_example_id_in_jsons(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        missing_ids = any("example_id" not in entry for entry in data)
        if missing_ids:
            print(f"File has missing (JSON) 'example_id' : {file_path}")

    except json.JSONDecodeError as e:
        logging.error(f"Error reading file JSON {file_path}: {e}")

def validate_example_id_in_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        missing_ids = any("example_id" not in entry for entry in data)
        if missing_ids:
            print(f"File has missing (Pickle) 'example_id' : {file_path}")
    except pickle.UnpicklingError as e:
        print(f"Error reading file Pickle {file_path}")

for root, dirs, files in os.walk(data_folder):
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith(".json") or file.endswith(".JSON"):
            file_path = os.path.join(root, file)
            print(f"Checking JSON file: {file_path}")
            validate_example_id_in_jsons(file_path)
        elif file.endswith(".pkl") or file.endswith(".PKL"):
            file_path = os.path.join(root, file)
            print(f"Checking Pickle file: {file_path}")
            validate_example_id_in_pickle(file_path)
