import json
import pickle
import random
import os

def sample_large_file(file_path, sample_size=10):
    """
    Extracts a random sample of entries from a large JSON or PKL file.
    
    Parameters:
    - file_path (str): Path to the file (JSON or PKL).
    - sample_size (int): Number of random entries to extract.
    
    Returns:
    - list or dict: A sample of entries from the file.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    sample = None

    if file_extension == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):  # JSON is an array of objects
                sample = random.sample(data, min(sample_size, len(data)))
            elif isinstance(data, dict):  # JSON is a dictionary
                keys = random.sample(list(data.keys()), min(sample_size, len(data)))
                sample = {key: data[key] for key in keys}
            else:
                print("Unexpected JSON structure. Could not sample.")
    elif file_extension == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):  # PKL is a list
                sample = random.sample(data, min(sample_size, len(data)))
            elif isinstance(data, dict):  # PKL is a dictionary
                keys = random.sample(list(data.keys()), min(sample_size, len(data)))
                sample = {key: data[key] for key in keys}
            else:
                print("Unexpected PKL structure. Could not sample.")
    else:
        print(f"Unsupported file format: {file_extension}")
    
    return sample

def display_sample(file_path, sample_size=10):
    """
    Displays a random sample of entries from a JSON or PKL file.
    
    Parameters:
    - file_path (str): Path to the file.
    - sample_size (int): Number of entries to sample and display.
    """
    sample = sample_large_file(file_path, sample_size)
    if sample is not None:
        #print(f"Sampled {sample_size} entries from {file_path}:")
        print(f"Sampled 1 entry from {file_path}:")
        if isinstance(sample, list):
            for i, entry in enumerate(sample, start=1):
                print(f"\nSample {i}:")
                print(json.dumps(entry, indent=4) if isinstance(entry, dict) else entry)
        elif isinstance(sample, dict):
            for i, (key, value) in enumerate(sample.items(), start=1):
                print(f"\nSample {i}:")
                print(f"Key: {key}")
                print(json.dumps(value, indent=4) if isinstance(value, dict) else value)
        else:
            print("Unexpected data structure in sample.")
    else:
        print("No sample could be extracted.")

# Replace this with the path to your file (JSON or PKL)
file_path = 'test_dataset.json'

# Adjust the sample size as needed
display_sample(file_path, sample_size=1)
