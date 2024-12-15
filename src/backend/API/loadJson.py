import os
import json

def load_json(path):
    # Define the file path
    file_path = os.path.join("test",path)

    try:
        # Open and load the JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return []
    
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {file_path}")
        return []
