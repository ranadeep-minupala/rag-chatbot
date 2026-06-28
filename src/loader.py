import os
import json

def load(data_dir):
    documents = []
    business_info = {}

    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)

        if filename.endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                text = f.read()                
            documents.append({"text": text, "source": filename})

        elif filename.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                business_info = json.load(f)          

    return documents, business_info