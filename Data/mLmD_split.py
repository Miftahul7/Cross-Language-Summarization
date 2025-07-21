#this script creates dataset for Multidomain-Multilingual experiemnt. We merge and shuffle train set from per domain Multilingual train sets to create train dataset. Similar approach goes for validation and test set. 
import os
import json
import random
from glob import glob

def load_jsonl(filepath):
    """Load a JSONL file into a list."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(data, filepath):
    """Save a list of dicts as JSONL."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def merge_all_domains_by_split(input_folder, output_folder, domains, splits, seed=42):
    os.makedirs(output_folder, exist_ok=True)
    random.seed(seed)

    for split in splits:
        merged = []
        for domain in domains:
            # Corrected path: includes subfolder named after the domain
            path = os.path.join(input_folder, domain, f"{domain}_{split}.json")
            if not os.path.exists(path):
                print(f" Missing: {path}")
                continue
            merged.extend(load_jsonl(path))

        random.shuffle(merged)

        out_path = os.path.join(output_folder, f"mLmD_{split}.json")
        write_jsonl(merged, out_path)
        print(f" Merged {len(merged)} items into {out_path}")

# 🛠️ Example usage
if __name__ == "__main__":
    input_folder = 'Data/mL_split'              
    output_folder = 'Data/combined_split'   
    domains = ['books', 'films', 'sportsman', 'writers']
    splits = ['train', 'val', 'test']

    merge_all_domains_by_split(input_folder, output_folder, domains, splits, seed=42)
