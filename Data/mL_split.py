#This creates split for Multilingual task for each domain. we first split each domain,language pair into train,test,split. then merge all train sets for each domain and shuffle. repeats for test and val. 
import os
import json
import random
from glob import glob

def read_and_annotate(filepath, tgt_lang):
    """Load a JSONL file and add src/tgt language."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            obj['src_lang'] = 'en'
            obj['tgt_lang'] = tgt_lang
            data.append(obj)
    return data

def merge_per_domain(input_root, output_root, langs, domains, splits, seed=42):
    os.makedirs(output_root, exist_ok=True)
    random.seed(seed)

    for domain in domains:
        for split in splits:
            merged = []

            for lang in langs:
                filename = f"{lang}_{domain}_{split}.json"
                path = os.path.join(input_root, lang, 'perDomainperLang', filename)

                if not os.path.exists(path):
                    print(f"Skipping missing: {path}")
                    continue

                annotated = read_and_annotate(path, lang)
                merged.extend(annotated)

            # Shuffle merged set
            random.shuffle(merged)

            # Save to output
            output_path = os.path.join(output_root, f"{domain}_{split}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in merged:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"Merged {len(merged)} items into: {output_path}")

# Example usage
if __name__ == "__main__":
    input_root = 'Data/md_split'
    output_root = 'Data/mL_split'
    langs = ['bn', 'hi', 'en']
    domains = ['books', 'films', 'sportsman', 'writers']
    splits = ['train', 'val', 'test']

    merge_per_domain(input_root, output_root, langs, domains, splits, seed=42)
