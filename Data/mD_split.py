import os
import json
import argparse
import random
from glob import glob

def split_data(data, seed):
    """Split data into 80/10/10 and return train/val/test."""
    random.seed(seed)
    random.shuffle(data)
    total = len(data)
    train_end = int(0.8 * total)
    val_end = train_end + int(0.1 * total)
    return data[:train_end], data[train_end:val_end], data[val_end:]

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

def main(input_folder, output_folder, seed):
    os.makedirs(output_folder, exist_ok=True)
    lang = os.path.basename(input_folder.rstrip("/"))

    json_files = glob(os.path.join(input_folder, '*.json'))
    train_all, val_all, test_all = [], [], []

    for file_path in json_files:
        filename = os.path.basename(file_path).replace('.json', '')
        try:
            file_lang, domain = filename.split('_', 1)
        except ValueError:
            print(f"Skipping malformed filename: {filename}")
            continue

        data = read_jsonl(file_path)

        for doc in data:
            doc['domain'] = domain  # Add domain key

        # Split into train/val/test
        train, val, test = split_data(data, seed)

        # Save intermediate per-domain splits
        write_jsonl(train, os.path.join(output_folder, f'{lang}_{domain}_train.json'))
        write_jsonl(val, os.path.join(output_folder, f'{lang}_{domain}_val.json'))
        write_jsonl(test, os.path.join(output_folder, f'{lang}_{domain}_test.json'))

        # Accumulate for merged output
        train_all.extend(train)
        val_all.extend(val)
        test_all.extend(test)

    # Shuffle merged sets
    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    # Save final merged splits
    write_jsonl(train_all, os.path.join(output_folder, f'{lang}_train.json'))
    write_jsonl(val_all, os.path.join(output_folder, f'{lang}_val.json'))
    write_jsonl(test_all, os.path.join(output_folder, f'{lang}_test.json'))

    print(f"\nAll splits saved in: {output_folder}")
    print(f" {lang}_train.json: {len(train_all)} samples")
    print(f" {lang}_val.json:   {len(val_all)} samples")
    print(f" {lang}_test.json:  {len(test_all)} samples\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and merge multilingual, multi-domain dataset.")
    parser.add_argument('--input_folder', required=True, help='Path to input JSON files (format: lang_domain.json)')
    parser.add_argument('--output_folder', required=True, help='Where to save output JSONL files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.seed)
