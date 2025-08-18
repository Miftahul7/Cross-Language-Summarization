"""Aggregate multidomain ROUGE-L report across languages and domains (mBART/mT5).

This script scans a folder of per-example prediction CSVs (3 per language per model in this project), computes average ROUGE-L F1 (%) per
(domain, language), per-language macro averages, and a global overall average,
then writes a single JSON summary.

It reads:
- A directory `ogpath` containing CSV files. Non-CSV (e.g., JSON) files are ignored.
- Each CSV is expected to include at least the following columns:
    - `domain`   : str — the domain/category label (e.g., "books", "films")
    - `rouge`    : str — a JSON-like string containing ROUGE scores for the row

How language is inferred:
- The language code is derived from each filename by `f.split('_')[2]`.
  Ensure filenames follow a convention where the third underscore-separated
  token is the language code (e.g., `something_something_en_XX_something.csv`).
  If your filenames differ, adjust the indexing logic accordingly.

It computes:
For each CSV file:
  1) Group rows by `domain`.
  2) For each (domain, row), parse `rouge` → extract ROUGE-L F1, multiply by 100.
  3) Store the per-domain average for that file's inferred language.
  4) Compute a per-language average across all its domains.
Finally:
  - Compute a global overall average across all languages and domains.
"""
from tqdm import tqdm
from icecream import ic
import pandas as pd
import os
import ast
import json
# Folder containing prediction CSVs (non-CSV files are skipped)
ogpath = 'abstractive/multidomain/predictions/mbart_predictions/'  # Adjust to your  folder
files = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

res = {}
overall_rl = []

for f in tqdm(files, desc='Processing files'):
    if not f.endswith('.json'):
        df = pd.read_csv(f'{ogpath}{f}')
        ic(f)

        grouped_df = df.groupby('domain')
        # Infer language code from filename convention: split by '_' and take index 2
        lang = f.split('_')[2]
        res[lang] = {}

        rl_lang = []
        # Per-domain aggregation within this file/language
        for domain, group in grouped_df:
            res[lang][domain] = {}
            rl_temp = []

            for i, row in group.iterrows():
                try:
                    # Parse JSON-like ROUGE cell (expects dict with ['rouge-l']['f'])
                    rouge_dict = ast.literal_eval(row['rouge'])
                    rl_score = 100 * rouge_dict['rouge-l']['f']
                    rl_temp.append(rl_score)
                except Exception as e:
                    ic(f'Error in file {f}, row {i}: {e}')
                    continue
            # Domain average (percent) or None if no valid rows
            if rl_temp:
                res[lang][domain]['rouge-l'] = sum(rl_temp) / len(rl_temp)
                rl_lang.extend(rl_temp)
            else:
                res[lang][domain]['rouge-l'] = None  
        # Per-language macro average across domains
        if rl_lang:
            res[lang]['average'] = {'rouge-l': sum(rl_lang) / len(rl_lang)}
            overall_rl.extend(rl_lang)
        else:
            res[lang]['average'] = {'rouge-l': None}

# Compute overall average if available
res['overall'] = {
    'rouge-l': sum(overall_rl) / len(overall_rl) if overall_rl else None
}

# Save result to JSON
output_path = 'abstractive/multidomain/predictions/mbart_predictions/mD_mbart_rouge_c-20.json'
with open(output_path, 'w') as fp:
    json.dump(res, fp, indent=2)

