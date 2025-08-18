"""Aggregate ROUGE-L report of prediction CSVs produced by the (abstractive/ML-MD/testing/testing_{mbart/mt5}.py).

This utility loads the per-example predictions CSV produced by the
`testing_mbart.py` or `testing_mt5.py` evaluation scripts and builds a compact
JSON report of average ROUGE-L F1 scores:
  - per (domain, target-language) pair
  - per domain (average over its languages)
  - overall average across all selected domains and languages

It reads:
A CSV at `file_path` containing at least the following columns:
- `domain`     : str  — domain/category label (e.g., "books", "films")
- `tgt_lang`   : str  — target language code (e.g., "en_XX", "hi_IN", "bn_IN")
- `pred_text`  : str  — model prediction (one example per row)
- `ref_text`   : str  — reference/ground-truth summary
Optional columns (ignored by this script): `src_lang`, `input_text`, etc.

It computes:
For each selected domain in `domains` and target language in `languages`:
- Iterates example rows, computes ROUGE-L F1 (via `rouge.get_scores`),
  multiplies by 100 (percentage), and averages across rows.
- Stores per-(domain, language) averages and the per-domain average (across
  its languages). Also computes a global overall average (macro across all
  collected per-example scores). 
  """
import pandas as pd
import json
from tqdm import tqdm
from rouge import Rouge
from icecream import ic

# Define file path
file_path = 'abstractive/ML-MD/predictions/mt5_prediction/lang_wise_combined_mt5.csv' 

# Load data
df = pd.read_csv(file_path)
rouge = Rouge()

# Define target domains and languages
domains = ['writers', 'books', 'sportsman','films']
languages = ['bn_IN', 'en_XX', 'hi_IN']

results = {}
overall_rouge = []

for domain in tqdm(domains, desc='Domain'):
    domain_data = df[df['domain'] == domain]
    results[domain] = {}
    domain_scores = []

    for lang in tqdm(languages, desc=f'{domain} - Language', leave=False):
        lang_data = domain_data[domain_data['tgt_lang'] == lang]
        rouge_scores = []

        for _, row in lang_data.iterrows():
            try:
                p = row['pred_text']
                r = row['ref_text']
                score = 100 * rouge.get_scores(p, r)[0]['rouge-l']['f']
                rouge_scores.append(score)
            except:
                continue

        if rouge_scores:
            avg_score = sum(rouge_scores) / len(rouge_scores)
            results[domain][lang] = avg_score
            domain_scores.extend(rouge_scores)
            ic(domain, lang, avg_score)

    if domain_scores:
        results[domain]['average'] = sum(domain_scores) / len(domain_scores)
        overall_rouge.extend(domain_scores)

# Compute overall
results['overall'] = sum(overall_rouge) / len(overall_rouge) if overall_rouge else None
ic('Overall Average ROUGE-L:', results['overall'])

# Save result
output_path = 'abstractive/ML-MD/predictions/mbart_prediction/rouge_mt5_mlmd_C20.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved results to: {output_path}")
