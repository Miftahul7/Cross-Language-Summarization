"""Aggregate Multilingual ROUGE-L F1 (percent) by domain and target language.

This script scans a directory of prediction CSVs produced by 
testing pipelines (abstractive/multilingual/testing/testing_{mbart/mt5}.py) and builds a JSON summary of ROUGE-L F1 averages for each model:

- Per domain & target language (e.g., writers → hi_IN).
- Per domain macro average (across the three languages).
- Overall macro average across all selected domains and languages.

Focus:
This script is intended for multilingual evaluation (cross-language within
each domain). It expects each CSV to contain per-example predictions and
references with target-language labels.

Inputs:
- `ogpath`: directory containing the prediction CSV files to aggregate.
- CSV schema (minimum expected columns):
    - `tgt_lang`   : str (e.g., "bn_IN", "en_XX", "hi_IN")
    - `pred_text`  : str (model prediction)
    - `ref_text`   : str (reference/ground-truth)
  Additional columns (e.g., `src_lang`, `input_text`, `domain`) may be present
  but are not required here, since the domain is inferred from the filename.

Domain inference:
- The domain is extracted from each filename with:
      `parts = f.split('_') ; domain = parts[2] if len(parts) > 2 else 'unknown'`
  This assumes your filenames include the domain as the third underscore-separated token. Adjust if your
  filename conventions differ.

Metric:
- Uses `rouge` package to compute ROUGE-L F1 per example, then multiplies
  by 100 to present results as percentages. Domain-level and overall averages
  are simple arithmetic means over the included examples.

Notes:
- Rows that raise an exception during ROUGE computation are skipped.
- If no valid scores are collected, the script writes `"overall": null`.
"""
from rouge import Rouge
import json
import os
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from icecream import ic
import json
# Directory containing multilingual prediction CSVs
ogpath = 'abstractive/multilingual/predictions/mbart_predictions'
files = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

rouge = Rouge()
overall_r = [] # collects all per-example RL scores across domains/langs
res = {} #final results dict

for f in tqdm(files, desc='{going through files}'):
    # Only consider files whose names indicate one of the target domains
    if any(x in f for x in ['books', 'films', 'sportsman', 'writers']):
        filepath = os.path.join(ogpath, f)
        df = pd.read_csv(filepath)
        ic(f)

        # # --- Infer domain from filename (3rd underscore-separated token) ---
        parts = f.split('_')
        domain = parts[2] if len(parts) > 2 else 'unknown'
        res[domain] = {}

        r_lang = [] # per-domain accumulator across languages
        # --- Aggregate by target language ---
        for lang in tqdm(['bn_IN', 'en_XX', 'hi_IN'], desc='langs'):
            temp = df[df['tgt_lang'] == lang]
            r_temp = [] # per-language accumulator (within this domain/file)

            for i, row in temp.iterrows():
                p = row.get('pred_text', '')
                r = row.get('ref_text', '')
                try:
                    score = rouge.get_scores(p, r)[0]['rouge-l']['f']
                    r_temp.append(100 * score)
                except:
                    continue

            if r_temp:
                avg_lang_score = sum(r_temp) / len(r_temp)
                ic(lang)
                ic(avg_lang_score)
                res[domain][lang] = avg_lang_score
                r_lang.extend(r_temp)
        # --- Domain-level average across languages ---
        if r_lang:
            avg_domain_score = sum(r_lang) / len(r_lang)
            res[domain]['average'] = avg_domain_score
            ic(domain)
            ic('average')
            ic(avg_domain_score)
            overall_r.extend(r_lang)

# # --- Overall macro average across all domains-languages ---
if overall_r:
    overall_avg = sum(overall_r) / len(overall_r)
    res['overall'] = overall_avg
    ic('overall average')
    ic(overall_avg)
else:
    res['overall'] = None
    ic('No valid ROUGE scores collected.')

# Save result to JSON
output_path = 'abstractive/multilingual/predictions/mbart_predictions/salience_mbart_Rouge_C20.json'
with open(output_path, 'w') as fp:
    json.dump(res, fp, indent=2)
print(f"Saved ROUGE summary to {output_path}")
