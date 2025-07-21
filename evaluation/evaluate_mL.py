from rouge import Rouge
import json
import os
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from icecream import ic
import json

ogpath = 'abstractive/multilingual/predictions/mbart_predictions'
files = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

rouge = Rouge()
overall_r = []
res = {}

for f in tqdm(files, desc='{going through files}'):
    if any(x in f for x in ['books', 'films', 'sportsman', 'writers']):
        filepath = os.path.join(ogpath, f)
        df = pd.read_csv(filepath)
        ic(f)

        # Extract domain name
        parts = f.split('_')
        domain = parts[2] if len(parts) > 2 else 'unknown'
        res[domain] = {}

        r_lang = []

        for lang in tqdm(['bn_IN', 'en_XX', 'hi_IN'], desc='langs'):
            temp = df[df['tgt_lang'] == lang]
            r_temp = []

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

        if r_lang:
            avg_domain_score = sum(r_lang) / len(r_lang)
            res[domain]['average'] = avg_domain_score
            ic(domain)
            ic('average')
            ic(avg_domain_score)
            overall_r.extend(r_lang)

# Save overall score
if overall_r:
    overall_avg = sum(overall_r) / len(overall_r)
    res['overall'] = overall_avg
    ic('overall average')
    ic(overall_avg)
else:
    res['overall'] = None
    ic('No valid ROUGE scores collected.')

# Save result to JSON
output_path = 'abstractive/multilingual/predictions/mt5_predictions/salience_mbart_Rouge_C20.json'
with open(output_path, 'w') as fp:
    json.dump(res, fp, indent=2)
print(f"Saved ROUGE summary to {output_path}")
