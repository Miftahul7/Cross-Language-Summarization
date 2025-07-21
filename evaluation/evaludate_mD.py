from tqdm import tqdm
from icecream import ic
import pandas as pd
import os
import ast
import json

ogpath = 'abstractive/multidomain/predictions/mbart_predictions/'  # Adjust to your  folder
files = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

res = {}
overall_rl = []

for f in tqdm(files, desc='Processing files'):
    if not f.endswith('.json'):
        df = pd.read_csv(f'{ogpath}{f}')
        ic(f)

        grouped_df = df.groupby('domain')
        lang = f.split('_')[2]
        res[lang] = {}

        rl_lang = []

        for domain, group in grouped_df:
            res[lang][domain] = {}
            rl_temp = []

            for i, row in group.iterrows():
                try:
                    rouge_dict = ast.literal_eval(row['rouge'])
                    rl_score = 100 * rouge_dict['rouge-l']['f']
                    rl_temp.append(rl_score)
                except Exception as e:
                    ic(f'Error in file {f}, row {i}: {e}')
                    continue

            if rl_temp:
                res[lang][domain]['rouge-l'] = sum(rl_temp) / len(rl_temp)
                rl_lang.extend(rl_temp)
            else:
                res[lang][domain]['rouge-l'] = None  

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

