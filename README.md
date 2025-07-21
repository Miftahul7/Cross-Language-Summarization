# Cross-Language-Summarization
## Repository Structure

### Root Files
- `README.md` – This file.
- `requirements.txt` – List of required Python libraries for this project.

---

## Data: Two Subsets of XWikiRef Dataset from https://github.com/DhavalTaunk08/XWikiGen/tree/main/XWikiRef

### `Data/Initial-20`
- Raw dataset, organized by language folders: `bn`, `en`, `hi`.
- Each language folder contains JSON files for:
  - `films`, `sportsman`, `books`, `writers`.

### `Data/Curated-20`
- Raw dataset, organized by language folders: `bn`, `en`, `hi`.
- Each language folder contains JSON files for:
  - `films`, `sportsman`, `books`, `writers`.
- Organized similarly by language and domain.
- `clean.py` script is used to preprocess raw data.
- Preprocessed data is saved inside the cleaned folder of each language (e.g., `Data/bn/cleaned/bn_writers.json`). 

### Scripts
- `clean.py` – Preprocesses each (language, domain) JSON file (e.g., `bn_writers.json`).
- `mD_split.py` – Splits data for Multidomain experiments (80:10:10). For a particular language, it takes each domain as input and splits it in an 80:10:10 ratio and saves it in the perDomainperLang folder. Then, it merges and shuffles all the domain’s training sets to create a training set (e.g., bn_train.json). A similar approach is used to get validation sets and test sets. This makes sure every domain is present in all the splits.
- `mL_split.py` – Splits data for Multilingual experiments. From the perDomainperLang folder, it merges the train sets of all languages for each domain to create a single train set (e.g., writers_train.json). A similar approach is used to get validation sets and test sets.
- `mLmD_split.py` – Aplits data for ML-MD experiments. It utilizes splits from the Multilingual task, merging the training sets of all domains to create a comprehensive training set comprising 12 domain-language pairs. A similar approach is used to get validation sets and test sets.

---

## Extractive Stage

### Folder: `extractive/`
- `extractive.py` / `extractive.sh` – Extracts top-k salient sentences from cleaned language-domain data.

---

## Abstractive Stage

Each experimental setting has its folder inside `abstractive/`.

### 1. `Multidomain/`
- `model/dataloader.py` – PyTorch Lightning `DataModule` for loading and tokenizing data.
- `model/model.py` – Summarization model using mBART or mT5.
- `train_mD.{py,sh}` – Training scripts.
- `testing/testing.{py, sh}` – loads trained mBart/mT5 model checkpoints, evaluates the summarization against the gold summary. Output a CSV file with the results, including the Rouge score.

### 2. `Multilingual/`
- `model/dataloader.py` – PyTorch Lightning `DataModule` for loading and tokenizing data.
- `model/model.py` – Summarization model using mBART or mT5.
- `testing/testing_mbart.{py, sh}` – loads trained mBart model checkpoints, evaluates the mT5 summarization against the gold summary. Output a CSV file with the results, including the Rouge score.
- `testing/testing_mt5.{py, sh}` – loads trained mT5 model checkpoints, evaluates the mT5 summarization against the gold summary. Output a CSV file with the results, including the Rouge score.

### 3. `ML-MD/` (Multilingual-Multidomain)
- `model/dataloader.py` – PyTorch Lightning `DataModule` for loading and tokenizing data.
- `model/model.py` – Summarization model using mBART or mT5.
- `train_mLmd.{py, sh}` – Training scripts.
- `testing/testing_mbart.{py, sh}` – loads trained mBart model checkpoints, evaluates the mT5 summarization against the gold summary. Output a CSV file with the results, including the Rouge score.
- `testing/testing_mt5.{py, sh}` – loads trained mT5 model checkpoints, evaluates the mT5 summarization against the gold summary. Output a CSV file with the results, including the Rouge score.

---

## Evaluation

Folder: `evaluation/`

Scripts that summarize ROUGE-L F1 scores for each experiment and model:
- `evaluate_mD.py` – For Multidomain setting.
- `evaluate_mL.py` – For Multilingual setting.
- `evaluate_mLmD.py` – For ML-MD setting.

Each script takes CSV predictions from testing and produces a domain/language-wise JSON report.

---

## Requirements

All dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
