"""This code is a modified version provided at https://github.com/DhavalTaunk08/XWikiGen/tree/main"""
"""Multilingual–Multidomain Abstractive Summarization Dataloaders.

This module provides utilities to load and tokenize a multilingual,
multi-domain abstractive summarization corpus stored as JSON Lines.
It supports encoder–decoder models such as mBART and mT5 via Hugging Face Transformers.

Components:
- Dataset1:
    A torch.utils.data.Dataset that reads JSONL examples, builds source/target
    texts, and tokenizes them. It also handles language codes for mBART and
    loss masking for mT5 (PAD → -100).
- DataModule:
    A PyTorch Lightning DataModule that instantiates train/val/test datasets
    and returns DataLoaders.
    
Language handling notes: 
- mBART: We set `tokenizer.tgt_lang` and return `tgt_lang` so that model can
  configure `forced_bos_token_id` accordingly.
- mT5: No special language tokens are required; we set label PAD tokens to -100
  so they are ignored by the loss.
"""
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import json
import torch

class Dataset1(Dataset):
    """JSONL-backed dataset for multilingual-multidomain summarization.
    This dataset builds the source text by concatenating `page_title`,
    `section_title`, and the joined `references`. The target text is the
    `content` field. Texts are tokenized to fixed lengths suitable for
    encoder–decoder models.
    """
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length, is_mt5):
        with open(data_path, 'r') as fp:
            self.df = [json.loads(line, strict=False) for line in fp.readlines()]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_mt5 = is_mt5
        self.languages_map = {
            'bn': 'bn_IN',
            'en': 'en_XX',
            'hi': 'hi_IN',
        }

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Tokenize and return a single training example.

        Builds the source text as:
            "page_title section_title <joined references>"
        Sets `tokenizer.tgt_lang` (for mBART) using mapped language codes.
        For mT5, replaces PAD ids in labels with -100 to mask them in the loss.

        Notes:
            - For mBART, callers typically use 'tgt_lang' to set
              `forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]`.
        """
        row = self.df[idx]
        input_text = ' '.join(row['references'])
        input_text = f"{row['page_title']} {row['section_title']} {input_text}"
        target_text = row['content']

        # Language IDs
        src_code = self.languages_map.get(row.get('src_lang', 'en'), 'en_XX')
        tgt_code = self.languages_map.get(row.get('tgt_lang', 'en'), 'en_XX')

        # Set tokenizer target language (only relevant for mBART)
        self.tokenizer.tgt_lang = tgt_code

        input_encoding = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(
                target_text,
                return_tensors='pt',
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True
            )

        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()

        if self.is_mt5:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'src_lang': src_code,
            'tgt_lang': tgt_code,  # required for forced_bos_token_id
            'domain': row.get('domain', 'unknown')
            }
  
class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for multilingual-multidomain summarization. It instantiates a single tokenizer and three Dataset1 splits, and
    provides DataLoaders for training/validation/testing/prediction."""
    def __init__(self, *args, **kwargs):
        """Save hyperparameters and initialize the tokenizer."""
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)

    def setup(self, stage=None):
        """Create train/val/test datasets."""
        self.train = Dataset1(
            self.hparams.train_path, self.tokenizer,
            self.hparams.max_source_length, self.hparams.max_target_length,
            self.hparams.is_mt5
        )
        self.val = Dataset1(
            self.hparams.val_path, self.tokenizer,
            self.hparams.max_source_length, self.hparams.max_target_length,
            self.hparams.is_mt5
        )
        self.test = Dataset1(
            self.hparams.test_path, self.tokenizer,
            self.hparams.max_source_length, self.hparams.max_target_length,
            self.hparams.is_mt5
        )

    def train_dataloader(self):
        """Return the training DataLoader.

        Returns:
            torch.utils.data.DataLoader: Shuffled training loader.
        """
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        """Return the validation DataLoader.

        Returns:
            torch.utils.data.DataLoader: Non-shuffled validation loader.
        """
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=1, shuffle=False)

    def test_dataloader(self):
        """Return the test DataLoader.

        Returns:
            torch.utils.data.DataLoader: Non-shuffled test loader.
        """
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=1, shuffle=False)

    def predict_dataloader(self):
        """Return the prediction DataLoader.
        Notes:
            Uses the test split for prediction by default.
        Returns:
            torch.utils.data.DataLoader: Non-shuffled prediction loader.
        """
        return self.test_dataloader()

