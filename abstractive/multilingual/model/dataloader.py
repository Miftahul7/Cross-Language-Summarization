"""This code is a modified version provided at https://github.com/DhavalTaunk08/XWikiGen/tree/main"""
"""Multilingual Abstractive Summarization Dataloaders.
This module defines utilities to load JSONL data and build PyTorch-Lightning
DataLoaders for multilingual abstractive summarization with seq2seq models such as
mBART and mT5.
Components:
- ``Dataset1``: A JSONL-backed ``torch.utils.data.Dataset`` that prepares source/target
  pairs, handles tokenizer calls (including mBART target language setting), and returns
  tensors for model consumption.
- ``DataModule``: A ``pytorch_lightning.LightningDataModule`` that owns a single shared
  tokenizer, instantiates ``Dataset1`` splits for train/val/test, and provides the
  corresponding DataLoaders.

Notes:
- For mBART model, ``tokenizer.tgt_lang`` is set using a minimal ISO-2 → mBART code map.
- For mT5 model, pad tokens in labels can be masked to ``-100`` (controlled by ``is_mt5``)
  so that they are ignored by the loss function.
"""

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import json
import torch

class Dataset1(Dataset):
    """JSONL-backed dataset for multilingual abstractive summarization.
    This dataset reads one JSON object per line and constructs source/target text pairs
    suitable for seq2seq summarization models. The source text is composed as:

        ``f"{page_title} {section_title} {' '.join(references)}"`` and the target text is the ``content`` field.
    Language handling:
        - For mBART: ``tokenizer.tgt_lang`` is set to the target language code mapped to
          an mBART language id (e.g., ``en_XX``). model can then use ``tgt_lang``
          to set ``forced_bos_token_id`` during training/inference.
        - For mT5: If ``is_mt5`` is ``True``, label pad tokens are replaced with ``-100``
          so they are ignored by the loss function.
    """
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length, is_mt5):
        """ Initialize the dataset.
        Args:
            data_path (str): Path to a JSONL file containing one example per line.
            tokenizer: A Hugging Face tokenizer compatible with the underlying model
                (e.g., mBART or mT5).
            max_source_length (int): Maximum sequence length for tokenized inputs;
                inputs are padded/truncated to this length.
            max_target_length (int): Maximum sequence length for tokenized targets;
                targets are padded/truncated to this length.
            is_mt5 (bool): If ``True``, pad tokens in ``labels`` are replaced with ``-100``
                (commonly used for T5-style models so the loss ignores pads).
        """
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
        """Return the number of examples in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Create and return a single tokenized training example.
        This method composes the source text from ``page_title``, ``section_title``, and
        the concatenated ``references`` list, sets the tokenizer target language (for
        mBART), tokenizes both source and target with padding/truncation to fixed lengths,
        and optionally masks target pad tokens with ``-100`` for mT5 training.

        Args:
            idx (int): Index of the example to fetch.
        Returns:
            dict: A dictionary with the following keys:
                - ``input_ids`` (LongTensor): Token ids for the source sequence (shape: ``[max_source_length]``).
                - ``attention_mask`` (LongTensor): Attention mask for the source (shape: ``[max_source_length]``).
                - ``labels`` (LongTensor): Token ids for the target sequence (shape: ``[max_target_length]``). 
                - ``src_lang`` (str): Source language code mapped to an mBART id (e.g., ``en_XX``).
                - ``tgt_lang`` (str): Target language code mapped to an mBART id (used downstream for ``forced_bos_token_id`` in mBART models).
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
            'tgt_lang': tgt_code  # required for forced_bos_token_id
        }
  

class DataModule(pl.LightningDataModule):
     """PyTorch Lightning DataModule for multilingual abstractive summarization.

    This module owns a single shared Hugging Face tokenizer and constructs train/val/test
    splits using ``Dataset1``. It exposes standard Lightning dataloader hooks."""
    def __init__(self, *args, **kwargs):
                """Initialize the DataModule and load the tokenizer.

        The tokenizer is created from ``self.hparams.tokenizer_name_or_path`` using
        ``transformers.AutoTokenizer.from_pretrained`` and shared across all dataset
        splits.
        """
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)

    def setup(self, stage=None):
        """Instantiate train/val/test ``Dataset1`` splits."""
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
        """ Returns:
            torch.utils.data.DataLoader: Batches training examples with shuffling enabled.
        """
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        """Returns:
            torch.utils.data.DataLoader: Batches validation examples with shuffling disabled.
        """
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=1, shuffle=False)

    def test_dataloader(self):
        """Returns:
            torch.utils.data.DataLoader: Batches test examples with shuffling disabled.
        """
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=1, shuffle=False)

    def predict_dataloader(self):
        """Alias to ``test_dataloader`` for Lightning's predict loop."""
        return self.test_dataloader()

