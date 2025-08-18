"""This code is a modified version provided at https://github.com/DhavalTaunk08/XWikiGen/tree/main"""
""" Minimal data loading pipeline for seq2seq summarization with PyTorch Lightning.
This module provides:
- `Dataset1`: reads JSONL data (one example per line), tokenizes, and returns tensors for seq2seq training.
- `DataModule`: a LightningDataModule that instantiates train/val/test datasets and exposes DataLoaders with configurable batch sizes.
Expected JSONL schema per line:
{
  "page_title": str,
  "section_title": str,
  "content": str,             # target text
  "references": List[str]     # list of reference sentences (will be concatenated)
}
"""
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import json
import torch
class Dataset1(Dataset):
    """ Dataset that reads JSONL examples and tokenizes them dynamically during data loading.
    Each example builds the encoder input as:
        "<page_title> <section_title> <references_joined>"
    and uses `content` as the decoder target.""" 
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length, target_lang, is_mt5):
        """ Initialize the dataset.
        Args:
            data_path (str): Path to a JSONL file (one JSON object per line).
            tokenizer (transformers.PreTrainedTokenizerBase): Initialized tokenizer.
            max_source_length (int): Max encoder sequence length.
            max_target_length (int): Max decoder sequence length.
            target_lang (str): Target language code (not used directly here).
            is_mt5 (bool): If True, replace padding IDs in labels with -100 for CE loss masking.
        """
        fp = open(data_path, 'r')
        self.df = [json.loads(line, strict=False) for line in fp.readlines()]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_mt5 = is_mt5
        print(f"Loaded dataset from {data_path}")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Create a tokenized example for index `idx`.
        Args:
            idx (int): Index of the example.
        Returns:
            dict: {
                "input_ids": torch.LongTensor [max_source_length],
                "attention_mask": torch.LongTensor [max_source_length],
                "labels": torch.LongTensor [max_target_length] 
            }
        """
        input_text = ' '.join(self.df[idx]['references'])
        input_text = str(self.df[idx]['page_title'] + ' ' + self.df[idx]['section_title'] + ' ' + input_text)
        target_text = self.df[idx]['content']

        input_encoding = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_source_length ,padding='max_length', truncation=True)
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(target_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)

        input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']
        labels = target_encoding['input_ids']

        if self.is_mt5:
            labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations

        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze()}    

class DataModule(pl.LightningDataModule):
     """PyTorch Lightning DataModule for the JSONL summarization dataset.

    Exposes train/val/test DataLoaders and constructs a tokenizer with
    `tgt_lang` (if applicable for the chosen model).
    """
    def __init__(self, *args, **kwargs):
        """Store hyperparameters and initialize tokenizer. """
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path, tgt_lang=self.hparams.target_lang)
        
    def setup(self, stage=None):
        """Create datasets for the given stage.
        Args:
            stage (str, optional): One of {"fit", "validate", "test", "predict", None}.
                - "fit"/None: create train and val datasets
                - "test"/"predict": create test dataset. """
        if stage == "fit" or stage is None:
            self.train = Dataset1(self.hparams.train_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.target_lang, self.hparams.is_mt5)
            self.val = Dataset1(self.hparams.val_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.target_lang, self.hparams.is_mt5)

        if stage == "test" or stage == "predict":
            self.test = Dataset1(self.hparams.test_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.target_lang, self.hparams.is_mt5)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=1,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=1,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=1,shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()
