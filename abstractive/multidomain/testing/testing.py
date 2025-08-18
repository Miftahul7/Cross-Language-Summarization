"""
Evaluation and inference for multidomain abstractive summarization.

This script loads a trained checkpoint, evaluates it on a JSONL test split, computes ROUGE, and writes detailed
predictions to a CSV. It mirrors the train-time data pipeline but focuses
on testing/inference.

Workflow:
1) Parse command-line args and infer language/model from checkpoint file name.
2) Build a Lightning `DataModule` with JSONL-backed `Dataset1`.
3) Load a `Summarizer` from checkpoint with the correct tokenizer/model.
4) Run `trainer.test()` → aggregates predictions and ROUGE in `on_test_epoch_end`.
5) Save a CSV with columns: [input_text, domain, ref_text, pred_text, rouge]. 
"""
from torch.utils.data import Dataset, DataLoader
import os
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import MBartForConditionalGeneration, MT5ForConditionalGeneration, AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer
import torch
import argparse
from rouge import Rouge

class Dataset1(Dataset):
    """ JSONL-backed seq2seq dataset for testing/evaluation.
    Builds encoder input as:
        "<page_title> <section_title> <references_joined>"
    and uses `content` as the decoder target. Includes `domain` in each item
    for domain-/language-wise reports at test time.
    """
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length, target_lang, is_mt5):
        """ Initialize the dataset.
        Args:
            data_path (str): Path to JSONL file (one example per line).
            tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer instance.
            max_source_length (int): Max encoder sequence length.
            max_target_length (int): Max decoder sequence length.
            target_lang (str): Target language code (not used here explicitly).
            is_mt5 (bool): If True, replace pad tokens in labels with -100 for loss masking.
        """
        fp = open(data_path, 'r')
        self.df = [json.loads(line, strict=False) for line in fp.readlines()]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_mt5 = is_mt5

    def __len__(self):
        """Return the number of examples."""
        return len(self.df)

    def __getitem__(self, idx):
        """Create a tokenized example for index `idx`."""
        input_text = ' '.join(self.df[idx]['references'])
        input_text = str(self.df[idx]['page_title'] + ' ' + self.df[idx]['section_title'] + ' ' + input_text)
        target_text = self.df[idx]['content']
        domain = self.df[idx]['domain']

        input_encoding = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_source_length ,padding='max_length', truncation=True)
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(target_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)

        input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']
        labels = target_encoding['input_ids']

        if self.is_mt5:
            labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations

        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze(), 'domain':domain}

class DataModule(pl.LightningDataModule):
    """ PyTorch Lightning DataModule for test-time evaluation."""
    def __init__(self, *args, **kwargs):
        """Store hyperparameters and initialize tokenizer."""
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path, tgt_lang=self.hparams.tgt_lang)
        
    def setup(self, stage=None):
        self.train = Dataset1(self.hparams.train_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.tgt_lang, self.hparams.is_mt5)
        self.val = Dataset1(self.hparams.val_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.tgt_lang, self.hparams.is_mt5)
        self.test = Dataset1(self.hparams.test_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.tgt_lang, self.hparams.is_mt5)

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=1,shuffle=True)

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=1,shuffle=False)

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=1,shuffle=False)

    def predict_dataloader(self):
        """Alias for the test dataloader (used by Lightning `predict`)."""
        return self.test_dataloader()
        

class Summarizer(pl.LightningModule):
    """ LightningModule wrapper for mBART/mT5 during inference/testing."""
    def __init__(self, *args, **kwargs):
        """Initialize the summarization model and ROUGE scorer."""
        super().__init__()
        self.save_hyperparameters()
        self.rouge = Rouge()
        if self.hparams.is_mt5:
            self.model = MT5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)

    def forward(self, input_ids, attention_mask, labels):
        """Forward pass for computing loss (not used during pure generation)."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def _step(self, batch):
        """Compute supervised loss (useful when labels are available). """
        input_ids, attention_mask, labels, domain = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['domain']
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs[0]
        return loss
    
    def _generative_step(self, batch):
        """ Generate summaries and decode input/pred/reference strings.
        Args:
            batch (dict): Contains input_ids, attention_mask, labels, domain.
        Returns:
            tuple(list[str], list[str], list[str], list[str]): (input_text, pred_text, ref_text, domain)
        """
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            use_cache=True,
            num_beams=self.hparams.eval_beams,
            max_length=self.hparams.tgt_max_seq_len 
            )

        input_text = self.hparams.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        pred_text = self.hparams.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if self.hparams.is_mt5:
            # Restore pad tokens before decoding references
            batch['labels'][batch['labels'] == -100] = self.hparams.tokenizer.pad_token_id
        ref_text = self.hparams.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        domain = batch['domain']

        return input_text, pred_text, ref_text, domain

    def training_step(self, batch, batch_idx):
        """(Optional) training step; present for API parity."""
        loss = self._step(batch)
        self.log("train_loss", loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """(Optional) validation step; present for API parity."""
        loss = self._step(batch)
        input_text, pred_text, ref_text = self._generative_step(batch)
        self.log("val_loss", loss, on_epoch=True)
        return 

    def validation_epoch_end(self, outputs):
        """(Optional) validation epoch end; left as in original implementation."""
        pred_text = []
        ref_text = []
        for x in outputs:
            pred = x['pred_text']
            if pred[0] == '':
                pred[0] = 'pred_text'
                pred_text.extend(pred)
            else:
                pred_text.extend(pred)

            ref = x['ref_text']
            if ref[0] == '':
                ref[0] = 'ref_text'
                ref_text.extend(ref)
            else:
                ref_text.extend(ref)

        rouge = self.rouge.get_scores(pred_text, ref_text, avg=True)

        self.log("val_rouge-1_prec", rouge['rouge-1']['p'])
        self.log("val_rouge-1_rec", rouge['rouge-1']['r'])
        self.log("val_rouge-1_f1", rouge['rouge-1']['f'])

        self.log("val_rouge-2_prec", rouge['rouge-2']['p'])
        self.log("val_rouge-2_rec", rouge['rouge-2']['r'])
        self.log("val_rouge-2_f1", rouge['rouge-2']['f'])

        self.log("val_rouge-l_prec", rouge['rouge-l']['p'])
        self.log("val_rouge-l_rec", rouge['rouge-l']['r'])
        self.log("val_rouge-l_f1", rouge['rouge-l']['f'])
        return


    def predict_step(self, batch, batch_idx):
        """Lightning predict step: return decoded input/pred/ref texts."""
        input_text, pred_text, ref_text = self._generative_step(batch)
        return {'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text}
    
    def on_test_start(self):
        """Initialize container for test outputs before evaluation begins."""
        self.test_outputs = []
        
    def test_step(self, batch, batch_idx):
        """Accumulate per-batch predictions and loss for later aggregation."""
        loss = self._step(batch)
        input_text, pred_text, ref_text, domain = self._generative_step(batch)
        output = {'test_loss': loss, 'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text, 'domain': domain}
        self.test_outputs.append(output)
        return output
    def on_test_epoch_end(self):
        """Aggregate test results, compute ROUGE, and save CSV to disk."""
        
        df_to_write = pd.DataFrame(columns=['input_text', 'domain', 'ref_text', 'pred_text', 'rouge'])
        input_texts = []
        pred_texts = []
        ref_texts = []
        rouge_scores = []
        domain = []
        for x in self.test_outputs:
            if x['pred_text'][0] == '':
                x['pred_text'][0] = 'pred_text'
            if x['ref_text'][0] == '':
                x['ref_text'][0] = 'ref_text'
            input_texts.extend(x['input_text'])
            pred_texts.extend(x['pred_text'])
            ref_texts.extend(x['ref_text'])
            domain.extend(x['domain'])
            
            try:
                rouge_scores.extend(self.rouge.get_scores(x['pred_text'][0], x['ref_text'][0]))
            except:
                rouge_scores.append({'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}})

        df_to_write['input_text'] = input_texts
        df_to_write['domain'] = domain
        df_to_write['ref_text'] = ref_texts
        df_to_write['pred_text'] = pred_texts
        df_to_write['rouge'] = rouge_scores
        
        model_nm = self.hparams.model_name_or_path.split('/')[-1].split('-')[0]
        save_path = f'abstractive/multidomain/predictions/{model_nm}_predictions/lang_wise_{self.hparams.target_lang}_{model_nm}.csv'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_to_write.to_csv(save_path, index=False)
        print(f" Saved detailed results to {save_path}")

        logger.log_text(f'lang_wise_{method}_{lang}_{model_nm}_predictions', dataframe=df_to_write)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model-specific CLI arguments (kept for parity with train-time)."""
        parser = parent_parser.add_argument_group('Bart Fine-tuning Parameters')
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--model_name_or_path', default='bart-base', type=str)
        parser.add_argument('--eval_beams', default=4, type=int)
        parser.add_argument('--tgt_max_seq_len', default=128, type=int)
        parser.add_argument('--tokenizer', default='bart-base', type=str)
        return parent_parser

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input parameters for extractive stage')
    parser.add_argument('--batch_size', default=1, type=int, help='test_batch_size')
    parser.add_argument('--train_path', default=None, help='path to input json file for a given domain in given language')
    parser.add_argument('--val_path', default=None, help='path to intermediate output json file for a given domain in given language')
    parser.add_argument('--test_path', default=None, help='path to output json file for a given domain in given language')
    parser.add_argument('--config', default=None, help='which config file to use')
    parser.add_argument('--tokenizer', default='facebook/mbart-large-50', help='which tokenizer to use')
    parser.add_argument('--model', default='facebook/mbart-large-50', help='which model to use')
    parser.add_argument('--target_lang', default='bn_IN', help='what is the target language')
    parser.add_argument('--ckpt_path', help='ckpt path')
    parser.add_argument('--is_mt5', type=int, help='is the model mt5')
    parser.add_argument('--prediction_path', default='preds_hi.txt', help='path to save prediction file')

    args = parser.parse_args()
    prediction_path = args.prediction_path
    # Infer settings
    ckpt_path = args.ckpt_path
    ckpt_path_1 = ckpt_path.split('/')[-1]
    method = ckpt_path_1.split('_')[2]
    lang = ckpt_path_1.split('_')[3]
    model_nm = ckpt_path_1.split('_')[4]

    if 'mt5' in model_nm:
        tokenizer = 'google/mt5-base'
        model_name = 'google/mt5-base'
        is_mt5 = 1
    else:
        tokenizer = 'facebook/mbart-large-50'
        model_name = 'facebook/mbart-large-50'
        is_mt5 = 0

    print('-----------------------------------------------------------------------------------------------------------')
    print(lang, model_name)
    # Build data split paths
    train_path = 'XWikiRef/md_split/' + lang + '/' + lang + '_train.json'
    val_path = 'XWikiRef/md_split/' + lang + '/' + lang + '_val.json'
    test_path = 'XWikiRef/md_split/' + lang + '/' + lang + '_test.json'

    lang_map = {
        'bn' : 'bn_IN',
        'en' : 'en_XX',
        'hi' : 'hi_IN',
    }
    # Lightning DataModule
    target_lang = lang_map[lang]
    dm_hparams = dict(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            tokenizer_name_or_path=tokenizer,
            is_mt5=is_mt5,
            max_source_length=512,
            max_target_length=256,
            train_batch_size=1,
            val_batch_size=1,
            test_batch_size=args.batch_size,
            tgt_lang=lang_map[lang])
    dm = DataModule(**dm_hparams)
    # Model config
    model_hparams = dict(
            learning_rate=1e-5,
            model_name_or_path=model_name,
            eval_beams=4,
            is_mt5=is_mt5,
            tgt_max_seq_len=128,
            tokenizer=dm.tokenizer,
        )

    # WandB / Trainer
    logger=WandbLogger(name='inference_' + method + '_' + lang +  '_' + model_name, save_dir='./', project='multidomain evaluation', log_model=False)
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=logger)
    # Load & run evaluation
    model = Summarizer.load_from_checkpoint(ckpt_path, **model_hparams)
    results = trainer.test(model=model, datamodule=dm, verbose=True)
    print('-----------------------------------------------------------------------------------------------------------')
