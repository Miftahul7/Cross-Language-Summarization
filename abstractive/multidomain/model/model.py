"""This code is a modified version provided at https://github.com/DhavalTaunk08/XWikiGen/tree/main"""
"""
Seq2Seq Summarization Model for PyTorch Lightning.

This module defines the `Summarizer` class, a LightningModule wrapper around
transformers-based seq2seq models (mBART or mT5). It is designed to be used
together with:
- `dataloader.py`: Provides the `DataModule` for reading and tokenizing JSONL data.
- `train_mD.py`: The main training script that instantiates `Summarizer` and `DataModule`, configures logging, checkpointing, distributed training, and launches training.

Key Features:
- Supports **mBART** and **mT5** architectures for multilingual summarization.
- Provides training, validation, testing, and prediction steps in the
  PyTorch Lightning lifecycle.
- Computes **ROUGE** scores during validation and testing for summarization
  quality evaluation.
- Writes test predictions and metrics to CSV files for later inspection. 
"""
import pytorch_lightning as pl
from transformers import MBartForConditionalGeneration, MT5ForConditionalGeneration, AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer
import torch
from rouge import Rouge
import json
import pandas as pd

class Summarizer(pl.LightningModule):
    """PyTorch Lightning wrapper for mBART/MT5 summarization models."""

    def __init__(self, *args, **kwargs):
        """Initialize the model and evaluation components."""
        super().__init__()
        self.save_hyperparameters()
        self.rouge = Rouge()
        self.config = AutoConfig.from_pretrained(self.hparams.config)
        self.validation_outputs = []
        if self.hparams.is_mt5:
            self.model = MT5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)

    def forward(self, input_ids, attention_mask, labels):
        """ Forward pass of the underlying seq2seq model.
        Args:
            input_ids (torch.LongTensor): Encoder input ids.
            attention_mask (torch.LongTensor): Encoder attention mask.
            labels (torch.LongTensor): Target sequence ids.
        Returns:
            transformers.modeling_outputs.Seq2SeqLMOutput: Model outputs including loss.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def _step(self, batch):
        """Compute loss for a batch."""
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs[0]
        return loss
    
    def _generative_step(self, batch):
        """ Generate summaries and decode input, prediction, and reference texts.
        Args:
            batch (dict): A batch with keys 'input_ids', 'attention_mask', 'labels'.
        Returns:
            tuple: (input_text, pred_text, ref_text) as lists of decoded strings.
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
            batch['labels'][batch['labels'] == -100] = self.hparams.tokenizer.pad_token_id
        ref_text = self.hparams.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        return input_text, pred_text, ref_text

    def training_step(self, batch, batch_idx):
        """ Training step: compute loss and log it."""
        loss = self._step(batch)
        self.log("train_loss", loss, on_epoch=True)
        return {'loss': loss}
    def on_validation_epoch_start(self):
        """ Reset storage for validation outputs at the start of an epoch."""
        self.validation_outputs = []

    def validation_step(self, batch, batch_idx):
        """ Compute loss and collect predictions for ROUGE."""
        loss = self._step(batch)
        input_text, pred_text, ref_text = self._generative_step(batch)
        self.log("val_loss", loss, on_epoch=True)
        self.validation_outputs.append({'pred_text': pred_text, 'ref_text': ref_text})
        return 

    def on_validation_epoch_end(self):
        """ Compute average ROUGE scores across validation outputs."""
        pred_text = []
        ref_text = []
        for x in self.validation_outputs:
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

        
        self.validation_outputs = []
    def predict_step(self, batch, batch_idx):
        """ Prediction step: return input, predicted, and reference texts."""
        input_text, pred_text, ref_text = self._generative_step(batch)
        return {'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text}

    def test_step(self, batch, batch_idx):
        """Test step: compute loss and return texts."""
        loss = self._step(batch)
        input_text, pred_text, ref_text = self._generative_step(batch)
        return {'test_loss': loss, 'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text}

    def test_epoch_end(self, outputs):
         """ Aggregate test results, save predictions to CSV, and log ROUGE + loss."""
        df_to_write = pd.DataFrame(columns=['input_text', 'ref_text', 'pred_text', 'rouge'])
        input_text = []
        langs = []
        pred_text = []
        ref_text = []
        rouge_scores = []
        for x in outputs:
            pred = x['pred_text']
            
            for i in range(len(pred)):
                if pred[i] == '':
                    pred[i] = 'default text'
            pred_text.extend(pred)
            
            input_text.extend(x['input_text'])
            
            ref = x['ref_text']
            for i in range(len(ref)):
                if ref[i] == '':
                    ref[i] = 'default text'
            ref_text.extend(ref)
            
            rouge_scores.extend(self.rouge.get_scores(pred, ref))

        df_to_write['input_text'] = input_text
        df_to_write['ref_text'] = ref_text
        df_to_write['pred_text'] = pred_text
        df_to_write['rouge'] = rouge_scores

        if self.hparams.is_mt5:
            df_to_write.to_csv(self.hparams.prediction_path + 'preds_mt5.csv', index=False)
        else:    
            df_to_write.to_csv(self.hparams.prediction_path + 'preds_mbart.csv', index=False)

        rouge = self.rouge.get_scores(pred_text, ref_text, avg=True)
        epoch_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        self.log("test_rouge-1_prec", rouge['rouge-1']['p'])
        self.log("test_rouge-1_rec", rouge['rouge-1']['r'])
        self.log("test_rouge-1_f1", rouge['rouge-1']['f'])

        self.log("test_rouge-2_prec", rouge['rouge-2']['p'])
        self.log("test_rouge-2_rec", rouge['rouge-2']['r'])
        self.log("test_rouge-2_f1", rouge['rouge-2']['f'])

        self.log("test_rouge-l_prec", rouge['rouge-l']['p'])
        self.log("test_rouge-l_rec", rouge['rouge-l']['r'])
        self.log("test_rouge-l_f1", rouge['rouge-l']['f'])
        
        self.log("epoch_test_loss", epoch_test_loss)

    def configure_optimizers(self):
        """ Configure optimizer (Adam)."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model-specific command line arguments.
        Args:
            parent_parser (argparse.ArgumentParser): Existing parser to extend.
        Returns:
            argparse.ArgumentParser: Updated parser with model args.
        """
        parser = parent_parser.add_argument_group('Bart Fine-tuning Parameters')
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--model_name_or_path', default='bart-base', type=str)
        parser.add_argument('--eval_beams', default=4, type=int)
        parser.add_argument('--tgt_max_seq_len', default=128, type=int)
        parser.add_argument('--tokenizer', default='bart-base', type=str)
        return parent_parser
