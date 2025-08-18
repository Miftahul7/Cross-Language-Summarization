"""This code is a modified version provided at https://github.com/DhavalTaunk08/XWikiGen/tree/main"""
"""
Training script for multidomain abstractive summarization.
This script handles model training using PyTorch Lightning. It ties together:
- `DataModule` (from `dataloader.py`): handles dataset loading and tokenization.
- `Summarizer` (from `model.py`): wraps mBART or mT5 for seq2seq summarization.
- Model checkpointing with PyTorch Lightning callbacks.
- GPU and distributed training configuration.

Features:
- Supports resuming training from the latest checkpoint automatically.
- Allows both full training and quick "sanity runs" for debugging.
- Saves best checkpoints and predictions to disk.
"""
from model.model import Summarizer 
from model.dataloader import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
import torch
import os
import sys
import glob
import time
import argparse
os.environ["WANDB_SILENT"] = "True"
import wandb
wandb.login(key=" ") # API key needed

def main(args):
    """Main training pipeline.
    This function:
    1. Sets up paths, hyperparameters, and language codes.
    2. Instantiates the `DataModule` (for train/val/test sets).
    3. Instantiates the `Summarizer` model with given hyperparameters.
    4. Configures checkpointing, logging, and trainer parameters.
    5. Resumes training from the last checkpoint if found, otherwise starts fresh.

    Args:
        args (argparse.Namespace): Command-line arguments parsed in `__main__`.
    """
    # Language mapping for mBART
    lang_map = {
        'bn' : 'bn_IN',
        'en' : 'en_XX',
        'hi' : 'hi_IN',
    }
    # Resolve input arguments
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    tokenizer_name_or_path = args.tokenizer
    model_name_or_path = args.model
    is_mt5 = args.is_mt5

    if args.config is not None:
        config = args.config
    else:
        config = model_name_or_path
    # Ensure prediction directory exists
    if not os.path.exists(args.prediction_path):
        os.system(f'mkdir -p {args.prediction_path}')
    # Training parameters
    n_gpus = args.n_gpus
    strategy = args.strategy
    EXP_NAME = args.exp_name
    save_dir = args.save_dir
    target_lang = lang_map[args.target_lang]
    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    test_batch_size = args.test_batch_size
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    prediction_path = args.prediction_path
    # DataModule hyperparameters
    dm_hparams = dict(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        is_mt5=is_mt5,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        target_lang=target_lang
    )
    dm = DataModule(**dm_hparams)
    # Model hyperparameters
    model_hparams = dict(
        learning_rate=1e-5,
        model_name_or_path=model_name_or_path,
        config = config,
        is_mt5=is_mt5,
        eval_beams=4,
        tgt_max_seq_len=max_target_length,
        tokenizer=dm.tokenizer,
        target_lang=target_lang,
        prediction_path=prediction_path
    )

    model = Summarizer(**model_hparams)
   # Handle sanity runs (small-scale debugging)
    if args.sanity_run=='yes':
        log_model = False
        limit_train_batches = 4
        limit_val_batches = 4
        limit_test_batches = 4
    else:
        log_model = True
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
    # Checkpoint filename pattern
    model_id = model_name_or_path.split('/')[-1].split('-')[0] 
    filename_pattern = f'{{epoch}}_{{step}}_salience_{args.target_lang}_{model_id}'

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=os.path.join(save_dir + EXP_NAME, 'lightning-checkpoints'),
        filename=filename_pattern,
        save_top_k=1,
        verbose=True,
        save_last=False,
        save_weights_only=False
    )

    # Check the number of available GPUs
    available_gpus = torch.cuda.device_count()
    if args.n_gpus > available_gpus:
        print(f"Requested {args.n_gpus} GPUs, but only {available_gpus} are available.")
        args.n_gpus = available_gpus

    # Setup trainer
    trainer_hparams = dict(
        devices=args.n_gpus,
        accelerator='gpu' if args.n_gpus > 0 else 'cpu',
        strategy=strategy,
        precision="16-mixed",
        max_epochs=num_epochs,
        num_sanity_val_steps=3,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        logger=WandbLogger(
            name=model_name_or_path.split('/')[-1],
            save_dir=save_dir + EXP_NAME,
            project=EXP_NAME,
            log_model=False
        ),
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches
    )

    # Create trainer
    trainer = pl.Trainer(**trainer_hparams)

    # Detect latest checkpoint
    checkpoint_dir = os.path.join(save_dir + EXP_NAME, 'lightning-checkpoints')
    latest_ckpt = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
            latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f" Found checkpoint to resume from: {latest_ckpt}")
        else:
            print("No checkpoints found in checkpoint directory.")
    else:
        print("Checkpoint directory does not exist yet.")

    # Train (resume if checkpoint found)
    if latest_ckpt:
        trainer.fit(model, dm, ckpt_path=latest_ckpt)
    else:
        trainer.fit(model, dm)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input parameters for extractive stage')
    parser.add_argument('--n_gpus', default=1, type=int, help='number of gpus to use')
    parser.add_argument('--train_path', help='path to input json file for a given domain in given language')
    parser.add_argument('--val_path', help='path to intermediate output json file for a given domain in given language')
    parser.add_argument('--test_path', help='path to output json file for a given domain in given language')
    parser.add_argument('--config', default=None, help='which config file to use')
    parser.add_argument('--tokenizer', default='facebook/mbart-large-50', help='which tokenizer to use')
    parser.add_argument('--model', default='facebook/mbart-large-50', help='which model to use')
    parser.add_argument('--is_mt5', type=int, help='is the model mt5')
    parser.add_argument('--exp_name', default='mbart-basline', help='experiment name')
    parser.add_argument('--save_dir', default='checkpoints/', help='where to save the logs and checkpoints')
    parser.add_argument('--target_lang', default='bn', help='what is the target language')
    parser.add_argument('--num_epochs', default=5, type=int, help='number of epochs')
    parser.add_argument('--train_batch_size', default=4, type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=4, type=int, help='val batch size')
    parser.add_argument('--test_batch_size', default=4, type=int, help='test batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--max_source_length', default=1024, type=int, help='max source length')
    parser.add_argument('--max_target_length', default=1024, type=int, help='max target length')
    parser.add_argument('--strategy', default='dp', help='which strategy to use')
    parser.add_argument('--sanity_run', default='no', help='which strategy to use')
    parser.add_argument('--prediction_path', default='abstractive/multidomain/predictions/', help='path to save prediction outputs') 

    args = parser.parse_args()

    main(args)
