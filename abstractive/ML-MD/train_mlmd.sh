#!/bin/bash

#SBATCH --job-name=c_mt5


# mbart
rm -rf salience_combined_mbart_multilingual.log; 
python abstractive/ML-MD/train_mLmd.py --train_path XWikiRef/combined_split/mLmD_train.json --val_path XWikiRef/combined_split/mLmD_train.json --test_path XWikiRef/combined_split/mLmD_train.json --tokenizer facebook/mbart-large-50 --model facebook/mbart-large-50 --is_mt5 0 --exp_name all_multilingual_salience_combined_mbart-summ --save_dir ./  --num_epochs 10 --train_batch_size 1 --val_batch_size 1 --test_batch_size 1  --gradient_accumulation_steps 16 --max_source_length 512 --max_target_length 256 --n_gpus 1 --strategy ddp --sanity_run no  2>&1|tee -a salience_combined_mbart_multilingual.log

# mt5
rm -rf salience_combined_mt5_multilingual.log; 
python abstractive/ML-MD/train_mLmd.py --train_path XWikiRef/combined_split/mLmD_train.json --val_path XWikiRef/combined_split/mLmD_val.json --test_path XWikiRef/combined_split/mLmD_test.json --tokenizer google/mt5-base --model google/mt5-base --is_mt5 1 --exp_name all_multilingual_salience_combined_mt5-summ --save_dir ./  --num_epochs 10 --train_batch_size 1 --val_batch_size 1 --test_batch_size 1 --gradient_accumulation_steps 16 --max_source_length 512 --max_target_length 256 --n_gpus 1 --strategy ddp --sanity_run no  2>&1|tee -a salience_combined_mt5_multilingual.log
