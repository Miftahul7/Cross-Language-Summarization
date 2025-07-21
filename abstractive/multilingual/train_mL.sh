#!/bin/bash

#SBATCH --job-name=mbart_mL_s

# mbart
for domain in books writers films sportsman; do rm -rf salience_${domain}_mbart_multilingual.log; python abstractive/multilingual/train_mL.py --train_path XWikiRef/mL_split/${domain}/${domain}_train.json --val_path XWikiRef/mL_split/${domain}/${domain}_val.json --test_path XWikiRef/mL_split/${domain}/${domain}_test.json --tokenizer facebook/mbart-large-50 --model facebook/mbart-large-50 --is_mt5 0 --exp_name all_multilingual_salience_${domain}_mbart-summ --save_dir ./ --domain ${domain} --num_epochs 10 --train_batch_size 1 --val_batch_size 1 --test_batch_size 1  --gradient_accumulation_steps 16 --max_source_length 512 --max_target_length 256 --n_gpus 1 --strategy ddp --sanity_run no  2>&1|tee -a salience_${domain}_mbart_multilingual.log;done



# mt5
for domain in books writers films sportsman; do rm -rf salience_${domain}_mt5_multilingual.log; python abstractive/multilingual/train_mL.py --train_path XWikiRef/mL_split/${domain}/${domain}_train.json --val_path XWikiRef/mL_split/${domain}/${domain}_val.json --test_path XWikiRef/mL_split/${domain}/${domain}_test.json --tokenizer google/mt5-base --model google/mt5-base --is_mt5 1 --exp_name all_multilingual_salience_${domain}_mt5-summ --save_dir ./ --domain ${domain} --num_epochs 10 --train_batch_size 1 --val_batch_size 1 --test_batch_size 1 --gradient_accumulation_steps 16 --max_source_length 512 --max_target_length 256 --n_gpus 1 --strategy ddp --sanity_run no  2>&1|tee -a salience_${domain}_mt5_multilingual.log;done
