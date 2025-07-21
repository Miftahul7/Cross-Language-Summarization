#!/bin/bash



#mbart
for lang in en bn hi; do rm -rf salience_${lang}_mbart_multidomain.log; python abstractive/multidomain/train_mD.py --train_path XWikiRef/md_split/${lang}/${lang}_train.json --val_path XWikiRef/md_split/${lang}/${lang}_val.json --test_path XWikiRef/md_split/${lang}/${lang}_test.json --tokenizer facebook/mbart-large-50 --model facebook/mbart-large-50 --is_mt5 0 --exp_name lang_wise_salience_${lang}_mbart-summ --save_dir ./ --target_lang ${lang} --num_epochs 10 --train_batch_size 1 --val_batch_size 1 --test_batch_size 1 --gradient_accumulation_steps 16 --max_source_length 512 --max_target_length 256 --n_gpus 1 --strategy ddp --sanity_run no 2>&1|tee -a logs/salience_${lang}_mbart_multidomain.log;done

echo 'Done'

#mT5

for lang in en bn hi; do rm -rf salience_${lang}_mt5_multidomain.log; python abstractive/multidomain/train_mD.py --train_path XWikiRef/md_split/${lang}/${lang}_train.json --val_path XWikiRef/md_split/${lang}/${lang}_val.json --test_path XWikiRef/md_split/${lang}/${lang}_test.json --tokenizer google/mt5-base --model google/mt5-base --is_mt5 1 --exp_name lang_wise_salience_${lang}_mt5-summ --save_dir ./ --target_lang ${lang} --num_epochs 10 --train_batch_size 1 --val_batch_size 1 --test_batch_size 1 --gradient_accumulation_steps 16 --max_source_length 512 --max_target_length 256 --n_gpus 1 --strategy ddp --sanity_run no 2>&1|tee -a salience_${lang}_mt5_multidomain.log;done
echo 'Done'