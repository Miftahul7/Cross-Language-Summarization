#!/bin/bash

#SBATCH --job-name=b_mt5_t



#mkdir mbart_predictions
dir="all_multilingual_salience_books_mt5-summ/lightning-checkpoints"
for f in "$dir"/*; do
	echo $f
   python abstractive/multilingual/testing/testing_mt5.py --ckpt_path $f 
done