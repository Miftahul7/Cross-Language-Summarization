#!/bin/bash

#SBATCH --job-name=s_test_mt5


#mkdir mbart_predictions
dir="all_multilingual_salience_sportsman_mbart-summ/lightning-checkpoints"
for f in "$dir"/*; do
	echo $f
   python abstractive/multilingual/testing/testing_mbart.py --ckpt_path $f 
done