#!/bin/bash

#SBATCH --job-name=mlmd_mbart_test



#mkdir mbart_predictions
dir="all_multilingual_salience_combined_mbart-summ/lightning-checkpoints/"
for f in "$dir"/*; do
	echo $f
   python abstractive/ML-MD/testing_mbart.py --ckpt_path $f 
done