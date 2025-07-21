#!/bin/bash



dir="all_multilingual_salience_combined_mt5-summ/lightning-checkpoints"
for f in "$dir"/*; do
	echo $f
   python abstractive/ML-MD/testing_mt5.py --ckpt_path $f 
done