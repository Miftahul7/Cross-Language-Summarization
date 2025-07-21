#!/bin/bash

# Run testing on each checkpoint file
dir="lang_wise_salience_hi_mt5-summ/lightning-checkpoints"
for f in "$dir"/*.ckpt; do
    echo "Running checkpoint: $f"
    python abstractive/multidomain/testing/testing.py --ckpt_path "$f" --target_lang hi_IN
done
