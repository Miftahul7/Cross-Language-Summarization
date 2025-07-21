#!/bin/bash
echo "Running extractive stage"


python extractive/extractive.py \
  --inp_file Data/Curated-20/hi/cleaned/hi_sportsman.json \
  --out_file extractive/output/mD/hi/cleaned/hi_sportsman.json \
  --top_k 20 \
  --tokenizer xlm-roberta-base \
  --model xlm-roberta-base

