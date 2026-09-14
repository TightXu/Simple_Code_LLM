#!/usr/bin/env bash
# Plan D1: A8 = 29.4M signature + 14.4M instruction (2:1)
set -u
cd "/c/Users/Tight/Documents/Train LLM/435M-v2/fine-tuning" || exit 1
echo "=== A8 (2:1) start $(date '+%F %T') ==="
py -3.12 sft_train.py \
  --bin-data data_mix_2to1/sft_train.bin \
  --sft-val-bin data_mix_2to1/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_a8 \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 100 --save-every 100 --val-batches 250 \
  --compile-mode default --seed 42
echo "A8_EXIT=$? $(date '+%F %T')"
