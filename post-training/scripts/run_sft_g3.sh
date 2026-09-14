#!/usr/bin/env bash
# Plan G3: gold standard : signature = 1:1 (29.4M each) -- a true 1:1 (A9's 1:1 used a small instruction seed, and G2 is 2.1:1)
set -u
cd "/c/Users/Tight/Documents/Train LLM/435M-v2/fine-tuning" || exit 1
echo "=== G3 (gold:sig = 1:1) start $(date '+%F %T') ==="
py -3.12 sft_train.py \
  --bin-data data_mix_g11/sft_train.bin \
  --sft-val-bin data_mix_g11/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_g3 \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 200 --save-every 200 --val-batches 250 \
  --compile-mode default --seed 42
echo "G3_EXIT=$? $(date '+%F %T')"
