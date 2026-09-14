#!/usr/bin/env bash
# Plan C: SFT fixed at 29.4M + instruction seed expanded to 9M (mix ratio 3.3:1) -- isolate how much instruction is most cost-effective
set -u
cd "/c/Users/Tight/Documents/Train LLM/435M-v2/fine-tuning" || exit 1
echo "=== A7 (big seed) start $(date '+%F %T') ==="
py -3.12 sft_train.py \
  --bin-data data_mix_seed9m/sft_train.bin \
  --sft-val-bin data_mix_seed9m/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_a7 \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 100 --save-every 100 --val-batches 250 \
  --compile-mode default --seed 42
echo "A7_EXIT=$? $(date '+%F %T')"
