#!/usr/bin/env bash
# Gold-standard arm G1: instruction data consisting only of "reference solution + tests passing 10/10" (62.5M tokens, 160,241 examples)
set -u
cd "$(dirname "$0")/.." || exit 1
echo "=== G1 (gold only) start $(date '+%F %T') ==="
py -3.12 sft_train.py \
  --bin-data data_gold/sft_train.bin \
  --sft-val-bin data_gold/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_g1 \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 200 --save-every 200 --val-batches 250 \
  --compile-mode default --seed 42
echo "G1_EXIT=$? $(date '+%F %T')"
