#!/usr/bin/env bash
# Plan B: Arm4 data (29.4M signature + 2.94M instruction) for 2 epochs -- only the number of passes changes
#   note: --t-max not passed -> total steps derived from epochs (the cosine schedule stretches accordingly)
set -u
cd "$(dirname "$0")/.." || exit 1
echo "=== A6 (2 epochs) start $(date '+%F %T') ==="
py -3.12 sft_train.py \
  --bin-data data_mix/sft_train.bin \
  --sft-val-bin data_mix/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_a6 \
  --epochs 2 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 200 --save-every 200 --val-batches 250 \
  --compile-mode default --seed 42
echo "A6_EXIT=$? $(date '+%F %T')"
