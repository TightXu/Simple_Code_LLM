#!/usr/bin/env bash
# Gold-standard arm G2: gold standard (62.5M) + signature data (29.4M) mixed = 91.9M tokens (ratio ~2.1:1, matching the strongest NL 2:1 in plan D)
set -u
cd "$(dirname "$0")/.." || exit 1
echo "=== G2 (gold+sig) start $(date '+%F %T') ==="
py -3.12 sft_train.py \
  --bin-data data_mix_gold/sft_train.bin \
  --sft-val-bin data_mix_gold/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_g2 \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 200 --save-every 200 --val-batches 250 \
  --compile-mode default --seed 42
echo "G2_EXIT=$? $(date '+%F %T')"
