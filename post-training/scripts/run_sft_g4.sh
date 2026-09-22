#!/usr/bin/env bash
# Plan G4 (two-stage): start from G1-bestval (strongest NL), short-train one round on pure signature data to restore the "signature completion" ability
#   start : ckpt_g1/best_val   <- G1 = pure gold standard (best NL column 43/55, but signature column only 13)
#   data : data/sft_train.bin (29.4M, pure signature -> function body)
#   idea : mixing within one round dilutes both sides (verified by G2) -> instead learn to follow instructions first, then restore the completion format separately
set -u
cd "$(dirname "$0")/.." || exit 1
echo "=== G4 (G1-bestval + pure-signature short train) start $(date '+%F %T') ==="
py -3.12 sft_train.py \
  --bin-data data/sft_train.bin \
  --sft-val-bin data/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ckpt_g1/best_val \
  --ckpt-dir ckpt_g4 \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 100 --save-every 100 --val-batches 250 \
  --compile-mode default --seed 42
echo "G4_EXIT=$? $(date '+%F %T')"
