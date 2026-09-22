#!/usr/bin/env bash
# Plan A: 147M signature data + 14.7M instruction (combining the two winners, mix ratio 10:1)
#   data : data_mix_big/sft_train.bin  <- data_big/sft_train.bin(147.0M) + data_instr_big/instr_train.bin(14.7M)
#   base : checkpoints_wsm/merged; all other hyperparameters same as Arm1/Arm4 (LR 2e-5 / warmup 50 / cosine / 1 epoch)
set -u
cd "$(dirname "$0")/.." || exit 1
echo "=== A5 (bigmix) start $(date '+%F %T') ==="
py -3.12 sft_train.py \
  --bin-data data_mix_big/sft_train.bin \
  --sft-val-bin data_mix_big/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_a5 \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 200 --save-every 200 --val-batches 250 \
  --compile-mode default --seed 42
echo "A5_EXIT=$? $(date '+%F %T')"
