#!/usr/bin/env bash
# SFT Arm 2 (plan A) -- only difference from Arm 1 = LR 5e-5 (the default in plan doc 435M-v2-SFT-plan-2026-09-11.md)
#   start : checkpoints_wsm/merged  <- same base as Arm 1 (so 2e-5 vs 5e-5 is a clean single-variable comparison)
#   data : fine-tuning/data/sft_train.bin (identical to Arm 1, 29.4M tokens / keep protocol)
#   rest : item-by-item identical to run_sft_arm1.sh (micro-batch 8 x accum 4 = 32 sequences / 1 epoch / warmup 50 /
#          cosine + eta-min 0.1 / compile default / seed 42 / eval+save every 100 / val-batches 250)
#   artifacts : ckpt_arm2/ (does not write ckpt/, to avoid touching Arm 1's artifacts)
#   rationale : the plan said lr 5e-5 (then 8e-5 on plateau); Arm 1 conservatively used 2e-5 -> this arm tests that decision
set -u
cd "$(dirname "$0")/.." || exit 1

echo "=== SFT Arm2 (plan A: base + lr5e-5) start $(date '+%Y-%m-%d %H:%M:%S') ==="
py -3.12 sft_train.py \
  --bin-data data/sft_train.bin \
  --sft-val-bin data/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_arm2 \
  --epochs 1 \
  --lr 5e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 100 --save-every 100 --val-batches 250 \
  --compile-mode default --seed 42
echo "SFT_ARM2_EXIT=$? $(date '+%Y-%m-%d %H:%M:%S')"
