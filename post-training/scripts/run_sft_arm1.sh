#!/usr/bin/env bash
# SFT Arm 1 -- the real supervised fine-tuning run for 435M v2
#   base : checkpoints_wsm/merged  (val loss 1.3716, 5/55 pass -- best of the three bases; merging was ruled out experimentally)
#   data : fine-tuning/data/sft_train.bin  (rebuilt with --crossing-policy keep: loss_ratio 0.8887, 0 zero-loss samples)
#   global batch : 8 (micro) x 4 (accum) = 32 sequences = 32,768 tokens / optimizer step
#               WARNING: micro-batch 16 x 2 fills all 32.6GB of VRAM -> allocator thrashing -> only 4.9K tok/s measured (GPU 99% but 200W)
#                  micro-batch 8 x 4 -> 21.6GB VRAM / 536W / ~46K tok/s, global batch unchanged
#   steps : 28,681 chunks / 32 ~ 896 steps = 1 epoch (29.4M tokens) ~ 11 minutes
#   LR   : 2e-5 (warmup 50 steps) -> cosine -> 10% floor (T_max auto = 896)
#          rationale: pretrain peak ~2.5e-4 / cooldown 1e-4 -> take ~1/12 of it; the 353M DAPT run at 8e-5 measured negative;
#                the base itself is weak and brittle (5/55), so the first arm targets "no regression + format alignment"
#   guards : every 100 steps evaluate both SFT masked val (protocol) and pretrain val (forgetting check), checkpoint every 100 steps, best_val kept automatically
#   not done : replay / shuffle -- run the most conservative path already covered by smoke tests
set -u
cd "$(dirname "$0")/.." || exit 1

echo "=== SFT Arm1 start $(date '+%Y-%m-%d %H:%M:%S') ==="
py -3.12 sft_train.py \
  --bin-data data/sft_train.bin \
  --sft-val-bin data/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 100 --save-every 100 --val-batches 250 \
  --compile-mode default --seed 42
echo "SFT_ARM1_EXIT=$? $(date '+%Y-%m-%d %H:%M:%S')"
