#!/usr/bin/env bash
# SFT Arm 3 -- large-dataset variant (150M tokens, "signature -> function body", same recipe at 5x volume)
#   only difference from Arm1 = the dataset (all other hyperparameters identical -> directly comparable)
#   base : checkpoints_wsm/merged (same as Arm1)
#   data : fine-tuning/data_big/sft_train.bin (--crossing-policy keep)
#   global batch: 8 x 4 = 32 sequences = 32,768 tokens/step (the Arm1 VRAM lesson)
#   steps : ~4,577 steps/epoch ~ 1 hour (at ~43K tok/s)
#   LR   : 2e-5 warmup 50 -> cosine->10% (same as Arm1)
#   guards : dual val every 200 steps (SFT masked + pretrain protocol) + checkpoint
#   WARNING: do not add --auto-resume: given together with --resume it silently trains from scratch (hit on 2026-09-12, 8 wasted minutes).
#      the script is fixed (falls back to --resume when there is no local ckpt), but we simply avoid it here.
#   WARNING: within 2 minutes of start, check the first log lines: loss should be ~1.6, pretrain_val_loss ~1.3;
#      if loss > 4 / pretrain_val > 6 = the base was not loaded, stop immediately.
set -u
cd "/c/Users/Tight/Documents/Train LLM/435M-v2/fine-tuning" || exit 1

echo "=== SFT Arm3 start $(date '+%Y-%m-%d %H:%M:%S') ==="
py -3.12 sft_train.py \
  --bin-data data_big/sft_train.bin \
  --sft-val-bin data_big/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_big \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 200 --save-every 200 --val-batches 250 \
  --compile-mode default --seed 42
echo "SFT_ARM3_EXIT=$? $(date '+%Y-%m-%d %H:%M:%S')"
