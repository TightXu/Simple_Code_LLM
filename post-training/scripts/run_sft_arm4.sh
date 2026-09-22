#!/usr/bin/env bash
# SFT Arm 4 -- instruction mix: SFT (29.4M) + instruction seed (2.94M) co-trained
#   start : checkpoints_wsm/merged (same base as Arm 1 -> the only variable is that 2.94M of instruction data)
#   data : data_mix/sft_train.bin (interleaved in 256-chunk blocks, 32.34M tokens / 31,552 chunks total)
#          corpus = data/sft_train.bin (signature->function body 29.4M) + data_instr_mixed/instr_train.bin (natural language->code 2.94M)
#   hyperparams : item-by-item identical to Arm 1/Arm 3 (LR 2e-5 won, see the three-arm conclusion)
#   artifacts : ckpt_arm4/
#   evaluation : must report both columns -- signature protocol (eval_sft.py, comparable with the first three arms) + natural-language protocol (instruction template)
set -u
cd "$(dirname "$0")/.." || exit 1

echo "=== SFT Arm4 (instruction mix) start $(date '+%Y-%m-%d %H:%M:%S') ==="
py -3.12 sft_train.py \
  --bin-data data_mix/sft_train.bin \
  --sft-val-bin data_mix/sft_val.bin \
  --val-bin ../data/all/val.bin \
  --resume ../checkpoints_wsm/merged \
  --ckpt-dir ckpt_arm4 \
  --epochs 1 \
  --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
  --batch-size 8 --grad-accum 4 \
  --eval-every 100 --save-every 100 --val-batches 250 \
  --compile-mode default --seed 42
echo "SFT_ARM4_EXIT=$? $(date '+%Y-%m-%d %H:%M:%S')"
