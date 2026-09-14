#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════
Ablation 1 - Data scaling (Data Scaling Ablation)
══════════════════════════════════════════════════════════════════════

Question: does more code data actually improve the model?

Arms (CodeLM-435M, 22L, d_ff=4096):
  Model A —   1B tokens  (baseline)
  Model B —   5B tokens
  Model C —  10B tokens

Controls:
  - same architecture (435M params)
  - same training hyperparameters
  - same filter policy (L1-L6)
  - all trained from scratch (old 353M checkpoints are architecturally incompatible)

Data source:
  - code_data/ (302G parquet) via online L1-L6 filtering
  - ~180M-250M tokens per pass; large scales need multiple passes

Usage:
  python ablation/ablation_data_scaling.py --scale 1B
  python ablation/ablation_data_scaling.py --scale 5B
  python ablation/ablation_data_scaling.py --scale 10B
══════════════════════════════════════════════════════════════════════
"""

import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_common import *

# config: use the 435M-v2 merged data (offline .bin, memmap subset)
TRAIN_BIN = DATA_DIR / "all" / "train.bin"
VAL_BIN = DATA_DIR / "all" / "val.bin"

SCALES = {
    "100M":    100_000_000,
    "500M":    500_000_000,
    "1B":    1_000_000_000,
    "5B":    5_000_000_000,
    "10B":  10_000_000_000,
}

# 435M arch params (match pretrain_train.py defaults)
MODEL_435M = {
    "d_model": 1024,
    "num_layers": 22,
    "num_heads": 16,
    "d_ff": 4096,
    "vocab_size": 32000,
}


def main():
    parser = argparse.ArgumentParser(description="Data scaling ablation - CodeLM-435M")
    parser.add_argument("--scale", choices=list(SCALES.keys()), required=True,
                       help="training data volume (100M/500M/1B/5B/10B)")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=8,
                       help="batch_size=8 recommended for 435M (NAS CPU)")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_tokens = SCALES[args.scale]
    exp_name = f"ablation_data_scaling_{args.scale}"
    exp_dir = ABLATION_DIR / exp_name

    print(f"{'='*60}")
    print(f"[INFO] data scaling ablation: {args.scale} ({target_tokens/1e9:.1f}B tokens)")
    print(f"arch: {sum(MODEL_435M.values())/1e6 if False else '435M'} params")
    print(f"{'='*60}")

    tokenizer = load_tokenizer()

    # validation set (use the merged val.bin directly)
    val_path = str(VAL_BIN) if VAL_BIN.exists() else None
    if not VAL_BIN.exists():
        print("[VAL] [WARN] no validation data, skipping eval")

    # data: first target_tokens of the offline .bin (memmap, no multi-pass)
    print(f"[DATA] taking {target_tokens/1e9:.2f}B tokens from {TRAIN_BIN.name}")
    data_gen = create_offline_dataloader(
        TRAIN_BIN, seq_len=args.seq_len, max_tokens=target_tokens)

    # model (435M)
    config = ModelConfig(**MODEL_435M)
    config.max_seq_len = args.seq_len

    state = run_ablation_training(
        exp_name=exp_name,
        exp_dir=str(exp_dir),
        data_gen=data_gen,
        val_bin=val_path,
        config=config,
        resume_from=None,  # train from scratch (old ckpt arch incompatible)
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_tokens=target_tokens,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=10,
        seed=args.seed,
    )

    # output summary
    print(f"\n{'='*60}")
    print(f"[OK] data scaling ablation done: {args.scale}")
    print(f"  final step: {state['step']}")
    print(f"  training tokens: {state['total_tokens']/1e9:.2f}B")
    print(f"  Best val loss: {state.get('best_val_loss', 'N/A')}")
    print(f"  model saved: {exp_dir}/final/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
