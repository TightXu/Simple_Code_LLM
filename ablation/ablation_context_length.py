#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════
Ablation 4 - Context length (Context Length Ablation)
══════════════════════════════════════════════════════════════════════

NOTE (2026-09): CLOSED without a training run — at the affordable budget (~50M tokens/arm)
the arms cannot produce an interpretable result (see ablation/README.md section 3-5).

Question: does a longer context improve code understanding/generation?

Arms (CodeLM-435M, 22L, d_ff=4096):
  Model A — seq_len =  512
  Model B — seq_len = 1024  (baseline)
  Model C — seq_len = 2048

Controls:
  - same architecture (only RoPE theta adjusts with seq_len)
  - same training tokens (50M)
  - same data / filtering
  - all trained from scratch

RoPE theta adjustment rule:
  - seq_len=512:  theta=10000  (standard)
  - seq_len=1024: theta=500000 (current default)
  - seq_len=2048: theta=1000000

Memory considerations (NAS CPU):
  - doubling seq_len roughly doubles per-step memory, so lower batch_size
  - 512: batch=16, ga=2
  - 1024: batch=8, ga=4
  - 2048: batch=4, ga=8

Compare:
  - training loss convergence
  - later: long-code completion accuracy, Long Range Arena-style tests

Usage:
  python ablation/ablation_context_length.py --ctx 512
  python ablation/ablation_context_length.py --ctx 1024
  python ablation/ablation_context_length.py --ctx 2048
  python ablation/ablation_context_length.py --ctx compare
══════════════════════════════════════════════════════════════════════
"""

import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_common import *

# paths
CLEAN_DATA = DATA_DIR / "codeparrot_clean"  # codeparrot_clean jsonl as ablation data
TRAIN_JSONL = CLEAN_DATA / "train.jsonl"
VAL_JSONL = CLEAN_DATA / "val.jsonl"
TOK_DATA_DIR = ABLATION_DIR / "tokenized_baseline"

# 435M architecture
MODEL_435M = {
    "d_model": 1024,
    "num_layers": 22,
    "num_heads": 16,
    "d_ff": 4096,
    "vocab_size": 32000,
}

TARGET_TOKENS = 50_000_000

# seq_len → (batch_size, grad_accum, rope_theta)
# keep effective batch_size ~32 as a control
CTX_CONFIGS = {
    512:  {"batch_size": 16, "grad_accum": 2,  "rope_theta": 10000.0},
    1024: {"batch_size": 8,  "grad_accum": 4,  "rope_theta": 500000.0},
    2048: {"batch_size": 4,  "grad_accum": 8,  "rope_theta": 1000000.0},
}


def prepare_shared_data(tokenizer, seq_len=1024, redo=False):
    """Prepare shared raw data for all context lengths (uses the baseline tokenizer)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(
        str(CLEAN_PRETRAIN_DIR / "tokenizer" / "tokenizer_435m.json"))
    if tok.token_to_id("<pad>") is None:
        tok.add_special_tokens(["<pad>"])

    train_bin = TOK_DATA_DIR / "train.bin"
    val_bin = TOK_DATA_DIR / "val.bin"

    if redo or not train_bin.exists():
        TOK_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # tokenize train (enough for 50M tokens, ~15K samples)
        from ablation_tokenizer import tokenize_subset
        print("[PREP] preparing tokenized data (enough for 50M tokens)...")
        tokenize_subset(
            TRAIN_JSONL, tok, train_bin,
            max_samples=50_000, seq_len=seq_len)
        tokenize_subset(
            VAL_JSONL, tok, val_bin,
            max_samples=5_000, seq_len=seq_len)

    return train_bin, val_bin


def main():
    parser = argparse.ArgumentParser(
        description="Context length ablation - CodeLM-435M")
    parser.add_argument("--ctx", type=str, default="compare",
                       help="seq_len: 512 / 1024 / 2048 / compare")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--tokens", type=int, default=TARGET_TOKENS,
                       help=f"training tokens (default {TARGET_TOKENS/1e6:.0f}M)")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--redo-data", action="store_true",
                       help="force re-tokenization of data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    # prepare shared data
    train_bin, val_bin = prepare_shared_data(tokenizer, redo=args.redo_data)

    if args.ctx == "compare":
        ctx_list = [512, 1024, 2048]
    else:
        ctx_list = [int(args.ctx)]

    results = {}
    for seq_len in ctx_list:
        cfg = CTX_CONFIGS[seq_len]
        exp_name = f"ablation_context_{seq_len}"
        exp_dir = ABLATION_DIR / exp_name

        print(f"\n{'='*60}")
        print(f"[INFO] context length ablation: seq_len={seq_len}")
        print(f"  rope_theta={cfg['rope_theta']:.0f}, "
              f"BS={cfg['batch_size']}xGA={cfg['grad_accum']}")
        print(f"  effective batch_size={cfg['batch_size']*cfg['grad_accum']}")
        print(f"{'='*60}")

        # data: same .bin, sliced at different seq_len
        data_gen = create_offline_dataloader(
            train_bin, seq_len=seq_len, max_tokens=args.tokens)

        config = ModelConfig(**MODEL_435M)
        config.max_seq_len = seq_len
        config.rope_theta = cfg["rope_theta"]

        state = run_ablation_training(
            exp_name=exp_name,
            exp_dir=str(exp_dir),
            data_gen=data_gen,
            val_bin=str(val_bin) if val_bin.exists() else None,
            config=config,
            resume_from=None,
            batch_size=cfg["batch_size"],
            grad_accum=cfg["grad_accum"],
            lr=args.lr,
            max_tokens=args.tokens,
            save_every=args.save_every,
            eval_every=args.save_every,
            log_every=10,
            seed=args.seed,
        )

        results[seq_len] = {
            "final_step": state["step"],
            "total_tokens": state["total_tokens"],
            "best_val_loss": state.get("best_val_loss", float("inf")),
        }

    # comparison report
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("\n[INFO] context length ablation - comparison")
        print(f"{'='*60}")
        for sl, r in sorted(results.items()):
            print(f"  seq_len={sl:4d}: step={r['final_step']:6d}, "
                  f"tokens={r['total_tokens']/1e6:.1f}M, "
                  f"best_val={r['best_val_loss']:.4f}")
        print(f"\n  note: val loss across seq_len is not strictly comparable")
        print(f"  (longer seq_len sees more context, so loss should be lower)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
