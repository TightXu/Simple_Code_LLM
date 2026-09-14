#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════
Ablation 3 - Tokenizer (Tokenizer Ablation)

NOTE (2026-09): CLOSED without a training run. Measured headroom for the only clean
same-family variant (vocab 32K -> 48K -> 64K) is +1.6% / +2.8% compression at
+7.5% / +15.1% parameter cost — below the noise floor at any affordable budget.
See ablation/README.md section 3-5 for the rationale and numbers.
══════════════════════════════════════════════════════════════════════

Question: does a code-optimized tokenizer beat a generic BPE?

Arms (CodeLM-435M):
  Model A - generic BPE tokenizer (baseline: 32K vocab, GPT-2 style)
  Model B - code-optimized BPE (same 32K vocab, but:
            - ByteLevel pre-tokenizer (keeps indentation/whitespace)
            - trained on clean code data
            - special tokens: INDENT, DEDENT, NEWLINE)

Metrics:
  - token count on the same code (compression)
  - training efficiency (effective tokens/sec; note token counts differ)
  - validation perplexity
  - vocabulary coverage

Key caveat:
  - different tokenizers produce different token counts, so at the same
    "50M tokens" the actual code covered differs
  - fix the character count, not the token count, for a fair comparison
  - this impl: read a fixed number of samples, compare on a character basis

Usage:
  # Step 1: analyze / train the code tokenizer
  python ablation/ablation_tokenizer.py --mode analyze

  # Step 2: train the baseline model
  python ablation/ablation_tokenizer.py --mode baseline

  # Step 3: train the code-optimized model
  python ablation/ablation_tokenizer.py --mode code_opt

  # run everything
  python ablation/ablation_tokenizer.py --mode compare
══════════════════════════════════════════════════════════════════════
"""

import sys, argparse, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_common import *

# paths
CLEAN_DATA = DATA_DIR / "codeparrot_clean"  # codeparrot_clean jsonl as ablation data
TRAIN_JSONL = CLEAN_DATA / "train.jsonl"
VAL_JSONL = CLEAN_DATA / "val.jsonl"
TOKENIZER_BASELINE = TOKENIZER_PATH  # the 435M-v2 tokenizer (baseline)
TOKENIZER_CODE_OPT = ABLATION_DIR / "tokenizer_code_opt.json"

# 435M architecture
MODEL_435M = {
    "d_model": 1024,
    "num_layers": 22,
    "num_heads": 16,
    "d_ff": 4096,
    "vocab_size": 32000,
}

# training samples (fixed so both tokenizers see the same code volume)
TRAIN_SAMPLES = 50_000  # ~50M tokens (baseline tokenizer); adjust as needed
VAL_SAMPLES = 1_000


def train_code_optimized_tokenizer(output_path, vocab_size=32000):
    """
    Train a byte-level BPE tokenizer on clean code data.
    Features:
      - ByteLevel pre-tokenizer (keeps indentation, whitespace, special chars)
      - code-specific special tokens
      - byte-level coverage of all Unicode
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

    special_tokens = [
        "<s>", "</s>", "<unk>", "<pad>", "<eos>",
        "<INDENT>", "<DEDENT>", "<NEWLINE>",
    ]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    # stream JSONL
    def data_iterator():
        with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                yield item["text"]

    print(f"[TOK-NEW] training code-optimized tokenizer (vocab={vocab_size})...")
    print(f"[TOK-NEW] data source: {TRAIN_JSONL}")

    tokenizer.train_from_iterator(
        data_iterator(), trainer,
        length=1_186_000  # total training rows
    )

    tokenizer.save(str(output_path))
    print(f"[TOK-NEW] [OK] saved: {output_path}")
    print(f"[TOK-NEW] vocab={tokenizer.get_vocab_size()}")
    return tokenizer


def tokenize_subset(jsonl_path, tokenizer, output_bin, max_samples,
                    seq_len=1024):
    """Tokenize a JSONL subset into .bin format."""
    import numpy as np
    from tqdm import tqdm

    output_bin = Path(output_bin)
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    all_ids = []
    total_lines = sum(1 for _ in open(jsonl_path))
    limit = min(max_samples, total_lines)

    print(f"[TOK] Tokenizing {limit}/{total_lines} samples → {output_bin}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=limit, desc="Tok", unit="line")):
            if i >= limit:
                break
            try:
                item = json.loads(line)
                ids = tokenizer.encode(item["text"]).ids
                all_ids.extend(ids)
            except Exception:
                continue

    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(str(output_bin))

    # write meta
    meta = {
        "total_tokens": len(arr),
        "vocab_size": tokenizer.get_vocab_size(),
        "dtype": "uint16",
        "source": str(jsonl_path),
        "samples": limit,
    }
    meta_path = output_bin.parent / f"{output_bin.stem}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"[TOK] [OK] {output_bin.name}: {len(arr):,} tokens "
          f"({output_bin.stat().st_size/1e6:.1f}MB)")
    print(f"[TOK] meta: {meta_path}")
    return output_bin


def analyze_tokenizers():
    """Compare the compression ratio of two tokenizers."""
    from tokenizers import Tokenizer
    import numpy as np

    # load baseline
    tok_base = Tokenizer.from_file(str(TOKENIZER_BASELINE))
    if tok_base.token_to_id("<pad>") is None:
        tok_base.add_special_tokens(["<pad>"])

    # train / load code-opt
    if not TOKENIZER_CODE_OPT.exists():
        print("[ANALYZE] code-optimized tokenizer does not exist, training now...")
        train_code_optimized_tokenizer(TOKENIZER_CODE_OPT)
    tok_code = Tokenizer.from_file(str(TOKENIZER_CODE_OPT))
    if tok_code.token_to_id("<pad>") is None:
        tok_code.add_special_tokens(["<pad>"])

    # sample the val set for comparison
    samples = []
    with open(VAL_JSONL, "r") as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            samples.append(json.loads(line)["text"])

    total_chars = sum(len(s) for s in samples)
    tokens_base = [len(tok_base.encode(s).ids) for s in samples]
    tokens_code = [len(tok_code.encode(s).ids) for s in samples]

    print(f"\n{'='*50}")
    print("[INFO] tokenizer comparison (200 samples)")
    print(f"{'='*50}")
    print(f"  total chars:          {total_chars:,}")
    print(f"  generic BPE tokens:  {sum(tokens_base):,} "
          f"({total_chars/sum(tokens_base):.1f} chars/tok)")
    print(f"  code BPE tokens:     {sum(tokens_code):,} "
          f"({total_chars/sum(tokens_code):.1f} chars/tok)")
    compression_gain = (1 - sum(tokens_code)/sum(tokens_base)) * 100
    print(f"  compression gain:    {compression_gain:+.1f}% "
          f"{'(code-optimized saves tokens)' if compression_gain > 0 else '(baseline saves)'}")

    # token distribution stats
    avg_base = np.mean(tokens_base)
    std_base = np.std(tokens_base)
    avg_code = np.mean(tokens_code)
    std_code = np.std(tokens_code)
    print(f"\n  avg tokens/sample:  generic={avg_base:.0f}±{std_base:.0f}, "
          f"code={avg_code:.0f}±{std_code:.0f}")
    print(f"  vocab size:          generic={tok_base.get_vocab_size()}, "
          f"code={tok_code.get_vocab_size()}")

    # test special-token encoding
    test_code = (
        "def foo():\n    x = 1\n    if x > 0:\n        return x\n    return 0\n"
    )
    base_ids = tok_base.encode(test_code).ids
    code_ids = tok_code.encode(test_code).ids
    print(f"\n  \"def foo():...\" tokens: generic={len(base_ids)}, "
          f"code={len(code_ids)}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Tokenizer ablation - CodeLM-435M")
    parser.add_argument("--mode",
                       choices=["baseline", "code_opt", "compare", "analyze"],
                       default="compare")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=TRAIN_SAMPLES,
                       help="training samples (fixed char volume, fair comparison)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_tokenizers()
        return

    modes = ["baseline", "code_opt"] if args.mode == "compare" else [args.mode]

    for mode in modes:
        exp_name = f"ablation_tokenizer_{mode}"
        exp_dir = ABLATION_DIR / exp_name

        print(f"\n{'='*60}")
        print(f"[INFO] tokenizer ablation: {mode}")
        print(f"{'='*60}")

        if mode == "baseline":
            tok_path = TOKENIZER_BASELINE
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(tok_path))
            if tokenizer.token_to_id("<pad>") is None:
                tokenizer.add_special_tokens(["<pad>"])
        else:
            if not TOKENIZER_CODE_OPT.exists():
                train_code_optimized_tokenizer(TOKENIZER_CODE_OPT)
            tok_path = TOKENIZER_CODE_OPT
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(tok_path))
            if tokenizer.token_to_id("<pad>") is None:
                tokenizer.add_special_tokens(["<pad>"])

        # re-encode data with the matching tokenizer
        tok_data_dir = ABLATION_DIR / f"tokenized_{mode}"
        train_bin = tok_data_dir / "train.bin"

        if not train_bin.exists():
            print(f"[TOK] encoding data with {mode} tokenizer ({args.samples} samples)...")
            tokenize_subset(
                TRAIN_JSONL, tokenizer, train_bin, args.samples,
                seq_len=args.seq_len,
            )

        data_gen = create_offline_dataloader(
            train_bin, seq_len=args.seq_len)

        config = ModelConfig(**MODEL_435M)
        config.vocab_size = tokenizer.get_vocab_size()
        config.max_seq_len = args.seq_len

        run_ablation_training(
            exp_name=exp_name,
            exp_dir=str(exp_dir),
            data_gen=data_gen,
            val_bin=None,  # val loss across tokenizers is not comparable
            config=config,
            resume_from=None,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            max_tokens=None,  # run through all data
            save_every=200,
            eval_every=0,  # no eval
            log_every=10,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
