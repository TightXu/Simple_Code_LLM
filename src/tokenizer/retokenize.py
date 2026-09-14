#!/usr/bin/env python3
"""Re-tokenize the three datasets' JSONL into .bin (streaming, memory-friendly)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretrain_data_stream import pre_tokenize_streaming, load_tokenizer

DATA = Path(__file__).resolve().parent / "data"
# Smallest first to see progress early
DATASETS = ["codeparrot_clean", "star_coder_data", "the_stack_v1"]


def main():
    tok = load_tokenizer()
    print("tokenizer vocab", tok.get_vocab_size(), flush=True)
    for ds in DATASETS:
        for split in ("train", "val"):
            j = DATA / ds / (split + ".jsonl")
            b = DATA / ds / (split + ".bin")
            if not j.exists():
                print("skip", ds, split, flush=True)
                continue
            print("tokenize", ds, split, "...", flush=True)
            pre_tokenize_streaming(str(j), str(b), tok, seq_len=1024)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
