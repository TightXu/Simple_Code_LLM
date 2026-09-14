#!/usr/bin/env python3
"""
nofilter data prep: codeparrot (raw) then star_coder (raw), minimal cleaning
================================================================
- no L1-L6 filtering (the core of the no-filter arm)
- minimal cleaning only: .py suffix (codeparrot has a path column) + >=50 chars
- star_coder has no path column: strip <reponame> metadata prefix, keep all
- same order as v2: codeparrot -> star_coder (the_stack skipped)
- stream from the raw parquet in place; only the .bin lands in the repo

Usage:
  py -3.12 ablation/prepare_nofilter_data.py --max-tokens 1100000000
"""
import os
import sys
import json
import glob
import argparse
import time
from pathlib import Path

import numpy as np

ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parent

# Raw parquet locations: override with --codeparrot-dir / --starcoder-dir or the
# CODEPARROT_DIR / STARCODER_DIR environment variables. Defaults are a relative layout,
# so a fresh clone never depends on the machine this was written on.
CODEPARROT_DIR = os.environ.get("CODEPARROT_DIR", str(PROJECT_ROOT / "data" / "raw" / "codeparrot"))
STARCODER_DIR = os.environ.get("STARCODER_DIR", str(PROJECT_ROOT / "data" / "raw" / "starcoderdata"))
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer_435m.json"
OUT_DIR = ABLATION_DIR / "data_nofilter"

MIN_CHARS = 50


def load_tokenizer():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    if tok.token_to_id("<pad>") is None:
        tok.add_special_tokens(["<pad>"])
    return tok


def strip_starcoder_meta(code):
    """Strip star_coder metadata prefixes such as <reponame>."""
    lines = code.split("\n")
    stripped = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("<reponame>"):
            # strip only the prefix tags, keep the code after
            rest = ln[ln.find(">") + 1:].lstrip()
            if rest:
                stripped.append(rest)
            continue
        if ln.startswith("<gh_stars>") or ln.startswith("<filename>"):
            rest = ln[ln.find(">") + 1:].lstrip()
            if rest:
                stripped.append(rest)
            continue
        stripped.append(ln)
    return "\n".join(stripped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=30_000_000_000,
                    help="total target tokens (30B floor; in practice limited by full dataset)"),
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--codeparrot-dir", type=str, default=CODEPARROT_DIR,
                    help="raw codeparrot parquet directory (default: $CODEPARROT_DIR or data/raw/codeparrot)")
    ap.add_argument("--starcoder-dir", type=str, default=STARCODER_DIR,
                    help="raw star_coder parquet directory (default: $STARCODER_DIR or data/raw/starcoderdata)")
    ap.add_argument("--start-file-index", type=int, default=0,
                    help="codeparrot start file index (resume from last checkpoint)")
    ap.add_argument("--append", action="store_true",
                    help="resume mode: open bin in append, infer total_tokens from existing bin")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from tqdm import tqdm

    codeparrot_dir, starcoder_dir = args.codeparrot_dir, args.starcoder_dir
    tokenizer = load_tokenizer()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_bin = OUT_DIR / "train.bin"

    target = args.max_tokens
    print(f"[PREP] nofilter data: target {target/1e9:.2f}B tokens (full two datasets, order codeparrot -> star_coder)")
    print(f"[PREP] tokenizer: {TOKENIZER_PATH.name}")
    print(f"[PREP] order: codeparrot({codeparrot_dir}) -> star_coder({starcoder_dir})")

    all_ids = []
    total_tokens = 0
    stats = {"codeparrot_files": 0, "codeparrot_samples": 0,
             "starcoder_files": 0, "starcoder_samples": 0,
             "rejected_short": 0, "rejected_nonpy": 0}

    # resume: infer produced tokens from the existing bin
    mode = "ab" if args.append else "wb"
    if args.append and out_bin.exists():
        total_tokens = int(out_bin.stat().st_size // 2)
        print(f"[RESUME] resume mode: existing bin has {total_tokens:,} tokens ({total_tokens/1e9:.2f}B), "
              f"continuing from codeparrot file #{args.start_file_index}")

    # stream to bin: flush every FLUSH_BATCH tokens, constant peak memory
    FLUSH_BATCH = 50_000_000  # 50M tokens ≈ 100MB
    fout = open(out_bin, mode)

    def flush_ids():
        nonlocal all_ids
        if all_ids:
            np.array(all_ids, dtype=np.uint16).tofile(fout)
            all_ids = []

    # Stage 1: codeparrot raw parquet full stream (has path column -> .py check)
    cp_files = sorted(glob.glob(codeparrot_dir + "/*.parquet"))
    start_idx = args.start_file_index
    print(f"[CP] {len(cp_files)} parquet files, full stream (from #{start_idx})")
    for idx in tqdm(range(start_idx, len(cp_files)), desc="codeparrot", unit="file"):
        pf = cp_files[idx]
        if total_tokens >= target:
            break
        try:
            table = pq.read_table(pf)
        except Exception as e:
            stats.setdefault("corrupt_files", []).append(pf.split("/")[-1])
            print(f"  [WARN] skipping corrupt file {pf.split('/')[-1]}: {str(e)[:80]}")
            continue
        paths = table.column("path")
        contents = table.column("content")
        for i in range(len(table)):
            try:
                if not str(paths[i].as_py()).endswith(".py"):
                    stats["rejected_nonpy"] += 1
                    continue
                code = str(contents[i].as_py())
                if len(code) < MIN_CHARS:
                    stats["rejected_short"] += 1
                    continue
                ids = tokenizer.encode(code).ids
                all_ids.extend(ids)
                stats["codeparrot_samples"] += 1
                total_tokens += len(ids)
                if len(all_ids) >= FLUSH_BATCH:
                    flush_ids()
                if total_tokens >= target:
                    break
            except Exception:
                continue
        flush_ids()
        stats["codeparrot_files"] += 1
        print(f"  ...{pf.split('/')[-1]}: cumulative {total_tokens/1e9:.2f}B tokens")
    print(f"[CP] stage done: {total_tokens/1e9:.2f}B tokens")

    # Stage 2: star_coder raw parquet full stream (no path column -> strip metadata, keep all)
    sc_files = sorted(glob.glob(starcoder_dir + "/*.parquet"))
    print(f"[SC] {len(sc_files)} parquet files, full stream")
    for pf in tqdm(sc_files, desc="starcoder", unit="file"):
        if total_tokens >= target:
            break
        try:
            table = pq.read_table(pf)
        except Exception as e:
            stats.setdefault("corrupt_files", []).append(pf.split("/")[-1])
            print(f"  [WARN] skipping corrupt file {pf.split('/')[-1]}: {str(e)[:80]}")
            continue
        contents = table.column("content")
        for i in range(len(table)):
            try:
                code = str(contents[i].as_py())
                code = strip_starcoder_meta(code)
                if len(code) < MIN_CHARS:
                    stats["rejected_short"] += 1
                    continue
                ids = tokenizer.encode(code).ids
                all_ids.extend(ids)
                stats["starcoder_samples"] += 1
                total_tokens += len(ids)
                if len(all_ids) >= FLUSH_BATCH:
                    flush_ids()
                if total_tokens >= target:
                    break
            except Exception:
                continue
        flush_ids()
        stats["starcoder_files"] += 1
        print(f"  ...{pf.split('/')[-1]}: cumulative {total_tokens/1e9:.2f}B tokens")
    print(f"[SC] stage done: {total_tokens/1e9:.2f}B tokens")
    flush_ids()
    fout.close()

    # write meta (the bin was already streamed)
    meta = {
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.get_vocab_size(),
        "dtype": "uint16",
        "source": "codeparrot_raw -> starcoder_raw (nofilter, minimal clean .py+>=50chars)",
        "order": "codeparrot -> star_coder (the_stack skipped)",
        "stats": stats,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(OUT_DIR / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] {out_bin}: {total_tokens:,} tokens ({total_tokens*2/1e9:.2f}GB)")
    print(f"[SAVE] meta: {OUT_DIR/'train_meta.json'}")
    print(f"[STATS] {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
