#!/usr/bin/env python3
"""
Streaming data filtering & pre-tokenization - memory-friendly.
Usage: python prepare_stream.py --data-dir DIR --output-dir OUT [--max-files N]
"""
import os, sys, json, hashlib, argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from array import array
import numpy as np

# reuse the filter rules from filter.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretrain_data import (
    load_tokenizer, is_filename_poison,
    check_l1_framework, check_l2_python2, check_l3_config,
    strip_all_copyright_comments, check_l4_tests, compute_quality_metrics,
    check_l6_autogen, verify_clean_data,
)

import pyarrow.parquet as pq
from tqdm import tqdm


def extract_streaming(data_dir, output_dir, max_files=0, start_file_index=0, val_split=0.02, seed=42):
    """Streaming extraction: filter as you go, write JSONL, no accumulation. Train/val split by hash."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(Path(data_dir).glob("*.parquet"))
    if start_file_index > 0:
        parquet_files = parquet_files[start_file_index:]
    print(f"[PACK] scanning {len(parquet_files)} parquet files (from #{start_file_index})...")

    mode = "a" if start_file_index > 0 else "w"
    train_f = open(output_dir / "train.jsonl", mode, encoding="utf-8")
    val_f = open(output_dir / "val.jsonl", mode, encoding="utf-8")

    stats = Counter()
    filter_hits = Counter()
    train_count = 0
    val_count = 0

    for pf in tqdm(parquet_files, desc="reading parquets"):
        try:
            table = pq.read_table(str(pf))
            columns = table.column_names
            path_col_name = "path" if "path" in columns else "max_stars_repo_path"
            path_col = table.column(path_col_name)
            content_col = table.column("content")

            for i in range(len(table)):
                filepath = str(path_col[i].as_py())
                code = str(content_col[i].as_py())

                if not filepath.endswith(".py"):
                    stats["non_python"] += 1; continue
                if is_filename_poison(filepath):
                    stats["filename_filtered"] += 1; continue
                if len(code) < 50:
                    stats["too_short"] += 1; continue

                bad, _ = check_l1_framework(code)
                if bad: stats["L1_framework"] += 1; continue

                bad, _ = check_l2_python2(code)
                if bad: stats["L2_python2"] += 1; continue

                bad, _ = check_l3_config(code)
                if bad: stats["L3_config"] += 1; continue

                stripped = strip_all_copyright_comments(code)
                if stripped != code:
                    stats["L3_stripped"] += 1
                    code = stripped

                bad, _ = check_l4_tests(code)
                if bad: stats["L4_tests"] += 1; continue

                passed, _ = compute_quality_metrics(code)
                if not passed: stats["L5_quality"] += 1; continue

                bad, _ = check_l6_autogen(code)
                if bad: stats["L6_autogen"] += 1; continue

                # [OK] passes - hash splits to train/val
                stats["accepted"] += 1
                h = int(hashlib.md5(code[:200].encode()).hexdigest()[:8], 16)
                is_val = (h % 50 == 0)  # ~2%
                line = json.dumps({"text": code}, ensure_ascii=False) + "\n"

                if is_val:
                    val_f.write(line); val_count += 1
                else:
                    train_f.write(line); train_count += 1

                if max_files and stats["accepted"] >= max_files:
                    break

        except Exception as e:
            stats["parse_error"] += 1
            print(f"  [WARN]  {pf.name}: {e}")

        if max_files and stats["accepted"] >= max_files:
            break

    train_f.close()
    val_f.close()

    # stats
    total_scanned = sum(stats[k] for k in ["non_python", "filename_filtered",
        "too_short", "L1_framework", "L2_python2", "L3_config",
        "L4_tests", "L5_quality", "L6_autogen", "accepted", "parse_error"])
    print(f"\n{'='*60}")
    print(f"[INFO] filtering stats")
    print(f"{'='*60}")
    print(f"  scanned:     {total_scanned:,}")
    print(f"  [OK] accepted: {stats['accepted']:,} ({stats['accepted']/max(total_scanned,1)*100:.1f}%)")
    print(f"  Train: {train_count:,}  Val: {val_count:,}")

    # save stats
    with open(output_dir / "filter_stats.json", "w") as f:
        json.dump({"stats": dict(stats), "timestamp": datetime.now().isoformat()}, f, indent=2)

    return train_count, val_count, stats


def pre_tokenize_streaming(input_path, output_bin, tokenizer, seq_len=1024, chunk_tokens=100_000_000):
    """Streaming tokenize: line-by-line JSONL, batch encode, chunked .bin write (memory-friendly)."""
    import tempfile, shutil
    all_ids = array('H')
    batch_texts = []
    batch_size = 256
    temp_files = []
    total_raw = 0

    with open(input_path, "r") as f:
        total_lines = sum(1 for _ in f)
    if total_lines == 0:
        print(f"  [FAIL] empty file: {input_path}")
        return 0

    print(f"  [WRITE] {total_lines:,} texts, tokenizing (chunk={chunk_tokens/1e6:.0f}M)...")

    def flush_chunk():
        nonlocal total_raw
        if len(all_ids) == 0:
            return
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".bin", dir=Path(output_bin).parent)
        ids = np.frombuffer(all_ids, dtype=np.uint16).copy()
        ids.tofile(tf.name)
        temp_files.append(tf.name)
        total_raw += len(all_ids)
        all_ids[:] = array('H')  # clear

    with open(input_path, "r") as f:
        pbar = tqdm(total=total_lines, desc=f"  Tokenizing {Path(input_path).name}",
                    unit="text", unit_scale=True)
        for line in f:
            item = json.loads(line)
            batch_texts.append(item["text"] + "<eos>")
            if len(batch_texts) >= batch_size:
                for enc in tokenizer.encode_batch(batch_texts):
                    all_ids.fromlist(enc.ids)
                pbar.update(len(batch_texts))
                batch_texts = []
                if len(all_ids) >= chunk_tokens:
                    flush_chunk()
        if batch_texts:
            for enc in tokenizer.encode_batch(batch_texts):
                all_ids.fromlist(enc.ids)
            pbar.update(len(batch_texts))
        flush_chunk()
        pbar.close()

    total_tokens = (total_raw // seq_len) * seq_len
    print(f"  [NUM] {total_raw:,} raw tokens -> truncate to {total_tokens:,} ({total_tokens/1e6:.1f}M)")

    # Concat temp files into final .bin, truncating to seq boundary
    remaining = total_tokens
    with open(output_bin, "wb") as out:
        for tf_path in temp_files:
            if remaining <= 0:
                break
            chunk = np.fromfile(tf_path, dtype=np.uint16)
            take = min(len(chunk), remaining)
            chunk[:take].tofile(out)
            remaining -= take
            os.unlink(tf_path)

    meta = {
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.get_vocab_size(),
        "seq_len": seq_len,
        "source": str(input_path),
        "dtype": "uint16",
        "created": datetime.now().isoformat(),
    }
    with open(str(output_bin).replace(".bin", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    file_size_mb = Path(output_bin).stat().st_size / 1e6
    print(f"  [OK] {total_tokens:,} tokens ({total_tokens/1e6:.1f}M), {file_size_mb:.1f} MB")
    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--start-file-index", type=int, default=0, help="which dataset index to start from (resume)")
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    print("=" * 60)
    print("streaming data prep (memory-friendly)")
    print(f"  source: {args.data_dir}")
    print(f"  output: {args.output_dir}")
    print("=" * 60)

    # Phase 1: extraction
    train_count, val_count, stats = extract_streaming(
        args.data_dir, args.output_dir, max_files=args.max_files,
        start_file_index=args.start_file_index)

    if args.start_file_index > 0:
        print(f"\n[OK] resume finished (from file #{args.start_file_index})")
        return 0

    # Phase 2: verification
    hits = verify_clean_data(str(Path(args.output_dir) / "train.jsonl"), sample_size=200)
    if hits > 0:
        print(f"\n[WARN]  {hits} contaminated rows remain, skipping pre-tokenization")
        return 1

    # Phase 3: pre-tokenization
    tokenizer = load_tokenizer()
    total_tokens = 0
    for name in ["train", "val"]:
        jsonl_path = Path(args.output_dir) / f"{name}.jsonl"
        bin_path = Path(args.output_dir) / f"{name}.bin"
        if jsonl_path.exists():
            tok = pre_tokenize_streaming(str(jsonl_path), str(bin_path), tokenizer, args.seq_len)
            total_tokens += tok

    print(f"\n{'='*60}")
    print(f"[OK] done!")
    print(f"   train: {train_count:,} rows  val: {val_count:,} rows")
    print(f"   total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"   output: {args.output_dir}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    exit(main())
