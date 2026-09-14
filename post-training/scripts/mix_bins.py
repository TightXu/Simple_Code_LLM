#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mix_bins.py — combine several (bin, mask) datasets into one at specified ratios, ready for sft_train.py to consume directly.

Contract (identical to build_sft_data.py / build_instr_data.py output):
  <stem>.bin       uint16 token id stream
  <stem>_mask.bin  uint8  per-token loss mask (1 = counts for loss)
  chunk_size = seq_len + 1 = 1025 (the file length must be a multiple of it, otherwise the script exits with an error)

Usage:
  py -3.12 mix_bins.py --out data_mix --parts data/sft_train.bin data_instr/instr_train.bin \
                       --mode interleave --seed 42
  py -3.12 mix_bins.py --out data_mix --parts ... --mode concat
It also merges the matching val sets when they exist: <stem> becomes the val name (--val-parts, optional).

Design trade-offs:
  - concat     : simple, preserves each input's internal order exactly (put the instruction data last)
  - interleave : block-interleave at chunk granularity (default block 256 chunks), so the instruction
                 format appears evenly along the stream and there is no distribution jump where
                 "the first 90% is signature completion and only the last 10% is instruction".
                 Order within a block is unchanged → no chunk is split, mask semantics unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

CHUNK = 1025  # seq_len 1024 + 1


def load_stream(bin_path: Path):
    data = bin_path.read_bytes()
    mask_path = bin_path.with_name(bin_path.stem + "_mask.bin")
    if not mask_path.exists():
        sys.exit(f"[ERR] missing mask file: {mask_path}")
    mask = mask_path.read_bytes()
    if len(data) % 2:
        sys.exit(f"[ERR] {bin_path.name} has an odd byte count (uint16 stream)")
    n_tok = len(data) // 2
    if len(mask) != n_tok:
        sys.exit(f"[ERR] {bin_path.name}: mask {len(mask)} != tokens {n_tok}")
    if n_tok % CHUNK:
        sys.exit(f"[ERR] {bin_path.name}: {n_tok} tokens is not a multiple of {CHUNK} (chunk alignment failed)")
    import numpy as np
    ids = np.frombuffer(data, dtype="<u2")
    msk = np.frombuffer(mask, dtype=np.uint8)
    return ids, msk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--parts", nargs="+", required=True, help="list of train bins (in the given order/ratio)")
    ap.add_argument("--val-parts", nargs="*", default=[], help="optional: matching val bin list")
    ap.add_argument("--mode", choices=["concat", "interleave"], default="concat")
    ap.add_argument("--block-chunks", type=int, default=256, help="interleave block size (in chunks)")
    ap.add_argument("--stem", default="sft_train")
    ap.add_argument("--val-stem", default="sft_val")
    args = ap.parse_args()

    import numpy as np
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parts = [load_stream(Path(p)) for p in args.parts]
    for p, (ids, _m) in zip(args.parts, parts):
        print(f"[IN ] {p}: {len(ids):,} tokens ({len(ids)//CHUNK:,} chunks)")

    if args.mode == "concat":
        ids = np.concatenate([p[0] for p in parts])
        msk = np.concatenate([p[1] for p in parts])
        plan = "concat"
    else:
        blocks = []
        for ids_i, msk_i in parts:
            n_ch = len(ids_i) // CHUNK
            for s in range(0, n_ch, args.block_chunks):
                e = min(s + args.block_chunks, n_ch)
                blocks.append((ids_i[s * CHUNK:e * CHUNK], msk_i[s * CHUNK:e * CHUNK]))
        ids = np.concatenate([b[0] for b in blocks])
        msk = np.concatenate([b[1] for b in blocks])
        plan = f"interleave(block={args.block_chunks} chunks, {len(blocks)} blocks)"

    (out / f"{args.stem}.bin").write_bytes(ids.astype("<u2").tobytes())
    (out / f"{args.stem}_mask.bin").write_bytes(msk.astype(np.uint8).tobytes())

    loss_tok = int(msk.sum())
    meta = {
        "total_tokens": int(len(ids)),
        "chunk_size": CHUNK,
        "seq_len": CHUNK - 1,
        "n_chunks": int(len(ids) // CHUNK),
        "loss_tokens": loss_tok,
        "loss_ratio": round(loss_tok / max(1, len(ids)), 4),
        "zero_loss_chunks": int(((msk.reshape(-1, CHUNK).sum(axis=1)) == 0).sum()),
        "mix": {"mode": plan, "parts": [str(p) for p in args.parts]},
        "dtype": "uint16", "mask_dtype": "uint8", "no_padding": True,
    }

    if args.val_parts:
        vids = np.concatenate([load_stream(Path(p))[0] for p in args.val_parts])
        vmsk = np.concatenate([load_stream(Path(p))[1] for p in args.val_parts])
        (out / f"{args.val_stem}.bin").write_bytes(vids.astype("<u2").tobytes())
        (out / f"{args.val_stem}_mask.bin").write_bytes(vmsk.astype(np.uint8).tobytes())
        meta["val_tokens"] = int(len(vids))
        meta["val_n_chunks"] = int(len(vids) // CHUNK)
        print(f"[OUT] {args.val_stem}.bin {len(vids):,} tokens")

    (out / f"{args.stem}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OUT] {args.stem}.bin {len(ids):,} tokens | {meta['n_chunks']:,} chunks | loss_ratio={meta['loss_ratio']} | zero_loss_chunks={meta['zero_loss_chunks']} | {plan}")
    print(f"[OUT] meta → {out / (args.stem + '_meta.json')}")


if __name__ == "__main__":
    main()
