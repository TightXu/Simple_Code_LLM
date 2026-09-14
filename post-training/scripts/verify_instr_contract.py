#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract verifier: reads the output of build_instr_data.py / build_sft_data.py using the **real consumer functions from sft_train.py**.

    py -3.12 verify_instr_contract.py --dir data_instr/dryrun --stems instr_train instr_val
    py -3.12 verify_instr_contract.py --dir data/sft_train.bin            # a bin file may be passed directly

What is verified is the consumer-side contract, not the builder self-check -- i.e. the functions in sft_train.py that are actually called:
resolve_loss_mask → _assert_bin_readable → _count_chunks → create_sft_dataloader → inspect_loss_mask.
CPU only (CUDA_VISIBLE_DEVICES=""), no model is built and the GPU is untouched.
Exit code 0 = everything passed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # never touch the GPU (training may be running elsewhere)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                    # noqa: E402
import sft_train as S                                 # noqa: E402
from tokenizers import Tokenizer                      # noqa: E402

DEFAULT_TOKENIZER = HERE.parent / "tokenizer" / "tokenizer.json"


def check_one(bin_path: Path, tok, vocab: int, seq_len: int):
    C = seq_len + 1
    fails, info = [], {}
    bin_path = Path(bin_path)
    mask_path = bin_path.parent / f"{bin_path.stem}_mask.bin"

    # ① mask-carrier resolution (the real sft_train entry point)
    kind, mpath = S.resolve_loss_mask(None, str(bin_path), vocab)
    info["resolve_loss_mask"] = [kind, str(mpath)]
    if kind != "file":
        fails.append(f"resolve_loss_mask -> {kind} (expected file: the companion file was not auto-discovered)")
    if mpath is None or Path(mpath) != mask_path:
        fails.append(f"mask path mismatch: {mpath} != {mask_path}")

    # ② layout: bin bytes == 2 × mask bytes == 2 × chunks × (seq_len+1)
    nb, nmb = bin_path.stat().st_size, mask_path.stat().st_size
    n_chunks = S._count_chunks(str(bin_path), seq_len)
    info["bytes"] = {"bin": nb, "mask": nmb, "chunks": n_chunks, "chunk_size": C}
    if nb != 2 * nmb:
        fails.append(f"byte count mismatch: bin={nb} != 2*mask={2*nmb}")
    if n_chunks * C != nb // 2:
        fails.append(f"chunks×chunk_size != token count: {n_chunks}×{C} != {nb // 2}")

    # ③ pull a few chunks from the real dataloader
    rows = []
    gen = S.create_sft_dataloader(str(bin_path), mask_kind=kind, mask_path=str(mpath),
                                  seq_len=seq_len, epochs=1, max_tokens=4 * seq_len)
    for i, (x, y, lm, ep) in enumerate(gen):
        r = {"i": i, "keep": int((lm == 1).sum()), "ignore": int((lm == 0).sum()),
             "x_dtype": str(x.dtype)}
        if tuple(x.shape) != (seq_len,) or tuple(y.shape) != (seq_len,) or tuple(lm.shape) != (seq_len,):
            fails.append(f"chunk {i} wrong shape: x={tuple(x.shape)} y={tuple(y.shape)} lm={tuple(lm.shape)}")
        if not bool((y[:-1] == x[1:]).all().item()):
            fails.append(f"chunk {i}: y is not x shifted by one (labels would be misaligned)")
        if str(x.dtype) != "torch.int64" or str(y.dtype) != "torch.int64":
            fails.append(f"chunk {i}: x/y dtype should be int64, got {x.dtype}/{y.dtype}")
        if int(x.max().item()) >= vocab:
            fails.append(f"chunk {i}: token id {int(x.max().item())} >= vocab({vocab})")
        if r["keep"] + r["ignore"] != seq_len:
            fails.append(f"chunk {i}: loss mask is not 0/1 (keep+ignore != {seq_len})")
        if r["keep"] == 0:
            fails.append(f"chunk {i}: the whole chunk is masked out (the prompt segment contributes no loss)")
        rows.append(r)
        if i >= 3:
            break
    info["chunks"] = rows
    if not rows:
        fails.append("create_sft_dataloader produced not a single chunk")

    # ④ startup self-check (ones==0 / zeros==0 triggers SystemExit directly)
    try:
        S.inspect_loss_mask(str(bin_path), kind, str(mpath), seq_len=seq_len, n_probe=16)
    except SystemExit as e:
        fails.append(f"inspect_loss_mask judged this a bug: {e}")

    # ⑤ spot check: the prompt segment (the mask=0 prefix) must be decodable and non-empty
    data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
    mask = np.memmap(str(mask_path), dtype=np.uint8, mode="r")
    ones = np.flatnonzero(np.asarray(mask, dtype=np.uint8) == 1)
    if len(ones):
        j = int(ones[0])
        head = tok.decode(np.asarray(data[:j], dtype=np.uint16).tolist())
        tail = tok.decode(np.asarray(data[j:j + 16], dtype=np.uint16).tolist())
        info["first_sample"] = {"prompt_head": head[:120], "code_head": tail}
        if not head.strip():
            fails.append("the prompt segment of the first sample decodes to empty")
        if int(np.asarray(data).max()) >= vocab:
            fails.append("an id >= vocab appears in the stream (it would be misread as a bit15 embedded mask)")
    return fails, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify the (ids, mask) contract with the real sft_train consumer functions")
    ap.add_argument("--dir", default=str(HERE / "data_instr" / "dryrun"),
                    help="directory (picks up <stems>.bin) or a single bin file")
    ap.add_argument("--stems", nargs="+", default=None, help="defaults to instr_train instr_val")
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--seq-len", type=int, default=1024)
    a = ap.parse_args(argv)

    p = Path(a.dir)
    if p.is_file():
        bins = [p]
    else:
        stems = a.stems or (["instr_train", "instr_val"]
                            if (p / "instr_train.bin").exists() else ["sft_train", "sft_val"])
        bins = [p / f"{s}.bin" for s in stems]

    tok = Tokenizer.from_file(a.tokenizer)
    vocab = tok.get_vocab_size()
    seq_len = a.seq_len or S.ModelConfig.max_seq_len
    print(f"[CFG] tokenizer vocab={vocab}  seq_len={seq_len}  chunk_size={seq_len+1}  "
          f"(ModelConfig.max_seq_len={S.ModelConfig.max_seq_len})")

    all_fails = {}
    for b in bins:
        print(f"\n===== {b} =====")
        if not b.exists():
            print("  [SKIP] does not exist")
            continue
        if b.stat().st_size == 0:
            print("  [SKIP] empty bin (0 bytes, less than one chunk)")
            all_fails[str(b)] = ["empty bin"]
            continue
        fails, info = check_one(b, tok, vocab, seq_len)
        print(f"  resolve_loss_mask: {info['resolve_loss_mask']}")
        print(f"  bytes: {info['bytes']}")
        for r in info["chunks"]:
            print(f"  chunk {r['i']}: keep={r['keep']} ignore={r['ignore']} dtype={r['x_dtype']}")
        if "first_sample" in info:
            print(f"  first prompt: {info['first_sample']['prompt_head']!r}")
            print(f"  first code  : {info['first_sample']['code_head']!r}")
        print("  " + ("✅ PASS" if not fails else "❌ FAIL"))
        for f in fails:
            print(f"    - {f}")
        all_fails[str(b)] = fails

    bad = sum(1 for v in all_fails.values() if v)
    print(f"\n[VERDICT] {'✅ all passed' if bad == 0 else f'❌ {bad} split(s) failed'} "
          f"(checked {sum(1 for b in bins if b.exists())} bin(s))")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
