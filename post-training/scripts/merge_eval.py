#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_eval.py — merge a new evaluation run into an existing summary (same model name is overwritten by the new result, everything else is kept).

Why needed: eval_sft.py / eval_nl.py overwrite their whole summary file on every run, and arms A/B/C are
run sequentially → without merging, the earlier arms' scores are lost.

Two shapes are supported:
  - list          : eval_sft_summary.json (elements contain "model")
  - {"meta":…, "models":[…]}, eval_nl_summary.json (models elements contain "model")

Usage:
  py -3.12 merge_eval.py --prev eval/_prev.json --new eval/eval_sft_summary.json --out eval/eval_sft_summary.json
  (when --prev does not exist, --new is copied straight to --out)
"""
import argparse
import json
import shutil
from pathlib import Path


def load(p):
    p = Path(p)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] cannot read {p}: {e}")
        return None


def merge_lists(prev, new, key="model"):
    new_names = {m.get(key) for m in new}
    return [m for m in prev if m.get(key) not in new_names] + new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    new = load(args.new)
    if new is None:
        raise SystemExit(f"[ERR] new result file does not exist or is unreadable: {args.new}")
    prev = load(args.prev)

    if prev is None:
        Path(args.out).write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[MERGE] no previous results → writing {args.out} directly")
        return

    if isinstance(new, dict) and "models" in new:
        merged = dict(new)
        merged["models"] = merge_lists(prev.get("models", []), new["models"])
    elif isinstance(new, list):
        merged = merge_lists(prev if isinstance(prev, list) else [], new)
    else:
        raise SystemExit("[ERR] unsupported result shape")

    Path(args.out).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    names = [m.get("model") for m in (merged["models"] if isinstance(merged, dict) else merged)]
    print(f"[MERGE] {args.out} ← {len(names)} models total: {', '.join(str(n) for n in names)}")


if __name__ == "__main__":
    main()
