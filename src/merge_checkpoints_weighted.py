#!/usr/bin/env python3
"""WSM weighted merge (streaming, weighted by LR segment).

Within the last --window-steps, each checkpoint exposes the optimizer LR. Two
segments are detected automatically:
  - cooldown segment (lr ~ 1e-4): gets --weight-cooldown of the total (default 0.30)
  - stale segment (lr ~ 2.5e-4):   gets the remaining (default 0.70)
Checkpoints inside a segment split that segment's weight evenly.

Streaming: load one checkpoint at a time, accumulate in-place, release it
immediately. Peak memory ~= one checkpoint + merged ~ 7GB (each checkpoint is
5.2GB: model 1.7GB + optimizer 3.5GB), which fits a 16GB machine.

Usage:
  py -3.12 merge_checkpoints_weighted.py --ckpt-dir checkpoints_wsm --window-steps 10000 --weight-cooldown 0.30 --out checkpoints_wsm/merged
"""
import argparse
import gc
from pathlib import Path
import torch


def find_steps(base):
    out = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                out.append((int(d.name.split("_")[1]), d))
            except (IndexError, ValueError):
                pass
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints_wsm")
    ap.add_argument("--window-steps", type=int, default=10000)
    ap.add_argument("--weight-cooldown", type=float, default=0.30,
                    help="total weight for the cooldown segment (lr=1e-4); stale segment gets 1 - this")
    ap.add_argument("--lr-threshold", type=float, default=1.5e-4,
                    help="LR below this counts as cooldown segment")
    ap.add_argument("--out", default="checkpoints_wsm/merged")
    args = ap.parse_args()

    base = Path(args.ckpt_dir)
    steps = find_steps(base)
    if not steps:
        print("no checkpoints:", base)
        return 1

    max_step = steps[-1][0]
    window = [(s, d) for s, d in steps if s >= max_step - args.window_steps]
    print(f"window (last {args.window_steps} steps): {len(window)} checkpoints")

    # First pass: read each checkpoint's LR, classify into segments
    cool, bug = [], []
    for s, d in window:
        ckpt = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
        lr = ckpt["optimizer"]["param_groups"][0]["lr"]
        (cool if lr < args.lr_threshold else bug).append(d)
        del ckpt
        gc.collect()
    print(f"cooldown segment (lr~1e-4): {len(cool)}")
    print(f"stale segment (lr~2.5e-4):  {len(bug)}")

    w_cool = args.weight_cooldown
    w_bug = 1.0 - w_cool
    if not cool:
        print("[WARN] no cooldown checkpoints, all weight to stale segment")
        w_bug = 1.0
    if not bug:
        print("[WARN] no stale checkpoints, all weight to cooldown segment")
        w_cool = 1.0

    w_cool_each = w_cool / len(cool) if cool else 0.0
    w_bug_each = w_bug / len(bug) if bug else 0.0
    print(f"per-checkpoint weight: cooldown {w_cool_each:.4f}, stale {w_bug_each:.4f}")

    # Second pass: weighted accumulation (in-place, streaming)
    merged = None
    for d in cool + bug:
        w = w_cool_each if d in cool else w_bug_each
        ckpt = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
        sd = ckpt["model"]
        if merged is None:
            merged = {k: v.float() * w for k, v in sd.items()}
        else:
            for k in merged:
                merged[k].add_(sd[k].float(), alpha=w)
        del ckpt, sd
        gc.collect()

    # Save: take the last checkpoint's structure, replace model with merged
    last = torch.load(window[-1][1] / "checkpoint.pt", map_location="cpu", weights_only=False)
    del last["model"]  # free the old model before placing merged, lowers peak save memory
    last["model"] = merged
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(last, out / "checkpoint.pt", _use_new_zipfile_serialization=False)
    print("[OK] saved:", out / "checkpoint.pt")


if __name__ == "__main__":
    main()
