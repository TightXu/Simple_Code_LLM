#!/usr/bin/env python3
"""WSM merge: uniformly average --n checkpoints from the last --window-steps.

Reference: arxiv 2507.17634 finds the merge window (duration) matters more
than how many checkpoints you average. If fewer checkpoints are available
than --n, use them all.

Usage:
  python merge_checkpoints.py --n 10 --window-steps 10000 --ckpt-dir checkpoints_wsm --out checkpoints_wsm/merged
"""
import argparse
from pathlib import Path
import torch


def find_milestones(base):
    steps = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                steps.append((int(d.name.split("_")[1]), d))
            except (IndexError, ValueError):
                pass
    steps.sort()
    return steps  # [(step, dir), ...] ascending by step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints_wsm")
    ap.add_argument("--n", type=int, default=10, help="how many checkpoints to average")
    ap.add_argument("--window-steps", type=int, default=10000,
                    help="only take checkpoints within the last N steps, then pick --n of them evenly")
    ap.add_argument("--out", default="checkpoints_wsm/merged")
    args = ap.parse_args()

    base = Path(args.ckpt_dir)
    ckpts = find_milestones(base)
    if not ckpts:
        print("no checkpoints to merge:", base)
        return 1

    if args.window_steps:
        max_step = ckpts[-1][0]
        ckpts = [(s, d) for s, d in ckpts if s >= max_step - args.window_steps]
        print(f"window (last {args.window_steps} steps): {len(ckpts)} checkpoints")

    # Pick --n evenly: include first and last, linear interpolation in between
    if len(ckpts) <= args.n:
        chosen = ckpts
    else:
        stride = (len(ckpts) - 1) / (args.n - 1)
        idxs = [int(round(i * stride)) for i in range(args.n)]
        chosen = [ckpts[i] for i in idxs]

    dirs = [d for _, d in chosen]
    print("merging", len(dirs), "checkpoints:", [d.name for d in dirs])

    merged = None
    for d in dirs:
        ckpt = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
        sd = ckpt["model"]
        if merged is None:
            merged = {k: v.float().clone() for k, v in sd.items()}
        else:
            for k in merged:
                merged[k] = merged[k] + sd[k].float()
    for k in merged:
        merged[k] = merged[k] / len(dirs)

    last = torch.load(dirs[-1] / "checkpoint.pt", map_location="cpu", weights_only=False)
    last["model"] = merged
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(last, out / "checkpoint.pt", _use_new_zipfile_serialization=False)
    print("saved:", out / "checkpoint.pt")


if __name__ == "__main__":
    main()
