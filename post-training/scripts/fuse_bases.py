"""Fused base: weighted average of the merged and best_val weights (model soup).

Purpose: user request -- fuse best_val and merged once more and see whether it beats the current base.
All outputs stay inside fine-tuning/ (isolated); checkpoints_wsm/ is untouched.

Usage:
  py -3.12 fuse_bases.py --w-best 0.25 0.5 0.75
     → ck_fusion/wbest025/checkpoint.pt, wbest050, wbest075
     (weight definition: out = (1-w) * merged + w * best_val)
"""
import argparse, gc, json, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
A_PATH = ROOT / "checkpoints_wsm" / "merged" / "checkpoint.pt"     # primary
B_PATH = ROOT / "checkpoints_wsm" / "best_val" / "checkpoint.pt"   # blend-in


def blend(dst, src, wa, wb, path=""):
    """Blend src into dst (in place) with weight wb; the dst side uses coefficient wa. Non-float tensors keep dst."""
    if isinstance(dst, dict):
        for k in list(dst.keys()):
            if k in src:
                blend(dst[k], src[k], wa, wb, path + "/" + str(k))
    elif isinstance(dst, list):
        for i, v in enumerate(dst):
            if i < len(src):
                blend(v, src[i], wa, wb, path + "[%d]" % i)
    elif torch.is_tensor(dst):
        if torch.is_floating_point(dst) and dst.shape == src.shape:
            dst.mul_(wa).add_(src, alpha=wb)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(A_PATH), help="primary checkpoint (default merged)")
    ap.add_argument("--b", default=str(B_PATH), help="checkpoint to blend in (default best_val)")
    ap.add_argument("--label", default="best", help="output directory prefix label")
    ap.add_argument("--w-best", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--out-dir", default=str(HERE / "ck_fusion"))
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    print("A (primary):", args.a)
    print("B (blend-in):", args.b)
    t0 = time.time()
    a = torch.load(args.a, map_location="cpu", weights_only=False)
    print("loaded merged  %.1fs" % (time.time() - t0))

    for wb in args.w_best:
        wa = 1.0 - wb
        t1 = time.time()
        b = torch.load(args.b, map_location="cpu", weights_only=False)
        blk = {k: (v.copy() if torch.is_tensor(v) else None) for k, v in a.items()}
        # restore a's original content into blk (copy.deepcopy is safer for dict/list deep copies)
        import copy
        blk = copy.deepcopy(a)
        blend(blk, b, wa, wb)
        del b
        gc.collect()
        ts = dict(blk.get("training_state", {}))
        _an, _bn = Path(args.a).parent.name, Path(args.b).parent.name
        ts.update({"step": 0, "total_tokens": 0, "fusion": "%s*%.2f + %s*%.2f" % (_an, wa, _bn, wb)})
        blk["training_state"] = ts
        out = out_root / ("%s%03d" % (args.label, round(wb * 100)))
        out.mkdir(parents=True, exist_ok=True)
        torch.save(blk, out / "checkpoint.pt")
        meta = {
            "out": str(out / "checkpoint.pt"),
            "formula": "out = %.2f * %s + %.2f * %s" % (wa, _an, wb, _bn),
            "w_a": wa, "w_b": wb, "a": _an, "b": _bn,
            "sources": {_an: args.a, _bn: args.b},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out / "fusion_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[OK] %s  (%.1fs)" % (out / "checkpoint.pt", time.time() - t1))
        del blk
        gc.collect()
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
