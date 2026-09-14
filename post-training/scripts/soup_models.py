"""Weight interpolation (model soup) between two SFT experts -- testing the "combination" route.

Background: so far every arm has been "one mixed dataset, one pass", and the NL and signature columns
fight each other for capacity along the same trade-off curve (the data layer has been explored
exhaustively). The weight layer has barely been tried: interpolating two experts that are each strong
somewhere should, in principle, deliver both capabilities without touching the data.

Differences from fuse_bases.py:
  - only ckpt["model"] is taken; optimizer/scheduler are dropped (saves memory, and they are not needed)
  - interpolation is done in float32 and cast back to the original dtype (avoids the precision loss of
    calling mul_ directly on bf16)
  - one output per alpha, each with a fusion_meta.json recording the formula and the source shas

Weight definition: out = (1 - alpha) * A + alpha * B   (alpha = share of B)

Example:
  py -3.12 soup_models.py --a ckpt_g8/best_val --b ckpt_a7/final \
        --alphas 0.2 0.35 0.5 0.65 --label g8xa7 --out-dir ck_soup
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent


def resolve(p: str) -> Path:
    """Accept a directory (checkpoint.pt appended automatically) or a file path."""
    q = Path(p)
    if q.is_dir():
        q = q / "checkpoint.pt"
    if not q.is_file():
        raise SystemExit(f"checkpoint not found: {q}")
    return q


def sha8(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:8]


def load_model_weights(p: Path):
    """Return (weights, config, training_state).

    Note: config must be carried into the output unchanged -- the eval side builds the model from the
    checkpoint config (the default ModelConfig has d_ff=3840 while the actual training config is 4096);
    dropping config causes a load_state_dict shape mismatch. optimizer/scheduler are deliberately
    dropped: they save memory and are not needed.
    """
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = ck.get("model")
    if sd is None:
        raise SystemExit(f"{p} has no 'model' key")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    cfg = dict(ck.get("config") or {})
    ts = dict(ck.get("training_state") or {})
    return sd, cfg, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="primary checkpoint (directory or checkpoint.pt)")
    ap.add_argument("--b", required=True, help="checkpoint to blend in")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                    help="share of B, any number of values; out = (1-a)*A + a*B")
    ap.add_argument("--label", default="soup", help="output directory prefix")
    ap.add_argument("--out-dir", default=str(HERE / "ck_soup"))
    args = ap.parse_args()

    pa, pb = resolve(args.a), resolve(args.b)
    print(f"A = {pa}\nB = {pb}")
    t0 = time.time()
    sa, cfg_a, ts_a = load_model_weights(pa)
    sb, cfg_b, ts_b = load_model_weights(pb)
    if cfg_a and cfg_b and cfg_a != cfg_b:
        print(f"[warn] A/B configs differ: A={cfg_a} B={cfg_b} (output keeps A's config)")
    print(f"loaded: A {len(sa)} tensors / B {len(sb)} tensors  ({time.time()-t0:.1f}s)")

    keys_a, keys_b = set(sa), set(sb)
    if keys_a != keys_b:
        only_a, only_b = sorted(keys_a - keys_b)[:5], sorted(keys_b - keys_a)[:5]
        print(f"[warn] key mismatch: A-only {only_a} / B-only {only_b} (these keys keep A's values)")

    # shape/dtype self-check: misaligned keys keep A and are recorded in the meta
    skipped = []
    for k in keys_a & keys_b:
        if sa[k].shape != sb[k].shape or sa[k].dtype != sb[k].dtype:
            skipped.append(k)
    if skipped:
        print(f"[warn] {len(skipped)} keys have a shape/dtype mismatch, keeping A: {skipped[:5]}")
        for k in skipped:
            sb.pop(k, None)

    blend_keys = [k for k in sa if k in sb and torch.is_floating_point(sa[k])]
    nonfloat = [k for k in sa if k in sb and not torch.is_floating_point(sa[k])]
    print(f"float tensors participating in the interpolation: {len(blend_keys)}; non-float (keep A): {len(nonfloat)}")

    A_SHA, B_SHA = sha8(pa), sha8(pb)   # hash each file once only (hashing in the loop would re-read tens of GB)
    # self-check snapshot: first 8 elements of every tensor to be interpolated (pre-build), compared against in and after the loop
    snap_a = {k: sa[k].flatten()[:8].clone() for k in blend_keys}
    snap_b = {k: sb[k].flatten()[:8].clone() for k in blend_keys}

    os.makedirs(args.out_dir, exist_ok=True)
    meta_all = []
    for a_b in args.alphas:
        t1 = time.time()
        a_a = 1.0 - a_b
        out = {"model": {}, "config": cfg_a}
        for k, v in sa.items():
            if k not in sb or k not in blend_keys:
                out["model"][k] = v.clone()
                continue
            f = v.clone().to(torch.float32)      # clone is mandatory: v.to(float32) returns v itself when v is already float32,
            f.mul_(a_a).add_(sb[k].to(torch.float32), alpha=a_b)   # so mul_/add_ directly would mutate parent A in place
            out["model"][k] = f.to(v.dtype)
            # self-check: reconcile against the pre-build A snapshot (this fires immediately if the parent was mutated in place)
            _got = out["model"][k].flatten()[:8].float()
            _want = (1 - a_b) * snap_a[k].float() + a_b * snap_b[k].float()
            _err = (_got - _want).abs().max().item()
            _tol = 5e-3 if v.dtype == torch.bfloat16 else 1e-4
            if _err > _tol:
                raise SystemExit(
                    f"[FATAL] self-check failed alpha={a_b} {k}: deviation from (1-alpha)*A+alpha*B is {_err:.2e} (tolerance {_tol:.0e}). "
                    "Usual cause: the parent was mutated in place, so the multiple alphas in one build compounded.")
        out["fusion"] = {"formula": f"out = {a_a:.2f} * A + {a_b:.2f} * B",
                         "A": str(pa), "B": str(pb), "A_sha8": A_SHA, "B_sha8": B_SHA}
        out["training_state"] = {
            "step": 0, "total_tokens": 0, "best_val_loss": float("nan"),
            "soup": f"{a_a:.2f}*{pa.parent.name or pa.name} + {a_b:.2f}*{pb.parent.name or pb.name}",
        }
        out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        d = Path(args.out_dir) / f"{args.label}{round(a_b*100):03d}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(out, d / "checkpoint.pt", _use_new_zipfile_serialization=False)
        meta = {
            "out": str(d / "checkpoint.pt"),
            "formula": f"out = {a_a:.2f} * A + {a_b:.2f} * B",
            "w_a": a_a, "w_b": a_b,
            "A": str(pa), "B": str(pb),
            "A_sha8": A_SHA, "B_sha8": B_SHA,
            "config_used": cfg_a,
            "n_blended": len(blend_keys), "keys_kept_from_A": skipped,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (d / "fusion_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_all.append(meta)
        print(f"[OK] alpha={a_b:.2f} -> {d}/checkpoint.pt  ({time.time()-t1:.1f}s)")

    # after the loop, confirm parent A was never mutated in place (read-only contract)
    for k in blend_keys:
        _d = (sa[k].flatten()[:8].float() - snap_a[k].float()).abs().max().item()
        if _d > 1e-6:
            raise SystemExit(f"[FATAL] parent A field {k} was mutated in place (deviation {_d:.2e}) -- interpolation must use a copy")

    (Path(args.out_dir) / f"{args.label}_all.json").write_text(
        json.dumps(meta_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
