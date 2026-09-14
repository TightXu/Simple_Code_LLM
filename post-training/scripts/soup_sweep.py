"""Weight soup sweep: interpolate the weights of two data-specialized experts, then evaluate both columns.

Background: so far every arm has been "one pass over one mixed dataset", so combination only happened
at the data layer. The weight layer has barely been tried -- A7-final is the signature expert (36/38)
and G8-bestval is the NL expert (47); both share the same root recipe and differ only in the data mix,
so weight interpolation is the only "get both without training" option left.

Stages:
  1) build  -- call soup_models.py to build candidates (pure CPU, outputs under ck_soup/<label>_<alpha>)
  2) eval   -- call eval_sft.py (signature column) and eval_nl.py (NL column), reusing the existing
               merge_eval merge/summary
  3) report -- print the comparison table (including reference rows)

Safety design:
  - refuse to start while training holds the GPU (unless --force), so we never fight G9/G10 for the
    card and slow both sides down
  - idempotent: candidates that already have a checkpoint.pt are skipped; eval results are merged into
    the existing summary instead of overwriting history
  - reads source only; writes only ck_soup/, eval/ and logs/, never checkpoints_wsm/

Usage:
  py -3.12 soup_sweep.py --stage all            # build + eval + summary (default)
  py -3.12 soup_sweep.py --stage build          # build only
  py -3.12 soup_sweep.py --stage eval           # eval only (assumes built)
  py -3.12 soup_sweep.py --pairs A7G8           # run a single pair
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGDIR = HERE / "logs"
EVAL = HERE / "eval"
SOUP_DIR = HERE / "ck_soup"
PY = sys.executable

# ── candidate definitions: (label, A, B, [alphas]), alpha = share of B; out = (1-alpha)A + alphaB ──
PAIRS = [
    ("A7G8",   "ckpt_a7/final",    "ckpt_g8/best_val", [0.20, 0.35, 0.50, 0.65, 0.80]),
    ("G5G8",   "ckpt_g5/final",    "ckpt_g8/best_val", [0.50]),
    ("G8self", "ckpt_g8/final",    "ckpt_g8/best_val", [0.50]),
    # Round 2: combine the two extremes of the ratio curve (G9 = best NL 50 / G5 = best signature 36)
    # Rationale: A7xG8 in round 1 had no exploitable trade-off (G8 dominates on every count); a real
    # "winning both columns" candidate has to come from the extreme pair.
    ("G9G5",   "ckpt_g9/best_val", "ckpt_g5/final",    [0.30, 0.50, 0.70]),
    ("G9G8",   "ckpt_g9/best_val", "ckpt_g8/best_val", [0.50]),
    ("G10G5",  "ckpt_g10/final",   "ckpt_g5/final",    [0.50]),
]
# model names used at eval time (must match those registered in eval_sft.py / eval_nl.py)
NAME_FMT = {
    "A7G8":   "SOUP-A7G8-%03d",
    "G5G8":   "SOUP-G5G8-%03d",
    "G8self": "SOUP-G8self-%03d",
    "G9G5":   "SOUP-G9G5-%03d",
    "G9G8":   "SOUP-G9G8-%03d",
    "G10G5":  "SOUP-G10G5-%03d",
}
DIR_FMT = {
    "A7G8":   "sA7G8_%03d",
    "G5G8":   "sG5G8_%03d",
    "G8self": "sG8self_%03d",
    "G9G5":   "sG9G5_%03d",
    "G9G8":   "sG9G8_%03d",
    "G10G5":  "sG10G5_%03d",
}
REFERENCE = ["G8-g3to1-bestval", "G5-g15-final", "A7-bigseed-final", "G3-g11-bestval",
             "A9-1to1-bestval", "G9-g5to1-bestval", "G10-g8to1-bestval"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_busy():
    """Return the python processes holding the GPU (used to tell whether training is running)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    hits = []
    for line in out.splitlines():
        if "python" in line.lower():
            hits.append(line.strip())
    return hits


def run(cmd, logfile):
    """Run a subprocess and tee its output into the log file as well."""
    with open(logfile, "a", encoding="utf-8", errors="replace") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def stage_build(labels, force):
    made = []
    for label, a, b, alphas in PAIRS:
        if label not in labels:
            continue
        todo = [x for x in alphas if force or not (SOUP_DIR / (DIR_FMT[label] % round(x * 100)) / "checkpoint.pt").is_file()]
        if not todo:
            log(f"build {label}: all already exist, skipping")
            continue
        pa, pb = HERE / a, HERE / b
        for p in (pa, pb):
            if not (p / "checkpoint.pt").is_file():
                raise SystemExit(f"missing source checkpoint: {p}/checkpoint.pt")
        log(f"build {label}: alpha={todo}  ← {a} ⊕ {b}")
        rc = run([PY, "soup_models.py", "--a", str(pa), "--b", str(pb),
                  "--alphas", *[str(x) for x in todo],
                  "--label", f"s{label}_", "--out-dir", str(SOUP_DIR)],  # the leading s must match DIR_FMT
                 LOGDIR / "soup_sweep.log")
        if rc != 0:
            raise SystemExit(f"soup_models failed rc={rc}, see logs/soup_sweep.log")
        made += [(label, x) for x in todo]
    return made


def soup_names(labels):
    names = []
    for label, _a, _b, alphas in PAIRS:
        if label in labels:
            names += [NAME_FMT[label] % round(x * 100) for x in alphas]
    return names


def stage_eval(labels):
    names = soup_names(labels)
    if not names:
        return
    spec = ",".join(names)
    for col, script, summary, gens, tag in (
        ("signature column", "eval_sft.py", "eval_sft_summary.json", "eval_sft_generations.jsonl", "sig"),
        ("NL column", "eval_nl.py", "nl_eval_summary.json", "nl_eval_generations.jsonl", "nl"),
    ):
        cur = EVAL / summary
        prev = EVAL / f"_soup_prev_{tag}.json"
        if cur.is_file():
            shutil.copy2(cur, prev)
        cmd = [PY, script, "--models", spec] + (["--skip-humaneval"] if script == "eval_sft.py" else [])
        log(f"eval {col}: {len(names)} models (this step holds the GPU)")
        rc = run(cmd, LOGDIR / f"soup_eval_{tag}.log")
        if rc != 0:
            log(f"{col} eval rc={rc}, not merged (see logs/soup_eval_{tag}.log)")
            continue
        if prev.is_file():
            run([PY, "merge_eval.py", "--prev", str(prev), "--new", str(cur), "--out", str(cur)],
                LOGDIR / "soup_sweep.log")
        gel = EVAL / gens
        if gel.is_file():
            shutil.copy2(gel, EVAL / f"generations_soup_{tag}.jsonl")
        log(f"{col} done and merged")


def stage_report(labels):
    def tot(m):
        return sum(int(v.split("/")[0]) for v in m["tasks"].values())

    sig = {m["model"]: m for m in json.load(open(EVAL / "eval_sft_summary.json", encoding="utf-8"))}
    nl = {m["model"]: m for m in json.load(open(EVAL / "nl_eval_summary.json", encoding="utf-8"))["models"]}
    names = soup_names(labels)
    rows = []
    for n in names + REFERENCE:
        has = (n in sig) or (n in nl)
        if has or n in names:      # list candidates even without results (so a missing run is obvious)
            rows.append((n, tot(nl[n]) if n in nl else None, tot(sig[n]) if n in sig else None))
    print("\n" + "=" * 78)
    print("model".ljust(24), " NL  sig  sum")
    print("-" * 78)
    for n, a, b in rows:
        mark = "  ←" if n in names else ""
        sa = "  ?" if a is None else f"{a:>3}"
        sb = "  ?" if b is None else f"{b:>3}"
        st = "  ?" if (a is None or b is None) else f"{a+b:>4}"
        print(f"{n:<24}{sa}{sb}{st}{mark}")
    print("=" * 78)
    out = EVAL / "soup_summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("| model | NL | sig | sum |\n|---|---|---|---|\n")
        for n, a, b in rows:
            f.write(f"| {n} | {a if a is not None else '—'} | {b if b is not None else '—'} | "
                    f"{(a+b) if (a is not None and b is not None) else '—'} |\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "build", "eval", "report"], default="all")
    ap.add_argument("--pairs", default=",".join(l for l, *_ in PAIRS), help="comma-separated, choose from " + ",".join(l for l, *_ in PAIRS))
    ap.add_argument("--force", action="store_true", help="skip the 'GPU is busy' check")
    args = ap.parse_args()
    labels = [s.strip() for s in args.pairs.split(",") if s.strip()]
    LOGDIR.mkdir(exist_ok=True)

    if args.stage in ("all", "eval") and not args.force:
        busy = gpu_busy()
        if busy:
            print("GPU is held by python processes; refusing to start evaluation (to avoid fighting training for the card):")
            for b in busy:
                print("   ", b)
            print("add --force if you really want to run it")
            return 2

    t0 = time.time()
    if args.stage in ("all", "build"):
        stage_build(labels, args.force)
    if args.stage in ("all", "eval"):
        stage_eval(labels)
    if args.stage in ("all", "report"):
        stage_report(labels)
    log(f"done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
