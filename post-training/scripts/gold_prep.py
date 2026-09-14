#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gold_prep.py — prerequisite for the gold-standard route: turns the problem set extracted by
`esd_pipeline extract` into the "candidates + results" pair that `pairs_to_bins.py --from-esd` needs.

Idea: gold standard = problems in the dataset whose "reference solution + tests pass 10/10" -- the
reference solution is itself an **execution-verified** answer, so using it as both the "candidate" and
the "result = passed" reuses the already-verified pairs_to_bins path and avoids writing a second
bin-generation implementation.

Usage:
  py -3.12 gold_prep.py --problems esd_gold/*.jsonl --outdir esd_gold
Outputs:
  esd_gold/candidates.jsonl   {id, completion}
  esd_gold/results.jsonl      {id, passed: True, reason: "gold reference solution"}
  esd_gold/problems_all.jsonl merged, deduplicated problem set
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    seen, problems = set(), []
    for f in args.problems:
        f = Path(f)
        if not f.exists():
            print(f"[WARN] skipping missing file: {f}")
            continue
        n0 = len(problems)
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = d.get("id")
            if not pid or pid in seen:
                continue
            if not d.get("ref_solution") or not d.get("prompt"):
                continue
            seen.add(pid)
            problems.append(d)
        print(f"[IN ] {f.name}: +{len(problems)-n0} problems")

    if not problems:
        raise SystemExit("[ERR] no usable problems → not writing empty files, exiting")

    with open(out / "problems_all.jsonl", "w", encoding="utf-8") as fp, \
         open(out / "candidates.jsonl", "w", encoding="utf-8") as fc, \
         open(out / "results.jsonl", "w", encoding="utf-8") as fr:
        for d in problems:
            fp.write(json.dumps(d, ensure_ascii=False) + "\n")
            fc.write(json.dumps({"id": d["id"], "completion": d["ref_solution"]}, ensure_ascii=False) + "\n")
            fr.write(json.dumps({"id": d["id"], "passed": True,
                                 "reason": "gold reference solution", "seconds": 0.0}, ensure_ascii=False) + "\n")

    n = len(problems)
    tok = sum(int(d.get("meta", {}).get("tokens_prompt", 0)) for d in problems)
    print(f"[OUT] {n} problems → problems_all.jsonl / candidates.jsonl / results.jsonl (deduplicated)")
    print(f"[OUT] prompt side totals roughly {tok/1e6:.1f}M tokens (solution side not counted)")
    print("\nNext step:\n  py -3.12 pairs_to_bins.py --from-esd 1 \\\n"
          f"    --problems {out/'problems_all.jsonl'} --candidates {out/'candidates.jsonl'} "
          f"--results {out/'results.jsonl'} \\\n    --out data_gold --crossing-policy keep")


if __name__ == "__main__":
    main()
