"""Build a "raw output index" per arm: arm → file → line range → pass count.

Why: reports only give aggregate scores, so checking "what did this arm actually write for this
problem" means digging through the jsonl yourself. This script scans every generation file under
eval/ (the in-line `model` field is the authoritative source) and produces:
  · eval/RAW-INDEX.md   human-readable table
  · eval/raw_index.json the same data, machine-readable
It modifies no existing file, uses no GPU, and can be re-run at any time.
"""
import json, os, glob, datetime

D = os.path.dirname(os.path.abspath(__file__))
# In the repo this script lives in post-training/scripts/ and indexes post-training/results/
EVAL = os.environ.get("EVAL_DIR", os.path.join(D, os.pardir, "results"))

COL = {"sig": "signature column", "nl": "NL column"}
def _guess_col(name):
    """Old files have no column field: only NL-column files carry a _nl / nl_ prefix, everything else is the signature column."""
    if "_nl" in name or name.startswith("nl_"):
        return "nl"
    return "sig"

files = sorted(glob.glob(os.path.join(EVAL, "**", "*.jsonl"), recursive=True))
rows, errs = [], []
for path in files:
    base = os.path.relpath(path, EVAL).replace(os.sep, "/")
    if base.startswith("_"):
        continue
    groups = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                errs.append(f"{base}:{i} {e}")
                continue
            m = (r.get("model") or "(no model field)", r.get("column"))
            g = groups.setdefault(m, {"n": 0, "pass": 0, "lenient": 0, "has_len": False,
                                      "first": i, "last": i, "seeds": set(), "tasks": set(),
                                      "col": r.get("column")})
            g["n"] += 1
            g["pass"] += 1 if r.get("passed") else 0
            if "passed_lenient" in r:
                g["has_len"] = True
                g["lenient"] += 1 if r.get("passed_lenient") else 0
            g["last"] = i
            if r.get("seed") is not None:
                g["seeds"].add(r["seed"])
            if r.get("name"):
                g["tasks"].add(r["name"])
    for (m, _c), g in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rows.append({"file": base, "model": m, "column": g["col"] or _guess_col(base),
                     "lines": [g["first"], g["last"]], "n": g["n"], "passed": g["pass"],
                     "lenient": g["lenient"] if g["has_len"] else None,
                     "seeds": len(g["seeds"]), "tasks": len(g["tasks"])})



def bucket(f):
    # use the basename: paths are relative to the results dir and may carry a
    # "generations/" prefix
    base = os.path.basename(f)
    if base.startswith("heldout"):
        return "held-out set (12 new tasks × 10 seeds)"
    if "noise20" in base:
        return "20-run precision pass (11 tasks × 20 seeds, isolated run)"
    return "5-run pass (11 tasks × 5 seeds)"


order = ["5-run pass (11 tasks × 5 seeds)", "20-run precision pass (11 tasks × 20 seeds, isolated run)", "held-out set (12 new tasks × 10 seeds)"]
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
out = [f"# Raw output index (arm → file → line range)", "",
       f"generated: {now} · script `fine-tuning/build_raw_index.py` (re-runnable, read-only)", "",
       "**How to use**: when a report shows a score for some arm, find it in the table below → open the",
       "matching file → `sed -n 'A,Bp'` to pull that arm's full raw generations (one JSON per line: `prompt` / `completion` / `passed`).",
       "Line numbers are **1-based physical lines**, because one file may mix two checkpoints of the same arm (`final` / `bestval`), so the in-line `model` field wins.", ""]
for b in order:
    sel = [r for r in rows if bucket(r["file"]) == b]
    if not sel:
        continue
    out += [f"## {b}", "", "| arm (in-line model field) | column | file | line range | n | passed | lenient | seeds / tasks |",
            "|---|---|---|---|---|---|---|---|"]
    for r in sorted(sel, key=lambda x: (x["model"], x["file"])):
        len_s = "—" if r["lenient"] is None else str(r["lenient"])
        out.append(f"| `{r['model']}` | {COL.get(r['column'], r['column'])} | `{r['file']}` | "
                   f"{r['lines'][0]}–{r['lines'][1]} | {r['n']} | {r['passed']} | {len_s} | "
                   f"{r['seeds']} / {r['tasks']} |")
    out.append("")

out += ["## Lenient pass (re-judged to allow the \"sort in place\" convention)", "",
        "The strict column requires the function to return a result, so a correct implementation that mutates the list in place and returns None is scored as failed.",
        "The lenient column re-judges **failed samples** only (when the sole argument is a list, in-place mutation also passes); the input is the same batch of generations files.", "",
        "| file | contents |", "|---|---|",
        "| `eval/lenient_results.json` | strict/lenient score per model (55 points per column) + source file name, produced by `rejudge_lenient.py` (CPU-only, re-runnable) |",
        "| the \"lenient\" column in the held-out section table | strict and lenient are both judged at generation time (field `passed_lenient`) |", ""]
out += ["## Summary file reference", "", "| file | contents |", "|---|---|",
        "| `eval_sft_summary.json` | signature-column summary (per-model per-task `x/5`) |",
        "| `nl_eval_summary.json` | NL-column summary |",
        "| `heldout_summary.json` / `heldout_results.json` | held-out set: strict/lenient for both columns, per model |",
        "| `noise20_*_{sig,nl}.json` | per-model summary of the 20-run precision pass |",
        "| `soup_summary.md` | weight soup candidate table |",
        f"", f"{len(rows)} records (arm × file × column) covering {len(set(r['model'] for r in rows))} distinct model names."]
if errs:
    out += ["", "## Parse warnings", ""] + [f"- `{e}`" for e in errs[:20]]
md = "\n".join(out) + "\n"
open(os.path.join(EVAL, "RAW-INDEX.md"), "w", encoding="utf-8").write(md)
json.dump(rows, open(os.path.join(EVAL, "raw_index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {os.path.join(EVAL, 'RAW-INDEX.md')} ({len(md)} chars) and eval/raw_index.json ({len(rows)} rows)")
print(f"covering {len(set(r['model'] for r in rows))} model names; {len(errs)} parse warnings")
for b in order:
    print("  ", b, len([r for r in rows if bucket(r['file']) == b]), "rows")