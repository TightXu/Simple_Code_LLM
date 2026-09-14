"""Offline lenient re-judge: re-judge existing generations without a GPU and without editing the eval scripts.

Background: under the strict criterion truncate() already cut the driver code after the function, so a
      mergesort/quicksort failure really means "sorted in place, returned None" -- a fully correct algorithm
      scored as failure, i.e. the signature column is polluted by "does it return". This script re-judges
      every failed sample with the lenient criterion:
  1) allow in-place mutation: for a test shaped like f(<list>) == <expected>, both f(args) == expected and args == expected
     pass (relaxed only when the sole argument is a list literal of length >= 2; empty/single-element cases stay strict)
  2) keep the strict-criterion code extraction (prompt + truncate(completion))
Output: strict / lenient score per model for both columns (rescaled to a 55-point scale)."""
import ast
import json
import os
import re
import subprocess
import sys

# scripts live next to the code they import (repo: post-training/scripts/)
FT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FT)
import eval_sft as E  # noqa: E402

TESTS = {}
for t in (E.ALGORITHMS + E.SIMPLE_TASKS):
    TESTS[t["name"]] = t["tests"]

CALL_RE = re.compile(r"^([A-Za-z_]\w*)\((.*)\)\s*==\s*(.+)$", re.S)


def lenient_test(t, i):
    """Rewrite `f(a, b) == exp` into an assertion that also allows in-place mutation."""
    m = CALL_RE.match(t.strip())
    if not m:
        return f"assert {t}"
    fname, argstr, exp = m.group(1), m.group(2), m.group(3)
    try:
        parsed = ast.literal_eval("(" + argstr + ",)")
    except Exception:
        parsed = None
    code = [f"_A{i} = ({argstr},)", f"_R{i} = {fname}(*_A{i})"]
    cond = f"(_R{i} == {exp})"
    # allow in-place mutation when the sole argument is a list (incl. empty/single-element: only a mutable object can reveal in-place behaviour there)
    if isinstance(parsed, tuple) and len(parsed) == 1 and isinstance(parsed[0], list):
        cond += f" or (_A{i}[0] == {exp})"
    return "\n".join(code) + f"\nassert {cond}"


def judge(full, tests):
    body = "\n".join(lenient_test(t, i) for i, t in enumerate(tests))
    script = "%s\n\n%s\nprint('PASS')\n" % (full, body)
    try:
        r = subprocess.run([sys.executable, "-c", script], timeout=5,
                           capture_output=True, text=True)
        return r.returncode == 0 and "PASS" in r.stdout
    except Exception:
        return False


def rejudge(tag, col):
    fn = os.path.join(FT, "eval", f"generations_noise20_{tag}_{col}.jsonl")
    if not os.path.exists(fn):
        return None
    n = n_strict = n_lenient = 0
    by_task = {}
    with open(fn, encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            n += 1
            strict = bool(m.get("passed"))
            lenient = strict
            if not strict:
                tests = TESTS.get(m["name"])
                if tests:
                    full = m["prompt"] + E.truncate(m["completion"])
                    lenient = judge(full, tests)
            n_strict += strict
            n_lenient += lenient
            t = by_task.setdefault(m["name"], [0, 0])
            t[1] += 1
            t[0] += strict
            if lenient and not strict:
                t.append(1) if False else None
    return n, n_strict, n_lenient, by_task


LABELS = [("G9-g5to1-s45-bestval", "g9s45"), ("G9-g5to1-s46-bestval", "g9s46"),
          ("G9-g5to1-s44-bestval", "g9s44"), ("G9-g5to1-s43-bestval", "g9s43"),
          ("G9-g5to1-bestval(s42)", "g9bv"), ("SOUP-G9x3-bestval", "g9x3bv"),
          ("SOUP-G9top2-bestval", "g9top2"), ("SOUP-G8x3-bestval", "soup3"),
          ("G8-g3to1-bestval(3:1 s42)", "g8bv"), ("SOUP-G5x3-final", "g5x3fi"),
          ("SOUP-G9G5-030", "g9g5_030")]

print(f"{'model':<26}{'col':<5}{'strict':>8}{'lenient':>8}{'delta':>8}{'strict(55)':>10}{'lenient(55)':>10}")
print("-" * 78)
rows = {}
for label, tag in LABELS:
    out = {}
    for col in ("sig", "nl"):
        r = rejudge(tag, col)
        if r is None:
            continue
        n, s, l, _ = r
        out[col] = (s, l, n)
        print(f"{label if col == 'sig' else '':<26}{col:<5}{f'{s}/{n}':>8}{f'{l}/{n}':>8}{l-s:>8}"
              f"{s/n*55:>10.1f}{l/n*55:>10.1f}")
    rows[tag] = out
    print()

dump = {"generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "note": "offline lenient re-judge (failed samples only; input = eval/*_generations*.jsonl);"
                " scores rescaled to a 55-point scale per column",
        "models": {}}
for label, tag in LABELS:
    o = rows.get(tag) or {}
    if "sig" in o and "nl" in o:
        st = (o["sig"][0] + o["nl"][0]) / (o["sig"][2] + o["nl"][2]) * 110
        ln = (o["sig"][1] + o["nl"][1]) / (o["sig"][2] + o["nl"][2]) * 110
        dump["models"][label] = {
            "tag": tag,
            "sig": {"n": o["sig"][2], "strict": o["sig"][0], "lenient": o["sig"][1],
                    "strict55": round(o["sig"][0] / o["sig"][2] * 55, 1),
                    "lenient55": round(o["sig"][1] / o["sig"][2] * 55, 1)},
            "nl": {"n": o["nl"][2], "strict": o["nl"][0], "lenient": o["nl"][1],
                   "strict55": round(o["nl"][0] / o["nl"][2] * 55, 1),
                   "lenient55": round(o["nl"][1] / o["nl"][2] * 55, 1)},
            "total": {"strict": round(st, 1), "lenient": round(ln, 1),
                      "delta": round(ln - st, 1)},
            "sources": {c: f"generations_{tag}_{c}.jsonl" for c in ("sig", "nl")},
        }
_json_path = FT + "/eval/lenient_results.json"
with open(_json_path, "w", encoding="utf-8") as _f:
    json.dump(dump, _f, ensure_ascii=False, indent=1)
print(f"\n[written] {_json_path} ({len(dump['models'])} models)")

print("=== both columns combined (55-point scale) ===")
print(f"{'model':<26}{'strict total':>10}{'lenient total':>10}{'delta':>8}")
for label, tag in LABELS:
    o = rows.get(tag) or {}
    if "sig" in o and "nl" in o:
        st = (o["sig"][0] + o["nl"][0]) / (o["sig"][2] + o["nl"][2]) * 110
        ln = (o["sig"][1] + o["nl"][1]) / (o["sig"][2] + o["nl"][2]) * 110
        print(f"{label:<26}{st:>10.1f}{ln:>10.1f}{ln-st:>+8.1f}")
