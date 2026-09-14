"""Held-out re-check: re-test candidate models on 12 **new** problems (none of the original 11) to answer "does the ranking generalise".

Why it is needed: the original 11 problems x 20 seeds were reused for model selection (seed picking, soup picking),
so the test set has been reused -- the held-out set is the only way to check "is the top model first only on the old problems".

Criterion matches the project:
  * signature column prompt = "def f(args):\\n    "            (same as eval_sft.ALGORITHMS)
  * NL column prompt        = "### Task\\n<requirement>\\n\\n### Code\\n"  (same as eval_nl.PROMPT_TEMPLATE)
  * judging: strict (same as eval_sft.check_algo: prompt/completion + truncate + asserts)
             lenient (in-place mutation allowed when the sole argument is a list) -- the two columns side by side
"""
import json
import os
import subprocess
import sys
import textwrap
import time

# scripts live next to the code they import (repo: post-training/scripts/)
FT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FT)
import torch  # noqa: E402
import eval_sft as E  # noqa: E402

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
NL_TPL = "### Task\n{instruction}\n\n### Code\n"

TASKS = [
    ("bubble_sort", "def bubble_sort(arr):\n    ",
     "Write the function `bubble_sort(arr)` that sorts a list of numbers in ascending order and returns it.",
     ["bubble_sort([5, 2, 9, 1]) == [1, 2, 5, 9]", "bubble_sort([]) == []", "bubble_sort([3, 3, 1]) == [1, 3, 3]"]),
    ("gcd", "def gcd(a, b):\n    ",
     "Write the function `gcd(a, b)` that returns the greatest common divisor of two integers.",
     ["gcd(12, 18) == 6", "gcd(17, 5) == 1", "gcd(0, 5) == 5"]),
    ("lcm", "def lcm(a, b):\n    ",
     "Write the function `lcm(a, b)` that returns the least common multiple of two positive integers.",
     ["lcm(4, 6) == 12", "lcm(7, 3) == 21", "lcm(5, 5) == 5"]),
    ("remove_duplicates", "def remove_duplicates(lst):\n    ",
     "Write the function `remove_duplicates(lst)` that removes duplicates from a list while keeping the original order.",
     ["remove_duplicates([1, 2, 1, 3, 2]) == [1, 2, 3]", "remove_duplicates([]) == []", "remove_duplicates([4, 4, 4]) == [4]"]),
    ("flatten", "def flatten(lst):\n    ",
     "Write the function `flatten(lst)` that flattens a list of lists by one level.",
     ["flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]", "flatten([]) == []", "flatten([[], [1]]) == [1]"]),
    ("is_anagram", "def is_anagram(s1, s2):\n    ",
     "Write the function `is_anagram(s1, s2)` that returns True if the two strings are anagrams of each other.",
     ["is_anagram('listen', 'silent') == True", "is_anagram('abc', 'abd') == False", "is_anagram('aab', 'aba') == True"]),
    ("caesar", "def caesar(s, k):\n    ",
     "Write the function `caesar(s, k)` that shifts each lowercase letter of s forward by k positions, wrapping around the alphabet.",
     ["caesar('abc', 1) == 'bcd'", "caesar('xyz', 3) == 'abc'", "caesar('abc', 0) == 'abc'"]),
    ("word_count", "def word_count(s):\n    ",
     "Write the function `word_count(s)` that returns a dict mapping each whitespace-separated word in s to how often it occurs.",
     ["word_count('a b a') == {'a': 2, 'b': 1}", "word_count('') == {}", "word_count('x') == {'x': 1}"]),
    ("longest_common_prefix", "def longest_common_prefix(strs):\n    ",
     "Write the function `longest_common_prefix(strs)` that returns the longest common prefix of a list of strings.",
     ["longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'", "longest_common_prefix([]) == ''", "longest_common_prefix(['dog', 'cat']) == ''"]),
    ("transpose", "def transpose(m):\n    ",
     "Write the function `transpose(m)` that returns the transpose of a matrix given as a list of rows.",
     ["transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]", "transpose([]) == []", "transpose([[1]]) == [[1]]"]),
    ("count_occurrences", "def count_occurrences(lst, x):\n    ",
     "Write the function `count_occurrences(lst, x)` that returns how many times x appears in lst.",
     ["count_occurrences([1, 2, 1, 1], 1) == 3", "count_occurrences([], 5) == 0", "count_occurrences(['a', 'b'], 'b') == 1"]),
    ("digit_sum", "def digit_sum(n):\n    ",
     "Write the function `digit_sum(n)` that returns the sum of the decimal digits of a non-negative integer n.",
     ["digit_sum(123) == 6", "digit_sum(0) == 0", "digit_sum(9999) == 36"]),
]
TESTS = {t[0]: t[3] for t in TASKS}

MODELS = ["G9-g5to1-s45-bestval", "G9-g5to1-s46-bestval", "G9-g5to1-s44-bestval",
          "G9-g5to1-bestval", "SOUP-G9top2-bestval", "SOUP-G8x3-bestval", "G8-g3to1-bestval"]

import ast
CALL_RE = __import__("re").compile(r"^([A-Za-z_]\w*)\((.*)\)\s*==\s*(.+)$", __import__("re").S)


def lenient_script(full, tests):
    lines = []
    for i, t in enumerate(tests):
        m = CALL_RE.match(t.strip())
        if not m:
            lines.append("assert " + t)
            continue
        fname, argstr, exp = m.groups()
        try:
            parsed = ast.literal_eval("(" + argstr + ",)")
        except Exception:
            parsed = None
        lines.append(f"_A{i} = ({argstr},)")
        lines.append(f"_R{i} = {fname}(*_A{i})")
        cond = f"(_R{i} == {exp})"
        if isinstance(parsed, tuple) and len(parsed) == 1 and isinstance(parsed[0], list):
            cond += f" or (_A{i}[0] == {exp})"
        lines.append(f"assert {cond}")
    return "%s\n\n%s\nprint('PASS')\n" % (full, "\n".join(lines))


def run_script(script, timeout=5):
    try:
        r = subprocess.run([sys.executable, "-c", script], timeout=timeout,
                           capture_output=True, text=True)
        return r.returncode == 0 and "PASS" in r.stdout
    except Exception:
        return False


def extract_nl(completion):
    """NL column: take the text after the last '### Code', strip fences, dedent, truncate."""
    code = completion.split("### Code")[-1] if "### Code" in completion else completion
    code = code.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])
    code = textwrap.dedent(code)
    return E.truncate(code).rstrip()


def main():
    paths = {n: (str(c), str(t)) for n, c, t in E.MODELS}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = E.load_tokenizer(paths[MODELS[0]][1])
    results = {}
    gens = []
    t0 = time.time()
    for name in MODELS:
        ckpt, _tok = paths[name]
        print(f"[{time.strftime('%H:%M:%S')}] loading {name}", flush=True)
        model = E.load_model(ckpt, device)
        for col in ("sig", "nl"):
            per_task = {}
            for tname, sig_prompt, instr, tests in TASKS:
                prompt = sig_prompt if col == "sig" else NL_TPL.format(instruction=instr)
                nok_s = nok_l = n = 0
                for seed in SEEDS:
                    comp = E.generate_one(model, tok, prompt, seed, device=device)
                    full = (prompt + E.truncate(comp)) if col == "sig" else extract_nl(prompt + comp)
                    ok_s = run_script("%s\n\n%s\nprint('PASS')\n" % (full, "\n".join("assert " + t for t in tests)))
                    ok_l = ok_s or run_script(lenient_script(full, tests))
                    n += 1
                    nok_s += ok_s
                    nok_l += ok_l
                    gens.append(dict(model=name, column=col, name=tname, seed=seed,
                                     prompt=prompt, completion=comp, passed=ok_s, passed_lenient=ok_l))
                per_task[tname] = (nok_s, nok_l, n)
            results[f"{name}|{col}"] = per_task
            s = sum(v[0] for v in per_task.values()); l = sum(v[1] for v in per_task.values())
            print(f"    {col}: strict {s}/{sum(v[2] for v in per_task.values())} | lenient {l}", flush=True)
        del model
        torch.cuda.empty_cache()

    out = os.path.join(FT, "eval", "heldout_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(FT, "eval", "heldout_generations.jsonl"), "w", encoding="utf-8") as fh:
        for g in gens:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")

    n_task = len(TASKS)
    print(f"\n=== held-out set ({n_task} problems x {len(SEEDS)} seeds, full marks {n_task*len(SEEDS)}/column) ===")
    print(f"{'model':<26}{'strict sig':>10}{'strict NL':>9}{'strict total':>10}{'lenient total':>10}")
    rows = []
    for name in MODELS:
        sg = results.get(f"{name}|sig", {}); nl = results.get(f"{name}|nl", {})
        s_sig = sum(v[0] for v in sg.values()); n_sig = sum(v[2] for v in sg.values())
        s_nl = sum(v[0] for v in nl.values()); n_nl = sum(v[2] for v in nl.values())
        l_sig = sum(v[1] for v in sg.values()); l_nl = sum(v[1] for v in nl.values())
        st = (s_sig + s_nl) / (n_sig + n_nl) * 110
        ln = (l_sig + l_nl) / (n_sig + n_nl) * 110
        rows.append((name, s_sig, n_sig, s_nl, n_nl, st, ln))
        print(f"{name:<26}{s_sig:>10}{s_nl:>9}{st:>10.1f}{ln:>10.1f}")
    print(f"\nElapsed {(time.time()-t0)/60:.1f} min; details written to eval/heldout_results.json")
    json.dump(rows, open(os.path.join(FT, "eval", "heldout_summary.json"), "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
