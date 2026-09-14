#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
435M-v2 - natural-language prompt-column evaluator (instruction-style)
================================================================
The question: **for the same model, if the request is switched from a "function signature" to a
"natural-language requirement" (the training-time instruction template), does it still work?**  -> sits next to eval_sft.py's signature column (the Arm4 instruction-mix conclusion rests on these two columns).

Relationship to eval_sft.py (**eval_sft.py is never modified**, only imported and reused; its main logic lives under `if __name__`):
  - Task table / seeds / model table / judge core all come from eval_sft, so the two columns are **item-by-item comparable**:
        ALGORITHMS(5) + SIMPLE_TASKS(6)   -> the same dicts are reused, tests is the **same list object**
        DEFAULT_SEEDS = [0, 42, 123, 999, 2024]
        truncate / run_script             -> the **same function objects** are reused
        generate_one's generation params  -> the same numbers (0.2 / 50 / 1.2 / 0.05 / 256)
  - The only differences = prompt + judging entry point:
        signature column  check_algo(prompt, completion, ...)   splices the prompt in as valid Python;
        natural-language column  the prompt is prose ("### Task\\n<requirement>\\n\\n### Code\\n"), and prose is not valid Python,
        so this file implements the NL judging path: **first extract the generated text after `### Code` as code**,
        then reuse the same run_script + tests logic that mirrors check_algo **statement by statement** (see below).

prompt template (**byte-for-byte** identical to build_instr_data.PROMPT_TEMPLATES["marker"], and also
identical to "prompt_template" in data_instr/*_meta.json; this file does an ast/byte comparison in its self-test):
        "### Task\\n{instruction}\\n\\n### Code\\n"

Judging criterion (mirrors eval_sft.check_algo statement by statement):
        code  = extract(after the last "### Code")  -> strip ``` fences -> dedent -> truncate -> rstrip
        script = code + "\\n\\n" + "\\n".join("assert " + t for t in tests) + "\\nprint('PASS')\\n"
        pass   = run_script(script)          # same run_script function object, same 3s timeout, same tests

Usage (this file runs without loading a model):
  py -3.12 eval_nl.py --self-test                 # (3) hand-built case self-test (5 classes x 11 tasks), **no GPU touched**
  py -3.12 eval_nl.py --models all --device cuda  # real run: 5(+2) models x 11 tasks x 5 seeds
  py -3.12 eval_nl.py --models SFT-bestval --seeds 0,42 --device cpu
  py -3.12 eval_nl.py --list-models

Artifacts (**does not overwrite** eval_sft's eval_sft_summary.json / eval_sft_generations.jsonl):
  eval/nl_eval_summary.json       per model x task -> passes / total
  eval/nl_eval_generations.jsonl  per-sample generations (raw completion + extracted code, for hand audit)
  eval/nl_dryrun.json             self-test report written by --self-test
"""

import argparse
import ast
import gc
import inspect
import json
import re
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:               # make eval_sft importable from any cwd
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402  (import only; no CUDA context, no model load)

import eval_sft  # noqa: E402
from eval_sft import (  # noqa: E402
    ALGORITHMS,
    DEFAULT_SEEDS,
    MODELS as SFT_MODELS,
    PROJECT_ROOT,
    SIMPLE_TASKS,
    check_algo,
    generate_one,
    load_model,
    load_tokenizer,
    run_script,
    truncate,
)

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer_435m.json"

# ===================================================================
# Template / generation params (same values as eval_sft; compared at runtime by the self-test)
# ===================================================================
PROMPT_TEMPLATE = "### Task\n{instruction}\n\n### Code\n"   # build_instr_data.PROMPT_TEMPLATES["marker"]
TEMPLATE_HEAD = "### Task\n"
TEMPLATE_TAIL = "### Code\n"
BUILD_INSTR_DATA = HERE / "build_instr_data.py"

# Same as eval_sft.main()'s argparse defaults; the self-test cross-checks via inspect.signature(eval_sft.generate_one)
GEN_DEFAULTS = dict(max_new=256, temperature=0.2, top_k=50, repetition_penalty=1.2, min_p=0.05)
JUDGE_TIMEOUT = 3          # same as the default timeout of eval_sft.check_algo -> run_script

DEFAULT_DEVICE = "cuda"    # same as eval_sft; this box has a 5090. For a CPU self-test add --device cpu or skip model loading

# Optional models: ckpt_arm4 (instruction-mix arm) joins --models all only if checkpoint.pt already exists
OPTIONAL_MODELS = [
    ("A4-instr-bestval", HERE / "ckpt_arm4" / "best_val"),
    ("A4-instr-final", HERE / "ckpt_arm4" / "final"),
    ("A5-bigmix-final", HERE / "ckpt_a5" / "final"),
    ("A5-bigmix-bestval", HERE / "ckpt_a5" / "best_val"),
    ("A6-2ep-final", HERE / "ckpt_a6" / "final"),
    ("A6-2ep-bestval", HERE / "ckpt_a6" / "best_val"),
    ("A7-bigseed-final", HERE / "ckpt_a7" / "final"),
    ("A7-bigseed-bestval", HERE / "ckpt_a7" / "best_val"),
    ("A8-2to1-final", HERE / "ckpt_a8" / "final"),
    ("A8-2to1-bestval", HERE / "ckpt_a8" / "best_val"),
    ("A9-1to1-final", HERE / "ckpt_a9" / "final"),
    ("A9-1to1-bestval", HERE / "ckpt_a9" / "best_val"),
    ("G1-gold-final", HERE / "ckpt_g1" / "final"),
    ("G1-gold-bestval", HERE / "ckpt_g1" / "best_val"),
    ("G2-goldsig-final", HERE / "ckpt_g2" / "final"),
    ("G2-goldsig-bestval", HERE / "ckpt_g2" / "best_val"),
    ("G3-g11-final", HERE / "ckpt_g3" / "final"),
    ("G3-g11-bestval", HERE / "ckpt_g3" / "best_val"),
    ("G4-2stage-final", HERE / "ckpt_g4" / "final"),
    ("G4-2stage-bestval", HERE / "ckpt_g4" / "best_val"),
    ("G5-g15-final", HERE / "ckpt_g5" / "final"),
    ("G5-g15-bestval", HERE / "ckpt_g5" / "best_val"),
    ("G6-g075-final", HERE / "ckpt_g6" / "final"),
    ("G6-g075-bestval", HERE / "ckpt_g6" / "best_val"),
    ("G7-g2to1-final", HERE / "ckpt_g7" / "final"),
    ("G7-g2to1-bestval", HERE / "ckpt_g7" / "best_val"),
    ("G8-g3to1-final", HERE / "ckpt_g8" / "final"),
    ("G8-g3to1-bestval", HERE / "ckpt_g8" / "best_val"),
    ("G9-g5to1-final", HERE / "ckpt_g9" / "final"),
    ("G9-g5to1-bestval", HERE / "ckpt_g9" / "best_val"),
    ("G10-g8to1-final", HERE / "ckpt_g10" / "final"),
    ("G10-g8to1-bestval", HERE / "ckpt_g10" / "best_val"),
    ("G8-g3to1-s43-final", HERE / "ckpt_g8_s43" / "final"),
    ("G8-g3to1-s43-bestval", HERE / "ckpt_g8_s43" / "best_val"),
    ("G8-g3to1-s44-final", HERE / "ckpt_g8_s44" / "final"),
    ("G8-g3to1-s44-bestval", HERE / "ckpt_g8_s44" / "best_val"),
    ("SOUP-G8x3-bestval", HERE / "ck_soup" / "sG8x3bv"),
    ("SOUP-G8x3-final", HERE / "ck_soup" / "sG8x3fi"),
    ("G9-g5to1-s43-final", HERE / "ckpt_g9_s43" / "final"),
    ("G9-g5to1-s43-bestval", HERE / "ckpt_g9_s43" / "best_val"),
    ("G9-g5to1-s44-final", HERE / "ckpt_g9_s44" / "final"),
    ("G9-g5to1-s44-bestval", HERE / "ckpt_g9_s44" / "best_val"),
    ("G5-g15-s43-final", HERE / "ckpt_g5_s43" / "final"),
    ("G5-g15-s43-bestval", HERE / "ckpt_g5_s43" / "best_val"),
    ("G5-g15-s44-final", HERE / "ckpt_g5_s44" / "final"),
    ("G5-g15-s44-bestval", HERE / "ckpt_g5_s44" / "best_val"),
    ("SOUP-G5x3-final", HERE / "ck_soup" / "sG5x3fi"),
    ("SOUP-G5x3-bestval", HERE / "ck_soup" / "sG5x3bv"),
    ("SOUP-G9x3-bestval", HERE / "ck_soup" / "sG9x3bv"),
    ("SOUP-G9top2-bestval", HERE / "ck_soup" / "sG9top2bv"),
    ("SOUP-G9x5-bestval", HERE / "ck_soup" / "sG9x5bv"),
    ("G9-g5to1-s45-final", HERE / "ckpt_g9_s45" / "final"),
    ("G9-g5to1-s45-bestval", HERE / "ckpt_g9_s45" / "best_val"),
    ("G9-g5to1-s46-final", HERE / "ckpt_g9_s46" / "final"),
    ("G9-g5to1-s46-bestval", HERE / "ckpt_g9_s46" / "best_val"),
    ("SOUP-A7G8-020", HERE / "ck_soup" / "sA7G8_020"),
    ("SOUP-G9G5-030", HERE / "ck_soup" / "sG9G5_030"),
    ("SOUP-G9G5-050", HERE / "ck_soup" / "sG9G5_050"),
    ("SOUP-G9G5-070", HERE / "ck_soup" / "sG9G5_070"),
    ("SOUP-G9G8-050", HERE / "ck_soup" / "sG9G8_050"),
    ("SOUP-G10G5-050", HERE / "ck_soup" / "sG10G5_050"),
    ("SOUP-A7G8-035", HERE / "ck_soup" / "sA7G8_035"),
    ("SOUP-A7G8-050", HERE / "ck_soup" / "sA7G8_050"),
    ("SOUP-A7G8-065", HERE / "ck_soup" / "sA7G8_065"),
    ("SOUP-A7G8-080", HERE / "ck_soup" / "sA7G8_080"),
    ("SOUP-G5G8-050", HERE / "ck_soup" / "sG5G8_050"),
    ("SOUP-G8self-050", HERE / "ck_soup" / "sG8self_050"),

]

# ===================================================================
# (1) Natural-language requirements (11 entries; each must contain the function name matching its entry, else the test cannot call it)
# ===================================================================
NL_INSTRUCTIONS = {
    "fibonacci":
        "Write the function `fibonacci(n)` that returns the n-th Fibonacci number, so that "
        "`fibonacci(0) == 0` and `fibonacci(1) == 1`.",
    "quicksort":
        "Write the function `quicksort(arr)` that returns a new list with the same elements as "
        "`arr` sorted in ascending order (for example `quicksort([3, 1, 4, 1]) == [1, 1, 3, 4]`).",
    "binary_search":
        "Write the function `binary_search(arr, target)` that searches the sorted list `arr` and "
        "returns the index of `target` in `arr`, or `-1` if `target` is not in `arr`.",
    "two_sum":
        "Write the function `two_sum(nums, target)` that returns the indices `[i, j]` of the two "
        "numbers in `nums` such that `nums[i] + nums[j] == target`.",
    "mergesort":
        "Write the function `mergesort(arr)` that returns a new list with the same elements as "
        "`arr` sorted in ascending order, implemented with the merge sort algorithm.",
    "is_palindrome":
        "Write the function `is_palindrome(s)` that returns `True` if the string `s` reads the "
        "same forwards and backwards, and `False` otherwise.",
    "is_prime":
        "Write the function `is_prime(n)` that returns `True` if the integer `n` is a prime "
        "number, and `False` otherwise.",
    "reverse_string":
        "Write the function `reverse_string(s)` that returns a new string with the characters of "
        "`s` in reverse order.",
    "count_vowels":
        "Write the function `count_vowels(s)` that returns how many characters of the string `s` "
        "are vowels ('a', 'e', 'i', 'o' or 'u').",
    "sum_list":
        "Write the function `sum_list(nums)` that returns the sum of all the numbers in the list "
        "`nums` (`0` for an empty list).",
    "factorial":
        "Write the function `factorial(n)` that returns `n!` (the factorial of `n`), with "
        "`factorial(0) == 1`.",
}

# NL task table: tests / entry reference the same dicts from eval_sft (object identity, not copies)
NL_TASKS = []
for _fam, _tasks in (("algo", ALGORITHMS), ("simple", SIMPLE_TASKS)):
    for _t in _tasks:
        assert _t["name"] in NL_INSTRUCTIONS, "missing natural-language requirement: %s" % _t["name"]
        _instr = NL_INSTRUCTIONS[_t["name"]]
        assert _t["entry"] in _instr, "%s: requirement text does not mention function name %s" % (_t["name"], _t["entry"])
        NL_TASKS.append({
            "name": _t["name"],
            "family": _fam,
            "entry": _t["entry"],                      # same string
            "tests": _t["tests"],                      # * same list object (identical to eval_sft)
            "instruction": _instr,
            "prompt": PROMPT_TEMPLATE.format(instruction=_instr),
            "sig_prompt": _t["prompt"],                # signature-column prompt (self-test cross-check only)
            "_src": _t,
        })


# ===================================================================
# (2) NL judging path (same skeleton as eval_sft.check_algo, only "where the code comes from" changes)
# ===================================================================
_FENCE_RX = re.compile(r"```[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*\n(.*?)(?:```|\Z)", re.S)
_FENCE_LINE_RX = re.compile(r"^```[A-Za-z0-9_+#.-]*$")


def strip_fences(text):
    """Strip ``` fences (semantically aligned with build_instr_data.extract_code(fence_policy='extract')):
    paired fences -> prefer python/py-tagged blocks, then untagged blocks, take the first;
    only stray fence lines (model closed but never opened) -> drop the leading/trailing fence lines; no fences -> as-is."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = _FENCE_RX.findall(text)
    if blocks:
        pylike = [b for tag, b in blocks if tag.lower().startswith(("python", "py", "py3"))]
        pool = pylike or [b for tag, b in blocks if tag == ""]
        return pool[0] if pool else text          # only non-python fences (java/cpp) -> as-is, let exec fail
    lines = text.split("\n")
    while lines and _FENCE_LINE_RX.match(lines[0].strip()):
        lines.pop(0)
    while lines and _FENCE_LINE_RX.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines)


def normalize_code(code):
    """dedent + per-line rstrip + drop leading/trailing blank lines (semantically aligned with build_instr_data.normalize_code)."""
    code = textwrap.dedent(code).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in code.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_code(full_text):
    """Take the content after the *last* "### Code\\n" -> strip fences -> normalize.
    Last one: the requirement body may itself contain the literal "### Code" (required by build_instr_data eval_hint)."""
    i = full_text.rfind(TEMPLATE_TAIL)
    body = full_text[i + len(TEMPLATE_TAIL):] if i != -1 else full_text
    return normalize_code(strip_fences(body))


def prepare_code(prompt, completion):
    """NL path: turn generated text into an executable code fragment. Order = extract marker -> strip fences/dedent -> truncate (same criterion) -> rstrip."""
    return truncate(extract_code(prompt + completion)).rstrip()


def judge_code(code, tests, timeout=JUDGE_TIMEOUT):
    """Statement-by-statement counterpart of eval_sft.check_algo's execution core (same run_script, same tests, same script skeleton)."""
    asserts = "\n".join("assert " + t for t in tests)
    script = "%s\n\n%s\nprint('PASS')\n" % (code, asserts)
    return run_script(script, timeout=timeout)


def check_algo_nl(prompt, completion, entry, tests, code=None):
    """NL version of check_algo: the prose prompt never enters the script; only text after `### Code` counts as code."""
    if code is None:
        code = prepare_code(prompt, completion)
    return judge_code(code, tests)


# ===================================================================
# (3) Model resolution (clear errors, no traceback)
# ===================================================================
def all_model_names():
    names = [n for n, _, _ in SFT_MODELS]
    for n, _p in OPTIONAL_MODELS:
        if n not in names:
            names.append(n)
    return names


def optional_model_ready(path):
    return (Path(path) / "checkpoint.pt").is_file()


def resolve_models(spec):
    known = {n: (ck, tok) for n, ck, tok in SFT_MODELS}
    for n, p in OPTIONAL_MODELS:
        known.setdefault(n, (p, TOKENIZER_PATH))

    if spec == "all":
        selected = [(n, ck, tok) for n, ck, tok in SFT_MODELS]
        for n, p in OPTIONAL_MODELS:
            if n in known:      # already in eval_sft.MODELS (auto-selected with SFT_MODELS) -> do not duplicate
                continue
            if optional_model_ready(p):
                selected.append((n, p, TOKENIZER_PATH))
                print("[MODEL] auto-included %s (%s already has checkpoint.pt)" % (n, p))
            else:
                print("[MODEL] skipping %s (%s has no checkpoint.pt yet)" % (n, p))
        return selected

    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit("[ERROR] unknown model name: %s\n        available: %s\n        (use all or comma-separated names)"
                         % (", ".join(unknown), ", ".join(all_model_names())))
    return [(n, known[n][0], known[n][1]) for n in names]


def validate_model(name, ckpt_dir, tok_path):
    if not Path(ckpt_dir).exists():
        raise SystemExit("[ERROR] checkpoint dir for model %s does not exist: %s\n"
                         "        (it may still be training; check the --ckpt path or wait until the dir is written)"
                         % (name, ckpt_dir))
    if not (Path(ckpt_dir) / "checkpoint.pt").is_file():
        raise SystemExit("[ERROR] model %s dir has no checkpoint.pt: %s"
                         % (name, ckpt_dir))
    if not Path(tok_path).is_file():
        raise SystemExit("[ERROR] tokenizer for model %s not found: %s" % (name, tok_path))


# ===================================================================
# (4) Self-test (--self-test): hand-built completions, 5 case classes x 11 tasks, never touches the GPU
# ===================================================================
SOLUTIONS = {
    "fibonacci": '''def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a''',
    "quicksort": '''def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)''',
    "binary_search": '''def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1''',
    "two_sum": '''def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        other = target - num
        if other in seen:
            return [seen[other], i]
        seen[num] = i
    return []''',
    "mergesort": '''def mergesort(arr):
    if len(arr) <= 1:
        return list(arr)
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    def merge(a, b):
        out = []
        i = 0
        j = 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                out.append(a[i])
                i += 1
            else:
                out.append(b[j])
                j += 1
        out.extend(a[i:])
        out.extend(b[j:])
        return out
    return merge(left, right)''',
    "is_palindrome": '''def is_palindrome(s):
    return s == s[::-1]''',
    "is_prime": '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True''',
    "reverse_string": '''def reverse_string(s):
    return s[::-1]''',
    "count_vowels": '''def count_vowels(s):
    count = 0
    for ch in s.lower():
        if ch in 'aeiou':
            count += 1
    return count''',
    "sum_list": '''def sum_list(nums):
    total = 0
    for num in nums:
        total += num
    return total''',
    "factorial": '''def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result''',
}

# (4) wrong return type (one typical failure mode per task: type/semantics wrong but syntax valid)
WRONG_TYPE = {
    "fibonacci": '''def fibonacci(n):
    seq = [0, 1]
    while len(seq) <= n:
        seq.append(seq[-1] + seq[-2])
    return seq''',
    "quicksort": '''def quicksort(arr):
    arr.sort()
    return None''',
    "binary_search": '''def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return True
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False''',
    "two_sum": '''def two_sum(nums, target):
    seen = {}
    for num in nums:
        if target - num in seen:
            return [target - num, num]
        seen[num] = True
    return []''',
    "mergesort": '''def mergesort(arr):
    if len(arr) <= 1:
        return tuple(arr)
    return tuple(sorted(arr))''',
    "is_palindrome": '''def is_palindrome(s):
    if s == s[::-1]:
        return "True"
    return "False"''',
    "is_prime": '''def is_prime(n):
    if n < 2:
        return "no"
    for i in range(2, n):
        if n % i == 0:
            return "no"
    return "yes"''',
    "reverse_string": '''def reverse_string(s):
    return list(reversed(s))''',
    "count_vowels": '''def count_vowels(s):
    return [ch for ch in s.lower() if ch in 'aeiou']''',
    "sum_list": '''def sum_list(nums):
    if not nums:
        return 0
    return max(nums)''',
    "factorial": '''def factorial(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total''',
}


def case_syntax(task):
    """(2a) syntax error: drop the colon at the end of the def line."""
    code = SOLUTIONS[task["name"]]
    head, sep, rest = code.partition("\n")
    return head.rstrip().rstrip(":") + sep + rest


def case_incomplete(task):
    """(2b) incomplete: generation hit max_new -- cut mid-line (half the body missing)."""
    lines = SOLUTIONS[task["name"]].rstrip().split("\n")
    keep = max(1, len(lines) // 2)
    return "\n".join(lines[:keep])


def case_head_only(task):
    """(2c) generation stopped right after the signature -> body missing."""
    return SOLUTIONS[task["name"]].rstrip().split("\n")[0]


def case_wrong_name(task):
    """(3) wrong function name: entry renamed (tests call the original name -> NameError)."""
    return re.sub(r"\b%s\b" % re.escape(task["entry"]), task["entry"] + "_v2",
                  SOLUTIONS[task["name"]])


def case_fence_python(task):
    """(5a) ```python fence."""
    return "```python\n" + SOLUTIONS[task["name"]] + "\n```"


def case_fence_bare(task):
    """(5b) untagged fence."""
    return "```\n" + SOLUTIONS[task["name"]] + "\n```"


def case_fence_tail_only(task):
    """(5c) closing fence only (a common way models end their output)."""
    return SOLUTIONS[task["name"]] + "\n```"


def run_script_detail(script, timeout=JUDGE_TIMEOUT):
    """Same params and semantics as run_script, but keeps stdout/stderr as self-test evidence."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, "-c", script], timeout=timeout,
                           capture_output=True, text=True)
        ok = r.returncode == 0 and "PASS" in r.stdout
        return ok, (r.stdout or ""), (r.stderr or "")
    except subprocess.TimeoutExpired:
        return False, "", "<TIMEOUT>"
    except Exception as e:  # pragma: no cover
        return False, "", "<%s: %s>" % (type(e).__name__, e)


def _sft_arg_defaults():
    """Read-only ast parse of eval_sft.py's argparse defaults (no import side effects, no execution)."""
    tree = ast.parse(eval_sft.__file__ and Path(eval_sft.__file__).read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument" and node.args \
                and isinstance(node.args[0], ast.Constant):
            flag = node.args[0].value
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    out[flag] = kw.value.value
    return out


def _build_instr_templates():
    """Read-only ast parse of build_instr_data.py's PROMPT_TEMPLATES / HEAD / TAIL literals."""
    src = BUILD_INSTR_DATA.read_text(encoding="utf-8")
    tree = ast.parse(src)
    want = {"PROMPT_TEMPLATES": {}, "TEMPLATE_TAIL": {}, "TEMPLATE_HEAD": {}}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in want:
            want[node.targets[0].id] = ast.literal_eval(node.value)
    return want


def _meta_template():
    for p in (HERE / "data_instr" / "instr_train_meta.json",
              HERE / "data_instr_mixed" / "instr_train_meta.json"):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        if "prompt_template" in d:
            return str(p), d["prompt_template"], d.get("prompt_template_name")
    return None, None, None


def self_test(out_path):
    print("=" * 72)
    print("eval_nl.py self-test (--self-test): hand-built completions x 5 case classes x 11 tasks, no model loaded")
    print("=" * 72)
    rep = {"gpu_used": False, "models_loaded": False, "cases": {}, "consistency": {}, "template": {}}

    # ---- Evidence A: template byte-for-byte ----
    tpl = _build_instr_templates()
    src_marker = tpl["PROMPT_TEMPLATES"].get("marker")
    meta_p, meta_tpl, meta_name = _meta_template()
    tpl_checks = {
        "eval_nl.PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        "build_instr_data.PROMPT_TEMPLATES['marker']": src_marker,
        "build_instr_data.TEMPLATE_HEAD['marker']": tpl["TEMPLATE_HEAD"].get("marker"),
        "build_instr_data.TEMPLATE_TAIL['marker']": tpl["TEMPLATE_TAIL"].get("marker"),
        "meta.json prompt_template (%s, name=%s)" % (meta_p, meta_name): meta_tpl,
    }
    tpl_ok = (src_marker == PROMPT_TEMPLATE == meta_tpl
              and tpl["TEMPLATE_HEAD"].get("marker") == TEMPLATE_HEAD
              and tpl["TEMPLATE_TAIL"].get("marker") == TEMPLATE_TAIL)
    rep["template"] = {"sources": tpl_checks, "byte_identical": bool(tpl_ok)}
    print("\n[template] byte-for-byte comparison: %s" % ("OK all identical" if tpl_ok else "MISMATCH"))
    for k, v in tpl_checks.items():
        print("   %-58s %r" % (k, v))

    # ---- Evidence B: generation params / judge core come from the same function objects ----
    sig = inspect.signature(generate_one)
    gen_now = {k: sig.parameters[k].default for k in GEN_DEFAULTS}
    argdef = _sft_arg_defaults()
    sft_args = {"max_new": argdef.get("--max-new"), "temperature": argdef.get("--temperature"),
                "top_k": argdef.get("--top-k"), "repetition_penalty": argdef.get("--repetition-penalty"),
                "min_p": argdef.get("--min-p")}
    tmo_default = inspect.signature(run_script).parameters["timeout"].default
    rep["consistency"] = {
        "gen_defaults": gen_now,
        "eval_sft_argparse_defaults": sft_args,
        "gen_match": gen_now == GEN_DEFAULTS == sft_args,
        "run_script_is_eval_sft_object": run_script is eval_sft.run_script,
        "truncate_is_eval_sft_object": truncate is eval_sft.truncate,
        "generate_one_is_eval_sft_object": generate_one is eval_sft.generate_one,
        "run_script_timeout_default": tmo_default,
        "judge_timeout": JUDGE_TIMEOUT,
        "tests_object_identity": all(t["tests"] is t["_src"]["tests"] for t in NL_TASKS),
        "task_count": len(NL_TASKS),
        "seeds": DEFAULT_SEEDS,
    }
    ok_b = (gen_now == GEN_DEFAULTS == sft_args and tmo_default == JUDGE_TIMEOUT
            and run_script is eval_sft.run_script and truncate is eval_sft.truncate
            and all(t["tests"] is t["_src"]["tests"] for t in NL_TASKS))
    print("\n[criterion] gen params %s | run_script/truncate same object %s/%s | timeout=%s=%s | tests same list object %s"
          % (gen_now, run_script is eval_sft.run_script, truncate is eval_sft.truncate,
             tmo_default, JUDGE_TIMEOUT, all(t["tests"] is t["_src"]["tests"] for t in NL_TASKS)))

    # ---- 5 case classes x 11 tasks ----
    classes = [
        ("1_correct",        "correct implementation -> PASS",         True,  lambda t: SOLUTIONS[t["name"]]),
        ("2a_syntax",        "syntax error (missing colon) -> FAIL",   False, case_syntax),
        ("2b_incomplete",    "incomplete (truncated mid-way) -> FAIL", False, case_incomplete),
        ("2c_head_only",     "signature only (no body) -> FAIL",       False, case_head_only),
        ("3_wrong_name",     "wrong function name -> FAIL",            False, case_wrong_name),
        ("4_wrong_type",     "wrong return type -> FAIL",              False, lambda t: WRONG_TYPE[t["name"]]),
        ("5a_fence_python",  "```python fence -> PASS",                True,  case_fence_python),
        ("5b_fence_bare",    "bare ``` fence -> PASS",                 True,  case_fence_bare),
        ("5c_fence_tail",    "closing fence only -> PASS",             True,  case_fence_tail_only),
    ]
    per_class = {}
    for key, title, expect, fn in classes:
        rows = []
        for t in NL_TASKS:
            comp = fn(t)
            code = prepare_code(t["prompt"], comp)
            ok = check_algo_nl(t["prompt"], comp, t["entry"], t["tests"], code=code)
            asserts = "\n".join("assert " + x for x in t["tests"])
            script = "%s\n\n%s\nprint('PASS')\n" % (code, asserts)
            _, _, err = run_script_detail(script)
            rows.append({"task": t["name"], "expected": expect, "actual": bool(ok),
                         "ok": bool(ok) == expect,
                         "code_head": code.split("\n")[0][:80],
                         "detail": (err.strip().split("\n")[-1][:160] if err and err.strip()
                                    and err.strip() != "<TIMEOUT>" else "")})
        n_ok = sum(1 for r in rows if r["ok"])
        per_class[key] = {"title": title, "n_pass": n_ok, "n_total": len(rows), "rows": rows}

    # ---- (6) NL judging path is necessary: a prose prompt fed straight to eval_sft.check_algo must fail ----
    need = []
    for t in NL_TASKS:
        ok_nl = check_algo_nl(t["prompt"], SOLUTIONS[t["name"]], t["entry"], t["tests"])
        ok_sig = check_algo(t["prompt"], SOLUTIONS[t["name"]], t["entry"], t["tests"])  # counter-example: signature column
        need.append({"task": t["name"], "nl_path": bool(ok_nl), "check_algo_on_prose": bool(ok_sig),
                     "ok": bool(ok_nl) and not ok_sig})
    per_class["6_nl_path_needed"] = {
        "title": "prose prompt via eval_sft.check_algo -> FAIL (hence the NL path is required); via the NL path -> PASS",
        "n_pass": sum(1 for r in need if r["ok"]), "n_total": len(need), "rows": need}

    # ---- (7) cross-criterion consistency: the same tests also PASS under eval_sft's signature column ----
    cross = []
    for t in NL_TASKS:
        lines = SOLUTIONS[t["name"]].split("\n")
        body = "\n".join(lines[1:])
        if body.startswith("    "):
            body = body[4:]
        ok = check_algo(t["sig_prompt"], body, t["entry"], t["tests"])
        cross.append({"task": t["name"], "ok": bool(ok), "body_head": body.split("\n")[0][:60]})
    per_class["7_sig_crosscheck"] = {
        "title": "same tests via eval_sft.check_algo (signature prompt + hand-written body) -> PASS",
        "n_pass": sum(1 for r in cross if r["ok"]), "n_total": len(cross), "rows": cross}

    rep["cases"] = per_class
    print("\n[cases] 5 classes (+2 evidence items) per task:")
    all_good = True
    for key, v in per_class.items():
        flag = "OK " if v["n_pass"] == v["n_total"] else "FAIL"
        all_good &= v["n_pass"] == v["n_total"]
        print("   %s %-20s %2d/%2d  %s" % (flag, key, v["n_pass"], v["n_total"], v["title"]))
    all_good &= bool(tpl_ok) and bool(ok_b)

    rep["all_ok"] = bool(all_good)
    rep["task_count"] = len(NL_TASKS)
    rep["note"] = ("this self-test only runs hand-built completions: no model load, no CUDA context, "
                   "only spawns pure-assert subprocesses (no torch import).")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[SELF-TEST %s] report -> %s" % ("PASS" if all_good else "FAIL", out_path))
    return 0 if all_good else 1


# ===================================================================
# (5) Main flow
# ===================================================================
def _sum_counts(tasks):
    """Aggregate {'fibonacci': '3/5', ...} into (hits, total)."""
    hit = tot = 0
    for v in (tasks or {}).values():
        try:
            a, b = str(v).split("/")
            hit += int(a)
            tot += int(b)
        except Exception:
            pass
    return hit, tot


def compare_with_sft(summaries, out_dir):
    """Read eval/eval_sft_summary.json and print the signature column side by side (without modifying it)."""
    p = Path(out_dir) / "eval_sft_summary.json"
    if not p.is_file():
        print("\n[compare] %s not found, skipping signature-column comparison (eval_sft.py writes that file; this script never does)" % p)
        return
    try:
        sig = {m["model"]: m for m in json.loads(p.read_text(encoding="utf-8"))}
    except Exception as e:
        print("\n[compare] failed to read %s: %s" % (p, e))
        return
    print("\n" + "=" * 72)
    print("natural-language column vs signature column (reading %s only)" % p.name)
    print("same tests list object / same run_script / same generation params; only prompt and judging entry differ")
    print("=" * 72)
    for s in summaries:
        o = sig.get(s["model"])
        nl_h, nl_t = _sum_counts(s["tasks"])
        if o:
            sg_h, sg_t = _sum_counts(o.get("tasks"))
            print("  %-18s NL %2d/%2d (%5.1f%%) | signature %2d/%2d (%5.1f%%)"
                  % (s["model"], nl_h, nl_t, 100.0 * nl_h / max(nl_t, 1),
                     sg_h, sg_t, 100.0 * sg_h / max(sg_t, 1)))
        else:
            print("  %-18s NL %2d/%2d (%5.1f%%) | signature N/A (name not in eval_sft_summary.json)"
                  % (s["model"], nl_h, nl_t, 100.0 * nl_h / max(nl_t, 1)))
    for s in summaries:
        o = sig.get(s["model"])
        if not o:
            continue
        print("\n  %s  per task (NL / signature):" % s["model"])
        for t in NL_TASKS:
            print("     %-16s %-5s %-5s" % (t["name"], s["tasks"][t["name"]],
                                            o.get("tasks", {}).get(t["name"], "-")))


def main():
    ap = argparse.ArgumentParser(description="natural-language prompt-column evaluator (instruction-style, template from build_instr_data)")
    ap.add_argument("--models", type=str, default="all",
                    help="all or comma-separated model names (see --list-models for available names)")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--max-new", type=int, default=GEN_DEFAULTS["max_new"])
    ap.add_argument("--temperature", type=float, default=GEN_DEFAULTS["temperature"])
    ap.add_argument("--top-k", type=int, default=GEN_DEFAULTS["top_k"])
    ap.add_argument("--repetition-penalty", type=float, default=GEN_DEFAULTS["repetition_penalty"])
    ap.add_argument("--min-p", type=float, default=GEN_DEFAULTS["min_p"])
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    ap.add_argument("--out-dir", type=str, default=str(HERE / "eval"))
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="run only the hand-built case self-test (no model load, no GPU)")
    ap.add_argument("--self-test-out", type=str, default=str(HERE / "eval" / "nl_dryrun.json"))
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.self_test_out)

    if args.list_models:
        for n, ck, _ in SFT_MODELS:
            print("%-18s %s" % (n, ck))
        for n, p in OPTIONAL_MODELS:
            print("%-18s %s%s" % (n, p, "" if optional_model_ready(p) else "   [missing checkpoint.pt]"))
        return 0

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    gen = dict(max_new=args.max_new, temperature=args.temperature, top_k=args.top_k,
               repetition_penalty=args.repetition_penalty, min_p=args.min_p, device=args.device)

    selected = resolve_models(args.models)
    if not selected:
        raise SystemExit("[ERROR] no model selected (--models %s)" % args.models)
    for name, ck, tok in selected:
        validate_model(name, ck, tok)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[DEV] %s | NL template %r | tasks %d (5 algorithms + 6 simple) x seeds %d = %d runs/model"
          % (args.device, PROMPT_TEMPLATE, len(NL_TASKS), len(seeds), len(NL_TASKS) * len(seeds)))

    all_samples = []
    summaries = []
    for name, ckpt_dir, tok_path in selected:
        print("\n" + "=" * 60)
        print("model: %s (%s)" % (name, ckpt_dir))
        print("=" * 60)
        t0 = time.time()
        try:
            tokenizer = load_tokenizer(tok_path)
            model = load_model(ckpt_dir, args.device)
        except Exception as e:
            print("[ERROR] failed to load model %s: %s: %s" % (name, type(e).__name__, e))
            print("        skipping this model (is --device %s available? is the checkpoint complete?)" % args.device)
            continue
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print("  params: %.0fM | vocab=%d | layers=%d | d_ff=%d" % (
            n_params, model.config.vocab_size, model.config.num_layers, model.config.d_ff))

        task_pass = {}
        n_hit = n_tot = 0
        for t in NL_TASKS:
            p = 0
            for seed in seeds:
                comp = generate_one(model, tokenizer, t["prompt"], seed, **gen)
                code = prepare_code(t["prompt"], comp)
                ok = check_algo_nl(t["prompt"], comp, t["entry"], t["tests"], code=code)
                p += (1 if ok else 0)
                n_hit += (1 if ok else 0)
                n_tot += 1
                all_samples.append({
                    "model": name, "column": "nl", "type": t["family"], "name": t["name"],
                    "seed": seed, "entry": t["entry"], "instruction": t["instruction"],
                    "prompt": t["prompt"], "completion": comp, "code": code,
                    "extraction_changed": (code != comp.rstrip()),
                    "passed": ok,
                })
            task_pass[t["name"]] = "%d/%d" % (p, len(seeds))
            print("  %-16s %s/%d" % (t["name"], p, len(seeds)))

        summaries.append({"model": name, "column": "nl", "ckpt": str(ckpt_dir),
                          "params_m": round(n_params, 1), "layers": model.config.num_layers,
                          "tasks": task_pass,
                          "total_nl": "%d/%d" % (n_hit, n_tot),
                          "elapsed_s": round(time.time() - t0, 1)})

        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    sp = out_dir / "nl_eval_summary.json"
    gp = out_dir / "nl_eval_generations.jsonl"
    payload = {
        "meta": {
            "column": "natural-language instruction",
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_template_source": "build_instr_data.PROMPT_TEMPLATES['marker']",
            "judge": "extract after last '### Code' -> strip fences -> dedent -> truncate -> run_script(timeout=3)",
            "tests_source": "eval_sft.ALGORITHMS + eval_sft.SIMPLE_TASKS (same list object)",
            "seeds": seeds, "gen": gen, "n_tasks": len(NL_TASKS),
            "comparable_to": "eval/eval_sft_summary.json (signature column)",
        },
        "models": summaries,
    }
    sp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(gp, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("\n[SAVE] %s + %s (%d rows)" % (sp.name, gp.name, len(all_samples)))
    compare_with_sft(summaries, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
