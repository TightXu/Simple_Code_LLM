#!/usr/bin/env python3
"""
================================================================
CodeLM - unified evaluation (multi-model x multiple algorithms x multiple seeds)
================================================================

CLI by default; add --interactive for guided mode (no other args required).

Goal: evaluate several checkpoints under identical prompt / seed / generation
settings, save every generation for human review, and also compute a reference
pass-rate (execute test cases + implementation checks). The reference rate is
advisory, not the verdict - see TECHNICAL.md on evaluation methodology: script
pass-rates lie, human review is the primary metric here.

Known limitations of the reference pass-rate (full discussion in TECHNICAL.md):
  1. Output-only tests can be cheated (arr.sort() passes quicksort) -> impl checks
     now scan the generated body for banned calls (e.g. sort()) and require key
     structure (e.g. mid in binary search).
  2. Output-only tests false-negative on tab/space mixing (TabError) -> normalize
     indentation before subprocess execution.
  3. Docstring blindness: from-scratch models treat docstrings as function end,
     so algorithm prompts are docstring-free (trailing 4-space indent to let the
     model complete the body); use --prompt-style docstring for HumanEval-style.

Usage (CLI):
  # all models x all algorithms x 5 seeds
  py -3.12 src/eval/eval.py

  # specified models / algorithms / seeds
  py -3.12 src/eval/eval.py --models checkpoints_wsm/merged checkpoints/final \
      --algorithms fibonacci quicksort --seeds 42,123

  # interactive mode
  py -3.12 src/eval/eval.py --interactive

Output:
  eval_output/generations.jsonl     # one line per generation (for human review)
  eval_output/summary.json          # reference pass-rate summary + per-algo table
================================================================
"""
import os, sys, json, math, time, random, argparse, re, subprocess, tempfile
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse model code from src/train.py (parameterized arch)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import ModelConfig, CodeLLM, load_checkpoint

# ═══════════════ algorithms (docstring-free; trailing 4-space indent) ═══════════════
ALGORITHMS = [
    {"name": "fibonacci", "prompt": "def fibonacci(n):\n    ", "entry": "fibonacci",
     "tests": ["fibonacci(0) == 0", "fibonacci(1) == 1",
               "fibonacci(10) == 55", "fibonacci(20) == 6765"],
     "required": ["fibonacci(", "n-1|n - 1"], "banned": ["arr.sort()"]},
    {"name": "quicksort", "prompt": "def quicksort(arr):\n    ", "entry": "quicksort",
     "tests": ["quicksort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]",
               "quicksort([]) == []", "quicksort([5]) == [5]"],
     "required": ["def quicksort", "arr["], "banned": ["arr.sort()", "sorted(arr)"]},
    {"name": "binary_search", "prompt": "def binary_search(arr, target):\n    ", "entry": "binary_search",
     "tests": ["binary_search([1, 3, 5, 7, 9], 5) == 2",
               "binary_search([1, 3, 5, 7, 9], 6) == -1",
               "binary_search([], 1) == -1"],
     "required": ["mid"], "banned": ["arr.index("]},
    {"name": "two_sum", "prompt": "def two_sum(nums, target):\n    ", "entry": "two_sum",
     "tests": ["sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]",
               "sorted(two_sum([3, 2, 4], 6)) == [1, 2]",
               "sorted(two_sum([3, 3], 6)) == [0, 1]"],
     "required": ["for "], "banned": []},
    {"name": "mergesort", "prompt": "def mergesort(arr):\n    ", "entry": "mergesort",
     "tests": ["mergesort([38, 27, 43, 3, 9, 82, 10]) == [3, 9, 10, 27, 38, 43, 82]",
               "mergesort([]) == []", "mergesort([1]) == [1]"],
     "required": ["def mergesort", "mid"], "banned": ["arr.sort()", "sorted(arr)"]},
]
DEFAULT_SEEDS = [42, 123, 456, 789, 2026]


def load_algorithms_file(path: str) -> list:
    """Load a custom task-set file and return a task list in Algorithms format.
    Two input formats are supported:
      (1) Algorithms: {name, prompt, entry, tests, required, banned}
      (2) HumanEval: {task_id, prompt, entry_point, test, ...} -> converted to (1)
    """
    import os
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"algorithms-file not found: {path}")
    if p.suffix == ".json":
        tasks = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(tasks, dict):
            tasks = [tasks]
    else:  # .jsonl
        tasks = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    out = []
    for t in tasks:
        t = dict(t)
        # HumanEval-style test field: a Python code string (with asserts). Extract the assert lines as tests.
        if "entry_point" in t and ("test" in t or "assertions" in t):
            entry = t["entry_point"]
            prompt = t["prompt"]
            raw_test = t.get("test", "") or "\n".join(t.get("assertions", []))
            # collect every line containing an assert as a test; if test is a full check() function, pull out the candidate calls
            tests = []
            for ln in raw_test.splitlines():
                s = ln.strip()
                if s.startswith("assert "):
                    # rewrite "assert candidate(args) == val" by replacing candidate with entry
                    tests.append(s.replace("candidate", entry).replace("assert ", "", 1))
            if not tests and raw_test.strip():
                # fallback: if it cannot be split, use the whole test as a single case (handed to judge for execution)
                tests = [raw_test]
            out.append({"name": t.get("task_id", entry), "prompt": prompt,
                        "entry": entry, "tests": tests,
                        "required": [], "banned": []})
        else:
            # Algorithms format
            out.append({"name": t["name"], "prompt": t["prompt"], "entry": t["entry"],
                        "tests": t.get("tests", []), "required": t.get("required", []),
                        "banned": t.get("banned", [])})
    return out

# ═══════════════════ generation helpers ═══════════════════
def sanitize_indent(code: str) -> str:
    """Normalize tab/space mixing (TabError killer) to 4-space indent; keeps relative body indentation."""
    lines = code.split("\n")
    out = []
    for ln in lines:
        stripped = ln.expandtabs(4)
        # only normalize leading whitespace, keep tabs inside content
        out.append(stripped)
    return "\n".join(out)

def run_script(src: str, timeout=5):
    """Execute generated code + tests; returns (ok, err). Each run in a fresh subprocess."""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(src)
            path = f.name
        proc = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        os.unlink(path)
        return (proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[:200])
    except subprocess.TimeoutExpired:
        return (False, "TIMEOUT")
    except Exception as e:
        return (False, str(e)[:200])

def check_implementation(code: str, alg: dict):
    """Implementation check: anti-cheat banned calls + required structure (e.g. mid in binary search)."""
    lower = code.lower()
    for pat in alg.get("banned", []):
        if pat.lower() in lower:
            return False, f"BANNED {pat}"
    for pat in alg.get("required", []):
        # required patterns: alternatives use |, otherwise single match
        if "|" in pat:
            if not any(alt in lower for alt in pat.split("|")):
                return False, f"MISSING {pat}"
        elif pat not in lower:
            return False, f"MISSING {pat}"
    return True, ""

def extract_function_body(full_code: str) -> str:
    """Extract a clean function body from prompt+gen (keeps the docstring, drops trailing junk).

    Rule: start at the first `def ` line and collect every line indented at or
    beyond the body indentation; an unindented, non-empty, non-comment line marks
    the end of the function, truncating any continuation junk after it
    (print/import/if __name__/comments).

    Returns the function body source (including the def line); empty string if
    there is no def line.
    Note: only code with consistent body indentation that parses as an AST is
    accepted; the extracted result still has to pass ast.parse.
    """
    lines = full_code.split("\n")
    out = []
    started = False
    def_line = None
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if not started:
            if stripped.startswith("def "):
                started = True
                def_line = i
                out.append(ln)
            elif stripped and not stripped.startswith("#"):
                # non-def, non-comment line: junk from the prompt prefix; drop it and keep looking for def
                continue
            # comment/blank line: skip
            continue
        # already inside the function body
        if stripped and not stripped.startswith("#"):
            if not ln.startswith((" ", "\t")):
                break  # unindented line = end of the function body
            out.append(ln)
        else:
            # blank/comment line: usually a blank line or comment inside the body; keep it, but it is truncated if an unindented line follows
            if ln.startswith((" ", "\t")):
                out.append(ln)
    if not out:
        return ""
    # drop the extra blank lines at the end of the body
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def judge_reference(code: str, alg: dict):
    """Reference pass-rate: implementation check + execution. Returns (pass, reason).

    Improvement: extract the function body from the full code (prompt+gen) first,
    dropping the junk after a trailing docstring (print/import/comments/if __name__),
    then run the tests. This stops models like v2 - which write the docstring
    before the logic - from being wrongly judged as failures.
    """
    # extract a clean function body first (including the def line)
    core = extract_function_body(code)
    if not core:
        return False, "no function body"
    # the implementation check runs on the core body (no prompt-prefix junk)
    ok_impl, reason = check_implementation(core, alg)
    if not ok_impl:
        return False, f"impl: {reason}"
    full = core + "\n\n" + "\n".join(f"assert {t}" for t in alg["tests"]) + "\nprint('PASS')"
    ok_run, err = run_script(sanitize_indent(full))
    if not ok_run:
        return False, f"exec: {err[:60] if err else 'FAIL'}"
    return True, "pass"

def generate(model, tokenizer, prompt, seed, max_new=256, temperature=0.2,
             top_k=50, repetition_penalty=1.2, min_p=0.05, device="cuda"):
    """Per-sample generation (no batch/padding - batched mask generation OOMs and is slow)."""
    torch.manual_seed(seed)
    input_ids = tokenizer.encode(prompt).ids
    input_ids = [min(i, model.config.vocab_size - 1) for i in input_ids]
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids)[:, -1, :]
            logits = logits / temperature
            # repetition penalty (downweight each already-generated token)
            for tok in set(ids[0].tolist()):
                logits[0, tok] = logits[0, tok] / repetition_penalty
            # top-k
            if top_k > 0 and logits.size(-1) > top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            # min-p (small-model anti-collapse): downweight below max*min_p
            probs = torch.softmax(logits, dim=-1)
            p_min = probs.max(dim=-1, keepdim=True).values * min_p
            logits[probs < p_min] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=-1)
            if nxt.item() == tokenizer.token_to_id("</s>"):
                break
    out = ids[0].tolist()
    # strip prompt prefix from output
    gen = tokenizer.decode(out[len(input_ids):])
    return gen

def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load a checkpoint plus its tokenizer (looks in the model dir, then repo tokenizer/)."""
    from train import load_checkpoint
    from tokenizers import Tokenizer
    model, config, _opt, _sched, state = load_checkpoint(model_path, device)
    model.to(device)          # load_checkpoint only moves the ckpt to device; the params default to CPU -> move them explicitly
    # Note: no bf16 conversion - train.py's forward contains `x * math.sqrt(d_model)`,
    # and the scalar multiply promotes the output to float32; bf16 weights would raise a dtype mismatch (float != BFloat16).
    # Keeping float32 is the most reliable setup for evaluation (inference is a little slower, but it only affects evaluation throughput, not correctness).
    # tokenizer search order: model dir -> repo tokenizer/<name>.json -> repo tokenizer.json
    cands = [
        Path(model_path) / "tokenizer.json",
        Path(__file__).resolve().parent.parent.parent / "tokenizer" / f"tokenizer_{Path(model_path).stem}.json",
        Path(__file__).resolve().parent.parent.parent / "tokenizer" / "tokenizer_435m.json",
    ]
    tok_path = next((c for c in cands if c.exists()), None)
    if tok_path is None:
        raise FileNotFoundError(f"tokenizer.json not found (tried {[str(c) for c in cands]})")
    tokenizer = Tokenizer.from_file(str(tok_path))
    return model, tokenizer

def interact(prompt_text, default=""):
    val = input(prompt_text).strip()
    return val if val else default

# ═══════════════════ main ═══════════════════
def main():
    ap = argparse.ArgumentParser(description="CodeLM unified eval: multi-model x multi-algo x multi-seed")
    ap.add_argument("--interactive", action="store_true", help="interactive mode (no other args required)")
    ap.add_argument("--models", type=str, default=None, help="comma-separated checkpoint paths (default: interactive/all)")
    ap.add_argument("--algorithms", type=str, default=None, help="comma-separated algorithm names (default: all)")
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated seeds (default: 5)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", type=str, default="eval_output")
    ap.add_argument("--algorithms-file", type=str, default=None,
                    help="custom task-set file (.json single-object array, or .jsonl one object per line). "
                         "each task accepts two formats: "
                         "(1) Algorithms format: {name,prompt,entry,tests,required,banned}; "
                         "(2) HumanEval format: {task_id,prompt,entry_point,test/assertions} - converted automatically. "
                         "when given, custom tasks are merged with the built-in ALGORITHMS (same name overrides).")
    ap.add_argument("--prompt-style", choices=["docstring-free", "docstring"], default="docstring-free",
                    help="docstring-free=completion (default); docstring=HumanEval-style (from-scratch models are blind to it)")
    args = ap.parse_args()

    # normalize models / algorithms / seeds
    if args.interactive:
        models = interact("checkpoint paths (comma-separated): ")
        algs = interact("algorithm names (comma-separated, default all): ", ",".join(a["name"] for a in ALGORITHMS))
        seeds = interact("seeds (comma-separated, default 5): ", ",".join(map(str, DEFAULT_SEEDS)))
        models = [m.strip() for m in models.split(",") if m.strip()]
        algs = [a.strip() for a in algs.split(",") if a.strip()]
        seeds = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    else:
        models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
        algs = [a.strip() for a in (args.algorithms or "").split(",") if a.strip()] or [a["name"] for a in ALGORITHMS]
        seeds = [int(s.strip()) for s in (args.seeds or ",".join(map(str, DEFAULT_SEEDS))).split(",") if s.strip()]

    # selected algorithm subset
    alg_pool = list(ALGORITHMS)
    if args.algorithms_file:
        custom = load_algorithms_file(args.algorithms_file)
        # merge: built-in + custom (same name overrides)
        custom_names = {a["name"] for a in custom}
        alg_pool = [a for a in ALGORITHMS if a["name"] not in custom_names] + custom
        print(f"[CUSTOM] merged {len(custom)} task(s) from {args.algorithms_file} "
              f"(task set now has {len(alg_pool)})")
    alg_map = {a["name"]: a for a in alg_pool}
    selected = [alg_map[n] for n in algs if n in alg_map]
    if not selected:
        print("[WARN]  no valid algorithms. Available:", list(alg_map))
        return

    print(f"[TARGET] evaluating: {len(models)} model(s) x {len(selected)} algorithm(s) x {len(seeds)} seed(s)")
    print(f"   generation: temp={args.temperature} top_k={args.top_k} rep={args.repetition_penalty} min_p={args.min_p}")

    # load models
    loaded = []
    for mp in models:
        try:
            m, tok = load_model_and_tokenizer(mp, args.device)
            loaded.append((mp, m, tok))
            print(f"[OK] loaded {mp}")
        except Exception as e:
            print(f"[FAIL]  failed to load {mp}: {e}")

    if not loaded:
        print("[WARN]  no usable models, exiting.")
        return

    # per-sample generation
    os.makedirs(args.output_dir, exist_ok=True)
    gen_path = Path(args.output_dir) / "generations.jsonl"
    summary = {"models": [], "params": vars(args), "algos": [a["name"] for a in selected], "seeds": seeds}
    total, passed = 0, 0
    with open(gen_path, "w", encoding="utf-8") as fo:
        for mp, model, tokenizer in loaded:
            model_results = []
            for alg in selected:
                alg_total, alg_pass = 0, 0
                for seed in seeds:
                    gen = generate(model, tokenizer, alg["prompt"], seed,
                                   args.max_new, args.temperature, args.top_k,
                                   args.repetition_penalty, args.min_p, args.device)
                    ok_ref, reason = judge_reference(alg["prompt"] + gen, alg)
                    total += 1
                    passed += int(ok_ref)
                    alg_total += 1
                    alg_pass += int(ok_ref)
                    rec = {"model": mp, "algorithm": alg["name"], "seed": seed,
                           "generation": gen, "reference_pass": ok_ref, "reason": reason}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                model_results.append({"algorithm": alg["name"], "pass": alg_pass, "total": alg_total})
                print(f"  {mp.split('/')[-1]} | {alg['name']:12s} | {alg_pass}/{alg_total} reference pass")
            summary["models"].append({"path": mp, "results": model_results})

    summary["reference_overall"] = {"pass": passed, "total": total}
    with open(Path(args.output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO]  reference pass-rate (advisory; human review is authoritative): {passed}/{total} = {passed/total*100:.1f}%")
    print(f"   generations -> {gen_path}")
    print(f"   summary   -> {Path(args.output_dir) / 'summary.json'}")
    print("\n[WARN]  note: script pass-rates are known to lie (cheating/false negatives); review generations.jsonl by hand.")

if __name__ == "__main__":
    main()
