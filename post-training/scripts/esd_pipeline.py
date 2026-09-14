#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
435M v2 · execution-feedback self-distillation (ESD / rejection sampling) pipeline scaffolding
==============================================================================================
Turns a public "code instruction + unit tests" parquet into an executable **problem set**, then
runs the candidate code produced by a model (or by the reference solution) against the unit tests
inside a **Docker sandbox** to obtain PASS/FAIL as the filtering signal -- that is the execution
feedback stage of rejection sampling. This file only does "scaffolding + verifiable sandbox
execution": it does not train, load models, or use the GPU.

Three (+1) stages
-----------------
  info           print the real parquet schema + 2 samples (look before you leap)
  extract        parquet -> normalized JSONL problem set
  sandbox-batch  run each candidate's tests in a Docker container -> {id, passed, reason, seconds}
  dryrun         self-check the whole chain on 20 problems: reference solutions should all PASS,
                 three kinds of deliberately broken solutions should all FAIL

Problem-set fields (extract output, one per line)
-------------------------------------------------
  id            id from the original parquet
  prompt        original input (natural-language requirement, verbatim)
  entry         name of the first def/class in the reference solution (the tests must call it)
  tests         normalized list of unit-test statements (each an independently executable Python
                statement string)
  ref_solution  Python code extracted from the original output (fence stripped), used by the
                dryrun self-check
  meta          {domain, generation_algorithm, average_test_score, prompt_templated,
                 prompt_template, n_tests, tokens_prompt, code_chars}

Execution semantics (isomorphic to check_algo in eval_sft.py, which is not modified)
-----------------------------------------------------------------------------------
  script text = code + "\\n\\n" + "\\n".join(tests) + "\\n"
  exit code 0 -> PASS; otherwise the last stderr line becomes reason; timeout -> TIMEOUT

Sandbox design (the important part)
-----------------------------------
  * Image python:3.12-slim, container `--network none` (no network), `--read-only` + tmpfs /tmp
    (candidate code is written to tmpfs, never to the mounted directories; the host directories
    only pass inputs in and collect results).
  * **Batching**: one container runs N samples (--batch-size), instead of one container per sample.
  * **Per-sample isolation**: the in-container driver uses subprocess with `start_new_session=True`,
    and on timeout `os.killpg(..., SIGKILL)` kills the **entire process group** -- an infinite loop
    or hang only destroys its own sample and cannot drag down the batch (dryrun verifies this with
    20 infinite-loop negatives).
  * Multithreaded concurrency inside the container (--jobs); each sample is still **timed
    individually**.

Usage
-----
    py -3.12 esd_pipeline.py --stage info
    py -3.12 esd_pipeline.py --stage extract --max-problems 200
    py -3.12 esd_pipeline.py --stage sandbox-batch --problems esd/problems.jsonl \\
        --candidates C.jsonl --out esd/results.jsonl
    py -3.12 esd_pipeline.py --stage dryrun --n 20

This file **only adds files**: code goes to esd_pipeline.py, data/reports go to the esd/ directory.
No existing file is modified (it only imports build_instr_data to reuse its 167-name eval-leak
blacklist).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../435M-v2/fine-tuning
PROJECT = HERE.parent                             # .../435M-v2
DEFAULT_PARQUET = HERE / "data_instr_raw" / "opencodeinstruct_shard0.parquet"
DEFAULT_OUTDIR = HERE / "esd"
DEFAULT_TOKENIZER = PROJECT / "tokenizer" / "tokenizer_435m.json"
DEFAULT_IMAGE = "python:3.12-slim"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Reuse the existing builder's "eval-leak" blacklist (167 names); read-only, never modified ──
sys.path.insert(0, str(HERE))
try:
    import build_instr_data as BI            # noqa: E402
    BLACKLIST = BI.build_blacklist(False)
    PROMPT_TEMPLATE = BI.PROMPT_TEMPLATES["marker"]      # "### Task\n{instruction}\n\n### Code\n"
except Exception as e:                                        # pragma: no cover
    BLACKLIST = set()
    PROMPT_TEMPLATE = "### Task\n{instruction}\n\n### Code\n"
    print(f"[warn] cannot import build_instr_data ({type(e).__name__}: {e}); blacklist falls back to empty",
          file=sys.stderr)

DEF_RX = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)", re.M)
FENCE_RX = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)
IMPORT_RX = re.compile(r"^[ \t]*(?:import[ \t]+([A-Za-z_][\w.]*)|from[ \t]+([A-Za-z_][\w.]*)[ \t]+import)", re.M)


# ═══════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def jloads(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def strip_fence(s: str) -> str:
    """Strip ```python ... ``` fences; return the text unchanged when there is no fence."""
    m = FENCE_RX.search(s or "")
    return (m.group(1) if m else (s or "")).replace("\r\n", "\n").strip("\n")


def norm_tests(raw) -> list[str]:
    """unit_tests field → normalized statement list (each entry stripped, internal newlines kept)."""
    v = jloads(raw) if isinstance(raw, str) else raw
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    out = []
    for t in v:
        if not isinstance(t, str):
            continue
        t = t.replace("\r\n", "\n").strip("\n").strip()
        if t:
            out.append(t)
    return out


def all_tests_passed(raw) -> bool:
    """tests_execution_status: in OpenCodeInstruct this is a per-test JSON list of 'pass'/'fail'."""
    v = jloads(raw) if isinstance(raw, str) else raw
    if isinstance(v, str):
        return v.strip().lower() in ("pass", "passed", "true", "1")
    if isinstance(v, list):
        return bool(v) and all(str(x).strip().lower() in ("pass", "passed", "true", "1") for x in v)
    return False


def entry_of(code: str):
    m = DEF_RX.search(code or "")
    return m.group(1) if m else None


def tests_call_entry(tests, entry) -> bool:
    rx = re.compile(r"\b%s\s*\(" % re.escape(entry or ""))
    return any(rx.search(t) for t in tests)


def nonstdlib_imports(tests) -> list[str]:
    """Root modules imported from non-stdlib packages in tests (the sandbox image is stdlib-only,
    so such problems can never pass)."""
    mods = set()
    for t in tests:
        for a, b in IMPORT_RX.findall(t):
            root = (a or b).split(".")[0]
            if root and root not in sys.stdlib_module_names:
                mods.add(root)
    return sorted(mods)


# Nondeterminism signals: used with no seed() anywhere → repeated runs of the same code differ
# (measured: the dice_game problem passed 5 of 12 runs)
NONDET_RX = re.compile(
    r"(?:\brandom\s*\.|(?:np|numpy)\s*\.\s*random|uuid\s*\.\s*uuid4|\bsecrets\s*\.|os\s*\.\s*urandom"
    r"|time\s*\.\s*(?:time|perf_counter)\s*\(|datetime\s*\.\s*(?:now|today)\s*\()")
SEED_RX = re.compile(r"\bseed\s*\(")


def nondeterminism_hint(text: str) -> str | None:
    """Return the matched nondeterminism marker; a seed() call makes it reproducible, so it does not count."""
    m = NONDET_RX.search(text or "")
    if not m:
        return None
    if SEED_RX.search(text or ""):
        return None
    return m.group(0).strip()[:32]


# ═══════════════════════════════════════════════════════════════════
# Docker
# ═══════════════════════════════════════════════════════════════════

DOCKER_CANDIDATES = [
    os.environ.get("ESD_DOCKER", ""),
    os.environ.get("DOCKER_BIN", ""),
    r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
    r"C:\Program Files\Docker\Docker\resources\bin\docker",
    "/c/Program Files/Docker/Docker/resources/bin/docker",
]


def find_docker() -> str:
    for c in DOCKER_CANDIDATES:
        if c and Path(c).exists():
            return c
    w = shutil.which("docker")
    if w:
        return w
    raise SystemExit("docker executable not found; set ESD_DOCKER=<path> or export PATH and retry")


def has_image(docker: str, image: str) -> bool:
    r = subprocess.run([docker, "image", "inspect", image],
                       capture_output=True, text=True)
    return r.returncode == 0


def ensure_image(docker: str, image: str, auto_pull: bool = True) -> None:
    if has_image(docker, image):
        return
    if not auto_pull:
        raise SystemExit(f"image {image} not found (--no-pull disables automatic pull)")
    eprint(f"[docker] pulling image {image} ...")
    r = subprocess.run([docker, "pull", image], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"docker pull failed: {r.stderr.strip()[-400:]}")


DRIVER_SRC = r'''# -*- coding: utf-8 -*-
"""In-container driver: run the unit tests item by item (concurrently), each with its own
timeout + killpg, appending results to a JSONL file.

Path conventions:
  /in   read-only mount (_driver.py + _items.jsonl)
  /out  writable mount (only _results.jsonl is written)
  /tmp  tmpfs (candidate code files go here; cwd is set here too, so relative-path writes
        inside tests land on the RAM disk and die with the container instead of polluting
        the host mount)
"""
import json, os, signal, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

IN = os.environ.get("ESD_IN", "/in")
OUT = os.environ.get("ESD_OUT", "/out")
ITEMS = os.path.join(IN, "_items.jsonl")
RESULTS = os.path.join(OUT, "_results.jsonl")


def classify(err):
    lines = [l for l in (err or "").strip().splitlines() if l.strip()]
    if not lines:
        return "ERR:nonzero-exit"
    return lines[-1][:180]


def run_one(it):
    body = it["code"] + "\n\n" + "\n".join(it["tests"]) + "\n"
    t0 = time.time()
    fd, src = tempfile.mkstemp(suffix=".py", dir="/tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        try:
            compile(body, src, "exec")
        except SyntaxError as e:
            return {"id": it["id"], "passed": False,
                    "reason": "SyntaxError: %s" % (str(e)[:160]),
                    "seconds": round(time.time() - t0, 4)}
        timeout = float(it.get("timeout", 10))
        p = subprocess.Popen([sys.executable, src], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True,
                             cwd="/tmp")
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            try:
                p.communicate(timeout=5)
            except Exception:
                pass
            return {"id": it["id"], "passed": False, "reason": "TIMEOUT",
                    "seconds": round(time.time() - t0, 4)}
        if p.returncode == 0:
            return {"id": it["id"], "passed": True, "reason": "ok",
                    "seconds": round(time.time() - t0, 4)}
        return {"id": it["id"], "passed": False, "reason": classify(err),
                "seconds": round(time.time() - t0, 4)}
    except Exception as e:
        return {"id": it["id"], "passed": False,
                "reason": "DRIVER-EXC: %s: %s" % (type(e).__name__, str(e)[:140]),
                "seconds": round(time.time() - t0, 4)}
    finally:
        try:
            os.unlink(src)
        except OSError:
            pass


def main():
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8") if l.strip()]
    jobs = max(1, int(os.environ.get("ESD_JOBS", "4")))
    with open(RESULTS, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            for r in ex.map(run_one, items):
                out.write(json.dumps(r) + "\n")
                out.flush()


main()
'''


def _docker_path(p) -> str:
    """Docker Desktop only accepts absolute paths with forward slashes; a relative path is taken
    as a named volume and errors out."""
    return str(Path(p).resolve()).replace("\\", "/")


def docker_run_cmd(docker: str, image: str, in_dir: Path, out_dir: Path, jobs: int,
                   mem: str = "1g", cpus: str = "2") -> list[str]:
    """Build the container command (as a list, so paths containing spaces are not split by a shell).

    The working directory is /tmp (tmpfs): relative-path writes inside tests (the dataset really
    contains them, e.g. `FileManager().write_file('test3.txt', ...)`) land on the RAM disk and
    never pollute the host mounts. The input directory is mounted **read-only** and results go to
    a separate writable directory -- candidate code cannot rewrite its own problem or inputs.
    """
    return [
        docker, "run", "--rm",
        "--network", "none",                       # no network
        "--read-only",                             # read-only root filesystem
        "--tmpfs", "/tmp:rw,size=128m,exec",       # the only large writable area (RAM disk)
        "--memory", mem, "--cpus", cpus, "--pids-limit", "512",
        "--security-opt", "no-new-privileges",
        "-e", f"ESD_JOBS={jobs}", "-e", "ESD_IN=/in", "-e", "ESD_OUT=/out",
        "-v", f"{_docker_path(in_dir)}:/in:ro",    # read-only inputs
        "-v", f"{_docker_path(out_dir)}:/out",     # separate writable results directory
        "-w", "/tmp",
        image, "python", "/in/_driver.py",
    ]


def sandbox_run(docker: str, image: str, items: list[dict], workdir: Path,
                jobs: int = 4, batch_tag: str = "batch") -> dict:
    """Run all items in **one** container and return ({id: result}, meta).

    Directory layout: <workdir>/in/ (read-only inputs) + <workdir>/out/ (writable results).
    """
    in_dir, out_dir = workdir / "in", workdir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / "_driver.py").write_text(DRIVER_SRC, encoding="utf-8")
    with open(in_dir / "_items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    res_path = out_dir / "_results.jsonl"
    res_path.write_text("", encoding="utf-8")

    cmd = docker_run_cmd(docker, image, in_dir, out_dir, jobs)
    inner = max(float(it.get("timeout", 10)) for it in items) if items else 10
    outer = max(90.0, (inner + 3.0) * len(items) / max(1, jobs) + 60.0)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=outer)
        rc, serr = p.returncode, (p.stderr or "")
    except subprocess.TimeoutExpired:
        rc, serr = -9, f"host-side outer timeout of {outer:.0f}s fired (some results may already be on disk)"
    elapsed = time.time() - t0

    out = {}
    for line in res_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                out[r["id"]] = r
            except Exception:
                pass
    got = len(out)
    print(f"[{batch_tag}] container rc={rc} elapsed {elapsed:.1f}s samples {got}/{len(items)}")
    if got < len(items):
        eprint(f"[{batch_tag}] warning: {len(items) - got} items have no result; container stderr tail: {serr.strip()[-300:]}")
    if items and got == 0 and rc != 0:
        raise SystemExit(
            f"[{batch_tag}] zero results for the whole batch (container rc={rc}) -- "
            f"the docker command itself failed, this is not a per-sample issue.\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr: {serr.strip()[-600:]}")
    return out, {"rc": rc, "elapsed": round(elapsed, 2), "n_items": len(items),
                 "n_results": got, "cmd": cmd, "stderr_tail": serr.strip()[-400:]}


def sandbox_batch_chunked(docker, image, items, workroot, batch_size, jobs, label):
    """Split into batches, one container each; return (merged_results, batch_metas)."""
    merged, metas = {}, []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        d = workroot / f"{label}_{i // batch_size:03d}"
        r, m = sandbox_run(docker, image, chunk, d, jobs=jobs,
                           batch_tag=f"{label}[{i // batch_size}]")
        m["items"] = [x["id"] for x in chunk]
        metas.append(m)
        for k, v in r.items():
            merged[k] = v
    return merged, metas


# ═══════════════════════════════════════════════════════════════════
# stage: info
# ═══════════════════════════════════════════════════════════════════

def stage_info(args):
    import pyarrow.parquet as pq
    p = Path(args.parquet)
    pf = pq.ParquetFile(p)
    print(f"== parquet: {p}")
    print(f"== rows={pf.metadata.num_rows}  row_groups={pf.metadata.num_row_groups}")
    print("== schema ==")
    print(pf.schema_arrow)
    rows = pf.read_row_group(0).slice(0, args.samples).to_pylist()
    for i, r in enumerate(rows):
        print(f"\n===== sample {i} (id={r.get('id')}) =====")
        for k, v in r.items():
            s = v if isinstance(v, str) else repr(v)
            print(f"--- {k} ({type(v).__name__}, len={len(s)}) ---")
            print(s[: args.chars])
    # actual shape of unit_tests
    ut = rows[0].get("unit_tests") if rows else None
    v = jloads(ut)
    print("\n== unit_tests parsing ==")
    print(f"   JSON parse: {type(v).__name__}, count={len(v) if isinstance(v, list) else 'N/A'}")
    if isinstance(v, list) and v:
        print(f"   element type={type(v[0]).__name__}; first repr={v[0]!r}")
    print(f"   first-sample tests_execution_status={rows[0].get('tests_execution_status')!r}")
    print(f"   → all_tests_passed()={all_tests_passed(rows[0].get('tests_execution_status'))}")


# ═══════════════════════════════════════════════════════════════════
# stage: extract
# ═══════════════════════════════════════════════════════════════════

def extract_problems(args, parquet=None, out_path=None, max_problems=None,
                     return_stats=False):
    import pyarrow.parquet as pq
    parquet = Path(parquet or args.parquet)
    max_problems = int(max_problems if max_problems is not None else args.max_problems)
    max_chars = args.max_input_chars
    max_tok = args.max_prompt_tokens if not args.no_token_filter else 0

    tok = None
    if max_tok:
        try:
            from tokenizers import Tokenizer
            if Path(args.tokenizer).exists():
                tok = Tokenizer.from_file(str(args.tokenizer))
            else:
                eprint(f"[extract] tokenizer not found at {args.tokenizer} → skipping token filter")
        except Exception as e:
            eprint(f"[extract] tokenizer unavailable ({type(e).__name__}) → skipping token filter")

    st = {"rows": 0, "drop_not_all_pass": 0, "drop_leak": 0, "drop_long_chars": 0,
          "drop_long_tokens": 0, "drop_no_entry": 0, "drop_no_tests": 0,
          "drop_entry_not_called": 0, "drop_nonstdlib": 0,
          "drop_nondeterministic": 0, "kept": 0}

    problems = []
    pf = pq.ParquetFile(parquet)
    for rg in range(pf.metadata.num_row_groups):
        if len(problems) >= max_problems:
            break
        for r in pf.read_row_group(rg).to_pylist():
            st["rows"] += 1
            if len(problems) >= max_problems:
                break
            if not all_tests_passed(r.get("tests_execution_status")):
                st["drop_not_all_pass"] += 1
                continue
            tests = norm_tests(r.get("unit_tests"))
            if not tests:
                st["drop_no_tests"] += 1
                continue
            inp = (r.get("input") or "").strip()
            if not inp or len(inp) > max_chars:
                st["drop_long_chars"] += 1
                continue
            code = strip_fence(r.get("output") or "")
            if not code:
                st["drop_no_entry"] += 1
                continue
            if BLACKLIST and set(DEF_RX.findall(code)).intersection(BLACKLIST):
                st["drop_leak"] += 1
                continue
            entry = entry_of(code)
            if not entry:
                st["drop_no_entry"] += 1
                continue
            if not tests_call_entry(tests, entry):
                st["drop_entry_not_called"] += 1
                continue
            if not args.no_dep_filter:
                # Dependencies can hide in the **reference solution** (e.g. Flask problems: the
                # tests import nothing, the reference solution does `import flask`), or in the
                # tests themselves -- scan both sides. The image is stdlib-only, so any non-stdlib
                # dependency is guaranteed to fail.
                deps = sorted(set(nonstdlib_imports(tests)).union(nonstdlib_imports([code])))
                if deps:
                    st["drop_nonstdlib"] += 1
                    continue
            if not args.keep_nondeterministic:
                # Nondeterministic problems (unseeded random / time / uuid) make PASS/FAIL jitter,
                # which directly corrupts the rejection-sampling filter signal -- dropped by default.
                if nondeterminism_hint(code + "\n" + "\n".join(tests)):
                    st["drop_nondeterministic"] += 1
                    continue
            prompt = PROMPT_TEMPLATE.format(instruction=inp)
            ntok = None
            if tok is not None:
                ntok = len(tok.encode(prompt, add_special_tokens=False).ids)
                if ntok > max_tok:
                    st["drop_long_tokens"] += 1
                    continue
            problems.append({
                "id": r.get("id"),
                "prompt": inp,
                "entry": entry,
                "tests": tests,
                "ref_solution": code,
                "meta": {
                    "domain": r.get("domain"),
                    "generation_algorithm": r.get("generation_algorithm"),
                    "average_test_score": r.get("average_test_score"),
                    "n_tests": len(tests),
                    "code_chars": len(code),
                    "prompt_chars": len(inp),
                    "tokens_prompt": ntok,
                    "prompt_template": PROMPT_TEMPLATE,
                    "prompt_templated": prompt,
                    "source_parquet": str(parquet),
                },
            })
            st["kept"] += 1

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for pr in problems:
                f.write(json.dumps(pr, ensure_ascii=False) + "\n")

    print(f"[extract] scanned {st['rows']} rows → kept {st['kept']} problems")
    print("[extract] filter funnel: " + "  ".join(f"{k}={v}" for k, v in st.items() if k != "rows" and v))
    if out_path is not None:
        print(f"[extract] → {out_path}")
    return (problems, st) if return_stats else problems


# ═══════════════════════════════════════════════════════════════════
# stage: sandbox-batch
# ═══════════════════════════════════════════════════════════════════

def load_jsonl(p):
    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def stage_sandbox_batch(args):
    docker = find_docker()
    ensure_image(docker, args.image, auto_pull=not args.no_pull)
    problems = {p["id"]: p for p in load_jsonl(args.problems)}
    cands = load_jsonl(args.candidates)
    print(f"[sandbox-batch] {len(problems)} problems, {len(cands)} candidates, image {args.image}")

    items, missing = [], []
    for c in cands:
        cid = c.get("id")
        pr = problems.get(cid)
        if pr is None:
            missing.append(cid)
            continue
        code = strip_fence(c.get("completion") or "")
        items.append({"id": cid, "code": code, "tests": pr["tests"], "timeout": args.timeout})

    workroot = Path(args.workdir) if args.workdir else (Path(args.out).parent / "_sandbox")
    results, metas = sandbox_batch_chunked(docker, args.image, items, workroot,
                                           args.batch_size, args.jobs, "sb")
    for cid in missing:
        results[cid] = {"id": cid, "passed": False, "reason": "NO_PROBLEM", "seconds": 0.0}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in cands:
            r = results.get(c.get("id")) or {"id": c.get("id"), "passed": False,
                                            "reason": "NO_RESULT", "seconds": 0.0}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    npass = sum(1 for r in results.values() if r.get("passed"))
    print(f"[sandbox-batch] PASS {npass}/{len(cands)} → {out}")
    meta_p = out.with_suffix(".meta.json")
    meta_p.write_text(json.dumps({"batches": metas, "n_candidates": len(cands),
                                  "n_pass": npass, "image": args.image,
                                  "docker": docker, "timeout": args.timeout},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sandbox-batch] batch metadata → {meta_p}")
    return results, metas


# ═══════════════════════════════════════════════════════════════════
# stage: dryrun -- negative-case construction + full-chain self-check
# ═══════════════════════════════════════════════════════════════════

def neg_syntax(code: str, entry: str = "", **_) -> str:
    """Negative 1: syntax error (blows up at compile time)."""
    return code + "\n\nthis is not valid python !!!(\n"


def neg_wrong_name(code: str, entry: str, **_):
    """Negative 2: rename the entry def/class → NameError when the tests call it.

    entry may be a **class name** (the dataset has BankAccount / Car / Rectangle and friends),
    so both def and class must be covered; patching only `def` leaves class-based problems
    effectively unmodified and yields a false PASS.
    """
    new, n = re.subn(
        r"^([ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+)%s\b" % re.escape(entry),
        r"\g<1>%s_RENAMED" % entry, code, count=1, flags=re.M)
    if n == 0:
        # fallback: if no definition is found, raise NameError explicitly
        return code + "\n\n%s\n" % ("%s()" % entry)
    return new


def neg_wrong_return(code: str, entry: str, **_):
    """Negative 3: override the entry function at the end so it returns the wrong type."""
    return code + ("\n\ndef %s(*args, **kwargs):\n"
                   "    return '__ESD_WRONG_RETURN__'\n" % entry)


def neg_infinite_loop(code: str, entry: str, **_):
    """Negative 4 (extra): override the entry function with an infinite loop → only the per-sample
    timeout can catch it."""
    return code + ("\n\ndef %s(*args, **kwargs):\n"
                   "    while True:\n        pass\n" % entry)


NEG_CLASSES = [
    ("syntax_error", "syntax error", neg_syntax),
    ("wrong_name", "wrong function name", neg_wrong_name),
    ("wrong_return", "wrong return type", neg_wrong_return),
    ("infinite_loop", "infinite loop (extra: verifies per-sample timeout isolation)", neg_infinite_loop),
]


def stage_dryrun(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    docker = find_docker()
    ensure_image(docker, args.image, auto_pull=not args.no_pull)
    print(f"[dryrun] docker={docker} image={args.image} n={args.n}")

    # 1) extract 20 problems
    prob_path = outdir / "problems.jsonl"
    problems, st = extract_problems(args, parquet=args.parquet, out_path=prob_path,
                                    max_problems=args.n, return_stats=True)
    if len(problems) < args.n:
        eprint(f"[dryrun] only {len(problems)} problems extracted (<{args.n})")

    # 2) candidates: reference solutions + four negative classes
    cands = []
    for p in problems:
        cands.append({"id": p["id"], "class": "ref", "completion": p["ref_solution"]})
    for name, label, fn in NEG_CLASSES:
        for p in problems:
            cands.append({"id": p["id"], "class": name,
                          "completion": fn(p["ref_solution"], p["entry"])})
    cand_path = outdir / "candidates.jsonl"
    with open(cand_path, "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[dryrun] {len(cands)} candidates → {cand_path}")

    # 3) run everything through the sandbox
    items = []
    for c in cands:
        p = next(x for x in problems if x["id"] == c["id"])
        items.append({"id": c["id"] + "|" + c["class"], "code": strip_fence(c["completion"]),
                      "tests": p["tests"], "timeout": args.timeout})
    workroot = outdir / "_sandbox"
    results, metas = sandbox_batch_chunked(docker, args.image, items, workroot,
                                           args.batch_size, args.jobs, "dry")

    res_path = outdir / "results.jsonl"
    with open(res_path, "w", encoding="utf-8") as f:
        for c in cands:
            key = c["id"] + "|" + c["class"]
            r = results.get(key) or {"id": key, "passed": False, "reason": "NO_RESULT",
                                     "seconds": 0.0}
            r["class"] = c["class"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4) verdict
    by_class = {}
    for c in cands:
        key = c["id"] + "|" + c["class"]
        r = results.get(key) or {"passed": False, "reason": "NO_RESULT", "seconds": 0.0}
        by_class.setdefault(c["class"], []).append(r)
    refs = by_class.get("ref", [])
    ref_pass = sum(1 for r in refs if r["passed"])
    verdict = {
        "ref": {"n": len(refs), "passed": ref_pass,
                "ok": ref_pass == len(refs),
                "expected": "all PASS"},
    }
    for name, label, _ in NEG_CLASSES:
        rs = by_class.get(name, [])
        npass = sum(1 for r in rs if r["passed"])
        reasons = {}
        for r in rs:
            reasons[r["reason"].split(":")[0][:28]] = reasons.get(r["reason"].split(":")[0][:28], 0) + 1
        verdict[name] = {"n": len(rs), "passed": npass, "failed": len(rs) - npass,
                         "ok": npass == 0, "expected": "all FAIL", "reasons": reasons}

    # 5) pull two complete pieces of evidence
    ev = []
    for want in ("ref", "infinite_loop"):
        for p in problems:
            key = p["id"] + "|" + want
            if key in results:
                ev.append({"class": want, "problem_id": p["id"], "entry": p["entry"],
                           "prompt": p["prompt"], "n_tests": len(p["tests"]),
                           "tests": p["tests"],
                           "code": strip_fence(
                               next(c["completion"] for c in cands
                                    if c["id"] == p["id"] and c["class"] == want)),
                           "result": results[key]})
                break

    payload = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t_start, 1),
        "docker": docker, "image": args.image,
        "timeout": args.timeout, "jobs": args.jobs, "batch_size": args.batch_size,
        "extract_stats": st,
        "n_problems": len(problems), "n_candidates": len(cands),
        "verdict": verdict,
        "all_ok": bool(verdict["ref"]["ok"]) and all(verdict[k]["ok"] for k, _, _ in NEG_CLASSES),
        "batches": metas,
        "evidence": ev,
        "problem_ids": [p["id"] for p in problems],
        "entry_points": {p["id"]: p["entry"] for p in problems},
    }
    js = outdir / "dryrun_result.json"
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ DRYRUN VERDICT ================")
    print(f"reference solutions:  {ref_pass}/{len(refs)} PASS  {'OK' if verdict['ref']['ok'] else '!! below 20/20'}")
    for name, label, _ in NEG_CLASSES:
        v = verdict[name]
        print(f"{label:<28} {v['failed']}/{v['n']} FAIL  {'OK' if v['ok'] else '!! unexpected PASS'}"
              f"  reason distribution={v['reasons']}")
    print(f"results JSON → {js}")
    write_report(outdir / "DRYRUN_REPORT.md", payload, docker)
    print(f"report → {outdir / 'DRYRUN_REPORT.md'}")
    return payload


def write_report(path: Path, p: dict, docker: str) -> None:
    v = p["verdict"]
    st = p["extract_stats"]
    L = []
    A = L.append
    A("# ESD pipeline dryrun report (execution-feedback self-distillation · rejection-sampling scaffolding)")
    A("")
    A(f"- generated: {p['when']}    elapsed: {p['elapsed_s']}s")
    A(f"- script: `fine-tuning/esd_pipeline.py` (**new**; no existing file was modified)")
    A(f"- artifact directory: `fine-tuning/esd/`")
    A(f"- sandbox: `{p['image']}` (Docker, `--network none`), "
      f"docker=`{docker}`, per-sample timeout={p['timeout']}s, in-container concurrency={p['jobs']}, "
      f"{p['batch_size']} per container")
    A(f"- **overall verdict: {'all as expected [OK]' if p['all_ok'] else 'deviations found [FAIL]'}**")
    A("")
    A("---")
    A("")
    A("## 1. Real format of the `unit_tests` field (measured, not guessed)")
    A("")
    A("Source: `data_instr_raw/opencodeinstruct_shard0.parquet` (100,000 rows, 1 row group).")
    A("")
    A("Real schema:")
    A("")
    A("```")
    A("id: string")
    A("input: string          # natural-language requirement (with **Input/Output/Constraints/Sample** sections)")
    A("output: string         # reference solution, **wrapped in a ```python fence**")
    A("domain: string         # generic / algorithmic")
    A("generation_algorithm: string   # self-instruct / evol-instruct")
    A("llm_judgement: string          # JSON (three scores + justification)")
    A("unit_tests: string     # ← see below")
    A("tests_execution_status: string # ← see below")
    A("average_test_score: double")
    A("```")
    A("")
    A("### `unit_tests`")
    A("")
    A("- It is a **string containing a JSON array**; deserializing yields `list[str]`.")
    A("- The array holds **exactly 10 entries** (4000-row sample: 100% are `list`, 100% have length 10).")
    A("- **Each entry is a standalone bare `assert` statement string**, like:")
    A("")
    A("```python")
    A('"\\nassert max_non_overlapping_tasks([(1, 3), (2, 4), (3, 5)]) == 2\\n"')
    A("```")
    A("")
    A("- **No pytest functions, no `import pytest`, no `def test_*`, no import header.**")
    A("- A few samples **embed** multi-statement blocks in one entry (`import io` / `import sys` /")
    A("  constructing objects / calling methods), but it is still a flat statement string that can be")
    A("  concatenated and executed directly (in the 4000-row sample, **27** samples contain at least")
    A("  one such entry; the `def` appearing inside them sits in a **string literal**, not a real")
    A("  function definition).")
    A("- **Execution semantics (as used by this pipeline, isomorphic to `eval_sft.py::check_algo`)**:")
    A("")
    A("```python")
    A('script = ref_or_candidate_code + "\\n\\n" + "\\n".join(tests) + "\\n"')
    A("passed = (exit code == 0)     # no output-marker dependency; relies on exception propagation")
    A("```")
    A("")
    A("- `output` (the reference solution) is fence-wrapped in 100% of cases (4000/4000); extraction strips it with a regex.")
    A("")
    A("### `tests_execution_status` (the critical gotcha)")
    A("")
    A("- It is **not** a scalar `\"passed\"`; it is a per-test JSON array string, e.g.")
    A("  `[\"pass\", \"pass\", \"fail\", \"pass\", ...]` (matching the 10 entries of `unit_tests`).")
    A("- 4000-row sample: only 1443 rows are **10/10 all pass**; the other 2557 contain a `fail`.")
    A("- This pipeline therefore keeps a row only if **every entry in the array is pass** (`all_tests_passed()`).")
    A("  That filter is **required**: in an early unfiltered trial over the first 3 rows, the reference")
    A("  solution for `id=4d61a8…` genuinely FAILED (`AssertionError`), consistent with the `fail` in")
    A("  its own `tests_execution_status`. With the filter on (40-problem sample) the reference")
    A("  solutions pass 39/40; the only failure is a missing `flask` module:")
    A("  `ModuleNotFoundError: No module named 'flask'`.")
    A("")
    A("### Extra filters derived from the format (`extract`)")
    A("")
    A("| Filter | Reason |")
    A("|---|---|")
    A("| `tests_execution_status` not all pass | The dataset ships failing assertions; the reference solution is guaranteed to fail |")
    A("| blacklist hit | Reuses `build_instr_data.build_blacklist()` (**167** names = 11 algorithm names + HumanEval entry_points) to prevent eval leakage |")
    A("| `input` too long | Hard context limit 1024; character cap 2000 + tokenizer token cap (default 640) |")
    A("| tests never call entry | The name of the reference solution's first def/class must appear in some test, otherwise \"PASS\" carries no information |")
    A("| tests **or reference solution** import a non-stdlib module | The sandbox image is `python:3.12-slim` (stdlib only). Both sides "
        "must be scanned: for Flask problems the unit tests import nothing, it is the **reference solution** that does `import flask` (on "
        "by default, `--no-dep-filter` turns it off) |")
    A("| nondeterministic problem | Unseeded `random` / `time.time()` / `datetime.now()` / `uuid4` → repeated runs of the same code flip "
        "between PASS/FAIL, directly corrupting the rejection-sampling filter signal (dropped by default, `--keep-nondeterministic` keeps "
        "them) |")
    A("")
    A("The last two filters were **forced by measurements**, not invented on a hunch:")
    A("")
    A("The first time all 200 reference solutions went through the sandbox, only **197/200 PASSed**. The 3 failures had two root causes:")
    A("")
    A("- 2 x `ModuleNotFoundError: No module named 'flask'` -- the dependency lives in the **reference solution**,")
    A("  and the original filter only scanned `tests`, so it missed it. → Scan both sides.")
    A("- 1 x `AssertionError` (`dice_game`, a dice-rolling simulation) -- running its reference solution")
    A("  **as the same code 12 times in a row** gave PASS 5 times / FAIL 7 times")
    A("  (`[T,T,F,F,F,T,F,T,F,F,F,T]`): the problem itself is nondeterministic.")
    A("  → Add the nondeterminism filter.")
    A("")
    A("With both filters added, the same 200 problems give **200/200 PASS** (see section 3.4).")
    A("")
    A("Measured filter funnel on a 3000-row sample (a separate sample from this dryrun):")
    A("")
    A("```")
    A("scanned 3000 rows → kept 941 items (≈31%)")
    A("drop_not_all_pass=1561  drop_leak=168  drop_long_chars=4  drop_entry_not_called=326")
    A("```")
    A("")
    A("Funnel for this dryrun's own 20 problems (same format, printed by the script at runtime):")
    A("")
    A("```")
    A(f"scanned {st['rows']} rows → kept {st['kept']} problems")
    A("  ".join(f"{k}={val}" for k, val in st.items() if k != "rows" and val))
    A("```")
    A("")
    A("---")
    A("")
    A("## 2. Docker commands (verbatim from the real run)")
    A("")
    A("### 2.1 Prepare the image")
    A("")
    A("```bash")
    A("# add docker to PATH first in git-bash")
    A('export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"')
    A("docker pull python:3.12-slim")
    A("```")
    A("")
    A("Observed output tail:")
    A("")
    A("```")
    A("Digest: sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea")
    A("Status: Downloaded newer image for python:3.12-slim")
    A("docker.io/library/python:3.12-slim")
    A("```")
    A("")
    A("### 2.2 How each sample runs inside the container (one container per batch)")
    A("")
    A("`esd_pipeline.py` builds the command with `subprocess` in **list form** (no shell involved, so spaces in paths are safe).")
    A("Below is the full command the script actually issues (`<IN>` / `<OUT>` = the batch's input/result directories):")
    A("")
    A("```bash")
    A(f'"{docker}" run --rm \\')
    A("  --network none \\")
    A("  --read-only \\")
    A('  --tmpfs /tmp:rw,size=128m,exec \\')
    A("  --memory 1g --cpus 2 --pids-limit 512 \\")
    A("  --security-opt no-new-privileges \\")
    A("  -e ESD_JOBS=4 -e ESD_IN=/in -e ESD_OUT=/out \\")
    A('  -v "<IN>:/in:ro" \\')
    A('  -v "<OUT>:/out" \\')
    A("  -w /tmp \\")
    A("  python:3.12-slim python /in/_driver.py")
    A("```")
    A("")
    A("The **first real command** of this dryrun (complete, unabridged, identical to what was issued):")
    A("")
    A("```bash")
    A(" ".join(p.get("batches", [{}])[0].get("cmd", ["<no batch record>"])))
    A("```")
    A("")
    A("Directory layout (one per batch):")
    A("")
    A("```")
    A("esd/_sandbox/dry_000/")
    A("  in/                        → mounted read-only (:ro) into the container at /in")
    A("    _driver.py               (built into the script, written to disk at runtime)")
    A("    _items.jsonl             (the {id, code, tests, timeout} to run in this batch)")
    A("  out/                       → mounted writable into the container at /out")
    A("    _results.jsonl           (the driver appends + flushes item by item)")
    A("```")
    A("")
    A("**Why inputs are read-only and cwd is /tmp**: the dataset contains tests that write files, e.g. the")
    A("unit test for the `FileManager` problem calls `write_file('test3.txt', ...)`. The first version used")
    A("a writable `/work` as cwd, and a `test3.txt` really did appear in the host directory (see section")
    A("4.5). Now inputs are read-only and cwd is tmpfs, so relative-path writes only land on the RAM disk")
    A("and die with the container.")
    A("")
    A("### 2.3 Isolation mechanism of the in-container driver")
    A("")
    A("`_driver.py` (an embedded constant, written to the working directory at runtime and then mounted into the container):")
    A("")
    A("```python")
    A("# per sample: write /tmp/case_*.py (tmpfs)")
    A("body = code + '\\n\\n' + '\\n'.join(tests) + '\\n'")
    A("compile(body, src, 'exec')            # syntax error → return SyntaxError at once, no process started")
    A("p = subprocess.Popen([sys.executable, src], start_new_session=True, cwd='/tmp')")
    A("p.communicate(timeout=timeout)       # ← per-sample timeout")
    A("# on timeout: os.killpg(os.getpgid(p.pid), SIGKILL)  ← kills the whole process group, grandchildren included")
    A("open('/out/_results.jsonl','a').write(json.dumps({id, passed, reason, seconds})+'\\n')")
    A("```")
    A("")
    A("- Per-item results are **flushed straight to disk**, so even if the whole batch is cut off by the host-side outer timeout, "
        "completed results are not lost.")
    A("- Inside the container, `ThreadPoolExecutor(max_workers=ESD_JOBS)` provides concurrency, and **each item's seconds are still timed individually**.")
    A("")
    A("### 2.4 Containers this dryrun actually started (measured per batch)")
    A("")
    A("| Batch | rc | elapsed(s) | samples | verdict |")
    A("|---|---|---|---|---|")
    for i, b in enumerate(p.get("batches", [])):
        A(f"| `dry[{i}]` | {b['rc']} | {b['elapsed']} | {b['n_results']}/{b['n_items']} | "
          f"{'[OK]' if b['n_results'] == b['n_items'] and b['rc'] == 0 else '[FAIL]'} |")
    A("")
    A(f"→ the 100 candidates needed only **{len(p.get('batches', []))} containers** ({p['batch_size']} each),"
      f" not one container per sample.")
    A("")
    A("| Batch | Full command |")
    A("|---|---|")
    for i, b in enumerate(p.get("batches", [])):
        A(f"| `dry[{i}]` | `{' '.join(b['cmd'])}` |")
    A("")
    A("---")

    A("## 3. Measured dryrun results")
    A("")
    A(f"**{p['n_problems']}** problems, **{p['n_candidates']}** candidates"
      f" (= {p['n_problems']} reference solutions + {p['n_problems']}×{len(NEG_CLASSES)} negatives).")
    A("")
    A("| Class | Expected | Measured | Verdict |")
    A("|---|---|---|---|")
    A(f"| reference solution `ref_solution` | 20/20 PASS | **{v['ref']['passed']}/{v['ref']['n']} PASS**"
      f" | {'[OK]' if v['ref']['ok'] else '[FAIL]'} |")
    for name, label, _ in NEG_CLASSES:
        vv = v[name]
        A(f"| negative · {label} (`{name}`) | all FAIL | **{vv['failed']}/{vv['n']} FAIL**"
          f" | {'[OK]' if vv['ok'] else '[FAIL]'} |")
    A("")
    A("Negative-failure reason distribution:")
    A("")
    A("```")
    for name, label, _ in NEG_CLASSES:
        A(f"{name:<16} {v[name]['reasons']}")
    A("```")
    A("")
    A("### 3.1 Evidence 1: the reference solution (should PASS)")
    A("")
    ev_ref = next((e for e in p["evidence"] if e["class"] == "ref"), None)
    if ev_ref:
        A(f"- problem_id: `{ev_ref['problem_id']}`    entry: `{ev_ref['entry']}`"
          f"    n_tests: {ev_ref['n_tests']}")
        A(f"- measured: `{json.dumps(ev_ref['result'], ensure_ascii=False)}`")
        A("")
        A("Submitted candidate code:")
        A("")
        A("```python")
        A(ev_ref["code"].rstrip())
        A("```")
        A("")
        A("Unit tests that ran (first 3):")
        A("")
        A("```python")
        for t in ev_ref["tests"][:3]:
            A(t)
        A("...")
        A("```")
    A("")
    A("### 3.2 Evidence 2: the infinite-loop negative (should FAIL with reason=TIMEOUT)")
    A("")
    ev_loop = next((e for e in p["evidence"] if e["class"] == "infinite_loop"), None)
    if ev_loop:
        A(f"- problem_id: `{ev_loop['problem_id']}`    entry: `{ev_loop['entry']}`")
        A(f"- measured: `{json.dumps(ev_loop['result'], ensure_ascii=False)}`")
        A("")
        A("Tail of the negative code (appended after the reference solution):")
        A("")
        A("```python")
        A(ev_loop["code"].rstrip()[-400:])
        A("```")
        A("")
        A("→ this item was **cut off by the per-sample timeout** while the other 79 items in the same"
          " batch returned results normally, proving that one hung sample does not bring down the batch.")
    A("")
    A("### 3.3 How the four negative classes are constructed")
    A("")
    A("| Class | Construction | Expected failure mechanism |")
    A("|---|---|---|")
    A("| `syntax_error` | append `this is not valid python !!!(` at the end | `SyntaxError` at the `compile()` stage |")
    A("| `wrong_name` | rename the entry `def/class <entry>` to `<entry>_RENAMED` | `NameError` when the tests call it |")
    A("| `wrong_return` | override the entry at the end with `def <entry>(*a, **k): return '__ESD_WRONG_RETURN__'` | for function-style "
        "problems the asserted value differs → `AssertionError`; for class-style problems the class is shadowed by the function → "
        "`AttributeError` |")
    A("| `infinite_loop` | override the entry at the end with `def <entry>(*a, **k): while True: pass` | per-sample timeout → `TIMEOUT` |")
    A("")
    A("Note that `wrong_name` must match both `def` and `class`: 7 of these 20 problems have a")
    A("**class name** as entry (`BankAccount`×2 / `Car` / `Rectangle` / `FileManager` / `Point2D` /")
    A("`Book`). Patching only `def` leaves those 7 \"completely unmodified\" and therefore falsely PASSing")
    A("-- exactly the real defect the first verification run turned up (see section 4).")
    A("")
    A("### 3.4 Scale check: 200 problems (not just 20 can run)")
    A("")
    A("The default of `--max-problems` is 200, so the whole chain was additionally verified at **default scale**:")
    A("")
    A("```bash")
    A("py -3.12 esd_pipeline.py --stage extract --max-problems 200 --out esd/problems_200.jsonl")
    A("py -3.12 esd_pipeline.py --stage sandbox-batch --problems esd/problems_200.jsonl \\")
    A("    --candidates esd/standalone_check/cands200.jsonl \\")
    A("    --out esd/standalone_check/results200.jsonl --timeout 10 --batch-size 64")
    A("```")
    A("")
    A("- extract: **1.1s** to collect the full 200 (525 rows scanned).")
    A("  funnel: `drop_not_all_pass=276  drop_leak=29  drop_long_chars=2  drop_entry_not_called=8  drop_nonstdlib=2  drop_nondeterministic=7`")
    A("- sandbox-batch: **4 containers** (64+64+64+8) cover the 200 candidates; a single container batch takes 0.5–1.0s,")
    A("  and total in-container execution time for the 200 samples is only **5.1s** (longest single item 0.28s).")
    A("- Result: **200/200 reference solutions PASS** (after adding the two filters from section 1; before that it was 197/200).")
    A("- Of these 200 entries, 154 are functions and 46 are **classes**, so both `def` and `class` forms ran through the whole chain.")
    A("")
    A("---")
    A("")
    A("## 4. Defects actually found and fixed during this verification round")
    A("")
    A("Scaffolding is not \"write it and ship it\" -- everything below was **exposed by a real run and confirmed fixed by re-running**:")
    A("")
    A("### 4.1 `-v` given a relative path → Docker treats it as a named volume, whole batch returns zero results")
    A("")
    A("The first `--stage sandbox-batch` run returned `NO_RESULT` for all 20, with container `rc=125`:")
    A("")
    A("```")
    A('docker: Error response from daemon: create esd\\_t\\_sandbox\\sb_000: '
      '"esd\\\\_t\\\\_sandbox\\\\sb_000" includes invalid characters for a local volume name, '
      'only "[a-zA-Z0-9][a-zA-Z0-9_.-]" are allowed. If you intended to pass a host directory, use absolute path')
    A("```")
    A("")
    A("Fix: `docker_run_cmd()` always applies `str(Path(host_dir).resolve()).replace(\"\\\\\", \"/\")`,")
    A("forcing \"absolute path + forward slashes\". A hard assertion was added at the same time: if a")
    A("non-empty batch returns `rc != 0` with zero results, the script raises `SystemExit` and prints")
    A("the command plus stderr, so it can **no longer silently degrade into a pile of `NO_RESULT`**.")
    A("After the fix the same command gives 20/20 PASS (one container, 0.6s).")
    A("")
    A("### 4.2 `neg_wrong_name` only patched `def`, so class-based entries falsely PASSed (7/20)")
    A("")
    A("First dryrun verdict: \"wrong function name 13/20 FAIL !! unexpected PASS\", and the 7 unexpected")
    A("PASSes were exactly the samples whose entry is a class (the rename regex never matched `class`,")
    A("leaving the candidate code **byte-for-byte identical** to the reference solution).")
    A("Fix: change the regex to `(?:def|class)`, and add a fallback for when `re.subn` reports 0")
    A("replacements (an explicit call that raises NameError). After re-running, 20/20 FAIL with")
    A("`NameError` as the reason for all of them.")
    A("")
    A("### 4.3 `neg_syntax()` signature did not match the uniform calling convention → TypeError crashed the dryrun")
    A("")
    A("Negative constructors are all called as `fn(code, entry)`, but `neg_syntax` accepted a single")
    A("positional argument, so the first dryrun died with")
    A("`TypeError: neg_syntax() takes 1 positional argument but 2 were given`.")
    A("Fix: unify the signature as `fn(code, entry=\"\", **_)`.")
    A("")
    A("### 4.4 (design note, not a defect) The criterion comes from the dataset's own consistency")
    A("")
    A("The dataset's first row `id=4d61a897…` has a `fail` in its `tests_execution_status`, and without")
    A("the filter its reference solution genuinely FAILs (`AssertionError`) -- so the execution semantics")
    A("**agree with** the dataset's own criterion, which is direct evidence that filtering on all-pass is")
    A("mandatory.")
    A("")
    A("### 4.5 File-writing tests wrote files into the host mount directory")
    A("")
    A("The dataset contains \"file operation\" problems whose unit tests write files themselves, e.g. the")
    A("`FileManager` problem:")
    A("")
    A("```python")
    A("assert FileManager().write_file('test.txt', 'Hello, world!') is None")
    A("```")
    A("")
    A("The first version mounted the batch directory **writable** at `/work` and set cwd to `/work`, so")
    A("relative-path writes from tests landed straight on the host disk -- after the dryrun there really")
    A("was an extra `test3.txt` in `esd/_sandbox/dry_000/` (contents `Python is fun!`). Trivial cache/junk")
    A("output, but it shows that candidate code can write to host mounts, which is unacceptable for a")
    A("sandbox.")
    A("")
    A("Fix (three changes made together):")
    A("")
    A("1. Mount the input directory **read-only (`:ro`)** at `/in` so candidate code can no longer rewrite")
    A("   its own problem or inputs.")
    A("2. Mount a separate small `/out` directory for results; the driver only appends to `/out/_results.jsonl`.")
    A("3. Set both the container's `-w` and the child's `cwd` to `/tmp` (tmpfs), so relative-path file writes")
    A("   inside tests only reach the RAM disk.")
    A("")
    A("After re-running, `esd/_sandbox/dry_000/` contains **only the `in/` and `out/` directories and no")
    A("files produced by tests**.")
    A("")
    A("### 4.6 The dependency scan only covered unit tests and missed the `import flask` hidden in the reference solution (197/200)")
    A("")
    A("Putting all 200 reference solutions through the sandbox left 3 failures, 2 of them")
    A("`ModuleNotFoundError: No module named 'flask'`. The investigation showed that those two problems'")
    A("`unit_tests` contain **not a single import**; the `import flask` sits in the **reference solution**")
    A("(the prompt really does ask for a web server built on Flask). A dependency filter that scans only")
    A("`tests` inherently misses this class -- and a model seeing the same prompt would also import flask,")
    A("so such problems have **no gradeable correct answer at all** in a stdlib-only image.")
    A("")
    A("Fix: `deps = nonstdlib_imports(tests) ∪ nonstdlib_imports([ref_code])`, scanning both sides.")
    A("")
    A("### 4.7 Nondeterministic problems make \"PASS\" itself untrustworthy (5/12 jitter)")
    A("")
    A("The one remaining failure was an `AssertionError`. Running its reference solution **as the same code 12 times in a row**:")
    A("")
    A("```")
    A("12 runs concatenated into one process: [True, True, False, False, False, True, False, True, False, False, False, True]")
    A("PASS count: 5 /12  →  unstable")
    A("```")
    A("")
    A("The problem simulates rolling two dice until the sum is 7 or 11; the code uses `random` but **never seeds it**.")
    A("For rejection sampling this is **fatal noise**: the same correct code is randomly judged FAIL and discarded.")
    A("(Viewed the other way it is just as dangerous: a wrong solution can happen to pass and be kept as a good sample.)")
    A("")
    A("Fix: add `nondeterminism_hint()` -- discard when it matches `random.` / `np.random` / `uuid4` / `secrets.` /")
    A("`os.urandom` / `time.time()` / `datetime.now()` **with no `seed(` anywhere in the text**; on by default,")
    A("`--keep-nondeterministic` keeps them.")
    A("")
    A("> This also refutes an initial hypothesis: I assumed the failure was caused by \"10 asserts")
    A("> concatenated into one process\" differing in semantics from the dataset's \"each executed")
    A("> independently\". The measurements **do not support** it -- both 12-run concatenated batches and per-item runs only jitter: both "
        "styles sometimes pass and are unstable. The real variable is the random numbers, not the concatenation style.")
    A("")
    A("---")
    A("")
    A("## 5. What was not verified (listed honestly)")
    A("")
    A("1. **No real model generation was run**: dryrun candidates are \"reference solutions + synthetic negatives\", not samples from the 435M model.")
    A("   This round forbade using the GPU or loading models, so \"the model's actual pass@1 / rejection-sampling retention rate\" is **untested**.")
    A("2. **No extract over the full 100k rows**: `--max-problems` exits early, so the dryrun scanned only 58 rows.")
    A("   The separate 300k → 3000-row sample gave `kept≈31%`, but the **full-scale funnel ratio is unmeasured**.")
    A("3. **The 640-token cap has no distribution validation**: only the \"over the cap → drop\" logic was confirmed; the 20 prompts in this round")
    A("   are 99–427 tokens, so **this filter dropped nothing at all** and its cost is unmeasured.")
    A("4. **Sandbox security was only addressed at the configuration level**: `--network none` / `--read-only` / `--pids-limit 512` /")
    A("   `no-new-privileges` / `--memory 1g` / `--cpus 2` / read-only inputs / cwd=tmpfs are all in place and measured to")
    A("   let the container run normally with **no test files leaking to the host**, but **no adversarial escape tests were done**")
    A("   (fork bombs, filling tmpfs, probing mount points and privilege-escalation attempts were all untried).")
    A("   The scope is running unit tests from a public dataset, so the risk is contained, but that does not justify claiming the sandbox is absolutely safe.")
    A("5. **The non-stdlib dependency filter relies on the host's `sys.stdlib_module_names`**: the host runs CPython 3.12.10 and the image is also 3.12,")
    A("   so the two should agree, but no per-module comparison was done; it was only verified to correctly block an obvious third-party module like `flask`")
    A("   (with the filter off, `ModuleNotFoundError: No module named 'flask'` was observed).")
    A("6. **The stronger form of \"one hung sample does not bring down the batch\" is unverified**: this round only tested 20 infinite "
        "loops with a 10s timeout")
    A("   (all `TIMEOUT`, with the other samples in the batch returning normally); **untested** are \"one sample exhausts the container's memory/process count\"")
    A("   or \"one sample fills up the tmpfs\" style resource exhaustion.")
    A("7. **The host-side outer timeout branch is unmeasured**: the")
    A("   `subprocess.TimeoutExpired` fallback in `sandbox_run()` (partial results written to disk) never fired in this round.")
    A("8. **Only the `python:3.12-slim` image was verified**: changing the image or base environment is untested.")
    A("9. **The three-part CLI for `--stage sandbox-batch` was exercised separately** (20/20 PASS under `esd/standalone_check/`,")
    A("   one container in 0.5–0.6s, so it is **not** an unverified item; recorded here only to avoid misjudging it).")
    A("10. **The `/out` mount is still writable by candidate code**: malicious code could in principle append")
    A("    forged JSON lines to `_results.jsonl` to fake a PASS. That attack was not actually tested; against unit tests from a public dataset the risk is very low,")
    A("    but a real self-distillation run that must defend against model code actively cheating should move to a driver-side encrypted/signed result channel.")
    A("")
    A("---")
    A("")
    A("## 6. Reproduction commands")
    A("")
    A("```bash")
    A('export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"')
    A("cd <repo>/post-training/scripts")
    A("")
    A("# (1) look at the real parquet structure first")
    A("py -3.12 esd_pipeline.py --stage info")
    A("")
    A("# (2) extract the problem set")
    A("py -3.12 esd_pipeline.py --stage extract --max-problems 200 --out esd/problems.jsonl")
    A("")
    A("# (3) run candidate code through the sandbox (each line of C.jsonl is {id, completion})")
    A("py -3.12 esd_pipeline.py --stage sandbox-batch \\")
    A("    --problems esd/problems.jsonl --candidates esd/C.jsonl --out esd/R.jsonl")
    A("")
    A("# (4) full-chain self-check (reference solutions 20/20 + all four negative classes FAIL)")
    A("py -3.12 esd_pipeline.py --stage dryrun --n 20")
    A("```")
    A("")
    A("Batch metadata (the command actually issued + rc + elapsed + sample coverage) is in the `batches` section of `esd/dryrun_result.json`.")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="435M v2 · execution-feedback self-distillation (rejection sampling) pipeline scaffolding")
    p.add_argument("--stage", required=True,
                   choices=["info", "extract", "sandbox-batch", "dryrun"])
    p.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--out", default=None, help="output file for extract / sandbox-batch")
    p.add_argument("--workdir", default=None, help="temp directory for sandbox mounts (default: next to out)")
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--samples", type=int, default=2, help="info: how many samples to print")
    p.add_argument("--chars", type=int, default=900, help="info: how many characters to print per field")

    # extract
    p.add_argument("--max-problems", type=int, default=200)
    p.add_argument("--max-input-chars", type=int, default=2000)
    p.add_argument("--max-prompt-tokens", type=int, default=640)
    p.add_argument("--no-token-filter", action="store_true")
    p.add_argument("--no-dep-filter", action="store_true",
                   help="do not drop problems whose tests/reference solution import non-stdlib modules")
    p.add_argument("--keep-nondeterministic", action="store_true",
                   help="keep nondeterministic problems (unseeded random/time/uuid) -- dropped by default because they make PASS/FAIL jitter")

    # sandbox
    p.add_argument("--problems", default=None)
    p.add_argument("--candidates", default=None)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--timeout", type=float, default=10.0, help="per-sample execution timeout (seconds)")
    p.add_argument("--jobs", type=int, default=4, help="in-container concurrency")
    p.add_argument("--batch-size", type=int, default=64, help="how many samples each container runs")
    p.add_argument("--no-pull", action="store_true", help="do not docker pull automatically when the image is missing")

    # dryrun
    p.add_argument("--n", type=int, default=20, help="how many problems dryrun uses")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stage == "info":
        return stage_info(args)
    if args.stage == "extract":
        out = args.out or str(Path(args.outdir) / "problems.jsonl")
        return extract_problems(args, out_path=out)
    if args.stage == "sandbox-batch":
        if not args.problems or not args.candidates:
            raise SystemExit("sandbox-batch requires --problems and --candidates")
        args.out = args.out or str(Path(args.outdir) / "results.jsonl")
        return stage_sandbox_batch(args)
    if args.stage == "dryrun":
        args.out = args.out or str(Path(args.outdir) / "results.jsonl")
        return stage_dryrun(args)
    raise SystemExit(f"unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
