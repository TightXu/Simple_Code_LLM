#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
esd_sample.py -- ESD (execution-feedback self-distillation / rejection sampling) pipeline: candidate generation
=====================================================================
Uses a **given checkpoint** to generate k candidate solutions for each problem produced by
esd_pipeline.py `extract`, then hands them to `esd_pipeline.py --stage sandbox-batch` for execution filtering.

This file does one thing only: sample. It touches no existing file, trains nothing, writes no aside artifacts.

Convention alignment (the reason this script exists)
----------------------------------
* model definition / loading / generation loop: **eval_sft.py is reused as-is**
  (load_tokenizer / load_model / ModelConfig / generate_one).
  The architecture is not reimplemented -- a private model definition here would drift from the training/eval code over time.
  eval_sft.py's main flow lives under `if __name__ == "__main__"`, so importing it is safe (no evaluation is triggered).
* prompt template: from build_instr_data.py's PROMPT_TEMPLATES (default 'marker'
  = `### Task\\n{instruction}\\n\\n### Code\\n`), **the same source** as the training data.
  If the problem set's meta carries `prompt_template` / `prompt_templated`, a **byte-for-byte assertion** runs and
  a mismatch aborts immediately (instead of quietly generating with a different template and then blaming the filter).
* generation params: --temperature / --top-k / --repetition-penalty / --min-p / --max-new
  are passed straight through to eval_sft.generate_one.

About --top-p
   eval_sft.generate_one()'s signature has **no** top_p (its sampling = top_k + min_p +
   repetition_penalty). This script keeps the contract and **does not touch eval_sft.py**, so --top-p < 1.0
   is **ignored with an explicit warning** (meta records top_p_applied=false).
   To really enable nucleus sampling, add the parameter to eval_sft.generate_one first -- this script then
   picks it up automatically; rewriting the sampling loop here would be the real mistake (generation would no longer match the evaluation parameter for parameter).

Usage
----
    # 0) dry-run: load no model, use each problem's reference solution as the candidate (self-check / downstream integration)
    py -3.12 esd_sample.py --dry-run --problems esd/problems.jsonl \
        --limit 5 --k 3 --out esd/cand_dry.jsonl

    # 1) real sampling (a GPU requires an explicit --device cuda; cpu is the default so a running training job is left alone)
    py -3.12 esd_sample.py --ckpt ckpt_a5/final --problems esd/problems.jsonl \
        --out esd/cand_a5.jsonl --k 4 --temperature 0.7 --device cuda

    # 2) throw them into the sandbox
    py -3.12 esd_pipeline.py --stage sandbox-batch --problems esd/problems.jsonl \
        --candidates esd/cand_a5.jsonl --out esd/results_a5.jsonl

Output (one row per line, always these 4 fields)
    {"id": <problem id>, "k_index": <0..k-1>, "completion": <candidate code>, "seed_used": <int>}

k > 1 and how it plugs into sandbox-batch (**read this**)
----------------------------------------
esd_pipeline.py's sandbox-batch uses a candidate's `id` both as the problem lookup key and as the result key,
so k candidates for the same problem overwrite each other (only the last one survives in the result dict). Two ways to use it:
  --id-mode plain  (default): id = the problem id → feed sandbox-batch directly (cleanest with k=1)
  --id-mode unique         : id = "<problem id>#k<k>", and --problems-out can export a
     problem set **expanded per candidate** (same id rule), so each of the k candidates is looked up and counted
     independently and a k>1 run does not bleed together. Feed sandbox-batch that expanded file via --problems.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:                                                    # Windows console: non-ASCII output
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── reuse existing modules (read-only, they are not modified) ──────────────────────────────────────────
import eval_sft as E                                    # noqa: E402  model definition + generation loop
import build_instr_data as BI                           # noqa: E402  single source of truth for templates

DEFAULT_TOKENIZER = HERE.parent / "tokenizer" / "tokenizer.json"
RECORD_KEYS = ("id", "k_index", "completion", "seed_used")


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def load_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[ERR] file not found: {p}")
    rows = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"[ERR] {p}:{i} is not valid JSON: {e}")
    return rows


def build_prompt(pr: dict, tmpl: str, tmpl_name: str) -> tuple[str, str]:
    """Return (prompt_text, problem id) and assert the template matches byte for byte."""
    if "id" not in pr:
        raise SystemExit(f"[ERR] problem has no id field: { {k: str(v)[:40] for k, v in pr.items()} }")
    instr = pr.get("prompt")
    if instr is None:
        instr = pr.get("instruction") or pr.get("input")
    if instr is None or not str(instr).strip():
        raise SystemExit(f"[ERR] problem {pr['id']} has no prompt/instruction")
    instr = str(instr)
    text = tmpl.format(instruction=instr)

    mt = pr.get("meta") or {}
    t_in_meta = mt.get("prompt_template")
    if t_in_meta not in (None, "", tmpl):
        raise SystemExit(
            f"[ERR] problem {pr['id']}: meta.prompt_template disagrees with this script's template -- generation would be misaligned.\n"
            f"      meta : {t_in_meta!r}\n      this script ({tmpl_name}): {tmpl!r}\n"
            f"      fix  : add --template <marker|fence|plain>, or use a template from the same source as the problem set.")
    pt = mt.get("prompt_templated")
    if pt is not None and pt != text:
        raise SystemExit(
            f"[ERR] problem {pr['id']}: the formatted template is **not byte-equal** to meta.prompt_templated.\n"
            f"      expected (meta) : {pt[:160]!r}\n      computed        : {text[:160]!r}")
    return text, str(pr["id"])


def expand_id(base_id: str, k_index: int, id_mode: str) -> str:
    return f"{base_id}#k{k_index}" if id_mode == "unique" else base_id


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="ESD generation step: generate k candidates per problem with a given ckpt (reuses eval_sft's model and generation params)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="checkpoint directory (containing checkpoint.pt); not needed for --dry-run")
    ap.add_argument("--problems", required=True, help="problem-set jsonl produced by esd_pipeline.py extract")
    ap.add_argument("--out", required=True, help="candidate output jsonl")
    ap.add_argument("--k", type=int, default=4, help="number of candidates per problem")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N problems (0 = all)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95, help="not supported by eval_sft.generate_one → warned about and ignored")
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0, help="base seed; candidate k uses seed + k_index")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="cpu by default: the local GPU is usually busy training; pass --device cuda explicitly to use it")
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--template", choices=sorted(BI.PROMPT_TEMPLATES), default="marker")
    ap.add_argument("--dry-run", action="store_true",
                    help="load no model: use each problem's ref_solution as the candidate (self-check / downstream integration)")
    ap.add_argument("--id-mode", choices=["plain", "unique"], default="plain",
                    help="plain: id=problem id (k=1 feeds sandbox-batch directly); unique: id=<problem id>#k<k>")
    ap.add_argument("--problems-out", default=None,
                    help="with --id-mode unique, additionally export a per-candidate expanded problem set (for sandbox-batch)")
    ap.add_argument("--truncate-eval", action="store_true",
                    help="truncate candidates with eval_sft.truncate() (same convention as eval_sft.check_algo); raw text is kept by default")
    a = ap.parse_args(argv)

    if a.k < 1:
        raise SystemExit("[ERR] --k must be >= 1")
    if not a.dry_run:
        if not a.ckpt:
            raise SystemExit("[ERR] --ckpt <checkpoint directory> is required (or use --dry-run for the reference-solution self-check)")
        ckpt = Path(a.ckpt)
        if not (ckpt / "checkpoint.pt").exists():
            raise SystemExit(f"[ERR] checkpoint does not exist: {ckpt / 'checkpoint.pt'}\n"
                             f"      --ckpt must point to a directory containing checkpoint.pt (e.g. ckpt_a5/final).")
    else:
        a.ckpt = a.ckpt or "<dry-run>"

    tmpl = BI.PROMPT_TEMPLATES[a.template]
    probs = load_jsonl(a.problems)
    if not probs:
        raise SystemExit(f"[ERR] empty problem set: {a.problems}")
    if a.limit and a.limit > 0:
        probs = probs[:a.limit]

    # ── byte-aligned prompts (template + meta assertions) ──
    prepared = []
    for pr in probs:
        text, pid = build_prompt(pr, tmpl, a.template)
        if a.dry_run and not (pr.get("ref_solution") or "").strip():
            raise SystemExit(f"[ERR] --dry-run needs problem {pid} to carry a ref_solution (this one has none)")
        prepared.append((pid, text, pr))
    print(f"[sample] {len(prepared)} problems x k={a.k} → {len(prepared) * a.k} expected candidates | "
          f"template={a.template} head={BI.TEMPLATE_HEAD[a.template]!r} tail={BI.TEMPLATE_TAIL[a.template]!r}")

    # ── tokenizer: length sanity check only (CPU, never touches the GPU); a failure here is not fatal ──
    tok = None
    try:
        if Path(a.tokenizer).exists():
            tok = E.load_tokenizer(a.tokenizer)
            lens = [len(tok.encode(t, add_special_tokens=False).ids) for _, t, _ in prepared]
            print(f"[sample] prompt tokens: max={max(lens)}  mean={sum(lens) / len(lens):.0f}  "
                  f"(hard context limit {E.ModelConfig.max_seq_len})")
            over = [pid for (pid, _, _), n in zip(prepared, lens) if n >= E.ModelConfig.max_seq_len]
            if over:
                eprint(f"[sample] WARN: {len(over)} prompts are already at the hard context limit: {over[:5]}"
                       f"(generation will be truncated by input_ids, so candidate quality is not trustworthy)")
        else:
            eprint(f"[sample] WARN: tokenizer not found {a.tokenizer} → skipping the prompt-token sanity check")
    except Exception as e:
        eprint(f"[sample] WARN: tokenizer unavailable ({type(e).__name__}: {e}) → skipping the prompt-token sanity check")

    # ── what --top-p actually does (honest warning) ──
    if a.top_p < 1.0:
        eprint(f"[sample] WARN: --top-p {a.top_p} is **ignored**: eval_sft.generate_one() has no top_p parameter"
               f"(its sampling is only top_k + min_p + repetition_penalty). This script keeps the contract and does not touch eval_sft.py,"
               f"preferring an explicit warning over quietly rewriting the sampling loop. meta records top_p_applied=false.")

    gen_kw = dict(max_new=a.max_new, temperature=a.temperature, top_k=a.top_k,
                  repetition_penalty=a.repetition_penalty, min_p=a.min_p, device=a.device)
    # Static contract check: every kwarg passed through here must really exist in eval_sft.generate_one's signature.
    # (If eval_sft.py renames or changes its signature this fails immediately -- no need to discover it after taking the GPU.)
    import inspect
    sig = inspect.signature(E.generate_one).parameters
    missing = [k for k in gen_kw if k not in sig]
    if missing:
        raise SystemExit(f"[ERR] eval_sft.generate_one has no such parameters: {missing}\n"
                         f"      actual signature: {list(sig)}\n"
                         f"      → eval_sft.py changed; the pass-through parameters here need syncing (do not rewrite the generation loop).")
    print(f"[sample] eval_sft.generate_one signature check OK: all pass-through args {sorted(gen_kw)} exist; "
          f"signature={list(sig)}")
    print(f"[sample] generation params (passed through to eval_sft.generate_one): {gen_kw}"
          f" | mode={'dry-run(reference solution)' if a.dry_run else 'ckpt=' + str(a.ckpt)}")

    model = None
    if not a.dry_run:
        print(f"[sample] loading tokenizer+model (device={a.device})...")
        t0 = time.time()
        if tok is None:
            tok = E.load_tokenizer(a.tokenizer)
        model = E.load_model(a.ckpt, a.device)
        print(f"[sample] load done in {time.time() - t0:.1f}s | "
              f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M "
              f"layers={model.config.num_layers} vocab={model.config.vocab_size}")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_records, n_empty, chars = 0, 0, 0
    seed_rule = "base_seed + k_index"
    t_start = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for pi, (pid, text, pr) in enumerate(prepared):
            t_p = time.time()
            for ki in range(a.k):
                seed_used = a.seed + ki
                if a.dry_run:
                    comp = pr.get("ref_solution") or ""
                else:
                    comp = E.generate_one(model, tok, text, seed_used, **gen_kw)
                    if a.truncate_eval:
                        comp = E.truncate(comp)
                rec = {"id": expand_id(pid, ki, a.id_mode), "k_index": ki,
                       "completion": comp, "seed_used": seed_used}
                assert tuple(rec.keys()) == RECORD_KEYS, tuple(rec.keys())
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_records += 1
                n_empty += (1 if not comp.strip() else 0)
                chars += len(comp)
            print(f"  [{pi + 1}/{len(prepared)}] {pid}  k={a.k}  "
                  f"took {time.time() - t_p:.1f}s  avg {chars / max(1, n_records):.0f} chars/candidate")
    print(f"[sample] → {out_path} ({n_records} candidates, {n_empty} empty, "
          f"took {time.time() - t_start:.1f}s)")

    # ── --id-mode unique: export a per-candidate expanded problem set (sandbox-batch's id lookup key) ──
    if a.problems_out:
        if a.id_mode != "unique":
            eprint("[sample] WARN: --problems-out only makes sense with --id-mode unique (otherwise ids already match the problem set) → skipping")
        else:
            pp = Path(a.problems_out)
            pp.parent.mkdir(parents=True, exist_ok=True)
            n = 0
            with open(pp, "w", encoding="utf-8") as f:
                for pid, _text, pr in prepared:
                    for ki in range(a.k):
                        q = dict(pr)
                        q["id"] = expand_id(pid, ki, a.id_mode)
                        q["base_id"] = pid
                        q["k_index"] = ki
                        f.write(json.dumps(q, ensure_ascii=False) + "\n")
                        n += 1
            print(f"[sample] → {pp} ({n} problems expanded per candidate, "
                  f"use --problems {pp} when feeding sandbox-batch)")

    meta = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "esd_sample.py",
        "mode": "dry_run" if a.dry_run else "model",
        "ckpt": str(a.ckpt),
        "device": a.device,
        "problems": str(a.problems),
        "n_problems": len(prepared),
        "k": a.k,
        "n_records": n_records,
        "n_empty_completions": n_empty,
        "avg_completion_chars": round(chars / max(1, n_records), 1),
        "id_mode": a.id_mode,
        "seed_base": a.seed,
        "seed_rule": seed_rule,
        "prompt_template_name": a.template,
        "prompt_template": tmpl,
        "prompt_template_head": BI.TEMPLATE_HEAD[a.template],
        "prompt_template_tail": BI.TEMPLATE_TAIL[a.template],
        "prompt_alignment": "template comes from build_instr_data.PROMPT_TEMPLATES; byte-for-byte assertion against meta.prompt_templated",
        "gen_params": gen_kw,
        "top_p_requested": a.top_p,
        "top_p_applied": False,
        "top_p_note": "eval_sft.generate_one has no top_p arg; this script does not touch eval_sft.py → ignored with a warning",
        "truncate_eval": bool(a.truncate_eval),
        "model_source": "eval_sft.load_tokenizer / load_model / generate_one (passed through parameter for parameter)",
        "record_fields": list(RECORD_KEYS),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sample] meta → {meta_path}")
    if n_records == 0:
        raise SystemExit("[ERR] no candidate was produced (problem set empty or cleared by --limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
