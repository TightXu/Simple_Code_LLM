#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairs_to_bins.py - ESD pipeline "training-data regression" stage
================================================
Turn **execution-filtered** (requirement, code) pairs into training bins that are **fully
contract-compatible** with data/ and data_instr/ -- i.e. rejection-sampling output back into SFT data sft_train.py can consume directly.

Contract (= the consumption contract of build_instr_data.py / build_sft_data.py)
----------------------------------------------------------
    <stem>.bin       flat uint16 token id stream
    <stem>_mask.bin  flat uint8 per-token loss mask (1 = counted in loss)
    <stem>_meta.json stats + config (isomorphic to data_instr/*_meta.json)
    chunk_size = seq_len + 1 = **1025** (not seq_len), **no padding**
    -> file size must be a multiple of 2*1025; a trailing partial chunk is cut (same policy as build_*)

    sample = encode(prompt).ids + encode(code + "\\n").ids + [<eos>]
             mask = 0 x len(prompt) + 1 x (len(code) + 1)     # only the solution part + <eos> is learned
    prompt = PROMPT_TEMPLATES[template].format(instruction=requirement)

**The prompt must be encoded separately from the solution**: encoding them together lets BPE merge the prompt tail ("### Code\\n")
with the first solution token -- the mask boundary would then be misaligned against the real token boundary.
This is a known trap; this script explicitly encodes the two parts separately.

Usage
----
    # 1) pairs given directly (each {"instruction": ..., "code": ...})
    py -3.12 pairs_to_bins.py --pairs esd/pairs.jsonl --out data_esd

    # 2) join pairs straight from the ESD triple (results.jsonl passed=true
    #    + candidates.jsonl completion + problems.jsonl prompt)
    py -3.12 pairs_to_bins.py --from-esd esd --out data_esd

    # 3) contract verification (**mandatory**; exit code 0 = all pass)
    py -3.12 verify_instr_contract.py --dir data_esd --stems sft_train sft_val

Filtering / error semantics
--------------
* Empty input, all samples filtered out, or (after truncation) not even one chunk left -> **hard error exit**,
  after clearing any 0-byte files that may have been written (never emit a "looks successful" empty dataset).
* Oversized samples (prompt+code+eos > seq_len) are **dropped and counted** by default (same policy as build_instr_data;
  reject_too_long_tokens is visible in the print output), not truncated into half a sample.
* val needs at least 1 complete chunk (otherwise the val bin is 0 bytes and verify fails outright):
  if val has less than one chunk after the hash split, the **longest** samples are moved over from train to top it up,
  and val_promoted_from_train is recorded in meta.rejection.

This file only adds itself and its own output directory; no existing file is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np                                      # noqa: E402
import build_instr_data as BI                           # noqa: E402  single source of truth for templates / fences / blacklist
from tokenizers import Tokenizer                        # noqa: E402

DEFAULT_TOKENIZER = HERE.parent / "tokenizer" / "tokenizer.json"

INSTR_ALIASES = ("instruction", "prompt", "input", "question", "task")
CODE_ALIASES = ("code", "completion", "response", "output", "solution", "answer")


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


def first_str(rec: dict, keys) -> str | None:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


# ===================================================================
# pair sources: (1) plain file  (2) join from ESD artifacts
# ===================================================================

def pairs_from_esd(esd_dir: Path, results=None, candidates=None, problems=None,
                   include_failed=False):
    """results.jsonl(passed) x candidates.jsonl(completion) x problems.jsonl(prompt) -> pairs.

    id is compatible with --id-mode unique ("<problem id>#k<k>"): if the problem is not found, strip the #k suffix and retry.
    """
    res_p = Path(results) if results else esd_dir / "results.jsonl"
    cnd_p = Path(candidates) if candidates else esd_dir / "candidates.jsonl"
    prb_p = Path(problems) if problems else esd_dir / "problems.jsonl"
    for p in (res_p, cnd_p, prb_p):
        if not p.exists():
            raise SystemExit(f"[ERR] --from-esd requires these files: {p} does not exist")

    problems = {str(p["id"]): p for p in load_jsonl(prb_p)}
    cands = {}
    for c in load_jsonl(cnd_p):
        cid = str(c.get("id"))
        cands.setdefault(cid, c.get("completion") or c.get("code") or "")

    st = {"results": 0, "skipped_failed": 0, "skipped_no_candidate": 0,
          "skipped_no_problem": 0, "kept": 0, "include_failed": bool(include_failed)}
    out = []
    for r in load_jsonl(res_p):
        st["results"] += 1
        if not r.get("passed") and not include_failed:
            st["skipped_failed"] += 1
            continue
        cid = str(r.get("id"))
        comp = cands.get(cid)
        if not comp:
            st["skipped_no_candidate"] += 1
            continue
        pr = problems.get(cid) or problems.get(cid.split("#k")[0])
        if pr is None:
            st["skipped_no_problem"] += 1
            continue
        instr = pr.get("prompt") or pr.get("instruction")
        if not instr:
            st["skipped_no_problem"] += 1
            continue
        out.append({"instruction": instr, "code": comp,
                    "source_id": cid, "entry": pr.get("entry"),
                    "group": cid.split("#k")[0]})
        st["kept"] += 1
    print("[pairs] from-esd " + "  ".join(f"{k}={v}" for k, v in st.items() if v))
    return out


# ===================================================================
# single pair -> (ids, mask)
# ===================================================================

def prep_pair(rec: dict, a, blacklist) -> tuple[dict | None, str | None]:
    """Clean one pair. Returns (sample dict, None) or (None, rejection reason)."""
    instr_raw = first_str(rec, INSTR_ALIASES)
    if instr_raw is None:
        return None, "reject_no_instruction"
    instr = instr_raw.strip()
    if any(lit in instr for lit in BI.TEMPLATE_LITERALS):
        return None, "reject_template_marker_in_instruction"
    if any(lit in instr for lit in BI.SPECIAL_TOKEN_TEXT):
        return None, "reject_special_token_literal"

    code_raw = first_str(rec, CODE_ALIASES)
    if code_raw is None:
        return None, "reject_no_code"
    text = code_raw
    marker = BI.TEMPLATE_TAIL[a.template]
    if marker.strip() and marker in text:
        # the candidate pasted the whole training prompt back -> take the content after the **last** marker
        text = text.rsplit(marker, 1)[-1]
    code, n_blocks, _had = BI.extract_code(text, a.fence_policy)
    if not code:
        return None, "reject_no_code"
    code = BI.normalize_code(code)
    if not code.strip():
        return None, "reject_no_code"
    if any(lit in code for lit in BI.TEMPLATE_LITERALS):
        return None, "reject_template_marker_in_code"
    if any(lit in code for lit in BI.SPECIAL_TOKEN_TEXT):
        return None, "reject_special_token_literal"
    if BI.effective_lines(code) < a.min_code_lines:
        return None, "reject_too_few_code_lines"
    if BI.leak_names(code, blacklist):
        return None, "reject_eval_leak_name"
    if a._compile_check:
        try:
            compile(code, "<pairs_to_bins>", "exec")
        except (SyntaxError, ValueError):
            return None, "reject_compile_fail"
    return {"instruction": instr, "code": code, "n_fence_blocks": n_blocks}, None


def encode_pair(sample: dict, a, tok, eos_id: int) -> tuple[array, bytes] | tuple[None, str]:
    prompt = BI.PROMPT_TEMPLATES[a.template].format(instruction=sample["instruction"])
    tail = "" if a.no_eos_newline else "\n"
    # * encode separately (never as one string): guarantees mask boundary == real token boundary
    pid = tok.encode(prompt).ids
    cid = tok.encode(sample["code"] + tail).ids
    n = len(pid) + len(cid) + 1
    if len(pid) > a.max_prompt_tokens:
        return None, "reject_prompt_too_long_tokens"
    if len(cid) > a.max_code_tokens:
        return None, "reject_code_too_long_tokens"
    if n > a.seq_len:
        return None, "reject_too_long_tokens"
    bad = [t for t in (pid + cid) if t in a._special_ids]
    if bad:
        return None, f"reject_special_id:{sorted(set(bad))}"
    ids = array("H", pid)
    ids.extend(cid)
    ids.append(eos_id)
    mask = b"\x00" * len(pid) + b"\x01" * (len(cid) + 1)
    sample["prompt_tokens"] = len(pid)
    sample["code_tokens"] = len(cid) + 1
    sample["total_tokens"] = n
    sample["prompt_text"] = prompt
    return ids, mask


# ===================================================================
# Writing (reuses build_instr_data.BinWriter -> byte-for-byte same cross-chunk policy)
# ===================================================================

def write_split(pool, out_dir: Path, stem: str, seq_len: int, policy: str):
    w = BI.BinWriter(out_dir, stem, seq_len)
    for ids, mask in pool:
        w.write_sample(ids, mask, policy)
    usable = w.finalize()
    return usable, w.n_samples, w.n_cross


def stats_from_disk(out_dir: Path, stem: str, seq_len: int, eos_id: int,
                    n_samples_written: int, tok=None, head_chunks: int = 8):
    """Read statistics back **from the final file** (numbers are only real after the tail is cut; no self-deception)."""
    C = seq_len + 1
    bpath, mpath = out_dir / f"{stem}.bin", out_dir / f"{stem}_mask.bin"
    n_bytes = bpath.stat().st_size if bpath.exists() else 0
    n_mask = mpath.stat().st_size if mpath.exists() else 0
    st = {"bin_bytes": n_bytes, "mask_bytes": n_mask, "aligned": bool(n_bytes and n_bytes % (2 * C) == 0
                                                                     and n_mask == n_bytes // 2)}
    if not n_bytes:
        st.update({"total_tokens": 0, "num_chunks": 0, "loss_tokens": 0, "loss_ratio": 0.0,
                   "samples_in_bin": 0, "zero_loss_chunks": 0, "first_chunks": []})
        return st
    ids = np.fromfile(bpath, dtype="<u2")
    mask = np.fromfile(mpath, dtype=np.uint8)
    C = seq_len + 1
    n_chunks = len(ids) // C
    eos_pos = np.flatnonzero(ids == eos_id).astype(np.int64)
    chunk_keep = mask[: n_chunks * C].reshape(-1, C).sum(axis=1)
    st.update({
        "total_tokens": int(len(ids)),
        "num_chunks": int(n_chunks),
        "loss_tokens": int(mask.sum()),
        "loss_ratio": round(float(mask.sum()) / max(1, len(ids)), 4),
        "samples_in_bin": int(len(eos_pos)),
        "samples_dropped_at_tail_truncation": int(max(0, n_samples_written - len(eos_pos))),
        "zero_loss_chunks": int((chunk_keep == 0).sum()),
        "min_chunk_keep": int(chunk_keep.min()) if n_chunks else 0,
        "max_token_id": int(ids.max()) if len(ids) else 0,
        "first_chunks": [int(x) for x in chunk_keep[:head_chunks]],
        "first_sample": None,
    })
    ones = np.flatnonzero(mask == 1)
    if len(ones) and tok is not None:
        j = int(ones[0])
        st["first_sample"] = {
            "prompt_head": tok.decode(np.asarray(ids[:j], dtype=np.uint16).tolist())[:120],
            "code_head": tok.decode(np.asarray(ids[j:j + 16], dtype=np.uint16).tolist())[:120],
            "prompt_tokens": j,
        }
    return st


def step_table(num_chunks: int, seq_len: int, epochs: int = 1) -> dict:
    """Give the step counts for a few common configs using sft_train.py's planned_steps formula:
    planned_steps = chunks_total*seq_len // (batch*seq_len*accum) = chunks_total // (batch*accum)."""
    out = {}
    for b, acc in ((8, 4), (8, 1), (4, 8), (16, 2)):
        per_optim = b * seq_len * acc
        steps = max(1, int(num_chunks * max(1, epochs)) // max(1, b * acc))
        out[f"batch{b}_accum{acc}"] = {"steps_per_epoch": steps,
                                       "tokens_per_step": per_optim,
                                       "steps_total": steps * max(1, epochs),
                                       "tokens_total": steps * per_optim * max(1, epochs)}
    return out


# ===================================================================
# main
# ===================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="write passed (requirement, code) pairs as training bins with the same contract as data/ and data_instr/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_argument_group("input")
    src.add_argument("--pairs", default=None, help='pairs jsonl, each line {"instruction":..., "code":...}')
    src.add_argument("--from-esd", default=None, metavar="ESD_DIR",
                     help="join pairs from the ESD artifacts dir (results.jsonl passed=true + candidates + problems)")
    src.add_argument("--results", default=None, help="results.jsonl for --from-esd (default <dir>/results.jsonl)")
    src.add_argument("--candidates", default=None, help="candidates.jsonl for --from-esd")
    src.add_argument("--problems", default=None, help="problems.jsonl for --from-esd")
    src.add_argument("--include-failed", action="store_true",
                     help="with --from-esd also accept passed=false (default: only execution-passed samples)")

    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--stem", default="sft_train")
    ap.add_argument("--val-stem", default="sft_val")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--no-val", action="store_true", help="do not produce val (verify with --stems sft_train only)")
    ap.add_argument("--seq-len", type=int, default=1024, help="chunk_size = seq_len + 1")
    ap.add_argument("--crossing-policy", choices=["mask", "keep"], default="keep",
                    help="tail piece of a cross-chunk sample: mask=zeroed (build_instr_data default) / keep=keep")
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--template", choices=sorted(BI.PROMPT_TEMPLATES), default="marker")
    ap.add_argument("--fence-policy", choices=["extract", "strip", "keep"], default="extract")
    ap.add_argument("--max-prompt-tokens", type=int, default=512)
    ap.add_argument("--max-code-tokens", type=int, default=768)
    ap.add_argument("--min-code-lines", type=int, default=1,
                    help="default 1 (a single correct line among self-distillation candidates still has value; build_instr_data uses 2)")
    ap.add_argument("--no-compile-check", action="store_true", help="do not require code to compile()")
    ap.add_argument("--no-eos-newline", action="store_true", help="do not append '\\n' to the solution (default: append, as build_instr_data does)")
    ap.add_argument("--no-blacklist", action="store_true", help="skip the eval-leak blacklist (applied by default)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="no dedup (default: dedup by (prompt+code) blake2b; duplicate candidates yield duplicate samples)")
    ap.add_argument("--instruction-dedup", action="store_true",
                    help="keep only one pair per requirement (default **off**: several different correct solutions per requirement are valuable for self-distillation)")
    ap.add_argument("--seed", type=int, default=42, help="used for logs/meta only (the split comes from blake2b and is reproducible)")
    a = ap.parse_args(argv)

    if not a.pairs and not a.from_esd:
        raise SystemExit("[ERR] need --pairs <pairs.jsonl> or --from-esd <esd dir>")
    if a.pairs and a.from_esd:
        raise SystemExit("[ERR] --pairs and --from-esd are mutually exclusive")

    # -- read pairs --
    if a.from_esd:
        raw = pairs_from_esd(Path(a.from_esd), a.results, a.candidates, a.problems,
                             include_failed=a.include_failed)
        src_desc = f"from-esd:{a.from_esd}"
    else:
        raw = load_jsonl(a.pairs)
        src_desc = str(a.pairs)
    if not raw:
        raise SystemExit(f"[ERR] 0 pairs in the input (source {src_desc}) -> refusing to emit an empty dataset.\n"
                         f"      check: with --from-esd, does results.jsonl contain any passed=true row?")

    # -- tokenizer / filters --
    if not Path(a.tokenizer).exists():
        raise SystemExit(f"[ERR] tokenizer not found: {a.tokenizer}")
    tok = Tokenizer.from_file(str(a.tokenizer))
    eos_id = tok.token_to_id("<eos>")
    if eos_id is None:
        raise SystemExit("[ERR] tokenizer has no <eos> (samples must end with <eos>)")
    vocab = tok.get_vocab_size()
    a._special_ids = {tok.token_to_id(t) for t in ("<s>", "<unk>", "<pad>")} | {eos_id}
    a._special_ids.discard(None)
    a._compile_check = not a.no_compile_check
    blacklist = BI.build_blacklist(a.no_blacklist)

    C = a.seq_len + 1
    print(f"[cfg] seq_len={a.seq_len} chunk_size={C} template={a.template} "
          f"crossing={a.crossing_policy} val_ratio={a.val_ratio} fence={a.fence_policy} "
          f"compile_check={not a.no_compile_check} blacklist={len(blacklist)} vocab={vocab} eos={eos_id}")

    st = {"rows": len(raw)}
    rejected = {}
    train_pool, val_pool = [], []
    seen, seen_instr = set(), set()
    train_tok = val_tok = 0

    for rec in raw:
        sample, why = prep_pair(rec, a, blacklist)
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        packed = encode_pair(sample, a, tok, eos_id)
        if packed[0] is None:
            rejected[packed[1]] = rejected.get(packed[1], 0) + 1
            continue
        ids, mask = packed
        digest = hashlib.blake2b((sample["prompt_text"] + sample["code"]).encode("utf-8", "replace"),
                                 digest_size=8).digest()
        if not a.no_dedup and digest in seen:
            rejected["reject_duplicate"] = rejected.get("reject_duplicate", 0) + 1
            continue
        ikey = hashlib.blake2b(sample["instruction"].encode("utf-8", "replace"), digest_size=8).digest()
        if a.instruction_dedup and ikey in seen_instr:
            rejected["reject_duplicate_instruction"] = rejected.get("reject_duplicate_instruction", 0) + 1
            continue
        seen.add(digest)
        seen_instr.add(ikey)
        h = int.from_bytes(digest, "big")
        if not a.no_val and (h % 10000) < int(a.val_ratio * 10000):
            val_pool.append((ids, mask))
            val_tok += len(ids)
            st["accepted_val"] = st.get("accepted_val", 0) + 1
        else:
            train_pool.append((ids, mask))
            train_tok += len(ids)
            st["accepted_train"] = st.get("accepted_train", 0) + 1
        st["accepted"] = st.get("accepted", 0) + 1
        st["accepted_tokens"] = st.get("accepted_tokens", 0) + len(ids)

    print(f"[prep] input {st['rows']} pairs -> {st.get('accepted', 0)} passed the filters"
          f" (train {st.get('accepted_train', 0)} / val {st.get('accepted_val', 0)})"
          f"  train {train_tok:,} tok / val {val_tok:,} tok")
    if rejected:
        print("[prep] rejection funnel: " + "  ".join(f"{k}={v}" for k, v in sorted(rejected.items())))

    if not train_pool and not val_pool:
        raise SystemExit(
            "[ERR] every pair was filtered out -> refusing to emit 0-byte bins.\n"
            "      rejection funnel: " + json.dumps(rejected, ensure_ascii=False) + "\n"
            "      most common causes: empty code / code that does not compile, oversized (prompt+code > seq_len),"
            "literal ### Task / <eos> markers inside the body.")

    # -- val top-up: val must form at least 1 complete chunk, else the val bin is 0 bytes (verify always fails) --
    promoted = 0
    if not a.no_val:
        while val_tok < C and train_pool:
            j = max(range(len(train_pool)), key=lambda i: len(train_pool[i][0]))
            s = train_pool.pop(j)
            val_pool.append(s)
            val_tok += len(s[0])
            train_tok -= len(s[0])
            promoted += 1
        if val_tok < C:
            raise SystemExit(f"[ERR] val has only {val_tok} token < one chunk({C}), "
                             f"and train has no sample left to move -> the val bin would be 0 bytes.\n"
                             f"      either supply more pairs or add --no-val (then verify with --stems sft_train only).")
        if promoted:
            st["val_promoted_from_train"] = promoted
            print(f"[prep] val is short of one chunk -> moved {promoted} longest samples from train (val={val_tok:,} tok)")

    if train_tok < C:
        raise SystemExit(f"[ERR] train has only {train_tok} token < one chunk({C}) -> refusing to write a 0-byte bin. "
                         f"(currently val={val_tok}, val_ratio={a.val_ratio})")

    # -- writing --
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_usable, train_n, train_cross = write_split(train_pool, out_dir, a.stem, a.seq_len, a.crossing_policy)
    val_usable = val_n = val_cross = 0
    if not a.no_val:
        val_usable, val_n, val_cross = write_split(val_pool, out_dir, a.val_stem, a.seq_len, a.crossing_policy)

    # -- read statistics back from the final file (numbers are real only after truncation) --
    tstat = stats_from_disk(out_dir, a.stem, a.seq_len, eos_id, train_n, tok=tok)
    vstat = (stats_from_disk(out_dir, a.val_stem, a.seq_len, eos_id, val_n, tok=tok)
             if not a.no_val else {})

    # build-time self-check (build_instr_data uses **the same function**; sft_train.py untouched, torch not imported)
    probe_names = [a.stem] + ([] if a.no_val else [a.val_stem])
    ccheck = BI.consumer_check(out_dir, probe_names, a.seq_len, vocab)
    print("[selfcheck] consumer_check: " + json.dumps(ccheck, ensure_ascii=False))

    for name, stat, usable in ((a.stem, tstat, train_usable), (a.val_stem, vstat, val_usable)):
        if a.no_val and name == a.val_stem:
            continue
        print(f"[out] {name}.bin {stat['bin_bytes']:,}B = {stat['total_tokens']:,} tok "
              f"| chunks={stat['num_chunks']:,} | samples={stat['samples_in_bin']:,} "
              f"| loss_ratio={stat['loss_ratio']} | zero_loss_chunks={stat['zero_loss_chunks']} "
              f"| first_chunks_keep={stat['first_chunks']} | aligned={stat['aligned']}")
        if stat.get("first_sample"):
            print(f"      first prompt: {stat['first_sample']['prompt_head']!r}")
            print(f"      first code  : {stat['first_sample']['code_head']!r}")
        if usable == 0 or stat["bin_bytes"] == 0:
            for f in (out_dir / f"{name}.bin", out_dir / f"{name}_mask.bin"):
                try:
                    f.unlink()
                except OSError:
                    pass
            raise SystemExit(f"[ERR] {name} came out 0 bytes (fewer tokens than one chunk) -> empty file removed.")

    # -- meta (isomorphic to data_instr/*_meta.json) --
    common = {
        "seq_len": a.seq_len,
        "chunk_size": C,
        "dtype": "uint16",
        "mask_dtype": "uint8",
        "mask_semantics": "1 = counted in loss (code part + <eos>); 0 = requirement part / cross-chunk tail / tail truncation",
        "task_type": "instruction",
        "sample_format": "prompt=PROMPT_TEMPLATES[t].format(instruction=requirement) + solution(code) + '<eos>'",
        "prompt_template": BI.PROMPT_TEMPLATES[a.template],
        "prompt_template_name": a.template,
        "prompt_template_head": BI.TEMPLATE_HEAD[a.template],
        "prompt_template_tail": BI.TEMPLATE_TAIL[a.template],
        "prompt_encoding": "tokenizer.encode(prompt).ids (encoded separately from the solution, so the mask boundary is token-aligned)",
        "dataloader_contract": (
            "ids=np.memmap(bin,dtype=np.uint16); mask=np.fromfile(mask_bin,dtype=np.uint8); "
            "C=seq_len+1; for off in range(0,len(ids),C): x=ids[off:off+seq_len]; "
            "y=ids[off+1:off+seq_len+1]; labels=np.where(mask[off+1:off+seq_len+1]==1,y,-100)"),
        "no_padding": True,
        "source": {"kind": "esd_pairs" if a.from_esd else "pairs_jsonl", "desc": src_desc,
                   "results": a.results, "candidates": a.candidates, "problems": a.problems,
                   "include_failed": bool(a.include_failed)},
        "config": {
            "val_ratio": a.val_ratio, "val": not a.no_val, "prompt_template": a.template,
            "fence_policy": a.fence_policy, "max_prompt_tokens": a.max_prompt_tokens,
            "max_code_tokens": a.max_code_tokens, "min_code_lines": a.min_code_lines,
            "compile_check": not a.no_compile_check, "eos_newline": not a.no_eos_newline,
            "blacklist_size": len(blacklist), "dedup": not a.no_dedup,
            "instruction_dedup": bool(a.instruction_dedup), "seed": a.seed,
            "crossing_policy": a.crossing_policy,
        },
        "rejection": dict(sorted(rejected.items())),
        "consumer_check": ccheck,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verify_cmd": f"py -3.12 verify_instr_contract.py --dir {out_dir} --stems "
                      + (a.stem if a.no_val else f"{a.stem} {a.val_stem}"),
    }
    for name, stat, n_written, n_cross, split in ((a.stem, tstat, train_n, train_cross, "train"),
                                                  (a.val_stem, vstat, val_n, val_cross, "val")):
        if a.no_val and split == "val":
            continue
        meta = dict(common)
        meta.update({
            "split": split,
            "total_tokens": stat["total_tokens"],
            "num_chunks": stat["num_chunks"],
            "num_samples_written": n_written,
            "samples_in_bin": stat["samples_in_bin"],
            "samples_dropped_at_tail_truncation": stat.get("samples_dropped_at_tail_truncation", 0),
            "crossing": {"policy": a.crossing_policy, "crossed_samples": n_cross},
            "crossing_counter": n_cross,
            "loss_tokens": stat["loss_tokens"],
            "loss_ratio": stat["loss_ratio"],
            "zero_loss_chunks": stat["zero_loss_chunks"],
            "min_chunk_keep": stat.get("min_chunk_keep"),
            "max_token_id": stat.get("max_token_id"),
            "aligned_to_chunk": stat["aligned"],
            "steps": step_table(stat["num_chunks"], a.seq_len),
            "verified": stat,
        })
        (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(f"[out] meta -> {out_dir / (a.stem + '_meta.json')}" +
          ("" if a.no_val else f" + {a.val_stem}_meta.json") + f"  ({time.time() - t0:.1f}s)")
    print(f"[out] verify: {common['verify_cmd']}")
    tsteps = step_table(tstat["num_chunks"], a.seq_len)
    print("[out] steps (epochs=1, sft_train.py planned_steps formula = chunks // (batch*accum)):"
          + "  ".join(f"{k}={v['steps_per_epoch']}steps({v['tokens_per_step'] // 1000}K tok/step)"
                      for k, v in tsteps.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
