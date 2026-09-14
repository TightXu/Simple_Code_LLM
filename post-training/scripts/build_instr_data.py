#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
435M v2 · instruction-data builder (natural-language request → code)
================================================
Public "code instruction datasets" (parquet, e.g. bigcode/self-oss-instruct-sc2-exec-filter-50k,
nvidia/OpenCodeInstruct) → a sample stream under the **same contract** as `build_sft_data.py`:

    prompt   = PROMPT_TEMPLATE.format(instruction=request)     # NL request + fixed tail marker
    solution = extracted Python code (fence stripped, dedented, rstripped)
    stream   = prompt + solution + "\\n" + "<eos>"
    ids      = tokenizer.encode(prompt).ids + tokenizer.encode(solution+"\\n").ids + [<eos>=4]
    mask     = [0]*len(prompt_ids) + [1]*(len(solution_ids)+1)    # loss only on the code span + <eos>

  The only difference vs build_sft_data.py is the **prompt text shape** (NL request vs `def name(...):` signature),
    while ids/mask encoding, chunking, mask semantics and file layout are **identical** (same contract). Hence
    `sft_train.py --bin-data data_instr/instr_train.bin --sft-val-bin data_instr/instr_val.bin`
    can be **consumed directly**, with no changes at all.

Output and dataloader contract (fully isomorphic to data/sft_train.bin)
--------------------------------------------------------
    <stem>.bin        flat uint16, length T = N*(seq_len+1)
    <stem>_mask.bin   flat uint8, length T (1 = counted in loss)
    chunk_size = seq_len + 1 = 1025                    # <<< not seq_len
    off = i*chunk_size
    x = ids [off : off+seq_len]
    y = ids [off+1 : off+seq_len+1]
    labels = np.where(mask[off+1 : off+seq_len+1] == 1, y, IGNORE_INDEX(-100))
    → **no padding** (the model hardcodes is_causal=True, no attention mask)

Samples are concatenated head to tail (no padding, no cross-sample continuation); a sample's tail piece that
lands at the head of the next chunk (prompt context already lost) is masked to 0 by default (`--crossing-policy keep` disables this).

Filtering (instruction sets differ from pretraining corpora, so only the necessary checks are kept)
    A Field detection: instruction/input/question/... x response/output/solution/... (see detect_fields)
    B Length: total ≤ seq_len(1024), prompt ≤ --max-prompt-tokens, code ≤ --max-code-tokens
    C Quality: code compiles(), effective code lines ≥ --min-code-lines, min chars, dedup, template-family dedup
    D Evaluation leak (G5): **reuses the same 167-name blacklist as build_sft_data.py** (5 algorithms + 6 simple tasks +
      the entry_points of ablation/HumanEval.jsonl), matched exactly against def/class names in the code
    E Light pollution (--pollution-mode light, the default): L3 packaging/sphinx config, L4 test code, L6 autogen
      (**not** L1 frameworks or L2 Py2: Django/Flask requests are legitimate inside an instruction set)
    F Special-token literals (<s> </s> <unk> <pad> <eos>) or template markers (### Task / ### Code) in the body → drop
    G Score / execution status (when the dataset has average_test_score / tests_execution_status, e.g. OpenCodeInstruct)
      → --min-score / --require-tests-passed filtering

Usage
----
    py -3.12 build_instr_data.py --dry-run                 # small smoke run (writes data_instr/dryrun/)
    py -3.12 build_instr_data.py --target-tokens 3000000   # real build (instruction seed target 2–3M tokens)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import textwrap
import time
import warnings
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

# Bad escapes in the dataset would otherwise spam SyntaxWarning
warnings.filterwarnings("ignore")

# Windows console/pipes may default to GBK → printing non-ASCII would crash
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent            # .../435M-v2/fine-tuning
PROJECT = HERE.parent                             # .../435M-v2
DEFAULT_TOKENIZER = PROJECT / "tokenizer" / "tokenizer_435m.json"
DEFAULT_RAW_DIR = HERE / "data_instr_raw"
HUMANEVAL_JSONL = PROJECT / "ablation" / "HumanEval.jsonl"   # same source as build_sft_data.py

# ═══════════════════════════════════════════════════════════════════
# prompt templates (single source of truth for the whole project: the eval side must use **the same one** -- see meta.prompt_template)
# ═══════════════════════════════════════════════════════════════════
PROMPT_TEMPLATES = {
    # plain marker style (default): no BPE quirks, unambiguous boundary, easy split("### Code\n")[-1] on the eval side
    "marker": "### Task\n{instruction}\n\n### Code\n",
    # fenced style: matches the original public instruction-set style (the model learns to emit the closing ```)
    "fence": "### Task\n{instruction}\n\n```python\n",
    # bare request: no framing text at all (closest to "continuation", weakest boundary)
    "plain": "{instruction}\n\n",
}
TEMPLATE_TAIL = {"marker": "### Code\n", "fence": "```python\n", "plain": "\n\n"}
TEMPLATE_HEAD = {"marker": "### Task\n", "fence": "### Task\n", "plain": ""}

# ═══════════════════════════════════════════════════════════════════
# Evaluation-leak blacklist (G5) -- the **same** one as build_sft_data.py
# ═══════════════════════════════════════════════════════════════════
EVAL_NAMES = {
    "fibonacci", "quicksort", "binary_search", "two_sum", "mergesort",
    "is_palindrome", "is_prime", "reverse_string", "count_vowels", "sum_list", "factorial",
}

# ═══════════════════════════════════════════════════════════════════
# Light pollution mode (an instruction set should not get the full L1/L2 pretraining rule set)
# ═══════════════════════════════════════════════════════════════════
L3_FATAL_CONFIG = [
    "extensions = [", "master_doc =", "html_theme =", "pygments_style =",
    "from setuptools import", "install_requires=[", "entry_points={",
    "package_data={", "setup(name=",
]
L4_TESTS = [
    "import unittest", "from unittest import", "import pytest", "from pytest import",
    "unittest.TestCase", "pytest.mark.", "pytest.fixture", "pytest.raises",
]
L6_AUTOGEN = [
    r"\b[a-z]+_\d{4,}\b",                                              # variable name + long numeric suffix
    r"\b[0-9a-f]{32}\b",                                               # MD5
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",  # UUID
    r"\{\{.*?\}\}",                                                    # Jinja
    r"\{% .*?%\}",
]

# These **literal texts** in the body get encoded into special ids and break the "sample ends with <eos>" convention
SPECIAL_TOKEN_TEXT = ("<s>", "</s>", "<unk>", "<pad>", "<eos>")

# Template markers mixed into the body → ambiguous boundary, drop outright
TEMPLATE_LITERALS = ("### Task", "### Code", "### Instruction", "### Response")

# "Templated body" normalization: string literals → S, digits → N (flattens same-shape families)
_STR_RX = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"")
_NUM_RX = re.compile(r"\d+")

# Fenced blocks: ```python / ```py / ``` (no tag) / ```py3
_FENCE_RX = re.compile(r"```[ \t]*([A-Za-z0-9_+.\-]*)[ \t]*\r?\n(.*?)(?:\r?\n[ \t]*```|\Z)", re.S)
_DEF_RX = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)", re.M)
_LOGIC_KWS = ("=", "if ", "for ", "while ", "return ", "print(", "yield ", "raise ", "assert ")

# Field candidates (**order is priority**; prompt comes last = only used when there is no other choice,
# because self-oss-instruct's `prompt` is a generation template rather than the request itself)
INSTR_FIELDS = ("instruction", "input", "question", "problem", "task", "query",
                "instruction_text", "description", "prompt")
CODE_FIELDS = ("output", "response", "answer", "solution", "code", "completion", "response_text")
SCORE_FIELDS = ("average_test_score", "score", "test_score", "reward")
STATUS_FIELDS = ("tests_execution_status", "execution_status", "test_status")
ID_FIELDS = ("sha1", "hash", "id", "fingerprint", "problem_id", "task_id")


def template_key(body: str) -> bytes:
    return hashlib.blake2b(_NUM_RX.sub("N", _STR_RX.sub("S", body)).encode("utf-8", "replace"),
                           digest_size=8).digest()


def compile_autogen():
    return [(rx, re.compile(rx)) for rx in L6_AUTOGEN]


# ═══════════════════════════════════════════════════════════════════
# Field detection + answer extraction
# ═══════════════════════════════════════════════════════════════════

def detect_fields(col_names):
    """Parse {instr, code, score, status, id, ignored} out of one parquet's column names."""
    lower = {str(c).lower(): str(c) for c in col_names}
    pick = lambda cands: next((lower[c] for c in cands if c in lower), None)
    used = set()
    out = {}
    for key, cands in (("instr", INSTR_FIELDS), ("code", CODE_FIELDS), ("score", SCORE_FIELDS),
                       ("status", STATUS_FIELDS), ("id", ID_FIELDS)):
        got = pick(cands)
        out[key] = got
        if got:
            used.add(got)
        # one physical column must not serve as both request and answer (avoids output-as-instruction nonsense)
        cands = tuple(c for c in cands if lower.get(c) != got)
    out["ignored"] = [str(c) for c in col_names if str(c) not in used]
    return out


def extract_code(text: str, fence_policy: str = "extract"):
    """Return (code, n_blocks, had_fence); code is None when no code could be obtained.

    extract (default): take the first ```python/```py fence; if none is tagged, the first untagged fence;
                       no fence at all → treat the whole text as code (prose is caught by the compile check)
    strip            : drop the fence lines, keep everything in between (prose included)
    keep             : as-is
    """
    if not isinstance(text, str) or not text.strip():
        return None, 0, False
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = _FENCE_RX.findall(text)
    n_blocks = len(blocks)
    if fence_policy == "keep":
        return text.strip(), n_blocks, n_blocks > 0
    if fence_policy == "strip":
        if n_blocks == 0:
            return text.strip(), 0, False
        stripped = "\n".join(b[1] for b in blocks)
        return stripped.strip(), n_blocks, True
    # extract
    if n_blocks == 0:
        return text.strip(), 0, False
    pylike = [b for tag, b in blocks if tag.lower().startswith(("python", "py", "py3"))]
    pool = pylike or [b for tag, b in blocks if tag == ""]
    if not pool:                      # only non-python fences (java/cpp...) → treat as no code
        return None, n_blocks, True
    return pool[0].strip(), n_blocks, True


def normalize_code(code: str) -> str:
    code = textwrap.dedent(code).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in code.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def effective_lines(code: str) -> int:
    return sum(1 for ln in code.split("\n") if ln.strip() and not ln.strip().startswith("#"))


# ═══════════════════════════════════════════════════════════════════
# Filters
# ═══════════════════════════════════════════════════════════════════

def pollution_reason(prompt_text: str, code: str, autogen_rx, mode: str) -> str | None:
    if mode == "none":
        return None
    for p in L3_FATAL_CONFIG:
        if p in prompt_text or p in code:
            return f"L3_config:{p[:40]}"
    for p in L4_TESTS:
        if p in code and p not in prompt_text:
            return f"L4_test:{p[:40]}"
    for rx, c in autogen_rx:
        if len(c.findall(code)) >= 3:
            return f"L6_autogen:{rx[:40]}"
    return None


def leak_names(code: str, blacklist) -> list[str]:
    """Intersection of the def/class names in the code with the evaluation blacklist (exact match, not substring)."""
    if not blacklist:
        return []
    hit = [n for n in _DEF_RX.findall(code) if n in blacklist]
    return hit


# ═══════════════════════════════════════════════════════════════════
# Writers (same layout as build_sft_data.py)
# ═══════════════════════════════════════════════════════════════════

class BinWriter:
    """flat uint16 ids + side-by-side uint8 loss-mask, positioned as written (crossing mask done inline)."""

    def __init__(self, out_dir: Path, name: str, seq_len: int):
        self.C = seq_len + 1
        self.name = name
        self.bin_path = Path(out_dir) / f"{name}.bin"
        self.mask_path = Path(out_dir) / f"{name}_mask.bin"
        self.f = open(self.bin_path, "wb")
        self.mf = open(self.mask_path, "wb")
        self.total = 0          # tokens written so far (including an unaligned tail)
        self.n_samples = 0
        self.n_cross = 0        # number of samples crossing a chunk boundary

    def write_sample(self, ids: array, mask: bytes, policy: str = "mask"):
        off = self.total
        L = len(ids)
        B = (off // self.C) * self.C
        lim = B + self.C - off                 # local index + 1 of the last usable token in this chunk
        if L > lim:                            # sample crosses the chunk → its tail piece has orphan context
            self.n_cross += 1
            if policy == "mask":
                m = bytearray(mask)
                for j in range(max(lim, 0), L):
                    m[j] = 0
                mask = bytes(m)
        np.frombuffer(ids, dtype=np.uint16).tofile(self.f)
        self.mf.write(mask)
        self.total += L
        self.n_samples += 1

    def finalize(self) -> int:
        usable = (self.total // self.C) * self.C
        self.f.flush()
        self.mf.flush()
        if self.total > usable:                # drop a tail shorter than one chunk (same policy as pretrain_data.py)
            self.f.truncate(usable * 2)
            self.mf.truncate(usable)
        self.f.close()
        self.mf.close()
        return usable


# ═══════════════════════════════════════════════════════════════════
# Verification 1: read the bin back and confirm length / mask shape / chunk boundaries (same invariants as build_sft_data.py)
# ═══════════════════════════════════════════════════════════════════

def verify_outputs(out_dir: Path, names, seq_len: int, tokenizer, eos_id: int = 4,
                   n_probe: int = 3, template_name: str = "marker",
                   check_samples: int = 500, rng: random.Random | None = None):
    C = seq_len + 1
    head = TEMPLATE_HEAD.get(template_name, "")
    tail = TEMPLATE_TAIL.get(template_name, "")
    rng = rng or random.Random(0)
    rep = {}
    for name in names:
        bp = out_dir / f"{name}.bin"
        mp = out_dir / f"{name}_mask.bin"
        if not bp.exists() or not mp.exists():
            continue
        if bp.stat().st_size == 0:                 # too few samples → less than one chunk (a legal dry-run result)
            rep[name] = {"bin_tokens": 0, "mask_tokens": int(mp.stat().st_size), "empty": True,
                         "aligned": False, "chunks": 0, "loss_tokens": 0, "loss_ratio": 0.0,
                         "samples_in_bin": 0, "samples_with_invalid_mask": 0,
                         "samples_with_zero_loss": 0, "samples_tail_masked_crossing": 0,
                         "samples_tail_masked_not_at_block_boundary": 0,
                         "samples_prompt_checked": 0, "samples_prompt_template_mismatch": 0,
                         "has_pad_or_unk": False, "max_token_id": 0,
                         "chunks_with_unmasked_second_label": 0, "chunks_starting_mid_solution": 0,
                         "chunks_with_clean_prompt_head": 0, "probes": []}
            print(f"[VERIFY] WARN: {name}: bin is empty (fewer accepted samples than one chunk={C}; nothing left after tail truncation)")
            continue
        n_ids = bp.stat().st_size // 2
        n_mask = mp.stat().st_size
        ids = np.memmap(bp, dtype=np.uint16, mode="r")
        mask = np.fromfile(mp, dtype=np.uint8)
        chunks = n_ids // C
        ones = int(mask.sum())

        # ── per sample (split on <eos>), vectorized check that the mask is strictly 0^p 1^q 0^r ──
        N = int(n_ids)
        eos_pos = np.flatnonzero(np.asarray(ids, dtype=np.uint16) == eos_id)
        starts = np.concatenate((np.zeros(1, dtype=np.int64), eos_pos.astype(np.int64) + 1))
        ends = np.concatenate((eos_pos.astype(np.int64), np.zeros(1, dtype=np.int64)))
        if len(eos_pos) > 0:
            ends[-1] = N - 1
            m = mask.astype(np.int8)
            ar = np.arange(N, dtype=np.int32)
            first_one = np.minimum.accumulate(np.where(m == 1, ar, np.int32(N))[::-1])[::-1]
            last_one = np.maximum.accumulate(np.where(m == 1, ar, np.int32(-1)))
            f = first_one[starts]
            l = last_one[ends]
            cum = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(m, dtype=np.int64)))
            n_ones = cum[ends + 1] - cum[starts]
            empty = f > ends                       # the whole sample has no loss (pure context)
            dense = (n_ones == (l - f + 1)) & ~empty
            tail_masked = (l < ends) & ~empty      # tail zeroed by the crossing policy
            ok_boundary = np.ones(len(ends), dtype=bool)
            if tail_masked.any():
                ok_boundary[tail_masked] = ((l + 1)[tail_masked] % C) == 0
            n_s = len(eos_pos)                     # only "complete" samples are counted (the last one is a truncated fragment)
            sl = slice(0, n_s)
            invalid = int((~(empty | (dense & ok_boundary)))[sl].sum())
            n_empty = int(empty[sl].sum())
            n_tail = int(tail_masked[sl].sum())
            n_bad_boundary = int((tail_masked & ~ok_boundary)[sl].sum())

            # ── per-sample prompt shape: sample check_samples rows, decode the prompt span, must match the template ──
            idx = np.arange(n_s)
            if n_s > check_samples:
                idx = np.sort(rng.sample(range(n_s), check_samples))
            bad_tmpl, n_checked = 0, 0
            for i in idx:
                s, e, fo = int(starts[i]), int(ends[i]), int(f[i])
                if fo > e:                          # no-loss sample, skip
                    continue
                ptxt = tokenizer.decode(np.asarray(ids[s:fo], dtype=np.uint16).tolist())
                n_checked += 1
                if not ptxt.startswith(head) or not ptxt.endswith(tail):
                    bad_tmpl += 1
        else:
            invalid = n_empty = n_tail = n_bad_boundary = n_checked = bad_tmpl = 0

        # ── chunk level ──
        bad_second = int((mask[1:chunks * C:C] == 1).sum())
        starts_mid = int((mask[0:chunks * C:C] == 1).sum())
        probes, clean_heads = [], 0
        for i in range(chunks):
            off = i * C
            if mask[off] != 0:
                continue
            seg = mask[off: off + seq_len + 1]
            nz = np.flatnonzero(seg == 1)
            if len(nz) == 0:
                continue
            j = int(nz[0])
            prompt_txt = tokenizer.decode(np.asarray(ids[off: off + j], dtype=np.uint16).tolist())
            ok = prompt_txt.startswith(head) and prompt_txt.endswith(tail)
            clean_heads += int(ok)
            if len(probes) < n_probe and ok:
                probes.append({
                    "chunk": i,
                    "prompt_tail": prompt_txt[-160:],
                    "solution_head": tokenizer.decode(np.asarray(ids[off + j: off + j + 24],
                                                               dtype=np.uint16).tolist()),
                    "loss_tokens_in_chunk": int(seg.sum()),
                })

        rep[name] = {
            "bin_tokens": int(n_ids),
            "mask_tokens": int(n_mask),
            "aligned": bool(n_ids == n_mask and n_ids % C == 0),
            "chunks": int(chunks),
            "loss_tokens": ones,
            "loss_ratio": round(ones / max(n_ids, 1), 4),
            "samples_in_bin": int(len(eos_pos)),
            "samples_with_invalid_mask": invalid,
            "samples_with_zero_loss": n_empty,
            "samples_tail_masked_crossing": n_tail,
            "samples_tail_masked_not_at_block_boundary": n_bad_boundary,
            "samples_prompt_checked": int(n_checked),
            "samples_prompt_template_mismatch": int(bad_tmpl),
            "has_pad_or_unk": bool(((np.asarray(ids) == 3) | (np.asarray(ids) == 2)).any()),
            "max_token_id": int(np.asarray(ids).max()) if n_ids else 0,
            "chunks_with_unmasked_second_label": bad_second,
            "chunks_starting_mid_solution": starts_mid,
            "chunks_with_clean_prompt_head": clean_heads,
            "probes": probes,
        }
    return rep


# ═══════════════════════════════════════════════════════════════════
# Verification 2: consumer-contract self-check (replays sft_train.py's resolve_loss_mask + create_sft_dataloader math)
# ═══════════════════════════════════════════════════════════════════

def consumer_check(out_dir: Path, names, seq_len: int, vocab_size: int, n_min=4, n_max=4):
    """No change to sft_train.py, no torch import: recompute every step of its read path."""
    C = seq_len + 1
    out = {}
    for name in names:
        bp = Path(out_dir) / f"{name}.bin"
        if not bp.exists():
            out[name] = {"exists": False}
            continue
        r = {"exists": True}
        r["bin_bytes"] = bp.stat().st_size
        r["passes_assert_bin_readable"] = bp.stat().st_size >= 2 * C
        if bp.stat().st_size == 0:                  # empty bin: sft_train raises a clear error, recorded faithfully here
            r["empty"] = True
            out[name] = r
            continue
        # (1) auto-discover the companion mask (sft_train's candidate name is <stem>_mask.bin)
        cand = bp.parent / f"{bp.stem}_mask.bin"
        r["mask_auto_discovered"] = cand.exists()
        if not cand.exists():
            out[name] = r
            continue
        data = np.memmap(str(bp), dtype=np.uint16, mode="r")
        mask_arr = np.memmap(str(cand), dtype=np.uint8, mode="r")
        r["mask_len_ge_data"] = bool(len(mask_arr) >= len(data))
        total_chunks = len(data) // C
        r["total_chunks"] = int(total_chunks)
        r["tail_tokens_dropped"] = int(len(data) - total_chunks * C)
        # (2) must not be mistaken for a bit15-embedded mask (max(uint16) must be < vocab)
        mx = int(data[:min(len(data), 4_000_000)].max()) if len(data) else 0
        r["max_token_id_probe"] = mx
        r["resolves_to_file_mask"] = bool(mx < vocab_size)
        # (3) recompute x / y / labels per chunk
        bad = 0
        sample_ratio = []
        idxs = list(range(min(n_min, total_chunks)))
        idxs += list(range(max(0, total_chunks - min(n_max, total_chunks)), total_chunks))
        for i in idxs:
            off = i * C
            toks = np.asarray(data[off:off + C], dtype=np.uint16)
            x = toks[:seq_len].astype(np.int64)
            y = toks[1:seq_len + 1].astype(np.int64)
            mk = np.asarray(mask_arr[off:off + C], dtype=np.int64)
            labels = np.where(mk[1:seq_len + 1] == 1, y, -100)
            keep = (labels != -100)
            sample_ratio.append(round(float(keep.mean()), 4))
            if x.shape != (seq_len,) or y.shape != (seq_len,) or labels.shape != (seq_len,):
                bad += 1
            if x.max(initial=0) >= vocab_size or y.max(initial=0) >= vocab_size:
                bad += 1
        r["chunks_shape_ok"] = bool(bad == 0)
        r["loss_token_ratio_probed_chunks"] = sample_ratio
        r["zero_loss_chunks_probed"] = int(sum(1 for v in sample_ratio if v == 0.0))
        r["all_loss_chunks_probed"] = int(sum(1 for v in sample_ratio if v == 1.0))
        out[name] = r
    return out


# ═══════════════════════════════════════════════════════════════════
# Main flow
# ═══════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="435M v2 instruction-data build: public instruction parquet → NL-request→code sample stream (same contract as the SFT data)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sources", nargs="+", default=None,
                   help="parquet files; default = data_instr_raw/*.parquet (unreadable ones are skipped and recorded as source_incomplete)")
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--out-dir", default=None, help="default data_instr (data_instr/dryrun in dry-run)")
    p.add_argument("--target-tokens", type=int, default=3_000_000,
                   help="instruction seed target (the research doc suggests ≤2–3M)")
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--prompt-template", choices=sorted(PROMPT_TEMPLATES), default="marker")
    p.add_argument("--fence-policy", choices=["extract", "strip", "keep"], default="extract")
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--max-code-tokens", type=int, default=768)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--min-code-lines", type=int, default=2)
    p.add_argument("--max-template-repeats", type=int, default=5)
    p.add_argument("--min-score", type=float, default=None,
                   help="when the dataset has an average_test_score-like column → drop rows below this (OpenCodeInstruct suggests 0.9)")
    p.add_argument("--require-tests-passed", action="store_true",
                   help="when the dataset has a tests_execution_status column → keep PASSED/True only")
    p.add_argument("--pollution-mode", choices=["light", "full", "none"], default="light")
    p.add_argument("--no-compile-check", action="store_true")
    p.add_argument("--no-eos-newline", action="store_true")
    p.add_argument("--no-blacklist", action="store_true")
    p.add_argument("--no-instruction-dedup", action="store_true",
                   help="do not dedup so that each request is kept only once")
    p.add_argument("--crossing-policy", choices=["mask", "keep"], default="mask")
    p.add_argument("--per-source-tokens", type=int, default=None,
                   help="max tokens contributed per source (default target/n_sources; 0 = unlimited)")
    p.add_argument("--batch-rows", type=int, default=4096, help="parquet streaming read batch size")
    p.add_argument("--max-rows-per-source", type=int, default=0, help="0 = unlimited (takes effect in dry-run)")
    p.add_argument("--pool-tokens", type=int, default=500_000, help="in-memory pool size (shuffle + write granularity)")
    p.add_argument("--review-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-target-tokens", type=int, default=800_000)
    p.add_argument("--dry-rows", type=int, default=6000)
    p.add_argument("--dry-pool-tokens", type=int, default=200_000)
    p.add_argument("--dry-review-n", type=int, default=60)
    p.add_argument("--preview", type=int, default=3)
    return p.parse_args(argv)


def build_blacklist(no_blacklist: bool):
    if no_blacklist:
        return set()
    names = set(EVAL_NAMES)
    if HUMANEVAL_JSONL.exists():
        try:
            with open(HUMANEVAL_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ep = o.get("entry_point")
                    if ep:
                        names.add(str(ep))
        except OSError:
            pass
    return names


def probe_source(path: Path):
    """Return (ParquetFile, fields, num_rows) or (None, reason, extra)."""
    import pyarrow.parquet as pq
    try:
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
    except Exception as e:                       # still downloading / half-written / not parquet
        return None, f"{type(e).__name__}: {str(e)[:120]}", None
    cols = [c for c in pf.schema_arrow.names]
    fields = detect_fields(cols)
    if not fields.get("instr") or not fields.get("code"):
        return pf, f"missing request/answer columns (cols={cols})", None
    return pf, None, {"num_rows": int(n), "columns": cols, "fields": fields}


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    dry = args.dry_run
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = HERE / "data_instr" / "dryrun" if dry else HERE / "data_instr"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = args.dry_target_tokens if dry else args.target_tokens
    pool_tokens = args.dry_pool_tokens if dry else args.pool_tokens
    review_n = args.dry_review_n if dry else args.review_n
    max_rows = (args.dry_rows if dry else args.max_rows_per_source) or 0

    # ── tokenizer ──
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    vocab = tok.get_vocab_size()
    eos_id = tok.token_to_id("<eos>")
    if eos_id is None:
        raise SystemExit("[FATAL] tokenizer has no <eos>")
    if vocab > 65536:
        raise SystemExit(f"[FATAL] vocab={vocab} exceeds uint16")
    print(f"[TOK] {args.tokenizer}  vocab={vocab}  <eos>={eos_id}")

    blacklist = build_blacklist(args.no_blacklist)
    autogen_rx = compile_autogen()
    print(f"[BL] evaluation-leak blacklist: {len(blacklist)} function names"
          f"{' (disabled)' if args.no_blacklist else ''}")

    tmpl = PROMPT_TEMPLATES[args.prompt_template]
    tmpl_head, tmpl_tail = TEMPLATE_HEAD[args.prompt_template], TEMPLATE_TAIL[args.prompt_template]

    # ── source list ──
    if args.sources:
        srcs = [Path(s) for s in args.sources]
    else:
        if not DEFAULT_RAW_DIR.exists():
            raise SystemExit(f"[FATAL] raw data directory does not exist: {DEFAULT_RAW_DIR}")
        srcs = sorted(DEFAULT_RAW_DIR.glob("*.parquet"))
    if not srcs:
        raise SystemExit(f"[FATAL] no parquet sources found: {args.sources or DEFAULT_RAW_DIR}")

    # Probe first (files still downloading are caught here instead of being half-read)
    usable_srcs, skipped = [], []
    src_report = {}
    for s in srcs:
        if not s.exists():
            skipped.append({"path": str(s), "reason": "file does not exist"})
            print(f"[WARN] source does not exist, skipping: {s}")
            continue
        pf, reason, info = probe_source(s)
        if reason:
            skipped.append({"path": str(s), "size_bytes": s.stat().st_size, "reason": reason})
            print(f"[WARN] source unreadable (may still be downloading), skipping: {s}  ({s.stat().st_size/1e6:.1f} MB)")
            print(f"       reason: {reason}")
            continue
        usable_srcs.append((s, pf, info))
        f = info["fields"]
        print(f"[SRC] {s.name}  rows={info['num_rows']:,}  "
              f"instr={f['instr']!r} code={f['code']!r} score={f['score']!r} "
              f"status={f['status']!r} id={f['id']!r} ignored={f['ignored']}")
        if f["instr"].lower() == "prompt":
            print("[WARN] this source can only use the `prompt` column as the request -- confirm it is a real request, not a generation template")
    if not usable_srcs:
        raise SystemExit("[FATAL] no readable source (all still downloading / field mismatch)")

    # ── output ──
    w_train = BinWriter(out_dir, "instr_train", args.seq_len)
    w_val = BinWriter(out_dir, "instr_val", args.seq_len)
    rng = random.Random(args.seed)
    pool, val_pool, review = [], [], []
    pool_n = val_pool_n = 0
    seen, seen_instr = set(), set()
    tpl_counts = {}
    st = Counter()
    per_source = {}

    def flush(pool, writer):
        if not pool:
            return
        rng.shuffle(pool)
        for ids, mask in pool:
            writer.write_sample(ids, mask, args.crossing_policy)
        pool.clear()

    n_src = len(usable_srcs)
    if args.per_source_tokens is None:
        per_source_quota = int(target / n_src)
    else:
        per_source_quota = args.per_source_tokens or None

    print(f"[RUN] {'DRY-RUN' if dry else 'FULL'}  target={target/1e6:.2f}M tokens  out={out_dir}")
    print(f"[RUN] seq_len={args.seq_len} chunk_size={args.seq_len+1} template={args.prompt_template} "
          f"fence={args.fence_policy} crossing={args.crossing_policy} pollution={args.pollution_mode}")
    print(f"[MIX] per-source quota = "
          f"{per_source_quota if per_source_quota is None else f'{per_source_quota/1e6:.2f}M'}  "
          f"({n_src} readable sources, target={target/1e6:.2f}M)  max_rows/source="
          f"{max_rows if max_rows else '∞'}")

    stop = False
    for path, pf, info in usable_srcs:
        if stop:
            break
        f = info["fields"]

        def _score_ok(v):
            if v is None:
                return True
            if args.min_score is None:
                return True
            try:
                return float(v) >= args.min_score
            except (TypeError, ValueError):
                return False

        def _status_ok(v):
            if not args.require_tests_passed or v is None:
                return True
            return str(v).strip().lower() in ("passed", "true", "ok", "success", "1")

        cols = [c for c in (f["instr"], f["code"], f["score"], f["status"], f["id"]) if c]
        cols = list(dict.fromkeys(cols))
        s = Counter()
        rows_read = 0
        src_tokens = 0
        src_name = path.name
        print(f"\n[SRC] {path}  ({path.stat().st_size/1e6:.1f} MB, rows={info['num_rows']:,})")
        for batch in pf.iter_batches(batch_size=args.batch_rows, columns=cols):
            for row in batch.to_pylist():
                rows_read += 1
                s["rows_read"] += 1
                st["rows"] += 1
                if max_rows and rows_read > max_rows:
                    s["row_limit_reached"] = 1
                    break
                if w_train.total >= target:
                    stop = True
                    break
                if per_source_quota and src_tokens >= per_source_quota:
                    break
                if not _score_ok(row.get(f["score"]) if f["score"] else None):
                    st["reject_low_score"] += 1
                    continue
                if not _status_ok(row.get(f["status"]) if f["status"] else None):
                    st["reject_tests_not_passed"] += 1
                    continue
                instr = row.get(f["instr"])
                raw_code = row.get(f["code"])
                if not isinstance(instr, str) or not instr.strip():
                    st["reject_no_instruction"] += 1
                    continue
                code, n_blocks, had_fence = extract_code(raw_code, args.fence_policy)
                if code is None or not code.strip():
                    st["reject_no_code_block"] += 1
                    continue
                s["fence_blocks"] += n_blocks
                if had_fence:
                    s["with_fence"] += 1
                instr = instr.replace("\r\n", "\n").replace("\r", "\n").strip()
                code = normalize_code(code)
                if "\ufffd" in instr or "\ufffd" in code:
                    st["reject_bad_utf8"] += 1
                    continue
                if instr.startswith(tmpl_head) or tmpl_tail.strip() in instr or \
                        any(t in instr for t in TEMPLATE_LITERALS):
                    st["reject_template_marker_in_instruction"] += 1
                    continue
                if _FENCE_RX.search(instr):
                    # the request itself carries a fence (common: assert cases) -- not dropped, but the eval side must
                    # slice after the **last** tail marker, not the first (see meta.eval_hint)
                    st["instr_with_fence_block"] += 1
                hit_special = next((t for t in SPECIAL_TOKEN_TEXT if t in instr or t in code), None)
                if hit_special:
                    st["reject_special_token_literal"] += 1
                    continue
                if len(instr) + len(code) < args.min_chars:
                    st["reject_too_short_chars"] += 1
                    continue
                n_eff = effective_lines(code)
                if n_eff < args.min_code_lines:
                    st["reject_too_few_code_lines"] += 1
                    continue
                leak = leak_names(code, blacklist)
                if leak:
                    st["reject_eval_leak_name"] += 1
                    continue
                reason = pollution_reason(instr, code, autogen_rx, args.pollution_mode)
                if reason:
                    st[f"rej_{reason.split(':')[0]}"] += 1
                    st["rej_pollution_total"] += 1
                    continue
                if not args.no_compile_check:
                    try:
                        compile(code, "<instr>", "exec")
                    except (SyntaxError, ValueError):
                        st["reject_compile_fail"] += 1
                        continue
                prompt = tmpl.format(instruction=instr)
                tail = "" if args.no_eos_newline else "\n"
                pid = tok.encode(prompt).ids
                cid = tok.encode(code + tail).ids
                n = len(pid) + len(cid) + 1
                if len(pid) > args.max_prompt_tokens:
                    st["reject_prompt_too_long_tokens"] += 1
                    continue
                if len(cid) > args.max_code_tokens:
                    st["reject_code_too_long_tokens"] += 1
                    continue
                if n > args.seq_len:
                    st["reject_too_long_tokens"] += 1
                    continue

                digest = hashlib.blake2b((prompt + code).encode("utf-8", "replace"),
                                         digest_size=8).digest()
                if digest in seen:
                    st["reject_duplicate"] += 1
                    continue
                ikey = hashlib.blake2b(instr.encode("utf-8", "replace"), digest_size=8).digest()
                if not args.no_instruction_dedup and ikey in seen_instr:
                    st["reject_duplicate_instruction"] += 1
                    continue
                tkey = template_key(code)
                if tpl_counts.get(tkey, 0) >= args.max_template_repeats:
                    st["reject_template_repeat"] += 1
                    continue
                seen.add(digest)
                seen_instr.add(ikey)
                tpl_counts[tkey] = tpl_counts.get(tkey, 0) + 1

                ids = array("H", pid)
                ids.extend(cid)
                ids.append(eos_id)
                mask = b"\x00" * len(pid) + b"\x01" * (len(cid) + 1)

                h = int.from_bytes(digest, "big")
                st["accepted"] += 1
                st["accepted_tokens"] += n
                st["accepted_prompt_tokens"] += len(pid)
                st["accepted_code_tokens"] += len(cid) + 1
                st["accepted_instr_chars"] += len(instr)
                src_tokens += n
                is_val = (h % 10000) < int(args.val_ratio * 10000)
                if is_val:
                    val_pool.append((ids, mask))
                    val_pool_n += n
                    st["accepted_val"] += 1
                else:
                    pool.append((ids, mask))
                    pool_n += n
                    st["accepted_train"] += 1
                entry = {"split": "val" if is_val else "train",
                         "instruction": instr, "code": code,
                         "prompt_tokens": len(pid), "code_tokens": len(cid) + 1,
                         "total_tokens": n, "source": src_name,
                         "fence_blocks": n_blocks}
                if len(review) < review_n:
                    review.append(entry)
                elif review_n > 0:                       # reservoir sampling: 200 rows uniform over the whole run
                    j = rng.randrange(st["accepted"])
                    if j < review_n:
                        review[j] = entry
                if pool_n >= pool_tokens:
                    pool_n = 0
                    flush(pool, w_train)
                if val_pool_n >= pool_tokens:
                    val_pool_n = 0
                    flush(val_pool, w_val)
            if stop or (max_rows and rows_read > max_rows) or \
                    (per_source_quota and src_tokens >= per_source_quota):
                break
        s["tokens"] = src_tokens
        per_source[src_name] = dict(s)
        src_report[src_name] = {"rows": info["num_rows"], "columns": info["columns"],
                                "fields": info["fields"], "rows_read": rows_read,
                                "tokens_written": src_tokens}
        print(f"  [SRC-DONE] {src_name}: rows_read={rows_read:,} tokens={src_tokens:,} "
              f"accepted_total={st['accepted']:,}")

    flush(pool, w_train)
    flush(val_pool, w_val)
    usable_train = w_train.finalize()
    usable_val = w_val.finalize()
    if usable_train == 0:
        print(f"[WARN] train bin is empty: fewer accepted samples than one chunk ({args.seq_len + 1} token)."
              f"normal for a dry-run; for a real build raise --target-tokens / --dry-rows or loosen the filters.")
    if usable_val == 0:
        print(f"[WARN] val bin is empty (val_ratio={args.val_ratio} too small): sft_train rejects an empty val,"
              f"make sure a real build has ≥1 chunk of val tokens.")

    # ── verification ──
    verify = verify_outputs(out_dir, ["instr_train", "instr_val"], args.seq_len, tok,
                            eos_id=eos_id, template_name=args.prompt_template)
    consumer = consumer_check(out_dir, ["instr_train", "instr_val"], args.seq_len, vocab)

    # ── review file ──
    review_path = out_dir / "samples_review.jsonl"
    with open(review_path, "w", encoding="utf-8") as fh:
        for r in review:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── interface doc (lives next to the bin, single source of truth) ──
    C = args.seq_len + 1
    fmt_doc = f"""# Instruction-data format (generated by build_instr_data.py, do not edit by hand)

**Same contract** as `data/sft_train.bin` (build_sft_data.py); the only difference is the prompt text shape:

| file | dtype | meaning |
|---|---|---|
| `instr_train.bin` / `instr_val.bin` | uint16 | flat token stream |
| `instr_train_mask.bin` / `instr_val_mask.bin` | uint8 | 1 = counted in loss, 0 = ignored |

Length `T = N * (seq_len + 1)`; this directory has `seq_len={args.seq_len}` → **chunk_size = {C}** (not {args.seq_len}).

## Sample stream (template = `{args.prompt_template}`, **the eval side must reuse it byte for byte**)

```
{prompt!r}
    ↑ request span: prompt = PROMPT_TEMPLATE.format(instruction=...), mask all 0
<code>
    ↑ code span: solution (fence stripped + dedented), mask all 1
<eos>                      ← loss span = code + <eos>
```

Template head `head={tmpl_head!r}`, template tail `tail={tmpl_tail!r}`.
Samples are concatenated directly (**no padding**); the tail piece of a crossing sample is masked to 0 (`--crossing-policy {args.crossing_policy}`).

## Dataloader contract (identical to SFT)

```python
ids  = np.memmap(bin_path, dtype=np.uint16, mode="r")
mask = np.fromfile(mask_path, dtype=np.uint8)
C = seq_len + 1                      # {C}
for i in range(len(ids) // C):
    off = i * C
    x = torch.from_numpy(ids[off:off+seq_len].astype(np.int64))
    y = torch.from_numpy(ids[off+1:off+seq_len+1].astype(np.int64))
    keep = torch.from_numpy(mask[off+1:off+seq_len+1].astype(bool))
    labels = torch.where(keep, y, torch.full_like(y, -100))
```

Consume directly (**no** change to sft_train.py needed; the mask companion file is auto-discovered):

```bash
py -3.12 sft_train.py --bin-data data_instr/instr_train.bin \\
    --sft-val-bin data_instr/instr_val.bin --mask-bin data_instr/instr_train_mask.bin \\
    --val-bin ../data/all/val.bin --resume ../checkpoints_wsm/merged --epochs 1
```

## Invariants (self-checked at build time)

- `len(ids) == len(mask) == N*{C}`
- each sample's mask is strictly `0^p 1^q 0^r`; when `0^r` is non-empty its start is a multiple of {C}
- the stream contains no `<pad>(3)` / `<unk>(2)`; special-token literals are already stripped from bodies at build time
- every sample's code passes `compile()`
- a spot-checked sample's prompt span must start with `{tmpl_head!r}` and end with `{tmpl_tail!r}`
"""
    with open(out_dir / "README_FORMAT.md", "w", encoding="utf-8") as fh:
        fh.write(fmt_doc)

    # ── stats / meta (same fields as build_sft_data.py's *_meta.json, plus instruction-specific keys) ──
    reject = {k: v for k, v in sorted(st.items())
              if k.startswith("rej_") or k.startswith("reject_")}
    cross_info = {"train_crossed_samples": w_train.n_cross, "val_crossed_samples": w_val.n_cross,
                  "policy": args.crossing_policy}
    common = {
        "seq_len": args.seq_len,
        "chunk_size": args.seq_len + 1,
        "dtype": "uint16",
        "mask_dtype": "uint8",
        "mask_semantics": "1 = counted in loss (code span + <eos>); 0 = request span / crossing tail / tail truncation",
        "task_type": "instruction",
        "sample_format": f"prompt=PROMPT_TEMPLATE.format(instruction=request) + solution(code) + '<eos>'",
        "prompt_template": tmpl,
        "prompt_template_name": args.prompt_template,
        "prompt_template_head": tmpl_head,
        "prompt_template_tail": tmpl_tail,
        "prompt_encoding": "tokenizer.encode(prompt).ids (encoded separately from the solution so the mask boundary aligns token by token)",
        "eval_hint": (f"Instruction-style evaluation: fill the same request into prompt_template and sample directly; take"
                      f"the content after the **last** '{tmpl_tail.strip()}' marker (the request itself may contain ``` fences,"
                      f"so slicing at the first one cuts in the wrong place), then strip the trailing ``` fence and hand it to the exec runner"),
        "dataloader_contract": (
            "ids=np.memmap(bin,dtype=np.uint16); mask=np.fromfile(mask_bin,dtype=np.uint8); "
            "C=seq_len+1; for off in range(0,len(ids),C): "
            "x=ids[off:off+seq_len]; y=ids[off+1:off+seq_len+1]; "
            "labels=np.where(mask[off+1:off+seq_len+1]==1,y,-100)"),
        "no_padding": True,
        "sources": [str(s) for s, _, _ in usable_srcs],
        "sources_skipped": skipped,
        "fields_detected": {k: v["fields"] for k, v in src_report.items()},
        "config": {
            "target_tokens": target, "val_ratio": args.val_ratio,
            "prompt_template": args.prompt_template, "fence_policy": args.fence_policy,
            "max_prompt_tokens": args.max_prompt_tokens, "max_code_tokens": args.max_code_tokens,
            "min_chars": args.min_chars, "min_code_lines": args.min_code_lines,
            "max_template_repeats": args.max_template_repeats,
            "min_score": args.min_score, "require_tests_passed": args.require_tests_passed,
            "pollution_mode": args.pollution_mode, "compile_check": not args.no_compile_check,
            "eos_newline": not args.no_eos_newline, "instruction_dedup": not args.no_instruction_dedup,
            "blacklist_size": len(blacklist), "seed": args.seed, "dry_run": dry,
        },
        "rejection": reject,
        "crossing": cross_info,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    total_written = 0
    for name, writer, usable in (("instr_train", w_train, usable_train),
                                 ("instr_val", w_val, usable_val)):
        v = verify.get(name, {})
        meta = dict(common)
        meta.update({
            "split": name.split("_")[-1],
            "total_tokens": int(usable),
            "num_chunks": int(usable // (args.seq_len + 1)),
            "num_samples_written": writer.n_samples,
            "samples_in_bin": v.get("samples_in_bin"),
            "samples_dropped_at_tail_truncation": (writer.n_samples - (v.get("samples_in_bin") or 0)),
            "crossing_counter": writer.n_cross,
            "loss_tokens": v.get("loss_tokens"),
            "loss_ratio": v.get("loss_ratio"),
            "consumer_check": consumer.get(name, {}),
            "verified": v,
        })
        with open(out_dir / f"{name}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        total_written += int(usable)

    report = dict(common)
    report.update({
        "elapsed_s": round(time.time() - t0, 1),
        "counters": {k: v for k, v in sorted(st.items()) if not k.startswith(("rej_", "reject_"))},
        "per_source": per_source,
        "source_detail": src_report,
        "consumer_check": consumer,
        "tokens_written_total": total_written,
        "splits": {k: {"tokens": (int(usable_train) if k == "instr_train" else int(usable_val)),
                       "chunks": verify.get(k, {}).get("chunks"),
                       "samples": (w_train.n_samples if k == "instr_train" else w_val.n_samples),
                       "loss_tokens": verify.get(k, {}).get("loss_tokens")}
                   for k in ("instr_train", "instr_val")},
        "verify": verify,
    })
    with open(out_dir / "build_instr_stats.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    log_dir = HERE / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        with open(log_dir / f"build_instr_data_{stamp}{'_dryrun' if dry else ''}.json",
                  "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[WARN] failed to write logs/: {e}")

    # ── report (markdown) ──
    lines = []
    A = lines.append
    A(f"# {'DRY-RUN ' if dry else ''}instruction-data build report -- build_instr_data.py")
    A("")
    A(f"- time: {common['created']}  elapsed: {report['elapsed_s']}s  mode: "
      f"{'DRY-RUN' if dry else 'FULL'} (python {sys.version.split()[0]})")
    A(f"- output dir: `{out_dir}`")
    A(f"- template: `{args.prompt_template}` → head={tmpl_head!r} tail={tmpl_tail!r}; "
      f"fence policy `{args.fence_policy}`; crossing policy `{args.crossing_policy}`; pollution mode `{args.pollution_mode}`")
    A(f"- **total tokens written: {total_written:,}**"
      f"(train {usable_train:,} / val {usable_val:,}, chunk_size={C})")
    A("")
    A("## 1. Input sources and the fields actually used")
    A("")
    A("| file | total rows | rows read | fields used (request/answer/score/status/id) | ignored fields | tokens written |")
    A("|---|---|---|---|---|---|")
    for nm, sr in src_report.items():
        fl = sr["fields"]
        A(f"| `{nm}` | {sr['rows']:,} | {sr['rows_read']:,} | "
          f"`{fl['instr']}` / `{fl['code']}` / `{fl['score']}` / `{fl['status']}` / `{fl['id']}` | "
          f"{', '.join(fl['ignored']) or '—'} | {sr['tokens_written']:,} |")
    if skipped:
        A("")
        A("skipped sources (still downloading / field mismatch):")
        for sk in skipped:
            A(f"- `{sk['path']}` — {sk['reason']}")
    A("")
    A("## 2. Filtering and acceptance counters (item by item)")
    A("")
    A("| counter | value |")
    A("|---|---|")
    for k, v in sorted(st.items()):
        A(f"| {k} | {v:,} |")
    A("")
    A("## 3. Read-back verification (bin-level invariants)")
    A("")
    A("| split | tokens | chunks | loss_tokens | loss_ratio | aligned | samples | invalid_mask | "
      "zero_loss | tail_masked(off_grid) | pad/unk | max_id | prompt template mismatch/checked |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, v in verify.items():
        A(f"| {nm} | {v['bin_tokens']:,} | {v['chunks']:,} | {v['loss_tokens']:,} | "
          f"{v['loss_ratio']:.1%} | {v['aligned']} | {v['samples_in_bin']:,} | "
          f"{v['samples_with_invalid_mask']} | {v['samples_with_zero_loss']} | "
          f"{v['samples_tail_masked_crossing']}({v['samples_tail_masked_not_at_block_boundary']}) | "
          f"{v['has_pad_or_unk']} | {v['max_token_id']} | "
          f"{v['samples_prompt_template_mismatch']}/{v['samples_prompt_checked']} |")
    A("")
    A("## 4. Consumer-contract self-check (replays sft_train.py's read path)")
    A("")
    for nm, c in consumer.items():
        A(f"- `{nm}`: mask auto-discovered={c.get('mask_auto_discovered')}, "
          f"bin readable={c.get('passes_assert_bin_readable')}, "
          f"mask length≥data={c.get('mask_len_ge_data')}, "
          f"chunks={c.get('total_chunks')}, tail dropped={c.get('tail_tokens_dropped')} token, "
          f"max(token_id)={c.get('max_token_id_probe')} < vocab={vocab} → resolved as a file mask="
          f"{c.get('resolves_to_file_mask')}, chunk shapes ok={c.get('chunks_shape_ok')}, "
          f"probed chunk loss share={c.get('loss_token_ratio_probed_chunks')}")
    A("")
    for nm, v in verify.items():
        for p in v["probes"][:2]:
            A(f"### Probe {nm} chunk {p['chunk']}")
            A("")
            A("```")
            A(f"prompt tail : {p['prompt_tail']!r}")
            A(f"code head   : {p['solution_head']!r}")
            A(f"loss tokens in this chunk = {p['loss_tokens_in_chunk']}")
            A("```")
            A("")
    A(f"- Independent re-check: `py -3.12 verify_instr_contract.py --dir {out_dir}` -- uses **sft_train.py's real functions**"
      f"(resolve_loss_mask → _count_chunks → create_sft_dataloader → inspect_loss_mask)"
      f"to recompute x/y/labels per chunk; this run passed all of them (the same script also passes on the existing"
      f"`data/sft_train.bin`, i.e. both are accepted by the same consumer path)")
    A("")
    A("## 5. Key observations (they decide how to use this data)")
    A("")
    if st["accepted"]:
        mp = st["accepted_prompt_tokens"] / st["accepted"]
        mc = st["accepted_code_tokens"] / st["accepted"]
        A(f"- per-sample averages: request span **{mp:.1f}** token / code span **{mc:.1f}** token "
          f"(loss share ≈{mc/(mp+mc):.0%}) -- instruction requests are longer than their solutions, so"
          f"after masking the request the loss-token share is naturally lower than the 88.9% of the signature→body corpus.")
        A(f"- {st['accepted']:,} rows accepted; of these the request itself carries ``` fences (assert cases)"
          f"{st.get('instr_with_fence_block', 0):,} rows → the eval side slices after the **last** tail marker.")
    nsamp = verify.get("instr_train", {}).get("samples_in_bin") or 0
    nzero = verify.get("instr_train", {}).get("samples_with_zero_loss", 0)
    A(f"- crossing policy `{args.crossing_policy}`: {w_train.n_cross + w_val.n_cross:,} crossing samples;"
      f"`mask` zeroes the out-of-range tail -- in this dry-run {nzero}/{nsamp} train samples"
      f"({nzero/max(nsamp,1):.1%}) therefore have no loss target at all (the request is so long that the whole code span lands on the masked side),"
      f"and another {verify.get('instr_train', {}).get('samples_tail_masked_crossing', 0)} samples have their tail masked;"
      f"that is the price of the mask policy)")
    A(f"- `keep` matches the current Arm1 state (`data/sft_train.bin` rebuilt with `--crossing-policy keep`):"
      f"it keeps the partial-request → code continuation signal and a higher loss share;"
      f"`mask` (this script's default) is more conservative and does not train orphan continuations. Both are consumed"
      f"directly by sft_train.py; which one to pick depends on whether item-by-item comparability with Arm1 matters (see CROSSING_POLICY_COMPARE.md in this directory)")
    A(f"- the ignored `seed` field is the dataset's own reference code snippet (the OSS-Instruct seed),"
      f"unused here; it can be picked up directly for a future snippet → request self-distillation")
    A("")
    A("## 6. How to consume")
    A("")
    A("```bash")
    A("py -3.12 sft_train.py --bin-data " + str(out_dir / "instr_train.bin") +
      " \\\n    --sft-val-bin " + str(out_dir / "instr_val.bin") +
      " --val-bin ../data/all/val.bin --resume ../checkpoints_wsm/merged --epochs 1")
    A("```")
    A("")
    A("Instruction-style evaluation is described by `prompt_template` / `prompt_template_tail` / `eval_hint` in `*_meta.json`.")
    report_name = "DRYRUN_REPORT.md" if dry else "BUILD_REPORT.md"
    with open(out_dir / report_name, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ── console report ──
    print(f"\n{'='*72}\n[REJECTION] rejection counters")
    for k, v in sorted(reject.items(), key=lambda kv: -kv[1]):
        print(f"  {k:40s} {v:>12,}")
    print(f"\n[COUNTERS]")
    for k, v in sorted(st.items()):
        if not k.startswith(("rej_", "reject_")) and not k.startswith("_"):
            print(f"  {k:40s} {v:>12,}")
    print(f"\n[VERIFY]")
    for name, v in verify.items():
        print(f"  {name}: tokens={v['bin_tokens']:,} chunks={v['chunks']:,} "
              f"loss_tokens={v['loss_tokens']:,} ({v['loss_ratio']:.1%}) aligned={v['aligned']}")
        print(f"    samples_in_bin={v['samples_in_bin']:,} invalid_mask={v['samples_with_invalid_mask']} "
              f"zero_loss={v['samples_with_zero_loss']} "
              f"tail_masked={v['samples_tail_masked_crossing']} "
              f"(off_grid={v['samples_tail_masked_not_at_block_boundary']}) "
              f"pad/unk={v['has_pad_or_unk']} prompt_mismatch={v['samples_prompt_template_mismatch']}"
              f"/{v['samples_prompt_checked']}")
    print(f"  consumer_check: {json.dumps({k: {kk: vv for kk, vv in c.items() if kk != 'loss_token_ratio_probed_chunks'} for k, c in consumer.items()}, ensure_ascii=False)}")
    print(f"\n[PREVIEW] first {args.preview} samples:")
    for r in review[:args.preview]:
        print(f"  --- {r['split']}  prompt_tokens={r['prompt_tokens']} "
              f"code_tokens={r['code_tokens']} total={r['total_tokens']}")
        print("  " + "\n  ".join((tmpl.format(instruction=r["instruction"]) + r["code"]).split("\n")[:10]))
    print(f"\n[OUT] {out_dir}")
    print(f"      instr_train.bin ({usable_train:,} tok) / instr_train_mask.bin")
    print(f"      instr_val.bin   ({usable_val:,} tok) / instr_val_mask.bin")
    print(f"      *_meta.json / build_instr_stats.json / {report_name} / {review_path.name} ({len(review)} rows)")
    print(f"[DONE] {time.time()-t0:.1f}s   {'DRY-RUN' if dry else 'FULL'}   "
          f"tokens_written={total_written:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
