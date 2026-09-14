#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fastgen.py — a faster autoregressive generation path that is **semantically equivalent** to eval_sft.generate_one
=========================================================================
Context: `esd_sample.py` calls `eval_sft.generate_one` directly; on an RTX 5090 that needs ~4 s for
256 tokens (≈65 tok/s) — unreasonably slow for a 435M model.

This file **modifies no** existing file (eval_sft.py / esd_sample.py / sft_train.py / data* /
ckpt* are all reused read-only). It only adds a new generation path that is token-by-token equivalent
(same seed; under greedy settings the output matches token by token), in three modes:

    --verify-equivalence   CPU + randomly initialized tiny model, token-by-token vs eval_sft.generate_one
    --bench                tokens/s on a real ckpt (**requires** an explicit --device cuda)
    --gen                  **CLI of the same shape** as esd_sample.py (--problems/--out/--k/--limit/…)

═══════════════════════════════════════════════════════════════════════════
1. Where the baseline is slow (two "structural" bottlenecks read out of eval_sft.generate_one)
═══════════════════════════════════════════════════════════════════════════

(a) **No KV cache**: every step calls `model(input_ids)`, recomputing the **whole prefix** from embed to
    lm_head. Step t costs ∝ t, so the whole sequence costs Σt ≈ L²/2 — for 256 new tokens (+ ~50
    prompt tokens) that is ~150× more compute than needed. This is wasted **compute**, independent of device.

(b) **repetition_penalty applied token by token in Python**:
        for tid in set(ids):
            if next_logits[tid] < 0: next_logits[tid] *= rp
            else:                    next_logits[tid] /= rp
    After 200 generated tokens `set(ids)` holds 200+ elements; `next_logits[tid] < 0` is a
    **0-dim tensor comparison → one host↔device sync per element**. That means hundreds of extra
    device→host round-trips plus hundreds of kernel launches (~10–20 µs each) → milliseconds per token,
    while the GPU sits completely idle. This is wasted **synchronization**.

A few smaller ones: `torch.cat` rebuilds input_ids every step, `.item()` syncs once per step (unavoidable,
it is bound to the generation length), and there is no `torch.inference_mode()`.

═══════════════════════════════════════════════════════════════════════════
2. What this file optimizes (each item can be disabled on its own, see --no-kv-cache / --no-tensor-params)
═══════════════════════════════════════════════════════════════════════════

1) **Preallocated KV cache + in-place writes** (disable with --no-kv-cache)
   Allocate the (L, B, H, cap, hd) K/V tensors once (cap = config.max_seq_len);
   each step writes only the new token's K/V into slot pos with `index_copy_`, and attention reads the cache.
   Per-token compute goes from O(t) to O(1) → the whole sequence from O(L²) to O(L).
   Prompt processing (prefill) is still one full-length forward pass, **bitwise identical to the baseline's
   first step** (same operators, same shapes, same is_causal=True SDPA, plus one cache write and read-back).

2) **Sampling parameters as tensors** (disable with --no-tensor-params)
   Maintain a vocab-sized bool mask `seen` (prompt + generated tokens).
        adj = where(raw < 0, raw * rp, raw / rp)      # computed over the whole vocab in one go
        next_logits = where(seen, adj, raw)           # applies only to tokens already seen
   Values are **bitwise identical** to the baseline (each branch uses the same operator: multiply negatives,
   divide positives — no reciprocal approximation), but it collapses "hundreds of syncs + hundreds of launches per token" into 2 tensor ops.
   Note `set(ids)` has **dedup** semantics; the mask dedups by design, so never apply the penalty once per repeated id.

3) **No synchronization inside the loop**: see `_sample()`. All that remains in the loop is
   (i) one `.item()` (the token value is needed to update the mask / test early stopping — cannot be removed),
   (ii) one scalar write `seen[tok]=True`, (iii) one `mask[..., pos]=True`.
   No `.cpu()` / `.tolist()` / element-wise indexing.

4) **inference_mode + eval** (disable with --no-inference-mode): no gradients, no version counters.

5) **Optional torch.compile (mode="reduce-overhead" → CUDA graphs)** (--compile)
   Once the KV cache brings per-token compute down to ~0.9 GFLOP, **kernel launch latency** becomes the new
   bottleneck (22 layers × ~10 kernels × ~5–8 µs ≈ 1–2 ms/token). CUDA graphs capture the whole
   single decode step into one graph, so one replay costs a single launch latency.
   Note: **in this repo reduce-overhead was actually slower during training** (training has long sequences,
   backward and the optimizer; graph capture fragments and every step differs in shape/randomness), while a single
   decode step looks completely different (fixed shapes, no backward, no dropout) — do not extrapolate → measure it (--bench).
   To be capturable by CUDA graphs, the decode step must have **static shapes**:
     * Cache writes use `index_copy_(dim=2, index=pos_buf)`, with pos a tensor rather than an int
       (an int would be guarded by torch.compile → recompile every step)
     * Cache reads use `decode_read="mask"`: read the whole cap-length cache + a bool mask,
       instead of a `[:n]` slice (dynamic n → dynamic shapes → not capturable). The mask flips only one
       scalar bit per step, cost described below.
     * All input buffers (tok_buf / pos_buf / mask / cache) are persistent tensors;
       graph capture records pointers, so in-place updates are visible to replay.
   Measured (CPU, counting backend='eager'): 16 decode steps compile **exactly 1 frame** → a tensor pos
   does avoid "a new shape every step → recompile every step".
   ⚠️ This machine has no C++ compiler (MSVC cl), so Inductor/CUDA graphs **cannot be verified
      on this machine**; on a failed first call the code warns explicitly and falls back to eager (numerics and semantics unchanged).
      Verify on real hardware: `--bench --device cuda` (look at the tok/s on the D row).

5b) **Two ways to read the cache in decode** (--decode-read; the measured difference matters a lot)
   * `slice` (default, eager only): read only the first n keys (`kc[:, :, :n, :]`),
     with a single query row → `is_causal=False`.
     Measured: against the last row of the baseline's full-length forward it is **32/32 bitwise identical in
     bf16** and differs by ~1 ulp in fp32. This is **numerically the read mode closest to the baseline**.
     ⚠️ Never pass `is_causal=True` for "1 query row + n key rows": PyTorch aligns the mask by the
     non-square rule and the measured result is completely wrong (Δ≈2.6) — a semantic error, not a precision issue.
   * `mask` (automatic with --compile): read the whole cap-sized cache + a bool mask → static shapes,
     which CUDA graphs require. Cost: the mask sends SDPA down a different kernel, and in bf16 roughly
     1/3 of cases show attention differences of 1e-4–1e-2 (fp32 is still only ~1 ulp).
     Measured CPU throughput is on par with slice (even slightly faster), so use it only when static shapes are needed.

**Not done** (important): batching prompts of different lengths together — this model's
attention has no padding mask (`F.scaled_dot_product_attention(..., is_causal=True)` offers
no protection against padding at all), so padding leaks straight into attention. This project has already hit that bug.

═══════════════════════════════════════════════════════════════════════════
3. Boundaries of semantic equivalence (stated honestly; measurements in fastgen_report.md)
═══════════════════════════════════════════════════════════════════════════
* **Sampling logic** (temperature / repetition_penalty / top_k / min_p / multinomial /
  early-stop condition) is copied operator by operator from the baseline with bitwise identical values; with `--no-kv-cache` the logits
  of both paths are bitwise identical too → in that case **even randomly sampled token sequences match token by token** (measured
  144/144, including random sampling at T=0.7/1.0).
* **Prefill is bitwise identical**: the full-prompt forward pass uses the same operator chain and the same shapes as the baseline's first step
  (measured max|Δ| = 0.000e+00, 4/4 configurations).
* For the **KV-cache decode step** the only remaining difference is "M=1 vs M=t shapes":
    - fp32: max|Δlogits| ≈ 1e-6 (that is 1 ulp) → 72/72 token-level match measured (including random sampling)
    - bf16: in a few cases Δ reaches tens of bf16 ulps (measured max 3.4), and random sampling is sensitive
      to RNG consumption order → 4 of 72 cases diverge after some steps; **under the equivalent-greedy setting 40/40 match**
    - Note this is unavoidable for **any** KV cache implementation (if the baseline recomputes the same prefix with a
      different batch shape, its result changes too); this file's contribution is to quantify that difference and pick
      the read mode closest to the baseline.
* **The temperature=0 trap**: the baseline's `logits / temperature` at temperature=0 is
  `x/0` → ±inf/NaN, softmax after top-k yields NaN, and the `min_p` filter removes every NaN →
  `probs.numel()==0` → **break, returning an empty string directly** (measured: the baseline returns an empty string in 32/32
  T=0 cases). So "temperature=0 (greedy)" is really a NaN degenerate path in the baseline, and both paths
  returning "" only counts as trivial equivalence (this file verifies those 32/32 anyway).
  Real "equivalent greedy" uses a tiny temperature (default 1e-6): logits/1e-6 magnifies the top-1 vs
  top-2 gap, softmax saturates to one-hot, and multinomial necessarily picks the argmax.
  --verify-equivalence runs both `0` and `1e-06` by default and requires agreement under both.
* **prompt+max_new > context limit**: every baseline step truncates input_ids to the last cap tokens
  (rope positions restart from 0 accordingly). fastgen takes the "recompute the whole window" branch; measured 16/16 match
  (including Δ=0.000e+00 in bf16). The cost is that this path degrades to the baseline's order of magnitude (recomputing the whole window each step),
  and it is only O(L) while prompt+max_new ≤ cap.

Usage
----
    # equivalence self-check (CPU + random tiny model; no GPU, no ckpt loaded)
    py -3.12 fastgen.py --verify-equivalence

    # tokens/s on a real ckpt (an explicit --device cuda is required; do not run while the GPU is busy training)
    py -3.12 fastgen.py --bench --device cuda --ckpt ckpt_a5/final

    # emit candidates with the same CLI shape as esd_sample.py (drop-in replacement)
    py -3.12 fastgen.py --gen --ckpt ckpt_a5/final --problems esd/problems.jsonl \
        --out esd/cand_a5_fast.jsonl --k 4 --temperature 0.7 --device cuda
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:  # Windows console encoding
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import eval_sft as E        # noqa: E402  reuse CodeLLM / ModelConfig / apply_rope / generate_one read-only
import build_instr_data as BI   # noqa: E402  single source of truth for prompt templates

DEFAULT_TOKENIZER = HERE.parent / "tokenizer" / "tokenizer.json"
REF_SIG_ARGS = ("max_new", "temperature", "top_k", "repetition_penalty", "min_p", "device")


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def _check_ref_signature():
    """Static contract check: fails immediately if eval_sft.generate_one changes signature or name."""
    sig = inspect.signature(E.generate_one).parameters
    missing = [k for k in REF_SIG_ARGS if k not in sig]
    if missing:
        raise SystemExit(
            f"[ERR] eval_sft.generate_one has no such parameters: {missing}\n"
            f"      actual signature: {list(sig)}\n"
            f"      → eval_sft.py changed; fastgen's equivalence claims must be re-verified.")
    return list(sig)


# ═══════════════════════════════════════════════════════════════════
# 1. KV cache generator
# ═══════════════════════════════════════════════════════════════════
class FastGen:
    """Replaces `eval_sft.generate_one`'s per-step full-prefix recomputation with incremental KV cache decoding.

    Semantic contract: the sampling part is copied operator by operator from the baseline; decode-step logits are
    mathematically equivalent to the baseline (they may differ in the last bit in floating point, see "Boundaries of semantic equivalence" at the top of the file).

    Each optimization can be disabled on its own, which makes --bench / --bench-cpu ablations possible:
        use_kv_cache=False      → fall back to "full-prefix forward per step" (= baseline compute), keeping only the sampling optimizations
        tensor_params=False     → fall back to the baseline's per-token Python penalty loop
        compile_mode="none"     → no compilation
        use_inference_mode=False→ fall back to no_grad
    """

    def __init__(self, model, device="cpu", *, use_kv_cache=True, tensor_params=True,
                 compile_mode="none", use_inference_mode=True, cache_len=None,
                 decode_read=None, compile_backend=None, warn=eprint):
        self.model = model
        self.device = torch.device(device)
        self.use_kv_cache = bool(use_kv_cache)
        self.tensor_params = bool(tensor_params)
        self.compile_mode = compile_mode
        self.use_inference_mode = bool(use_inference_mode)
        self._warn = warn
        self.trace = None            # when set to a list, records per-step logits (for verification)

        cfg = model.config
        self.H = int(cfg.num_heads)
        self.hd = int(cfg.head_dim)
        self.L = int(cfg.num_layers)
        self.cap = int(cache_len or cfg.max_seq_len)
        if self.cap > int(cfg.max_seq_len):
            raise SystemExit(f"[ERR] --cache-len {self.cap} exceeds the model max context "
                             f"{cfg.max_seq_len}; the excess cannot be equivalent to the baseline.")
        self.dtype = next(model.parameters()).dtype
        self.model.eval()

        # ── persistent buffers: CUDA graphs need stable pointers, so allocate once here ──
        n = 1                          # B=1 (the baseline also generates one sequence at a time)
        self._k = [torch.zeros(n, self.H, self.cap, self.hd, dtype=self.dtype, device=self.device)
                   for _ in range(self.L)]
        self._v = [torch.zeros(n, self.H, self.cap, self.hd, dtype=self.dtype, device=self.device)
                   for _ in range(self.L)]
        # Two ways to read the cache in the decode step (measurements in fastgen_report.md):
        #   "slice": read only the first n keys (a [:n] view, dynamic length) — in bf16 it is
        #            **bitwise identical** to the baseline's full-length forward (CPU measured 32/32), ~1 ulp off in fp32. Eager only.
        #   "mask" : read the whole cap-length cache + a bool mask (static shapes) — required by CUDA graphs,
        #            at the cost of sending SDPA down a different kernel (in bf16 about 1/3 of cases show
        #            attention differences of ~1e-4–1e-2).
        self.decode_read = decode_read or ("mask" if compile_mode != "none" else "slice")
        self._read_explicit = decode_read is not None
        self._n_valid = 0
        # the decode step reads the whole cache + a bool mask (static shapes; True = participates in attention)
        self._mask = torch.zeros(1, 1, 1, self.cap, dtype=torch.bool, device=self.device)
        self._tok_buf = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self._pos_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self._pos_arange = torch.arange(self.cap, dtype=torch.long, device=self.device)

        self._decode_fwd = self._decode_step
        self.compile_failed = None
        self._compile_pending = False
        if compile_mode != "none" and self.use_kv_cache:
            backend = compile_backend
            mode = compile_mode
            if compile_mode == "eager" and backend is None:
                # dynamo-only: needs no C++ compiler (this machine has no MSVC cl → Inductor unavailable)
                # used on compiler-less machines to check "can dynamo capture it / does it recompile per step / do results match"
                backend, mode = "eager", None
            elif compile_mode == "eager":
                mode = None
            try:
                if backend is not None:
                    self._decode_fwd = torch.compile(self._decode_step, backend=backend)
                else:
                    self._decode_fwd = torch.compile(self._decode_step, mode=mode, fullgraph=False)
                self._compile_pending = True      # torch.compile is lazy: only the first call reveals whether it works
            except Exception as e:                # pragma: no cover - depends on the environment
                self.compile_failed = f"{type(e).__name__}: {e}"
                self._warn(f"[fastgen] ⚠️ torch.compile(mode={compile_mode}) unavailable → falling back to eager: "
                           f"{self.compile_failed}")
                self._decode_fwd = self._decode_step
                self.compile_mode = "none"

    # ── forward pass: reuses the model's own submodules (same weights/operators as the baseline) ──
    def _attn(self, attn, x, cos_sel, sin_sel, layer, pos_idx, key_len, causal):
        B, T, D = x.shape
        qkv = attn.qkv(x).view(B, T, self.H, 3, self.hd)
        q = E.apply_rope(qkv[..., 0, :], cos_sel, sin_sel)   # same function and same cos/sin values as the baseline
        k = E.apply_rope(qkv[..., 1, :], cos_sel, sin_sel)
        v = qkv[..., 2, :]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        self._k[layer].index_copy_(2, pos_idx, k)            # preallocated + in-place write
        self._v[layer].index_copy_(2, pos_idx, v)
        if causal:      # prefill: slicing + is_causal=True — identical parameters to the baseline CausalSelfAttention
            out = F.scaled_dot_product_attention(
                q, self._k[layer][:, :, :key_len, :], self._v[layer][:, :, :key_len, :],
                is_causal=True)
        elif self.decode_read == "mask":
            # static shapes (capturable by CUDA graphs): read the whole cache + bool mask
            out = F.scaled_dot_product_attention(q, self._k[layer], self._v[layer],
                                                 attn_mask=self._mask)
        else:
            # read only the valid prefix; a single query row → is_causal=False (causality is enforced by the key length)
            # ⚠️ never pass is_causal=True for 1 query row + n key rows: PyTorch aligns the mask by the
            #    non-square rule (measured result completely wrong, Δ≈2.6)
            n = self._n_valid
            out = F.scaled_dot_product_attention(
                q, self._k[layer][:, :, :n, :], self._v[layer][:, :, :n, :], is_causal=False)
        return attn.out_proj(out.transpose(1, 2).reshape(B, T, D))

    def _blocks(self, tok, pos_idx, key_len, causal, cos_sel, sin_sel):
        cfg = self.model.config
        x = self.model.embed(tok) * (cfg.d_model ** 0.5)
        for i, blk in enumerate(self.model.blocks):
            x = x + self._attn(blk.attn, blk.attn_norm(x), cos_sel, sin_sel,
                               i, pos_idx, key_len, causal)
            x = x + blk.ffn(blk.ffn_norm(x))
        return self.model.lm_head(self.model.final_norm(x))

    # ── two entry points: prefill (T=P, once) and decode (T=1, once per token) ──
    def prefill(self, ids):
        """Full-prefix forward pass + cache fill. Operators/shapes/mask identical to the baseline's first step."""
        tok = torch.tensor([ids], dtype=torch.long, device=self.device)
        P = tok.shape[1]
        pos_idx = self._pos_arange[:P]
        cos_sel = self.model.cos.index_select(0, pos_idx)   # == cos[:P], bitwise identical values
        sin_sel = self.model.sin.index_select(0, pos_idx)
        # decode mask: the first P slots are valid, everything after is masked out (stale values from the previous round)
        self._mask.fill_(False)
        self._mask[:, :, :, :P] = True
        logits = self._blocks(tok, pos_idx, key_len=P, causal=True, cos_sel=cos_sel, sin_sel=sin_sel)
        return logits

    def decode(self, pos):
        """Writes the single token in self._tok_buf into cache slot pos and computes logits."""
        cos_sel = self.model.cos.index_select(0, self._pos_buf)
        sin_sel = self.model.sin.index_select(0, self._pos_buf)
        if self._compile_pending:
            # torch.compile is lazy: failure can only surface on the first call (e.g. no MSVC cl here)
            self._compile_pending = False
            try:
                return self._decode_fwd(cos_sel, sin_sel)
            except Exception as e:            # pragma: no cover - depends on the environment
                self.compile_failed = f"{type(e).__name__}: {str(e)[:200]}"
                self._warn(f"[fastgen] ⚠️ torch.compile(mode={self.compile_mode}) failed on the first call → "
                           f"falling back to eager (numerics and semantics unchanged): {self.compile_failed}")
                self.compile_mode = "none"
                self._decode_fwd = self._decode_step
                if not self._read_explicit and self.decode_read == "mask":
                    # mask is only a static-shape compromise (needed by CUDA graphs); with eager there is no reason to keep it:
                    # the slice read (valid prefix only) is both faster and numerically closer to the baseline
                    self.decode_read = "slice"
                    self._warn("[fastgen] ⚠️ compile unavailable → decode cache read falls back to slice (valid prefix only)")
        return self._decode_fwd(cos_sel, sin_sel)

    def _decode_step(self, cos_sel, sin_sel):
        # the arguments contain **no** Python int / dynamic shape → no guard that would recompile every step
        pos_idx = self._pos_buf
        return self._blocks(self._tok_buf, pos_idx, key_len=0, causal=False,
                            cos_sel=cos_sel, sin_sel=sin_sel)

    # ── sampling: operator by operator identical to eval_sft.generate_one, just tensorized ────────
    @staticmethod
    def _sample(next_logits, seen, top_k, min_p):
        """Returns (tok 0-dim tensor, whether min_p filtered everything out). Copies the baseline's operator order."""
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, min(top_k, next_logits.numel()))
            probs = F.softmax(topk_vals, dim=-1)
            if min_p > 0:
                keep = probs >= min_p * probs.max()
                probs = probs[keep]
                topk_idx = topk_idx[keep]
                if probs.numel() == 0:
                    return None, True
                probs = probs / probs.sum()
            return topk_idx[torch.multinomial(probs, 1)], False
        probs = F.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, 1), False

    # ── main loop: maps one-to-one onto eval_sft.generate_one's control flow ───────────
    def generate_ids(self, tokenizer, prompt, seed, max_new=256, temperature=0.2,
                     top_k=50, repetition_penalty=1.2, min_p=0.05):
        if seed is not None:
            torch.manual_seed(seed)                       # same seed handling as the baseline
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)

        pad_id = tokenizer.token_to_id("<pad>") or 0      # the baseline has `or 0` here; copied verbatim
        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is None:
            eos_id = tokenizer.token_to_id("</s>")

        ids = list(tokenizer.encode(prompt).ids)
        V = int(self.model.config.vocab_size)
        cap = self.cap
        # tensorized repetition_penalty: the equivalent of `set(ids)` (dedup)
        seen = torch.zeros(V, dtype=torch.bool, device=self.device)
        if ids:
            seen[torch.tensor(ids, dtype=torch.long, device=self.device)] = True

        generated = []
        n_cached = 0
        prefilled = False
        ctx = torch.inference_mode() if self.use_inference_mode else torch.no_grad()

        with ctx:
            for _ in range(max_new):
                L = len(ids)
                window = min(L, cap)
                if (not self.use_kv_cache):
                    # ablation baseline: full-prefix forward every step (= baseline compute), sampling optimizations only
                    tok = torch.tensor([ids[-window:]], dtype=torch.long, device=self.device)
                    logits = self.model(tok)
                    n_cached = window
                elif (not prefilled) or not (window == L and n_cached == L - 1):
                    # first step / sliding window (prompt+max_new > cap; the baseline truncates the prefix) → recompute everything
                    logits = self.prefill(ids[-window:])
                    n_cached = window
                    prefilled = True
                else:
                    # normal path: feed only the previous token into the single decode step
                    self._pos_buf[0] = n_cached
                    self._n_valid = n_cached + 1              # for the "slice" read: number of valid keys
                    if self.decode_read == "mask":            # for the "mask" read: mark one slot valid
                        self._mask[0, 0, 0, n_cached] = True
                    logits = self.decode(n_cached)
                    n_cached += 1
                if self.trace is not None:
                    self.trace.append(logits[0, -1, :].float().clone())

                next_logits = logits[0, -1, :].float() / temperature
                if self.tensor_params:
                    if repetition_penalty != 1.0:
                        # bitwise identical to the baseline's per-token loop: multiply negatives, divide positives (no reciprocal)
                        adj = torch.where(next_logits < 0,
                                          next_logits * repetition_penalty,
                                          next_logits / repetition_penalty)
                        next_logits = torch.where(seen, adj, next_logits)
                else:
                    for tid in set(ids):                     # the baseline's literal implementation (for ablation)
                        if next_logits[tid] < 0:
                            next_logits[tid] *= repetition_penalty
                        else:
                            next_logits[tid] /= repetition_penalty

                tok_t, empty = self._sample(next_logits, seen, top_k, min_p)
                if empty:
                    break                                    # baseline: probs.numel()==0 → break
                tok = tok_t.item()                           # the only synchronization point in the loop (required)

                if tok == pad_id or (eos_id is not None and tok == eos_id):
                    break
                generated.append(tok)
                ids.append(tok)
                if seen.dim() > 0:
                    seen[tok] = True                         # in-place scalar write, no sync
                self._tok_buf[0, 0] = tok                    # for the next decode step
                if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                    break
        return generated

    def generate(self, tokenizer, prompt, seed, **kw):
        gen = self.generate_ids(tokenizer, prompt, seed, **kw)
        return tokenizer.decode(gen) if gen else ""


def fast_generate_one(model, tokenizer, prompt, seed, max_new=256, temperature=0.2,
                      top_k=50, repetition_penalty=1.2, min_p=0.05, device="cuda",
                      **fast_kw):
    """Drop-in version with the same signature/semantics as `eval_sft.generate_one` (caches FastGen per model)."""
    fg = getattr(model, "_fastgen", None)
    if fg is None or fg.device.type != str(device):
        fg = FastGen(model, device, **fast_kw)
        model._fastgen = fg
    return fg.generate(tokenizer, prompt, seed, max_new=max_new, temperature=temperature,
                       top_k=top_k, repetition_penalty=repetition_penalty, min_p=min_p)


# ═══════════════════════════════════════════════════════════════════
# 2. Equivalence verification (CPU + random tiny model; no ckpt, no GPU)
# ═══════════════════════════════════════════════════════════════════
class DummyTokenizer:
    """For equivalence verification only: encode treats "1 2 3" as an id sequence; decode inverts it (injective).

    Hence "decode(generated) strings are equal" ⟺ "the generated token sequences are identical".
    pad/eos use very high ids so a randomly initialized model does not hit early stopping immediately and make the test trivial.
    """

    def __init__(self, vocab):
        self.vocab = int(vocab)
        self._pad = self.vocab - 1
        self._eos = self.vocab - 2

    def token_to_id(self, s):
        return {"<pad>": self._pad, "<eos>": self._eos, "</s>": self._eos}.get(s)

    def encode(self, text):
        return SimpleNamespace(ids=[int(x) for x in str(text).split()])

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


TINY_CONFIGS = [
    # (name, config kwargs) — driven by eval_sft.ModelConfig class attributes + the CodeLLM(config) constructor signature
    ("tiny-2L-d64-h4-ctx64", dict(vocab_size=256, d_model=64, num_layers=2, num_heads=4,
                                  d_ff=128, max_seq_len=64, rope_theta=10000.0)),
    ("tiny-3L-d96-h6-ctx48", dict(vocab_size=512, d_model=96, num_layers=3, num_heads=6,
                                  d_ff=192, max_seq_len=48, rope_theta=500000.0)),
]


def build_tiny_model(cfg_kw, seed=0, dtype=torch.float32, device="cpu"):
    config = E.ModelConfig()
    for k, v in cfg_kw.items():
        if not hasattr(config, k):
            raise SystemExit(f"[ERR] ModelConfig has no attribute {k} (do not guess config names)")
        setattr(config, k, v)
    torch.manual_seed(seed)
    model = E.CodeLLM(config)
    model.eval()
    model.to(device).to(dtype)
    return model


def _run_ref_with_trace(model, tokenizer, prompt, seed, **gen):
    """Runs the **real** eval_sft.generate_one and records its per-step logits along the way.

    Implemented by attaching a forward wrapper to the instance (nn.Module.__call__ goes through self.forward,
    so an instance attribute takes effect) → what is recorded are the baseline's real values, not a rewritten approximation.
    """
    orig_fwd = model.forward
    rec = []

    def wrapped(input_ids, *a, **k):
        out = orig_fwd(input_ids, *a, **k)
        rec.append(out[0, -1, :].float().clone())
        return out

    model.forward = wrapped
    try:
        text = E.generate_one(model, tokenizer, prompt, seed, **gen)
    finally:
        model.forward = orig_fwd
    return text, rec


def _seq(text):
    return [int(x) for x in text.split()] if text.strip() else []


def _first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def _prefill_bitwise_delta(model, ids, fg):
    """Prefill uses exactly the same operators/shapes as the baseline's first step and should be bitwise identical (Δ should be 0)."""
    with torch.no_grad():
        ref = model(torch.tensor([ids], dtype=torch.long, device=fg.device))
        got = fg.prefill(ids)
    return (ref.float() - got.float()).abs().max().item()


def _gemm_shape_sensitivity(model, T=48, seed=0):
    """Given **the same input row and the same weights**, are the M=1 and M=T GEMMs bitwise identical?

    That is the only structural difference between incremental decoding and a full-length forward: the M dimension
    of the q/k/v projections and of SDPA (1 vs t). If the same input row already differs with M in bf16/fp32,
    then no KV cache path **can** be bitwise identical to "recompute everything" — that is not an implementation bug.
    """
    W = model.blocks[0].attn.qkv.weight.detach()
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, T, W.shape[1], generator=g).to(W.dtype)
    with torch.no_grad():
        full = F.linear(x, W)[0, -1]
        row = F.linear(x[:, -1:, :], W)[0, 0]
    return (full.float() - row.float()).abs().max().item()


def _trace_stats(ref_logits, trace):
    """Compares logits step by step: returns (max|Δ|, top-1 agreement rate, steps compared, steps agreeing)."""
    n = min(len(ref_logits), len(trace))
    if not n:
        return 0.0, 0.0, 0, 0
    dmax, agree = 0.0, 0
    for x, y in zip(ref_logits[:n], trace[:n]):
        dmax = max(dmax, (x - y).abs().max().item())
        agree += int(x.argmax() == y.argmax())
    return dmax, agree / n, n, agree


def _one_case(model, tok, prompt, seed, gen_kw, cfg_name, dt_name, tag, rows, verbose=True):
    """Runs one three-way comparison: baseline / fastgen(kv off) / fastgen(kv on), appending a diagnostic row."""
    max_new, temperature = gen_kw["max_new"], gen_kw["temperature"]
    ids_kw = {k: v for k, v in gen_kw.items() if k != "device"}   # generate_ids has no device parameter
    try:
        ref_text, ref_logits = _run_ref_with_trace(model, tok, prompt, seed, **gen_kw)
        ref_err = None
    except Exception as e:
        ref_text, ref_logits, ref_err = None, [], f"{type(e).__name__}: {e}"
    fg_off = FastGen(model, "cpu", use_kv_cache=False, tensor_params=True)
    try:
        off_ids = fg_off.generate_ids(tok, prompt, seed, **ids_kw)
        off_err = None
    except Exception as e:
        off_ids, off_err = None, type(e).__name__
    fg_on = FastGen(model, "cpu", use_kv_cache=True, tensor_params=True, compile_mode="none")
    fg_on.trace = []
    try:
        on_ids = fg_on.generate_ids(tok, prompt, seed, **ids_kw)
        on_err = None
    except Exception as e:
        on_ids, on_err = None, type(e).__name__

    ref_ids = _seq(ref_text) if ref_text is not None else None
    if ref_err:
        # the baseline raises under this parameter set (e.g. T=0 + min_p=0 → NaN probability tensor):
        # the criterion is that fastgen raises the **same exception type** (matching behavior), not that any exception is tolerated
        ref_kind = ref_err.split(":")[0]
        ok_a = (off_err == ref_kind)
        ok_b = (on_err == ref_kind)
        detail = (f"baseline raised [{ref_err[:70]}]; kv_off={off_err} / kv_on={on_err}"
                  f" (only the same exception type counts as agreement)")
    elif float(temperature) == 0.0:
        ok_a = (off_ids == []) and (on_ids == [])
        ok_b = (on_ids == [])
        detail = (f"T=0 degenerate path: ref={len(ref_ids)} tok / kv_off={len(off_ids or [])} / "
                  f"kv_on={len(on_ids or [])}")
    else:
        ok_a = off_ids == ref_ids
        ok_b = on_ids == ref_ids
        detail = (f"A(kv off) first divergence idx={_first_diff(ref_ids, off_ids or [])}; "
                  f"B(kv on) first divergence idx={_first_diff(ref_ids, on_ids or [])}")

    dmax, top1, n_cmp, n_agree = _trace_stats(ref_logits, fg_on.trace)
    # run the "mask" read (required by CUDA graphs) separately: default parameter set only, to bound verification time
    m_fields = {}
    if tag == "default":
        fg_m = FastGen(model, "cpu", use_kv_cache=True, tensor_params=True, decode_read="mask")
        fg_m.trace = []
        try:
            m_ids = fg_m.generate_ids(tok, prompt, seed, **ids_kw)
            m_err = None
        except Exception as e:
            m_ids, m_err = None, type(e).__name__
        if ref_err:
            ok_m = (m_err == ref_err.split(":")[0])
            m_ids = m_ids if m_ids is not None else ([] if ok_m else None)
        else:
            ok_m = (m_ids == ref_ids)
        dm, tm, nm, am = _trace_stats(ref_logits, fg_m.trace)
        m_fields = dict(kv_on_mask_ok=bool(ok_m), mask_max_dlogits=dm,
                        mask_top1_agree=tm, mask_top1_steps=nm, mask_top1_agree_steps=am,
                        mask_first_diff=_first_diff(ref_ids or [], m_ids or []))
    rows.append(dict(cfg=cfg_name, dtype=dt_name, tag=tag, temperature=temperature, seed=seed,
                     kv_off_ok=bool(ok_a), kv_on_ok=bool(ok_b), n_ref=len(ref_ids or []),
                     n_on=len(on_ids or []), n_off=len(off_ids or []),
                     first_diff_on=_first_diff(ref_ids or [], on_ids or []), max_dlogits=dmax,
                     top1_agree=(n_agree / n_cmp if n_cmp else 0.0), top1_steps=n_cmp,
                     top1_agree_steps=n_agree,
                     prompt_len=len(prompt.split()), note=detail, **m_fields))
    if verbose and not ok_b:
        print(f"  ✗ {cfg_name}/{dt_name}/{tag}/T={temperature}/seed={seed}: "
              f"ref={len(ref_ids or [])} tok, kv_on={len(on_ids or [])} tok, "
              f"first divergence={_first_diff(ref_ids or [], on_ids or [])}, max|Δlogits|={dmax:.3e}")
    if not ok_a:
        print(f"  ✗ sampling side not equivalent (bug, should not happen) {cfg_name}/{dt_name}/{tag}/T={temperature}"
              f"/seed={seed}: {detail}")
    return ok_a, ok_b


class _RecordingTokenizer:
    """Forwards to the real tokenizer but records the token ids generate_one decodes at the end.

    That makes token-level (rather than string-level) comparison possible against the **real tokenizer / real ckpt** —
    a real BPE decode is not injective, so equal strings do not imply equal tokens.
    """

    def __init__(self, tok):
        self._tok = tok
        self.last_ids = None

    def __getattr__(self, k):
        return getattr(self._tok, k)

    def decode(self, ids, *a, **k):
        self.last_ids = [int(i) for i in ids]
        return self._tok.decode(ids, *a, **k)


def verify_equivalence_ckpt(args):
    """Equivalence on a real ckpt + real tokenizer (token level). Pass --device cuda explicitly to use the GPU.

    The reference path recomputes the whole prefix every step → expensive on a 435M model, so max_new and the prompt count stay small by default.
    """
    if not args.ckpt:
        raise SystemExit("[ERR] --verify-equivalence needs --ckpt when using a real ckpt")
    ckpt = Path(args.ckpt)
    if not (ckpt / "checkpoint.pt").exists():
        raise SystemExit(f"[ERR] checkpoint not found: {ckpt / 'checkpoint.pt'}")
    device = args.device
    real_tok = E.load_tokenizer(args.tokenizer)
    rec_tok = _RecordingTokenizer(real_tok)
    model = E.load_model(ckpt, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    temps = [float(x) for x in args.verify_temperatures.split(",")]
    seeds = [int(x) for x in args.verify_seeds.split(",")]
    max_new = args.verify_max_new
    if args.problems:
        probs = load_jsonl(args.problems)[:4]
        tmpl = BI.PROMPT_TEMPLATES[args.template]
        prompts = [build_prompt(pr, tmpl, args.template)[0] for pr in probs]
    else:
        prompts = BENCH_PROMPTS[:3]
    print("=" * 76)
    print(f"equivalence verification (real ckpt, token level): {ckpt}")
    print(f"  {n_params:.0f}M | device={device} | dtype={next(model.parameters()).dtype} | "
          f"prompts={len(prompts)} | max_new={max_new} | temps={temps} | seeds={seeds}")
    print("=" * 76)
    n_ok = n_tot = 0
    for temperature in temps:
        for seed in seeds:
            for pi, prompt in enumerate(prompts):
                plen = len(real_tok.encode(prompt).ids)
                if plen + max_new > model.config.max_seq_len:
                    print(f"  ⚠️ prompt {plen} tok + max_new {max_new} > context "
                          f"{model.config.max_seq_len} → taking the truncation branch (slower but still semantically equivalent)")
                ref_text, ref_logits = _run_ref_with_trace(
                    model, rec_tok, prompt, seed, max_new=max_new, temperature=temperature,
                    top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                    min_p=args.min_p, device=device)
                ref_ids = rec_tok.last_ids or []
                outs = {}
                for tag, kw in (("slice", dict(decode_read="slice")),
                                ("mask", dict(decode_read="mask"))):
                    fg = FastGen(model, device, use_kv_cache=True, tensor_params=True,
                                 compile_mode="none", **kw)
                    fg.trace = []
                    try:
                        outs[tag] = (fg.generate_ids(
                            real_tok, prompt, seed, max_new=max_new, temperature=temperature,
                            top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                            min_p=args.min_p), fg.trace)
                    except Exception as e:
                        outs[tag] = (None, [], f"{type(e).__name__}: {e}")
                for tag in ("slice", "mask"):
                    got, trace = outs[tag][0], outs[tag][1]
                    ok = (got == ref_ids)
                    d, t1, ns, ag = _trace_stats(ref_logits, trace)
                    n_ok += int(ok)
                    n_tot += 1
                    print(f"  T={temperature:<7} seed={seed} prompt#{pi}({plen}tok) "
                          f"decode_read={tag:5s} ref={len(ref_ids)}tok fast="
                          f"{len(got) if got is not None else 'ERR'}tok "
                          f"{'match ✓' if ok else f'diverged idx={_first_diff(ref_ids, got or [])} ✗'} "
                          f"top1={ag}/{ns} max|Δlogits|={d:.3e}")
    print("-" * 76)
    print(f"token-level matches on the real ckpt: {n_ok}/{n_tot}")
    print("=" * 76)
    if device != "cuda":
        print("[hint] running 435M on CPU is slow; real hardware (bf16/cuda) is where actual generation precision lives.")
    return n_ok, n_tot


def verify_equivalence(args):
    """Phase A: sampling logic equivalence (kv off; bitwise identical logits → even random sampling must match token by token)
       Phase B: KV cache path equivalence (kv on; greedy must match token by token, random sampling is checked for divergence)"""
    _check_ref_signature()
    if args.ckpt:
        return verify_equivalence_ckpt(args)
    temps = [float(x) for x in args.verify_temperatures.split(",")]
    seeds = [int(x) for x in args.verify_seeds.split(",")]
    max_new = args.verify_max_new
    print("=" * 76)
    print("equivalence verification: fastgen vs eval_sft.generate_one (CPU + random tiny model)")
    print("=" * 76)
    print(f"  torch={torch.__version__} | max_new={max_new} | seeds={seeds} | "
          f"temperatures={temps}")
    print(f"  settings note: at temperature=0 the baseline is a NaN degenerate path (logits/0 → all NaN after top-k → "
          f"the min_p filter removes everything → empty string returned);\n"
          f"            temperature=1e-06 = **equivalent greedy** (softmax saturates to one-hot, so argmax is always taken).")

    rows = []
    n_pass = n_fail = 0
    prefill_deltas = []
    gemm_deltas = []
    for cfg_name, cfg_kw in TINY_CONFIGS:
        for dtype, dt_name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            model = build_tiny_model(cfg_kw, seed=7, dtype=dtype)
            V = int(model.config.vocab_size)
            cap = int(model.config.max_seq_len)
            tok = DummyTokenizer(V)
            rng = random.Random(1234)

            # ── Phase 0: is prefill **bitwise** identical to the baseline's first step ──
            probe = [rng.randrange(V) for _ in range(min(cap, 16))]
            fg_probe = FastGen(model, "cpu")
            d_pre = _prefill_bitwise_delta(model, probe, fg_probe)
            prefill_deltas.append((cfg_name, dt_name, d_pre))
            d_gemm = _gemm_shape_sensitivity(model)
            print(f"  [{cfg_name}/{dt_name}] prefill vs model(ids) bitwise delta max|Δ|={d_pre:.3e}"
                  f"  → {'bitwise identical ✓' if d_pre == 0.0 else 'differs (see report)'}")
            print(f"  [{cfg_name}/{dt_name}] same weights and input row, M=1 vs M={48} GEMM delta "
                  f"max|Δ|={d_gemm:.3e} → {'bitwise identical' if d_gemm == 0.0 else '**the M dimension changes the result**'}"
                  f" (this is the only source of numerical difference between the KV cache and full recomputation)")
            gemm_deltas.append((cfg_name, dt_name, d_gemm))

            # ── Phase A/B: normal length (no prefix truncation) ──
            for top_k, min_p, rp, tag in ((50, 0.05, 1.2, "default"),
                                          (50, 0.0, 1.0, "min_p/rp off"),
                                          (1, 0.05, 1.2, "top_k=1"),
                                          (0, 0.0, 1.2, "no top_k (full-vocab softmax)")):
                for temperature in temps:
                    for seed in seeds:
                        plen = rng.randint(5, max(6, cap - max_new - 2))
                        prompt = " ".join(str(rng.randrange(V)) for _ in range(plen))
                        gen_kw = dict(max_new=max_new, temperature=temperature, top_k=top_k,
                                      repetition_penalty=rp, min_p=min_p, device="cpu")
                        ok_a, ok_b = _one_case(model, tok, prompt, seed, gen_kw,
                                               cfg_name, dt_name, tag, rows)
                        n_pass += int(ok_b)
                        n_fail += int(not ok_b)

            # ── Phase C: prompt + max_new > cap → the baseline truncates the prefix, fastgen recomputes the whole window ──
            for plen, ctag in ((max(1, cap - 4), "sliding window(prompt=cap-4)"),
                               (max(1, cap - 1), "sliding window(prompt=cap-1)")):
                prompt = " ".join(str(rng.randrange(V)) for _ in range(plen))
                for temperature in (1e-06, temps[-1]):
                    gen_kw = dict(max_new=max_new, temperature=temperature, top_k=50,
                                  repetition_penalty=1.2, min_p=0.05, device="cpu")
                    ok_a, ok_b = _one_case(model, tok, prompt, seeds[0], gen_kw,
                                           cfg_name, dt_name, ctag, rows)
                    n_pass += int(ok_b)
                    n_fail += int(not ok_b)
            del model
    # ── summary ──
    nA = sum(1 for r in rows if r["kv_off_ok"])
    nB = sum(1 for r in rows if r["kv_on_ok"])
    print("-" * 76)
    print(f"Phase A (kv off: bitwise identical logits → even random sampling must match token by token): "
          f"{nA}/{len(rows)} ✓")
    print(f"Phase B (kv on, decode reads the cache with **slice**, valid prefix only): {nB}/{len(rows)} ✓")
    mrows = [r for r in rows if r.get("kv_on_mask_ok") is not None]
    if mrows:
        print(f"Phase B' (kv on, decode reads the cache with **mask**, whole cache + mask = required by CUDA graphs): "
              f"{sum(1 for r in mrows if r['kv_on_mask_ok'])}/{len(mrows)} ✓"
              f" (default parameter set only)")
    print(f"Phase C is included in Phase B (the prefix-truncation path when prompt+max_new > context limit)")
    print(f"Phase 0 (bitwise prefill): "
          f"{sum(1 for _, _, d in prefill_deltas if d == 0.0)}/{len(prefill_deltas)} bitwise identical")
    for cfg_name, dt_name, d in gemm_deltas:
        print(f"  GEMM M=1 vs M=48 (same weights, same input row) {cfg_name}/{dt_name}: max|Δ|={d:.3e}"
              f" → {'bitwise identical' if d == 0.0 else 'shape sensitive'}")
    for tag_m, dkey, tkey, akey in (
            ("slice", "max_dlogits", "top1_steps", "top1_agree_steps"),
            ("mask", "mask_max_dlogits", "mask_top1_steps", "mask_top1_agree_steps")):
        sub = [r for r in rows if dkey in r]
        if not sub:
            continue
        steps = sum(r.get(tkey, 0) for r in sub)
        agree = sum(r.get(akey, 0) for r in sub)
        print(f"  decode read mode={tag_m:5s}: {steps} decode steps vs the baseline's full-length forward → "
              f"top-1 agreement {agree}/{steps} ({100.0 * agree / max(steps, 1):.1f}%), "
              f"max max|Δlogits|={max(r[dkey] for r in sub):.3e}")
    for dt_name in ("fp32", "bf16"):
        sub = [r for r in rows if r["dtype"] == dt_name]
        if sub:
            print(f"    {dt_name}: max|Δlogits|={max(r['max_dlogits'] for r in sub):.3e}")
    t0_rows = [r for r in rows if float(r["temperature"]) == 0.0]
    print(f"**contract setting T=0 (the baseline's NaN degenerate path; both sides return an empty string)**: "
          f"{sum(1 for r in t0_rows if r['kv_on_ok'])}/{len(t0_rows)} match")
    greedy = [r for r in rows if float(r["temperature"]) == 1e-06]
    if greedy:
        print(f"equivalent-greedy setting (T=1e-06, softmax saturates to one-hot → argmax): "
              f"{sum(1 for r in greedy if r['kv_on_ok'])}/{len(greedy)} token-by-token matches")
        for r in [r for r in greedy if not r["kv_on_ok"]][:6]:
            print(f"    ✗ {r['dtype']}/{r['tag']}/seed={r['seed']}: first divergence idx={r['first_diff_on']}"
                  f", max|Δlogits|={r['max_dlogits']:.3e}")
    stoch = [r for r in rows if float(r["temperature"]) not in (0.0, 1e-06)]
    if stoch:
        bad = [r for r in stoch if not r["kv_on_ok"]]
        print(f"random-sampling setting: {len(stoch) - len(bad)}/{len(stoch)} token-by-token matches"
              + (f"; {len(bad)} diverged (last-bit logit differences amplified by multinomial)" if bad else ""))

    if args.verify_compile:
        print("-" * 76)
        print("extra: equivalence under torch.compile + whether it recompiles every step")
        print("      without a C++ compiler (MSVC cl) Inductor is unavailable; at minimum use backend='eager' here to")
        print("      check that the single decode step is dynamo-capturable and does not recompile as pos changes.")
        cfg_name, cfg_kw = TINY_CONFIGS[0]
        model = build_tiny_model(cfg_kw, seed=7, dtype=torch.float32)
        V = int(model.config.vocab_size)
        tok = DummyTokenizer(V)
        rng = random.Random(9)
        prompt = " ".join(str(rng.randrange(V)) for _ in range(12))
        kw = dict(max_new=16, temperature=1e-6, top_k=50, repetition_penalty=1.2, min_p=0.05)
        # the read mode is fixed to mask on both sides, so the only variable here is whether compile wraps the step
        fg_e = FastGen(model, "cpu", compile_mode="none", decode_read="mask")
        fg_e.trace = []
        ids_e = fg_e.generate_ids(tok, prompt, 0, **kw)
        print(f"  eager reference (decode_read=mask): {len(ids_e)} tok = {ids_e[:10]}…")
        # dynamo calls the backend once per compiled frame → a counting backend measures recompiles as pos changes
        try:
            from torch._dynamo.backends.registry import lookup_backend
            eager_backend = lookup_backend("eager")
            nframes = {"n": 0}

            def counting_backend(gm, example_inputs, _n=nframes, _b=eager_backend):
                _n["n"] += 1
                return _b(gm, example_inputs)
        except Exception as e:            # pragma: no cover
            counting_backend, nframes = None, {"n": -1}
            print(f"  ⚠️ dynamo counting backend unavailable: {type(e).__name__}: {e}")
        for mode in ("eager", "default", "reduce-overhead"):
            cb = counting_backend if mode == "eager" else None
            fg = FastGen(model, "cpu", compile_mode=mode, decode_read="mask", compile_backend=cb)
            fg.trace = []
            before = nframes["n"]
            t0 = time.time()
            try:
                ids_c = fg.generate_ids(tok, prompt, 0, **kw)
                dt = time.time() - t0
                n_new = nframes["n"] - before if nframes["n"] >= 0 else -1
                status = "match ✓" if ids_c == ids_e else "mismatch ✗"
                extra = ""
                if fg.compile_failed:
                    extra = f"  ⚠️ compile failed, fell back to eager: {fg.compile_failed[:80]}"
                print(f"  mode={mode:16s} {status}  tokens={len(ids_c)}/{len(ids_e)}  "
                      f"new compiled frames={n_new} (=1 means 16 decode steps captured one graph, no recompiles as pos changes)"
                      f"  time {dt:.1f}s{extra}")
            except Exception as e:
                print(f"  mode={mode:16s} ✗ {type(e).__name__}: {str(e)[:160]}")

    print("=" * 76)
    return rows, n_pass, n_fail


# ═══════════════════════════════════════════════════════════════════
# 3. Benchmarks
# ═══════════════════════════════════════════════════════════════════
def _bench_variants():
    """Ablation variants: (name, FastGen kwargs, note)"""
    return [
        ("ref-equivalent", dict(use_kv_cache=False, tensor_params=False, compile_mode="none"),
         "= baseline compute + the baseline's Python penalty loop (should ≈ eval_sft.generate_one)"),
        ("A +tensor-params", dict(use_kv_cache=False, tensor_params=True, compile_mode="none"),
         "removes only the per-token indexing/synchronization"),
        ("B +kv-cache", dict(use_kv_cache=True, tensor_params=True, compile_mode="none"),
         "incremental KV cache decoding (removes the O(t) recompute); decode reads the valid prefix only"),
        ("B2 +kv-cache(mask read)", dict(use_kv_cache=True, tensor_params=True,
                                        compile_mode="none", decode_read="mask"),
         "same cache, but decode reads the whole cap-sized cache + mask (required by CUDA graphs)"),
        ("C B+compile(default)", dict(use_kv_cache=True, tensor_params=True, compile_mode="default"),
         "plus Inductor compilation"),
        ("D B+reduce-overhead", dict(use_kv_cache=True, tensor_params=True,
                                     compile_mode="reduce-overhead"),
         "plus CUDA graphs (CUDA only; this mode sets decode_read=mask → static shapes)"),
    ]


BENCH_PROMPTS = [
    "### Task\nWrite a function is_prime(n) that returns True if n is a prime number.\n\n### Code\n",
    "### Task\nWrite a function mergesort(arr) that sorts a list of numbers.\n\n### Code\n",
    "### Task\nImplement two_sum(nums, target) returning indices of the two numbers.\n\n### Code\n",
    "### Task\nWrite a function count_vowels(s) returning the number of vowels.\n\n### Code\n",
]


def bench(args, device):
    """tokens/s on a real ckpt. --bench enforces --device cuda (the local GPU is usually busy training)."""
    _check_ref_signature()
    if device != "cuda":
        raise SystemExit(
            "[ERR] --bench must run on a real ckpt and requires an explicit --device cuda per the contract.\n"
            "      Do not run it while the GPU is busy training; for CPU relative ordering use --bench-cpu (clearly labelled as not GPU-representative).")
    if not torch.cuda.is_available():
        raise SystemExit("[ERR] --device cuda was requested but torch.cuda.is_available() is False.")
    if not args.ckpt:
        raise SystemExit("[ERR] --bench requires --ckpt <checkpoint dir>")
    tokenizer = E.load_tokenizer(args.tokenizer)
    model = E.load_model(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("=" * 76)
    print(f"bench: {args.ckpt} | {n_params:.0f}M | device={device} | "
          f"torch={torch.__version__} | GPU={torch.cuda.get_device_name(0)}")
    print(f"  max_new={args.max_new} prompts={len(BENCH_PROMPTS)} temperature={args.temperature}")
    print("=" * 76)

    # baseline: the real generate_one (timed as well, to serve as the yardstick)
    E.generate_one(model, tokenizer, BENCH_PROMPTS[0], args.seed, max_new=min(8, args.max_new),
                   temperature=args.temperature, top_k=args.top_k,
                   repetition_penalty=args.repetition_penalty, min_p=args.min_p, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for i, p in enumerate(BENCH_PROMPTS):
        E.generate_one(model, tokenizer, p, args.seed + i, max_new=args.max_new,
                       temperature=args.temperature, top_k=args.top_k,
                       repetition_penalty=args.repetition_penalty, min_p=args.min_p,
                       device=device)
    dt = time.time() - t0
    print(f"  eval_sft.generate_one (baseline)        "
          f"{dt / len(BENCH_PROMPTS):7.2f} s/seq   {args.max_new * len(BENCH_PROMPTS) / dt:7.1f} tok/s")
    torch.cuda.synchronize()

    for name, kw, note in _bench_variants():
        for attempt in (1, 2):
            try:
                fg = FastGen(model, device, **kw)
                # warmup (not timed): keeps the first compile / graph capture / memory pool allocation out of the measurement
                fg.generate(tokenizer, BENCH_PROMPTS[0], args.seed,
                            max_new=min(8, args.max_new), temperature=args.temperature,
                            top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                            min_p=args.min_p)
                torch.cuda.synchronize()
                t = time.time()
                for i, p in enumerate(BENCH_PROMPTS):
                    fg.generate(tokenizer, p, args.seed + i, max_new=args.max_new,
                                temperature=args.temperature, top_k=args.top_k,
                                repetition_penalty=args.repetition_penalty, min_p=args.min_p)
                torch.cuda.synchronize()
                dt = time.time() - t
                if fg.compile_failed:
                    print(f"  ⚠️ {name}: compile failed, fell back to eager — {fg.compile_failed[:120]}")
                print(f"  {name:38s} {dt / len(BENCH_PROMPTS):7.2f} s/seq   "
                      f"{args.max_new * len(BENCH_PROMPTS) / dt:7.1f} tok/s   "
                      f"(decode_read={fg.decode_read}; {note})")
                del fg
                break
            except Exception as e:
                if kw.get("compile_mode") not in (None, "none") and attempt == 1:
                    print(f"  ⚠️ {name}: {type(e).__name__}: {str(e)[:200]} → retrying without compile")
                    kw = dict(kw, compile_mode="none")
                    continue
                print(f"  ✗ {name} failed: {type(e).__name__}: {str(e)[:300]}")
                break
    print("=" * 76)
    print("Note: tok/s is computed from max_new (early stopping makes the real tok/s higher); both paths use the same seed"
          "and prompt, and early-stop behavior matches, so the relative comparison is valid.")


def bench_cpu(args):
    """Variant ablation on CPU: only **relative ordering** and structural conclusions matter (absolute tok/s is not GPU-representative)."""
    _check_ref_signature()
    print("=" * 76)
    print("bench-cpu: variant ablation on CPU (absolute tok/s is meaningless; only relative ordering/mechanism matters)")
    print(f"  torch={torch.__version__} threads={torch.get_num_threads()} "
          f"max_new={args.max_new} seqs={args.bench_seqs}")
    print("=" * 76)
    if args.ckpt:
        tokenizer = E.load_tokenizer(args.tokenizer)
        model = E.load_model(args.ckpt, "cpu").float()
        print(f"  using real ckpt {args.ckpt} (CPU float32)")
    else:
        cfg_kw = dict(vocab_size=1024, d_model=256, num_layers=6, num_heads=8, d_ff=1024,
                      max_seq_len=256, rope_theta=500000.0)
        model = build_tiny_model(cfg_kw, seed=1, dtype=torch.float32)
        tokenizer = DummyTokenizer(int(model.config.vocab_size))
        print("  using a randomly initialized medium model (6L/d256/h8, CPU float32; not a ckpt, for relative comparison only)")
    V = int(model.config.vocab_size)
    rng = random.Random(0)
    prompts = [" ".join(str(rng.randrange(V)) for _ in range(32)) for _ in range(args.bench_seqs)]

    # warmup (not timed)
    E.generate_one(model, tokenizer, prompts[0], args.seed, max_new=min(8, args.max_new),
                   temperature=args.temperature, top_k=args.top_k,
                   repetition_penalty=args.repetition_penalty, min_p=args.min_p, device="cpu")
    t0 = time.time()
    for i, p in enumerate(prompts):
        E.generate_one(model, tokenizer, p, args.seed + i, max_new=args.max_new,
                       temperature=args.temperature, top_k=args.top_k,
                       repetition_penalty=args.repetition_penalty, min_p=args.min_p,
                       device="cpu")
    dt = time.time() - t0
    base = args.max_new * len(prompts) / dt
    print(f"  eval_sft.generate_one (baseline)        {dt / len(prompts):7.3f} s/seq   "
          f"{base:8.1f} tok/s   1.00x")
    for name, kw, note in _bench_variants():
        try:
            fg = FastGen(model, "cpu", **kw)
            # warmup (not timed): drops the cost of the first compile attempt / memory allocation
            fg.generate(tokenizer, prompts[0], args.seed, max_new=min(8, args.max_new),
                        temperature=args.temperature, top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty, min_p=args.min_p)
            t0 = time.time()
            for i, p in enumerate(prompts):
                fg.generate(tokenizer, p, args.seed + i, max_new=args.max_new,
                            temperature=args.temperature, top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty, min_p=args.min_p)
            dt = time.time() - t0
            rate = args.max_new * len(prompts) / dt
            print(f"  {name:38s} {dt / len(prompts):7.3f} s/seq   {rate:8.1f} tok/s   "
                  f"{rate / base:5.2f}x   (decode_read={fg.decode_read}; {note})")
            del fg
        except Exception as e:
            print(f"  ✗ {name} failed: {type(e).__name__}: {str(e)[:200]}")
    print("=" * 76)
    print("⚠️ CPU conclusions ≠ GPU conclusions: on CPU the benefit of removing per-token sync is amplified by the Python"
          "interpreter, while the CUDA graphs benefit does not exist on CPU (nothing can be compiled to a graph). GPU conclusions require --bench.")


# ═══════════════════════════════════════════════════════════════════
# 4. Generate candidates (same CLI shape as esd_sample.py, drop-in replacement)
# ═══════════════════════════════════════════════════════════════════
def load_jsonl(path):
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


def build_prompt(pr, tmpl, tmpl_name):
    """Same template assertion logic as esd_sample.build_prompt (byte-for-byte identical)."""
    if "id" not in pr:
        raise SystemExit(f"[ERR] problem is missing the id field: { {k: str(v)[:40] for k, v in pr.items()} }")
    instr = pr.get("prompt")
    if instr is None:
        instr = pr.get("instruction") or pr.get("input")
    if instr is None or not str(instr).strip():
        raise SystemExit(f"[ERR] problem {pr['id']} has no prompt/instruction")
    text = tmpl.format(instruction=str(instr))
    mt = pr.get("meta") or {}
    t_in_meta = mt.get("prompt_template")
    if t_in_meta not in (None, "", tmpl):
        raise SystemExit(
            f"[ERR] problem {pr['id']}: meta.prompt_template does not match this script's template — the generation settings would be misaligned.\n"
            f"      meta : {t_in_meta!r}\n      this script ({tmpl_name}): {tmpl!r}")
    pt = mt.get("prompt_templated")
    if pt is not None and pt != text:
        raise SystemExit(f"[ERR] problem {pr['id']}: the formatted template is not byte-for-byte equal to meta.prompt_templated.")
    return text, str(pr["id"])


def expand_id(base_id, k_index, id_mode):
    return f"{base_id}#k{k_index}" if id_mode == "unique" else base_id


def run_gen(args):
    _check_ref_signature()
    if not args.ckpt:
        raise SystemExit("[ERR] --gen requires --ckpt <checkpoint dir>")
    ckpt = Path(args.ckpt)
    if not (ckpt / "checkpoint.pt").exists():
        raise SystemExit(f"[ERR] checkpoint not found: {ckpt / 'checkpoint.pt'}")
    tmpl = BI.PROMPT_TEMPLATES[args.template]
    probs = load_jsonl(args.problems)
    if args.limit and args.limit > 0:
        probs = probs[:args.limit]
    prepared = [(*build_prompt(pr, tmpl, args.template), pr) for pr in probs]
    if args.top_p < 1.0:
        eprint(f"[fastgen] ⚠️ --top-p {args.top_p} ignored: the baseline generate_one samples with only "
               f"top_k + min_p + repetition_penalty (same settings as esd_sample.py).")
    print(f"[fastgen] {len(prepared)} problems × k={args.k} | template={args.template} | "
          f"device={args.device} | kv_cache={not args.no_kv_cache} "
          f"tensor_params={not args.no_tensor_params} compile={args.compile}")
    tokenizer = E.load_tokenizer(args.tokenizer)
    model = E.load_model(args.ckpt, args.device)
    fg = FastGen(model, args.device, use_kv_cache=not args.no_kv_cache,
                 tensor_params=not args.no_tensor_params, compile_mode=args.compile,
                 use_inference_mode=not args.no_inference_mode,
                 cache_len=(args.cache_len or None),
                 decode_read=(None if args.decode_read == "auto" else args.decode_read))
    print(f"[fastgen] decode cache read mode={fg.decode_read} (slice=valid prefix only / mask=whole cache+mask)")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rec, n_empty, chars, n_tok = 0, 0, 0, 0
    t_start = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for pi, (text, pid, pr) in enumerate(prepared):
            t_p = time.time()
            for ki in range(args.k):
                seed_used = args.seed + ki
                ids = fg.generate_ids(tokenizer, text, seed_used, max_new=args.max_new,
                                      temperature=args.temperature, top_k=args.top_k,
                                      repetition_penalty=args.repetition_penalty,
                                      min_p=args.min_p)
                comp = tokenizer.decode(ids) if ids else ""
                if args.truncate_eval:
                    comp = E.truncate(comp)
                f.write(json.dumps({"id": expand_id(pid, ki, args.id_mode), "k_index": ki,
                                    "completion": comp, "seed_used": seed_used},
                                   ensure_ascii=False) + "\n")
                n_rec += 1
                n_tok += len(ids)
                n_empty += (1 if not comp.strip() else 0)
                chars += len(comp)
            print(f"  [{pi + 1}/{len(prepared)}] {pid}  k={args.k}  "
                  f"elapsed {time.time() - t_p:.1f}s  avg {chars / max(1, n_rec):.0f} chars/candidate")
    dt = time.time() - t_start
    print(f"[fastgen] → {out_path} ({n_rec} candidates, {n_empty} empty, {n_tok} tokens generated, "
          f"elapsed {dt:.1f}s → {n_tok / max(dt, 1e-9):.1f} tok/s)")
    meta = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": "fastgen.py",
            "mode": "model", "ckpt": str(args.ckpt), "device": args.device,
            "problems": str(args.problems),
            "n_problems": len(prepared), "k": args.k, "n_records": n_rec,
            "n_empty_completions": n_empty, "avg_completion_chars": round(chars / max(1, n_rec), 1),
            "generated_tokens": n_tok,
            "tokens_per_s": round(n_tok / max(dt, 1e-9), 1),
            "id_mode": args.id_mode, "seed_base": args.seed, "seed_rule": "base_seed + k_index",
            "record_fields": ["id", "k_index", "completion", "seed_used"],
            "truncate_eval": bool(args.truncate_eval),
            "prompt_template_name": args.template, "prompt_template": tmpl,
            "prompt_template_head": BI.TEMPLATE_HEAD[args.template],
            "prompt_template_tail": BI.TEMPLATE_TAIL[args.template],
            "prompt_alignment": ("template comes from build_instr_data.PROMPT_TEMPLATES; "
                                 "asserted byte-for-byte against meta.prompt_templated (same as esd_sample.py)"),
            "gen_params": dict(max_new=args.max_new, temperature=args.temperature,
                               top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                               min_p=args.min_p),
            "fast_path": dict(kv_cache=not args.no_kv_cache,
                              tensor_params=not args.no_tensor_params,
                              compile=args.compile,
                              decode_read=getattr(fg, "decode_read", None),
                              inference_mode=not args.no_inference_mode),
            "top_p_requested": args.top_p, "top_p_applied": False,
            "model_source": "eval_sft.load_tokenizer / load_model (the model definition is not rewritten)",
            "generator": "fastgen.FastGen (semantically equivalent to eval_sft.generate_one; see fastgen_report.md)" if not args.no_kv_cache
                         else "eval_sft.generate_one (--no-kv-cache)",
            "elapsed_s": round(dt, 1)}
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fastgen] meta → {out_path.with_suffix('.meta.json')}")
    return 0


# ═══════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════
def build_parser():
    ap = argparse.ArgumentParser(
        description="A generation path that is semantically equivalent to eval_sft.generate_one but faster (KV cache / tensorized sampling / "
                    "optional CUDA graphs), including --verify-equivalence and --bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--verify-equivalence", action="store_true",
                    help="verify token-by-token equivalence with eval_sft.generate_one on CPU + a random tiny model")
    ap.add_argument("--bench", action="store_true", help="measure tokens/s on a real ckpt (needs --device cuda)")
    ap.add_argument("--bench-cpu", action="store_true", help="CPU variant ablation (relative ordering only)")
    ap.add_argument("--gen", action="store_true", help="generate candidates (same CLI shape as esd_sample.py)")
    # generation parameters (same names and defaults as esd_sample.py)
    ap.add_argument("--ckpt", default=None, help="checkpoint directory (containing checkpoint.pt)")
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="defaults to cpu: the local GPU is usually busy training; pass --device cuda explicitly to use it")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95, help="⚠️ not supported by the baseline → warns and is ignored")
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    # optimization switches
    ap.add_argument("--no-kv-cache", action="store_true", help="disable the KV cache (back to a full-prefix forward every step)")
    ap.add_argument("--no-tensor-params", action="store_true", help="disable tensorized sampling parameters (back to the baseline Python loop)")
    ap.add_argument("--no-inference-mode", action="store_true", help="use no_grad instead of inference_mode")
    ap.add_argument("--compile", choices=["none", "eager", "default", "reduce-overhead"],
                    default="none",
                    help="torch.compile mode for the decode step (reduce-overhead = CUDA graphs; "
                         "eager = dynamo capture only, needs no C++ compiler)")
    ap.add_argument("--cache-len", type=int, default=0, help="KV cache capacity (default = model max_seq_len)")
    ap.add_argument("--decode-read", choices=["auto", "slice", "mask"], default="auto",
                    help="how the decode step reads the cache: slice=valid prefix only (bitwise identical to the baseline in bf16, "
                         "eager only); mask=whole cache+mask (required by CUDA graphs); auto=chosen from --compile")
    # --gen (same shape as esd_sample.py)
    ap.add_argument("--problems", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", choices=sorted(BI.PROMPT_TEMPLATES), default="marker")
    ap.add_argument("--id-mode", choices=["plain", "unique"], default="plain")
    ap.add_argument("--truncate-eval", action="store_true")
    # verification / benchmark details
    ap.add_argument("--verify-seeds", default="0,42")
    ap.add_argument("--verify-max-new", type=int, default=24)
    ap.add_argument("--verify-temperatures", default="0,1e-06,0.7,1.0")
    ap.add_argument("--verify-compile", action="store_true", help="additionally verify equivalence with compile=default (slow)")
    ap.add_argument("--bench-seqs", type=int, default=4)
    return ap


def _contract_verdict(rows):
    """Contract-setting verdict (drives the exit code):

    Three categories that must pass 100%:
      * Phase A (kv off: bitwise identical logits → even random sampling must match)
      * T=0 (copies the baseline's NaN degenerate path)
      * T=1e-06 (equivalent greedy → argmax)
    Random-sampling divergences in bf16 (T=0.7/1.0) do **not** affect the exit code: they are the inherent
    "M=1 vs M=t shapes" effect in bf16 amplified by multinomial, quantified in the report.
    """
    bad = [r for r in rows
           if (not r["kv_off_ok"]) or
           (float(r["temperature"]) in (0.0, 1e-06) and not r["kv_on_ok"])]
    n_contract = sum(1 for r in rows
                     if r["kv_off_ok"] and float(r["temperature"]) in (0.0, 1e-06))
    stoch_bad = [r for r in rows if float(r["temperature"]) not in (0.0, 1e-06)
                 and not r["kv_on_ok"]]
    return (not bad), n_contract, len(bad), len(stoch_bad)


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.verify_equivalence:
        res = verify_equivalence(a)
        if a.ckpt:                      # the real-ckpt path returns (n_ok, n_tot)
            n_ok, n_tot = res
            return 0 if n_ok == n_tot else 1
        rows, n_pass, n_fail = res
        out = HERE / "fastgen_equiv.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[fastgen] details → {out} ({len(rows)} rows)")
        ok, n_contract, n_bad, n_stoch_bad = _contract_verdict(rows)
        print(f"[fastgen] contract settings (Phase A + T=0 + equivalent greedy), {n_contract} cases: "
              f"{'all passed ✓' if ok else f'failed {n_bad} ✗'}"
              f"; bf16 random-sampling divergences: {n_stoch_bad} (known, does not affect the verdict, see report)")
        return 0 if ok else 1
    if a.bench:
        bench(a, a.device)
        return 0
    if a.bench_cpu:
        bench_cpu(a)
        return 0
    if a.gen:
        if not a.problems or not a.out:
            raise SystemExit("[ERR] --gen requires --problems and --out")
        return run_gen(a)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
