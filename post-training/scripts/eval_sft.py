#!/usr/bin/env python3
"""
Cross-model comparison eval: 353M@8B / 353M@16B / 435M-v2@31B
================================================================
Goal: evaluate the three checkpoints under identical prompt, seed, and generation params,
and record every output for hand audit (never conclude from script pass rates alone).

Eval set:
  - 5 basic algorithms x 5 seeds (same as ablation_eval.py, so existing results are comparable)
  - HumanEval 164 problems x 1 seed (pass rate is reference only)
  - 6 simple tasks with docstrings x 5 seeds (tests docstring understanding and code generation)

Usage (Windows):
  py -3.12 ablation/eval_cross_models.py
  py -3.12 ablation/eval_cross_models.py --models 353M-8B,435M-31B
  py -3.12 ablation/eval_cross_models.py --skip-humaneval
"""

import sys
import json
import time
import argparse
import subprocess
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parent
HUMANEVAL_PATH = PROJECT_ROOT / "ablation" / "HumanEval.jsonl"

MODELS = [
    ("BASE-merged", PROJECT_ROOT / "checkpoints_wsm" / "merged", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SFT-final",   PROJECT_ROOT / "fine-tuning" / "ckpt" / "final",    PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SFT-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # Arm 3 = big dataset (147M tokens, same hyperparams, data only) -> answers "is 5x data worth it"
    ("A3-big-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_big" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # Arm 2 (plan A) = same data / same base as Arm 1, only LR differs (5e-5)
    ("A2-lr5e5-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_arm2" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # Arm 4 = instruction mix: SFT 29.4M + instruction seed 2.94M (the only variable is the instruction data)
    ("A4-instr-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_arm4" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("A4-instr-final",   PROJECT_ROOT / "fine-tuning" / "ckpt_arm4" / "final",    PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # Plans A/B/C (2026-09-13): A5 = 147M signature + 14.7M instruction (two winners merged); A6 = Arm4 data 2 epochs; A7 = 29.4M + 9M instruction (big seed)
    ("A5-bigmix-final",  PROJECT_ROOT / "fine-tuning" / "ckpt_a5" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("A5-bigmix-bestval",PROJECT_ROOT / "fine-tuning" / "ckpt_a5" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("A6-2ep-final",     PROJECT_ROOT / "fine-tuning" / "ckpt_a6" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("A6-2ep-bestval",   PROJECT_ROOT / "fine-tuning" / "ckpt_a6" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("A7-bigseed-final", PROJECT_ROOT / "fine-tuning" / "ckpt_a7" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("A7-bigseed-bestval",PROJECT_ROOT / "fine-tuning" / "ckpt_a7" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # Plan D (2026-09-13): instruction-ratio sweep -- A8 = 29.4M signature + 14.4M instruction (2:1); A9 = 29.4M + 29.4M (1:1)
    ("A8-2to1-final",    PROJECT_ROOT / "fine-tuning" / "ckpt_a8" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("A8-2to1-bestval",  PROJECT_ROOT / "fine-tuning" / "ckpt_a8" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("A9-1to1-final",    PROJECT_ROOT / "fine-tuning" / "ckpt_a9" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("A9-1to1-bestval",  PROJECT_ROOT / "fine-tuning" / "ckpt_a9" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # Gold-standard route (2026-09-13): G1 = instruction data from "reference solution + 10/10 tests passed" only (62.5M); G2 = gold + signature (91.9M)
    ("G1-gold-final",    PROJECT_ROOT / "fine-tuning" / "ckpt_g1" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G1-gold-bestval",  PROJECT_ROOT / "fine-tuning" / "ckpt_g1" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G2-goldsig-final", PROJECT_ROOT / "fine-tuning" / "ckpt_g2" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G2-goldsig-bestval",PROJECT_ROOT / "fine-tuning" / "ckpt_g2" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # G3 = gold:sig 1:1; G4 = two-stage (G1-bestval start + short pure-signature run)
    ("G3-g11-final",     PROJECT_ROOT / "fine-tuning" / "ckpt_g3" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G3-g11-bestval",   PROJECT_ROOT / "fine-tuning" / "ckpt_g3" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G4-2stage-final",  PROJECT_ROOT / "fine-tuning" / "ckpt_g4" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G4-2stage-bestval",PROJECT_ROOT / "fine-tuning" / "ckpt_g4" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # G5/G6/G7 = ratio scan (strict-criterion gold standard, 1.5:1 / 0.75:1 / 1:1)
    ("G5-g15-final",     PROJECT_ROOT / "fine-tuning" / "ckpt_g5" / "final",   PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G5-g15-bestval",   PROJECT_ROOT / "fine-tuning" / "ckpt_g5" / "best_val",PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G6-g075-final",    PROJECT_ROOT / "fine-tuning" / "ckpt_g6" / "final",   PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G6-g075-bestval",  PROJECT_ROOT / "fine-tuning" / "ckpt_g6" / "best_val",PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G7-g2to1-final",   PROJECT_ROOT / "fine-tuning" / "ckpt_g7" / "final",   PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G7-g2to1-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g7" / "best_val",PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G8-g3to1-final",   PROJECT_ROOT / "fine-tuning" / "ckpt_g8" / "final",   PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G8-g3to1-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g8" / "best_val",PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # G9/G10 = ratios after pool expansion (gold 147M / 235M : signature 29.4M ~ 5:1 / 8:1)
    ("G9-g5to1-final",   PROJECT_ROOT / "fine-tuning" / "ckpt_g9" / "final",   PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G9-g5to1-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g9" / "best_val",PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G10-g8to1-final",  PROJECT_ROOT / "fine-tuning" / "ckpt_g10" / "final",  PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G10-g8to1-bestval",PROJECT_ROOT / "fine-tuning" / "ckpt_g10" / "best_val",PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    # -- model-soup candidates: weight interpolation of two data-specialised experts; no training, eval only --
    # A7-final = signature expert (36); G8-bestval = NL expert (47); G5-final = more balanced (43/36)
    ("G8-g3to1-s43-final",   PROJECT_ROOT / "fine-tuning" / "ckpt_g8_s43" / "final",    PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G8-g3to1-s43-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g8_s43" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G8-g3to1-s44-final",   PROJECT_ROOT / "fine-tuning" / "ckpt_g8_s44" / "final",    PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G8-g3to1-s44-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g8_s44" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G8x3-bestval",    PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG8x3bv",      PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G8x3-final",      PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG8x3fi",      PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G9-g5to1-s43-final", PROJECT_ROOT / "fine-tuning" / "ckpt_g9_s43" / "final", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G9-g5to1-s43-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g9_s43" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G9-g5to1-s44-final", PROJECT_ROOT / "fine-tuning" / "ckpt_g9_s44" / "final", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G9-g5to1-s44-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g9_s44" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G5-g15-s43-final", PROJECT_ROOT / "fine-tuning" / "ckpt_g5_s43" / "final", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G5-g15-s43-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g5_s43" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G5-g15-s44-final", PROJECT_ROOT / "fine-tuning" / "ckpt_g5_s44" / "final", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G5-g15-s44-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g5_s44" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G5x3-final",    PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG5x3fi", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G5x3-bestval",  PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG5x3bv", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G9x3-bestval",  PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG9x3bv", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G9top2-bestval", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG9top2bv", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G9x5-bestval",   PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG9x5bv",   PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G9-g5to1-s45-final", PROJECT_ROOT / "fine-tuning" / "ckpt_g9_s45" / "final", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G9-g5to1-s45-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g9_s45" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G9-g5to1-s46-final", PROJECT_ROOT / "fine-tuning" / "ckpt_g9_s46" / "final", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("G9-g5to1-s46-bestval", PROJECT_ROOT / "fine-tuning" / "ckpt_g9_s46" / "best_val", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-A7G8-020", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sA7G8_020", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),  # noqa
    ("SOUP-G9G5-030", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG9G5_030", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G9G5-050", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG9G5_050", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G9G5-070", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG9G5_070", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G9G8-050", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG9G8_050", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G10G5-050", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG10G5_050", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-A7G8-035", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sA7G8_035", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-A7G8-050", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sA7G8_050", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-A7G8-065", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sA7G8_065", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-A7G8-080", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sA7G8_080", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G5G8-050", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG5G8_050", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
    ("SOUP-G8self-050", PROJECT_ROOT / "fine-tuning" / "ck_soup" / "sG8self_050", PROJECT_ROOT / "tokenizer" / "tokenizer.json"),
]
DEFAULT_SEEDS = [0, 42, 123, 999, 2024]


# ===================================================================
# Generic architecture (read from checkpoint config; supports 353M 18L and 435M 22L)
# ===================================================================
class ModelConfig:
    vocab_size = 32000
    d_model = 1024
    num_layers = 18
    num_heads = 16
    d_ff = 3840
    max_seq_len = 1024
    rope_theta = 500000.0
    dropout_rate = 0.0

    @property
    def head_dim(self):
        return self.d_model // self.num_heads


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return ((x / rms) * self.scale.float()).to(orig)


def precompute_rope(dim, max_len, theta=500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    B, T, H, D = x.shape
    cos = cos[:T, :].view(1, T, 1, D // 2)
    sin = sin[:T, :].view(1, T, 1, D // 2)
    xr = x.view(B, T, H, 2, D // 2)
    x_cos = xr[..., 0, :] * cos - xr[..., 1, :] * sin
    x_sin = xr[..., 1, :] * cos + xr[..., 0, :] * sin
    return torch.stack([x_cos, x_sin], dim=-2).reshape(B, T, H, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.d_model
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        H = self.config.num_heads
        hd = self.config.head_dim
        qkv = self.qkv(x).view(B, T, H, 3, hd)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(attn_out.transpose(1, 2).reshape(B, T, D))


class SwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CodeLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed(input_ids) * (self.config.d_model ** 0.5)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.lm_head(self.final_norm(x))


# ===================================================================
# Load
# ===================================================================
def load_tokenizer(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    if tok.token_to_id("<pad>") is None:
        tok.add_special_tokens(["<pad>"])
    return tok


def load_model(ckpt_dir, device):
    ckpt = torch.load(str(Path(ckpt_dir) / "checkpoint.pt"),
                      map_location="cpu", weights_only=False)
    config = ModelConfig()
    for k, v in ckpt.get("config", {}).items():
        if hasattr(config, k):
            setattr(config, k, v)
    model = CodeLLM(config)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    del ckpt, sd
    gc.collect()
    model.eval()
    model.to(device).to(torch.bfloat16)
    return model


# ===================================================================
# Single-sample generation (same params as ablation_eval.py)
# ===================================================================
@torch.no_grad()
def generate_one(model, tokenizer, prompt, seed, max_new=256, temperature=0.2,
                 top_k=50, repetition_penalty=1.2, min_p=0.05, device="cuda"):
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

    pad_id = tokenizer.token_to_id("<pad>") or 0
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("</s>")

    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new):
        if input_ids.shape[1] > model.config.max_seq_len:
            input_ids = input_ids[:, -model.config.max_seq_len:]
        logits = model(input_ids)
        next_logits = logits[0, -1, :].float() / temperature

        for tid in set(ids):
            if next_logits[tid] < 0:
                next_logits[tid] *= repetition_penalty
            else:
                next_logits[tid] /= repetition_penalty

        if top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, min(top_k, next_logits.numel()))
            probs = F.softmax(topk_vals, dim=-1)
            if min_p > 0:
                keep = probs >= min_p * probs.max()
                probs = probs[keep]
                topk_idx = topk_idx[keep]
                if probs.numel() == 0:
                    break
                probs = probs / probs.sum()
            tok = topk_idx[torch.multinomial(probs, 1)].item()
        else:
            probs = F.softmax(next_logits, dim=-1)
            tok = torch.multinomial(probs, 1).item()

        if tok == pad_id or (eos_id is not None and tok == eos_id):
            break
        generated.append(tok)
        ids.append(tok)
        input_ids = torch.cat([input_ids, torch.tensor([[tok]], device=device)], dim=1)
        if len(generated) >= 5 and len(set(generated[-5:])) == 1:
            break

    return tokenizer.decode(generated) if generated else ""


# ===================================================================
# Execution check (reference only; conclusions rest on hand audit)
# ===================================================================
def truncate(completion):
    stops = ["\ndef ", "\nclass ", "\n# ", "\n@", "\nif __name__", "\nprint("]
    idx = len(completion)
    for s in stops:
        i = completion.find(s)
        if i != -1:
            idx = min(idx, i)
    return completion[:idx]


def run_script(script, timeout=3):
    try:
        r = subprocess.run([sys.executable, "-c", script], timeout=timeout,
                           capture_output=True, text=True)
        return r.returncode == 0 and "PASS" in r.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def check_algo(prompt, completion, entry, tests):
    full = prompt + truncate(completion)
    asserts = "\n".join("assert " + t for t in tests)
    return run_script("%s\n\n%s\nprint('PASS')\n" % (full, asserts))


def check_humaneval(prompt, completion, entry, test_code):
    full = prompt + truncate(completion)
    return run_script("%s\n\n%s\n\ncheck(%s)\nprint('PASS')\n" % (full, test_code, entry),
                      timeout=5)


# ===================================================================
# Eval set
# ===================================================================
ALGORITHMS = [
    {"name": "fibonacci", "prompt": "def fibonacci(n):\n    ", "entry": "fibonacci",
     "tests": ["fibonacci(0) == 0", "fibonacci(1) == 1",
               "fibonacci(10) == 55", "fibonacci(20) == 6765"]},
    {"name": "quicksort", "prompt": "def quicksort(arr):\n    ", "entry": "quicksort",
     "tests": ["quicksort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]",
               "quicksort([]) == []", "quicksort([5]) == [5]"]},
    {"name": "binary_search", "prompt": "def binary_search(arr, target):\n    ",
     "entry": "binary_search",
     "tests": ["binary_search([1, 3, 5, 7, 9], 5) == 2",
               "binary_search([1, 3, 5, 7, 9], 6) == -1",
               "binary_search([], 1) == -1"]},
    {"name": "two_sum", "prompt": "def two_sum(nums, target):\n    ", "entry": "two_sum",
     "tests": ["sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]",
               "sorted(two_sum([3, 2, 4], 6)) == [1, 2]",
               "sorted(two_sum([3, 3], 6)) == [0, 1]"]},
    {"name": "mergesort", "prompt": "def mergesort(arr):\n    ", "entry": "mergesort",
     "tests": ["mergesort([38, 27, 43, 3, 9, 82, 10]) == [3, 9, 10, 27, 38, 43, 82]",
               "mergesort([]) == []", "mergesort([1]) == [1]"]},
]

# Simple tasks with docstrings: tests docstring understanding + general code generation
SIMPLE_TASKS = [
    {"name": "is_palindrome", "prompt": "def is_palindrome(s):\n    \"\"\"Return True if s is a palindrome.\"\"\"\n    ",
     "entry": "is_palindrome",
     "tests": ["is_palindrome('racecar') == True", "is_palindrome('hello') == False",
               "is_palindrome('') == True"]},
    {"name": "is_prime", "prompt": "def is_prime(n):\n    \"\"\"Return True if n is a prime number.\"\"\"\n    ",
     "entry": "is_prime",
     "tests": ["is_prime(2) == True", "is_prime(17) == True", "is_prime(15) == False",
               "is_prime(1) == False"]},
    {"name": "reverse_string", "prompt": "def reverse_string(s):\n    \"\"\"Return the reverse of s.\"\"\"\n    ",
     "entry": "reverse_string",
     "tests": ["reverse_string('hello') == 'olleh'", "reverse_string('') == ''",
               "reverse_string('a') == 'a'"]},
    {"name": "count_vowels", "prompt": "def count_vowels(s):\n    \"\"\"Return the number of vowels in s.\"\"\"\n    ",
     "entry": "count_vowels",
     "tests": ["count_vowels('hello') == 2", "count_vowels('aeiou') == 5",
               "count_vowels('xyz') == 0"]},
    {"name": "sum_list", "prompt": "def sum_list(nums):\n    \"\"\"Return the sum of all numbers in nums.\"\"\"\n    ",
     "entry": "sum_list",
     "tests": ["sum_list([1, 2, 3]) == 6", "sum_list([]) == 0", "sum_list([-1, 5]) == 4"]},
    {"name": "factorial", "prompt": "def factorial(n):\n    \"\"\"Return n! (n factorial).\"\"\"\n    ",
     "entry": "factorial",
     "tests": ["factorial(0) == 1", "factorial(5) == 120", "factorial(3) == 6"]},
]


def load_humaneval(path):
    problems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


# ===================================================================
# Main flow
# ===================================================================
def main():
    ap = argparse.ArgumentParser(description="cross-model comparison eval")
    ap.add_argument("--models", type=str, default="all")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-humaneval", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    device = args.device
    gen = dict(max_new=args.max_new, temperature=args.temperature, top_k=args.top_k,
               repetition_penalty=args.repetition_penalty, min_p=args.min_p, device=device)

    model_map = {name: (ck, tok) for name, ck, tok in MODELS}
    if args.models == "all":
        selected = MODELS
    else:
        selected = [(n, ck, tok) for n, ck, tok in MODELS if n in args.models.split(",")]
        if not selected:      # a typo used to silently evaluate 0 models and write an empty summary -> now it fails hard
            print("[ERROR] no model name matched: %s" % args.models)
            print("        available (first 12):", ", ".join(n for n, _c, _t in MODELS[:12]), "...")
            sys.exit(2)

    he_problems = load_humaneval(HUMANEVAL_PATH) if not args.skip_humaneval else []
    print("[DEV] %s | HumanEval %d problems | seeds=%s" % (device, len(he_problems), seeds))

    all_samples = []
    summaries = []

    for name, ckpt_dir, tok_path in selected:
        print("\n" + "=" * 60)
        print("model: %s (%s)" % (name, ckpt_dir))
        print("=" * 60)
        t0 = time.time()
        tokenizer = load_tokenizer(tok_path)
        model = load_model(ckpt_dir, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print("  params: %.0fM | vocab=%d | layers=%d | d_ff=%d" % (
            n_params, model.config.vocab_size, model.config.num_layers, model.config.d_ff))

        algo_pass = {}
        for algo in ALGORITHMS + SIMPLE_TASKS:
            p = 0
            for seed in seeds:
                comp = generate_one(model, tokenizer, algo["prompt"], seed, **gen)
                ok = check_algo(algo["prompt"], comp, algo["entry"], algo["tests"])
                p += (1 if ok else 0)
                all_samples.append({"model": name, "type": "task", "name": algo["name"],
                                    "seed": seed, "prompt": algo["prompt"],
                                    "completion": comp, "passed": ok})
            algo_pass[algo["name"]] = "%d/%d" % (p, len(seeds))
            print("  %-16s %s/%d" % (algo["name"], p, len(seeds)))

        he_pass = 0
        if he_problems:
            for prob in he_problems:
                comp = generate_one(model, tokenizer, prob["prompt"], seeds[0], **gen)
                ok = check_humaneval(prob["prompt"], comp, prob["entry_point"], prob["test"])
                he_pass += (1 if ok else 0)
                all_samples.append({"model": name, "type": "humaneval",
                                    "name": prob["task_id"], "seed": seeds[0],
                                    "entry_point": prob["entry_point"],
                                    "prompt": prob["prompt"],
                                    "completion": comp, "passed": ok})
            print("  HumanEval pass@1: %d/%d (%.2f%%)" % (he_pass, len(he_problems),
                                                          100.0 * he_pass / len(he_problems)))

        summaries.append({"model": name, "params_m": round(n_params, 1),
                          "layers": model.config.num_layers,
                          "tasks": algo_pass,
                          "humaneval": "%d/%d" % (he_pass, len(he_problems)) if he_problems else "N/A",
                          "elapsed_s": round(time.time() - t0, 1)})

        del model
        gc.collect()
        torch.cuda.empty_cache()

    out_dir = Path(__file__).resolve().parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_sft_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    with open(out_dir / "eval_sft_generations.jsonl", "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("\n[SAVE] eval_sft_summary.json + eval_sft_generations.jsonl (%d rows)" % len(all_samples))


if __name__ == "__main__":
    main()
