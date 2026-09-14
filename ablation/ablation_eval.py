#!/usr/bin/env python3
"""
435M-v2 data-scaling ablation evaluation (clean rewrite from scratch)
========================================================================

- evaluate 6 checkpoints one at a time (never all in VRAM at once)
- 5 algorithms x N seeds + HumanEval pass@1
- per-sample generation (no batch, no padding, no attention mask - simplest and most stable)
- execution via subprocess + timeout to prevent infinite loops
- saves: summary json + detailed generations jsonl

Usage (Windows):
  py -3.12 ablation/ablation_eval.py
  py -3.12 ablation/ablation_eval.py --checkpoints 1B,31B
  py -3.12 ablation/ablation_eval.py --skip-humaneval
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

# paths
ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parent
CKPT_BASE = PROJECT_ROOT / "checkpoints_wsm"
HUMANEVAL_PATH = ABLATION_DIR / "HumanEval.jsonl"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer_435m.json"

CHECKPOINTS = [
    ("1B", "step_020000"),
    ("2B", "step_045000"),
    ("5B", "step_110000"),
    ("10B", "step_210000"),
    ("20B", "step_420000"),
    ("31B", "merged"),
]
DEFAULT_SEEDS = [0, 42, 123, 999, 2024]


# ═══════════════════════════════════════════════════════════════════
# model architecture (inline, no attn_mask; RMSNorm kept in float to avoid dtype mismatch)
# ═══════════════════════════════════════════════════════════════════
class ModelConfig:
    vocab_size = 32000
    d_model = 1024
    num_layers = 22
    num_heads = 16
    d_ff = 4096
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


# ═══════════════════════════════════════════════════════════════════
# load
# ═══════════════════════════════════════════════════════════════════
def load_tokenizer():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
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


# ═══════════════════════════════════════════════════════════════════
# per-sample generation (no batch / padding / mask)
# ═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════
# execution (subprocess isolation + timeout to prevent hangs)
# ═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════
# 5 algorithms (docstring-free, trailing 4-space indent to complete the body)
# ═══════════════════════════════════════════════════════════════════
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


def load_humaneval(path):
    problems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="435M-v2 data-scaling ablation evaluation")
    ap.add_argument("--checkpoints", type=str, default="all")
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

    ckpt_map = dict(CHECKPOINTS)
    if args.checkpoints == "all":
        selected = CHECKPOINTS
    else:
        selected = [(k, ckpt_map[k]) for k in args.checkpoints.split(",") if k in ckpt_map]

    he_problems = load_humaneval(HUMANEVAL_PATH) if not args.skip_humaneval else []
    tokenizer = load_tokenizer()

    print("[DEV] %s" % device)
    print("[HUMANEVAL] %d problems" % len(he_problems))

    all_results = []
    for label, ckpt_dir in selected:
        ckpt_path = CKPT_BASE / ckpt_dir
        if not (ckpt_path / "checkpoint.pt").exists():
            print("[SKIP] %s does not exist: %s" % (label, ckpt_path))
            continue

        print("\n" + "=" * 60)
        print("Checkpoint: %s (%s)" % (label, ckpt_dir))
        print("=" * 60)
        model = load_model(ckpt_path, device)

        samples = []
        algo_pass = {}
        for algo in ALGORITHMS:
            p = 0
            for seed in seeds:
                comp = generate_one(model, tokenizer, algo["prompt"], seed, **gen)
                ok = check_algo(algo["prompt"], comp, algo["entry"], algo["tests"])
                p += (1 if ok else 0)
                samples.append({"type": "algorithm", "name": algo["name"], "seed": seed,
                                "prompt": algo["prompt"], "completion": comp, "passed": ok})
            algo_pass[algo["name"]] = "%d/%d" % (p, len(seeds))

        he_pass = 0
        if he_problems:
            for prob in he_problems:
                comp = generate_one(model, tokenizer, prob["prompt"], seeds[0], **gen)
                ok = check_humaneval(prob["prompt"], comp, prob["entry_point"], prob["test"])
                he_pass += (1 if ok else 0)
                samples.append({"type": "humaneval", "name": prob["task_id"], "seed": seeds[0],
                                "entry_point": prob["entry_point"], "prompt": prob["prompt"],
                                "completion": comp, "passed": ok})

        res = {
            "label": label,
            "ckpt": ckpt_dir,
            "algorithms": algo_pass,
            "humaneval": "%d/%d" % (he_pass, len(he_problems)) if he_problems else "N/A",
            "humaneval_pass1": round(100.0 * he_pass / max(len(he_problems), 1), 2)
            if he_problems else None,
            "samples": samples,
        }
        all_results.append(res)

        print("  algorithm: " + "  ".join("%s=%s" % (k, v) for k, v in algo_pass.items()))
        if he_problems:
            print("  HumanEval pass@1: %s (%.2f%%)" % (res["humaneval"], res["humaneval_pass1"]))

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # summary table
    print("\n" + "=" * 60)
    print("\n[A] ablation summary (data scaling)")
    print("=" * 60)
    algo_names = [a["name"] for a in ALGORITHMS]
    header = "%-6s" % "scale"
    for n in algo_names:
        header += "%-12s" % n
    header += "%-12s" % "HumanEval"
    print(header)
    for res in all_results:
        row = "%-6s" % res["label"]
        for n in algo_names:
            row += "%-12s" % res["algorithms"].get(n, "N/A")
        row += "%-12s" % res["humaneval"]
        print(row)

    # save
    summary = [{k: v for k, v in res.items() if k != "samples"} for res in all_results]
    with open(ABLATION_DIR / "ablation_scaling_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    n = 0
    with open(ABLATION_DIR / "ablation_generations.jsonl", "w", encoding="utf-8") as f:
        for res in all_results:
            for s in res["samples"]:
                s["checkpoint"] = res["label"]
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                n += 1
    print("\n[SAVE] summary: ablation_scaling_results.json")
    print("[SAVE] generations: %d rows -> ablation_generations.jsonl" % n)


if __name__ == "__main__":
    main()
