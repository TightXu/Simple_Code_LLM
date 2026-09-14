#!/usr/bin/env python3
"""
Shared training infrastructure for all ablation scripts.
"""

import os, sys, json, math, time, random, argparse, glob
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# paths (relative to this file, portable to Windows)
ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parent          # repo root/ (LLM github/)
DATA_DIR = PROJECT_ROOT / "experiments" / "data"   # training data lives under experiments/ (gitignored)
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer_435m.json"  # this repo layout
CODE_LLM_PROJECT = PROJECT_ROOT              # compat: legacy references
CLEAN_PRETRAIN_DIR = PROJECT_ROOT            # compat: legacy references (old scripts named the project root this way)

sys.path.insert(0, str(PROJECT_ROOT))

# model architecture (self-contained inline; 435M: 22 layers / d_ff=4096)
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    num_layers: int = 22
    num_heads: int = 16
    d_ff: int = 4096
    max_seq_len: int = 1024
    rope_theta: float = 500000.0
    dropout_rate: float = 0.0

    @property
    def head_dim(self):
        return self.d_model // self.num_heads

    @property
    def num_params(self):
        emb = self.vocab_size * self.d_model
        per_layer = 4 * self.d_model ** 2 + 3 * self.d_model * self.d_ff + 4 * self.d_model
        return int(emb + self.num_layers * per_layer + self.d_model * self.vocab_size)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


def precompute_rope_frequencies(dim: int, max_len: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    B, T, H, D = x.shape
    cos = cos[:T, :].view(1, T, 1, D // 2)
    sin = sin[:T, :].view(1, T, 1, D // 2)
    x_rot = x.view(B, T, H, 2, D // 2)
    x_cos = x_rot[..., 0, :] * cos - x_rot[..., 1, :] * sin
    x_sin = x_rot[..., 1, :] * cos + x_rot[..., 0, :] * sin
    return torch.stack([x_cos, x_sin], dim=-2).reshape(B, T, H, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        cos, sin = precompute_rope_frequencies(
            config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed(input_ids) * math.sqrt(self.config.d_model)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.lm_head(self.final_norm(x))

# ── Tokenizer ──
def load_tokenizer():
    from tokenizers import Tokenizer
    for tok_path in [TOKENIZER_PATH]:
        if tok_path.exists():
            tok = Tokenizer.from_file(str(tok_path))
            if tok.token_to_id("<pad>") is None:
                tok.add_special_tokens(["<pad>"])
            print(f"[TOK] vocab={tok.get_vocab_size()}, path={tok_path.name}")
            return tok
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")


# offline data loader
def create_offline_dataloader(bin_path, seq_len=1024, start_offset=0, max_tokens=None):
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f".bin not found: {bin_path}")
    
    meta_path = bin_path.parent / f"{bin_path.stem}_meta.json"
    total_str = "?"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        total_str = f"{meta['total_tokens']/1e9:.2f}B"
    
    file_gb = bin_path.stat().st_size / 1e9
    usable = (int(bin_path.stat().st_size / 2) - start_offset) // (seq_len + 1)
    print(f"[DATA] {bin_path.name}: {file_gb:.2f}GB, {total_str} tokens, ~{usable:,} chunks")
    
    data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
    chunk_size = seq_len + 1
    total_chunks = len(data) // chunk_size
    
    def gen():
        start_chunk = start_offset // seq_len
        yielded = 0
        for i in range(start_chunk, total_chunks):
            if max_tokens and yielded >= max_tokens:
                break
            offset = i * chunk_size
            chunk = data[offset:offset + chunk_size]
            x = torch.tensor(chunk[:seq_len], dtype=torch.long)
            y = torch.tensor(chunk[1:seq_len + 1], dtype=torch.long)
            yielded += seq_len
            yield x, y
        print(f"[DATA] stream finished: {yielded:,} tokens")
    
    return gen()


# online data loader (for quality ablation: no-filter version)
def create_online_dataloader(data_dir, tokenizer, seq_len=1024, 
                              max_tokens=None, apply_filter=True):
    """Stream parquet online, optionally applying L1-L6 filtering."""
    import pyarrow.parquet as pq
    
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    print(f"[DATA] online mode: {len(parquet_files)} files, filter={'L1-L6' if apply_filter else 'off'}")
    
    if apply_filter:
        from pretrain_data import (
            check_l1_framework, check_l2_python2, check_l3_config,
            check_l4_tests, compute_quality_metrics, check_l6_autogen,
        )
    
    def filter_code(code):
        if not apply_filter:
            return True
        for check in [check_l1_framework, check_l2_python2, 
                       check_l3_config, check_l4_tests]:
            bad, _ = check(code)
            if bad: return False
        passed, _ = compute_quality_metrics(code)
        if not passed: return False
        bad, _ = check_l6_autogen(code)
        return not bad
    
    buffer = []
    total_yielded = 0
    
    def gen():
        nonlocal total_yielded, buffer
        for pf in tqdm(parquet_files, desc="Online", unit="file"):
            table = pq.read_table(pf)
            path_col = table.column("path")
            content_col = table.column("content")
            for i in range(len(table)):
                try:
                    if not str(path_col[i].as_py()).endswith(".py"):
                        continue
                    code = str(content_col[i].as_py())
                    if len(code) < 50:
                        continue
                    if not filter_code(code):
                        continue
                    ids = tokenizer.encode(code).ids
                    buffer.extend(ids)
                    while len(buffer) >= seq_len + 1:
                        chunk = buffer[:seq_len + 1]
                        buffer = buffer[seq_len:]
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        total_yielded += seq_len
                        yield x, y
                        if max_tokens and total_yielded >= max_tokens:
                            return
                except Exception:
                    continue
        print(f"[DATA] online stream finished: {total_yielded:,} tokens")
    
    return gen()


# ── Checkpoint I/O ──
def save_checkpoint(model, optimizer, scheduler, config, state, path):
    import shutil
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "config": vars(config) if hasattr(config, '__dict__') else config,
        "training_state": state,
        "timestamp": datetime.now().isoformat(),
    }
    t0 = time.time()
    torch.save(ckpt, tmp / "checkpoint.pt")
    
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)
    
    file_size = (path / "checkpoint.pt").stat().st_size / 1e9
    print(f"[SAVE] {path.name}: step={state['step']}, tokens={state['total_tokens']/1e9:.2f}B, "
          f"{file_size:.2f}GB, save={time.time()-t0:.1f}s")


def load_checkpoint(path_or_dir, device="cpu"):
    path = Path(path_or_dir)
    ckpt_file = path / "checkpoint.pt" if path.is_dir() else path
    print(f"[LOAD] {ckpt_file} ({ckpt_file.stat().st_size/1e9:.2f}GB)")
    ckpt = torch.load(str(ckpt_file), map_location=device, weights_only=False)
    
    config = ModelConfig()
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    model = CodeLLM(config)
    model.load_state_dict(ckpt["model"])
    
    state = ckpt.get("training_state", {
        "step": ckpt.get("step", 0),
        "total_tokens": ckpt.get("total_tokens", 0),
        "best_val_loss": float("inf"),
    })
    
    print(f"[LOAD] step={state['step']}, tokens={state['total_tokens']/1e9:.2f}B")
    return model, config, ckpt.get("optimizer"), ckpt.get("scheduler"), state


# evaluation
@torch.no_grad()
def evaluate(model, val_bin, config, device, seq_len=1024, max_batches=50, pad_id=0):
    model.eval()
    if not Path(val_bin).exists():
        model.train()
        return None
    
    data = np.memmap(str(val_bin), dtype=np.uint16, mode='r')
    chunk_size = seq_len + 1
    total_chunks = min(len(data) // chunk_size, max_batches)
    
    total_loss = 0.0
    total_tokens = 0
    for i in range(total_chunks):
        offset = i * chunk_size
        chunk = data[offset:offset + chunk_size]
        x = torch.tensor(chunk[:seq_len], dtype=torch.long).unsqueeze(0).to(device)
        y = torch.tensor(chunk[1:seq_len + 1], dtype=torch.long).unsqueeze(0).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1), ignore_index=pad_id)
        n_tok = (y != pad_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    
    model.train()
    return total_loss / max(total_tokens, 1)


# training loop (core)
def run_ablation_training(
    # experiment tag
    exp_name: str,
    exp_dir: str,
    # data
    data_gen,
    val_bin: str = None,
    # model
    config: ModelConfig = None,
    resume_from: str = None,
    # hyperparameters
    batch_size: int = 10,
    grad_accum: int = 4,
    lr: float = 2.5e-4,
    warmup_steps: int = 500,
    t_max: int = 50000,
    max_tokens: int = None,
    max_steps: int = None,
    # logging
    save_every: int = 500,
    eval_every: int = 500,
    log_every: int = 10,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEV] {device}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "training_log.csv"
    
    # model
    if config is None:
        config = ModelConfig()
    
    state = {"step": 0, "total_tokens": 0, "best_val_loss": float("inf")}
    
    if resume_from:
        model, loaded_config, opt_state, sched_state, state = load_checkpoint(resume_from, device)
        # use the checkpoint's architecture
        for k in ["d_model", "num_layers", "num_heads", "d_ff"]:
            setattr(config, k, getattr(loaded_config, k))
    else:
        model = CodeLLM(config).to(device)
        print(f"[INIT] training from scratch: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    config.max_seq_len = config.max_seq_len  # keep current
    
    n_params = sum(p.numel() for p in model.parameters())
    per_step_tok = batch_size * config.max_seq_len
    per_optim_tok = per_step_tok * grad_accum
    
    print(f"[ARCH] {n_params/1e6:.1f}M params, d={config.d_model}, L={config.num_layers}, "
          f"h={config.num_heads}, d_ff={config.d_ff}")
    print(f"[HP] BS={batch_size}×GA={grad_accum}, LR={lr:.1e}, warmup={warmup_steps}")
    
    if max_tokens:
        print(f"[GOAL] target tokens: {max_tokens/1e9:.2f}B (~{max_tokens//per_optim_tok} steps)")
    
    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=lr * 0.1)
    
    if resume_from:
        try:
            optimizer.load_state_dict(opt_state)
            scheduler.load_state_dict(sched_state)
            print("[RESUME] optimizer + scheduler restored")
        except Exception as e:
            print(f"[RESUME] [WARN]  restore failed: {e}")
    
    # ── Dtype ──
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"[DTYPE] {dtype}, AMP={'[OK]' if use_amp else '[FAIL]'}")
    
    # ── Tokenizer (for pad_id) ──
    tokenizer = load_tokenizer()
    pad_id = tokenizer.token_to_id("<pad>") or 0
    
    # training loop
    print(f"\n{'='*60}")
    print(f"[START] {exp_name}")
    print(f"{'='*60}\n")
    
    model.train()
    step = state["step"]
    total_tokens = state["total_tokens"]
    best_val_loss = state["best_val_loss"]
    
    step_losses = []
    accum_loss = 0.0
    accum_count = 0
    cycle_tokens = 0
    batch_xs, batch_ys = [], []
    
    t_start = time.time()
    first_step = True
    optimizer.zero_grad()
    
    try:
        for x_seq, y_seq in data_gen:
            batch_xs.append(x_seq)
            batch_ys.append(y_seq)
            
            if len(batch_xs) < batch_size:
                continue
            
            x_batch = torch.stack(batch_xs).to(device)
            y_batch = torch.stack(batch_ys).to(device)
            batch_xs, batch_ys = [], []
            
            # Warmup
            if step < warmup_steps:
                wlr = lr * (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = wlr
            
            # Forward + backward
            with torch.amp.autocast("cuda", dtype=dtype):
                logits = model(x_batch)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    y_batch.view(-1), ignore_index=pad_id)
                if grad_accum > 1:
                    loss = loss / grad_accum
            
            loss.backward()
            accum_loss += loss.item() * (grad_accum if grad_accum > 1 else 1)
            accum_count += 1
            cycle_tokens += x_batch.shape[0] * x_batch.shape[1]
            
            # Optimizer step
            if accum_count >= grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if step >= warmup_steps:
                    scheduler.step()
                
                step += 1
                total_tokens += cycle_tokens
                avg_loss = accum_loss / grad_accum
                step_losses.append(avg_loss)
                accum_loss, accum_count, cycle_tokens = 0.0, 0, 0
                
                state["step"] = step
                state["total_tokens"] = total_tokens
                
                # first-step confirmation
                if first_step:
                    first_step = False
                    elapsed = time.time() - t_start
                    tok_per_sec = total_tokens / max(elapsed, 1)
                    print(f"[1ST] Step {step} | {total_tokens/1e6:.1f}M tok | "
                          f"Loss:{avg_loss:.4f} | LR:{optimizer.param_groups[0]['lr']:.2e} | "
                          f"{tok_per_sec:,.0f} tok/s")
                
                # logging
                if step % log_every == 0:
                    elapsed = time.time() - t_start
                    tok_per_sec = total_tokens / max(elapsed, 1)
                    recent_loss = np.mean(step_losses[-log_every:])
                    lr_now = optimizer.param_groups[0]["lr"]
                    eta = ""
                    if max_tokens:
                        eta_h = (max_tokens - total_tokens) / max(tok_per_sec, 1) / 3600
                        eta = f" ETA:{eta_h:.1f}h"
                    print(f"Step {step:6d} | {total_tokens/1e9:.2f}B tok | "
                          f"Loss:{recent_loss:.4f} | LR:{lr_now:.2e} | {tok_per_sec:,.0f} tok/s{eta}")
                
                # validation
                if val_bin and eval_every and step % eval_every == 0:
                    val_loss = evaluate(model, val_bin, config, device, 
                                       seq_len=config.max_seq_len, pad_id=pad_id)
                    if val_loss is not None:
                        print(f"  [CHART] Val: {val_loss:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            state["best_val_loss"] = best_val_loss
                
                # save checkpoint
                if step % save_every == 0:
                    save_dir = exp_dir / f"step_{step:06d}"
                    save_checkpoint(model, optimizer, scheduler, config, state, save_dir)
                
                # termination condition
                if max_tokens and total_tokens >= max_tokens:
                    print(f"\n[OK] reached target tokens: {total_tokens/1e9:.2f}B")
                    break
                if max_steps and step >= max_steps:
                    print(f"\n[OK] reached step limit: {step}")
                    break
    
    except KeyboardInterrupt:
        print("\n[PAUSE] manually interrupted")
    
    # final save
    final_path = exp_dir / "final"
    save_checkpoint(model, optimizer, scheduler, config, state, final_path)
    
    elapsed = time.time() - t_start
    print(f"\n[OK] {exp_name} finished")
    print(f"   Steps: {step}, Tokens: {total_tokens/1e9:.2f}B, Time: {elapsed/3600:.1f}h")
    if best_val_loss < float("inf"):
        print(f"   Best val: {best_val_loss:.4f}")
    print(f"   Output: {final_path}")
    
    return state
