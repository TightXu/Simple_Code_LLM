#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════
CodeLM-435M — SFT supervised fine-tuning (prompt-span loss mask + packing + epochs)
══════════════════════════════════════════════════════════════════════
Copied from pretrain_train_wsm.py with minimal changes (2026-09-11, isolated 435M-v2/fine-tuning/ workspace)

Changes relative to the pretraining script:
  1. loss mask: compute loss only on "answer-span" tokens, ignore every prompt token (per-token mask)
  2. packing: the data is already a fixed-length stream of concatenated samples, never padded
     (model forward hardcodes is_causal=True and takes no attention_mask → padding always leaks silently)
  3. --epochs: reread the same flat .bin for several passes on small datasets (no cross-sample continuation)
  4. FT default hyperparameters: lr 5e-5 / warmup 20 / epochs 1 / save+eval every 200 steps / bf16 / grad-clip 1.0
  5. --reset-optimizer plus step reset by default (SFT takes the pretrained weights only, inherits no step/tokens/max_tokens)
  6. ckpt defaults to fine-tuning/ckpt/, log to fine-tuning/logs/ (with an isolation guard)

Usage:
  py -3.12 sft_train.py --bin-data data/sft_train.bin --mask-bin data/sft_train_mask.bin \
      --val-bin ../data/all/val.bin --sft-val-bin data/sft_val.bin \
      --resume ../checkpoints_wsm/merged --epochs 1 --ckpt-dir ckpt --log-dir logs
══════════════════════════════════════════════════════════════════════
"""

import os, sys, json, math, time, random, argparse, glob
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ═══════════════════ Dependency check (Windows-compatible) ═══════════════════

def _check_deps():
    import importlib, subprocess
    deps = {"torch": "torch", "datasets": "datasets", "tokenizers": "tokenizers",
            "tqdm": "tqdm", "numpy": "numpy"}
    ok, missing = [], []
    for mod, pkg in deps.items():
        try:
            importlib.import_module(mod)
            ok.append(mod)
        except ImportError:
            missing.append((mod, pkg))
            print(f"[DEPS] Installing {pkg}...")
            cmd = [sys.executable, "-m", "pip", "install",
                   "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", pkg]
            if sys.platform != "win32":
                cmd.insert(4, "--break-system-packages")
            subprocess.run(cmd, check=True)
            ok.append(mod)
    print(f"[DEPS] Loaded: {', '.join(ok)}")

_check_deps()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ═══════════════════ Path resolution ═══════════════════

CLEAN_PRETRAIN_DIR = Path(__file__).resolve().parent

print(f"[PATH] Script:  {Path(__file__).resolve()}")
print(f"[PATH] Project: {CLEAN_PRETRAIN_DIR}")


# ═══════════════════ Model architecture (self-contained) ═══════════════════

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    num_layers: int = 18
    num_heads: int = 16
    d_ff: int = 3840
    max_seq_len: int = 1024
    rope_theta: float = 500000.0
    dropout_rate: float = 0.0

    @property
    def head_dim(self): return self.d_model // self.num_heads

    @property
    def num_params(self):
        emb = self.vocab_size * self.d_model
        per_layer = 4 * self.d_model**2 + 3 * self.d_model * self.d_ff + 4 * self.d_model
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
        H, D = config.num_heads, config.d_model
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
        q, k, v = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
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
        cos, sin = precompute_rope_frequencies(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
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

# ═══════════════════ Tokenizer ═══════════════════

def load_tokenizer():
    """Load the tokenizer: prefer this project, fall back to the previous version."""
    from tokenizers import Tokenizer
    candidates = [
        str(CLEAN_PRETRAIN_DIR / "tokenizer" / "tokenizer_435m.json"),           # if a tokenizer was copied into this dir
        str(CLEAN_PRETRAIN_DIR.parent / "tokenizer" / "tokenizer_435m.json"),    # read-only reference to ../tokenizer/ (the pretraining copy)
    ]
    for i, tok_path in enumerate(candidates):
        tok = Path(tok_path)
        exists = tok.exists()
        print(f"[TOK] candidate{i+1}: {tok} (exists={exists}, size={tok.stat().st_size if exists else 'N/A'})")
        if exists and tok.is_file():
            tokenizer = Tokenizer.from_file(tok_path)
            vocab = tokenizer.get_vocab_size()
            specials = {t: tokenizer.token_to_id(t) for t in ["<s>", "</s>", "<unk>", "<pad>", "<eos>"]}
            print(f"[TOK] [OK] loaded: vocab={vocab}, specials={specials}")
            if tokenizer.token_to_id("<pad>") is None:
                tokenizer.add_special_tokens(["<pad>"])
                print(f"[TOK] [WARN]  added the missing <pad> token")
            return tokenizer
    raise FileNotFoundError(f"Tokenizer not found. Searched: {candidates}")

# ═══════════════════ Constants ═══════════════════

DEFAULT_CKPT_DIR = CLEAN_PRETRAIN_DIR / "ckpt"      # SFT: writes only under fine-tuning/ckpt/
DEFAULT_LOG_DIR = CLEAN_PRETRAIN_DIR / "logs"       # SFT: writes only under fine-tuning/logs/
# Guard: ckpts in these parent dirs are pretraining/ablation assets; SFT may not write there by default
FORBIDDEN_CKPT_PATTERNS = ("checkpoints_wsm", "tokenizer_ablation", "ablation")
IS_WINDOWS = sys.platform == "win32"


def _guard_ckpt_dir(ckpt_dir: Path, allow_foreign: bool = False):
    """SFT isolation guard: checkpoints must land inside fine-tuning/ (README isolation rules 1/2)."""
    p = Path(ckpt_dir).resolve()
    if str(p).lower().startswith(str(CLEAN_PRETRAIN_DIR.resolve()).lower()):
        return
    bad = [pat for pat in FORBIDDEN_CKPT_PATTERNS if pat in p.parts]
    detail = f"hits pretraining/ablation asset dirs {bad}" if bad else "not inside the fine-tuning/ dir"
    msg = (f"[GUARD] [FAIL] --ckpt-dir={p} rejected ({detail}).\n"
           f"        SFT checkpoints always go under {DEFAULT_CKPT_DIR}; pass --allow-foreign-ckpt-dir explicitly if you really need this.")
    if allow_foreign:
        print("[GUARD] [WARN]  " + msg.replace("\n", "\n        "))
    else:
        raise SystemExit(msg)


def _count_chunks(bin_path, seq_len: int) -> int:
    """Number of complete chunks in a flat uint16 .bin (chunk = seq_len+1; a trailing remainder shorter than one chunk is dropped, no padding)."""
    p = Path(bin_path)
    if not p.exists():
        return 0
    return (p.stat().st_size // 2) // (seq_len + 1)


def _fmt_tokens(n: int) -> str:
    """SFT runs at M scale, not B -- do not print 6M tokens as 0.00B."""
    n = int(n or 0)
    return f"{n/1e9:.2f}B" if n >= 1e9 else f"{n/1e6:.2f}M"
print(f"[ENV] platform: {'Windows' if IS_WINDOWS else 'Linux/Mac'}, Python: {sys.version.split()[0]}")


# ═══════════════════ Windows compatibility: LATEST.txt instead of symlink ═══════════════════

def _set_latest_link(ckpt_base: Path, target_name: str):
    latest_file = ckpt_base / "LATEST.txt"
    if IS_WINDOWS:
        latest_file.write_text(target_name)
        print(f"[LATEST] Windows: LATEST.txt → {target_name}")
    else:
        latest_link = ckpt_base / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(target_name, target_is_directory=True)
        print(f"[LATEST] Linux: symlink latest → {target_name}")


def _read_latest_link(ckpt_base: Path) -> Optional[str]:
    if not IS_WINDOWS:
        latest_link = ckpt_base / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            try:
                return os.readlink(str(latest_link))
            except Exception:
                pass
    latest_file = ckpt_base / "LATEST.txt"
    if latest_file.exists():
        return latest_file.read_text().strip()
    return None


# ═══════════════════ Training state (incl. data position) ═══════════════════

@dataclass
class DataPosition:
    source_type: str = ""
    file_index: int = 0
    files_total: int = 0
    rows_skipped: int = 0
    bin_path: str = ""
    token_offset: int = 0

    def to_dict(self) -> dict: return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataPosition":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingState:
    step: int = 0
    total_tokens: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    max_tokens: int = 0
    round_start_tokens: int = 0  # cumulative tokens at start of current round (for multi-dataset sequential training)
    data_position: DataPosition = None

    def __post_init__(self):
        if self.data_position is None:
            self.data_position = DataPosition()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_position"] = self.data_position.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingState":
        dp = d.pop("data_position", {})
        state = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if dp:
            state.data_position = DataPosition.from_dict(dp)
        return state


# ═══════════════════ Checkpoint I/O ═══════════════════

def save_checkpoint(model, optimizer, scheduler, config, state: TrainingState, path: Path):
    import shutil
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    sd = model.state_dict()
    # torch.compile adds an _orig_mod. prefix to state_dict keys on save; strip it so checkpoints stay clean and loadable
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    ckpt = {
        "model": sd,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "config": vars(config),
        "training_state": state.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }
    t0 = time.time()
    torch.save(ckpt, tmp / "checkpoint.pt", _use_new_zipfile_serialization=False)
    save_time = time.time() - t0

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)

    file_size = (path / "checkpoint.pt").stat().st_size / 1e9
    dp = state.data_position
    pos_info = f"file#{dp.file_index}" if dp.source_type == "parquet" else f"tok={dp.token_offset:,}"
    print(f"Ckpt: {path.name} | step={state.step} | "
          f"tokens={_fmt_tokens(state.total_tokens)} | pos={pos_info} | "
          f"{file_size:.2f}GB | save={save_time:.1f}s")


def load_checkpoint(path, device="cpu") -> Tuple[nn.Module, ModelConfig, dict, dict, TrainingState]:
    path = Path(path)
    ckpt_file = path / "checkpoint.pt" if path.is_dir() else path
    file_size = ckpt_file.stat().st_size / 1e9
    print(f"[LOAD] {ckpt_file} ({file_size:.2f}GB)")
    ckpt = torch.load(str(ckpt_file), map_location=device, weights_only=False)

    config = ModelConfig()
    if "config" in ckpt and ckpt["config"]:
        for k, v in ckpt["config"].items():
            if hasattr(config, k):
                setattr(config, k, v)
    print(f"[LOAD] architecture: d_model={config.d_model}, layers={config.num_layers}, "
          f"heads={config.num_heads}, d_ff={config.d_ff}")

    model = CodeLLM(config)
    sd = ckpt["model"]
    # Compatibility with old checkpoints carrying the _orig_mod. prefix (produced when saving under torch.compile)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)

    state = TrainingState.from_dict(ckpt.get("training_state", {}))
    print(f"[LOAD] training state: step={state.step}, tokens={state.total_tokens/1e9:.2f}B, "
          f"best_val={state.best_val_loss:.4f}")
    return model, config, ckpt.get("optimizer"), ckpt.get("scheduler"), state


def find_latest_ckpt(base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_CKPT_DIR
    base_dir = Path(base_dir)
    print(f"[FIND] searching checkpoints: {base_dir} (exists={base_dir.exists()})")
    if not base_dir.exists():
        print("[FIND] directory does not exist, no checkpoint")
        return None

    target_name = _read_latest_link(base_dir)
    if target_name:
        target = base_dir / target_name
        print(f"[FIND] pointer target: {target_name} (exists={target.exists()})")
        if target.exists() and (target / "checkpoint.pt").exists():
            print(f"[FIND] [OK] found: {target}")
            return target

    # Collect every dir holding a checkpoint.pt (including nested run_*/step_*)
    candidates = []

    def _scan_dir(d: Path, depth=0):
        if depth > 2:
            return
        for child in sorted(d.iterdir()):
            if child.is_dir():
                ckpt_file = child / "checkpoint.pt"
                if ckpt_file.exists():
                    try:
                        parts = child.name.split("_")
                        step = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
                        candidates.append((step, child))
                    except (IndexError, ValueError):
                        pass
                _scan_dir(child, depth + 1)

    _scan_dir(base_dir)
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_step, best_dir = candidates[0]
        print(f"[FIND] [OK] scan found {len(candidates)} dirs, latest: step_{best_step} ({best_dir})")
        return best_dir
    print("[FIND] [FAIL] no checkpoint found")
    return None


# ═══════════════════ Data loaders ═══════════════════

def create_online_dataloader(data_dir, tokenizer, state: DataPosition, seq_len=1024):
    import pyarrow.parquet as pq
    import threading, queue
    from pretrain_data import (
        check_l1_framework, check_l2_python2, check_l3_config,
        check_l4_tests, compute_quality_metrics, check_l6_autogen,
    )

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    total_files = len(parquet_files)
    if state.file_index > 0:
        print(f"[DATA] skipping the first {state.file_index}/{total_files} parquet files")
        parquet_files = parquet_files[state.file_index:]
    state.files_total = total_files

    if not parquet_files:
        raise FileNotFoundError(f"no parquet files: {data_dir}")
    print(f"[DATA] online mode: {len(parquet_files)} files (out of {total_files}, starting at #{state.file_index})")
    print(f"[DATA] filters: L1-L6 (same as pretrain_data.py)")

    q = queue.Queue(maxsize=3)
    stop = threading.Event()
    filter_stats = {"total": 0, "passed": 0, "rejected": 0}

    def reader():
        batch = []
        for pf in parquet_files:
            if stop.is_set(): break
            try:
                table = pq.read_table(pf)
                path_col = table.column("path")
                content_col = table.column("content")
                for i in range(len(table)):
                    if stop.is_set(): break
                    try:
                        if not str(path_col[i].as_py()).endswith(".py"): continue
                        code = str(content_col[i].as_py())
                        if len(code) < 50: continue
                        filter_stats["total"] += 1
                        bad = False
                        for check in [check_l1_framework, check_l2_python2,
                                       check_l3_config, check_l4_tests]:
                            is_bad, _ = check(code)
                            if is_bad: bad = True; break
                        if bad: filter_stats["rejected"] += 1; continue
                        passed, _ = compute_quality_metrics(code)
                        if not passed: filter_stats["rejected"] += 1; continue
                        is_bad, _ = check_l6_autogen(code)
                        if is_bad: filter_stats["rejected"] += 1; continue
                        filter_stats["passed"] += 1
                        batch.append(code)
                        if len(batch) >= 128:
                            q.put(batch); batch = []
                    except Exception:
                        continue
            except Exception as e:
                print(f"  [WARN]  {Path(pf).name}: {e}")
        if batch and not stop.is_set():
            q.put(batch)
        q.put(None)
        s = filter_stats
        print(f"[DATA] filter stats: total={s['total']}, passed={s['passed']} "
              f"({s['passed']/max(s['total'],1)*100:.1f}%), rejected={s['rejected']}")

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    buffer = []
    total = 0

    def gen():
        nonlocal total
        pbar = tqdm(desc="Online", unit="tok", unit_scale=True)
        while True:
            batch = q.get()
            if batch is None: break
            for enc in tokenizer.encode_batch(batch):
                buffer.extend(enc.ids)
                while len(buffer) >= seq_len + 1:
                    chunk = buffer[:seq_len + 1]
                    buffer = buffer[seq_len:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    total += seq_len
                    pbar.update(seq_len)
                    yield x, y
        pbar.close()
        print(f"[DATA] online data stream finished: {total:,} tokens")

    return gen(), state.files_total


def create_lang_only_online_dataloader(data_dirs, tokenizer, state, seq_len=1024,
                                        min_chars=50, shuffle=True):
    """Language-only filter (keep .py) + multiple data dirs + shuffle + streaming online tokenize.
    No L1-L6 -- used for the "fair nofilter baseline": drop other languages only,
    apply no other quality filter. Column names adapt to path / max_stars_repo_path.
    data_dirs: list of dirs (or a comma-separated string)."""
    import pyarrow.parquet as pq
    import threading, queue, random
    if isinstance(data_dirs, str):
        data_dirs = [d.strip() for d in data_dirs.split(",") if d.strip()]
    parquet_files = []
    for d in data_dirs:
        parquet_files += sorted(glob.glob(os.path.join(d, "*.parquet")))
    total_files = len(parquet_files)
    if state.file_index > 0:
        parquet_files = parquet_files[state.file_index:]
    state.files_total = total_files
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files: {data_dirs}")
    print(f"[DATA] lang-only online mode: {len(parquet_files)}/{total_files} files, "
          f"language-only filter (.py) + ≥{min_chars} chars, shuffle={shuffle}")

    q = queue.Queue(maxsize=3)
    stop = threading.Event()

    def strip_meta(code):
        """Strip the star_coder <reponame> / <gh_stars> / <filename> metadata prefixes"""
        out = []
        for ln in code.split("\n"):
            ls = ln.strip()
            if ls.startswith("<reponame>") or ls.startswith("<gh_stars>") or ls.startswith("<filename>"):
                rest = ls[ls.find(">") + 1:].lstrip()
                if rest:
                    out.append(rest)
                continue
            out.append(ln)
        return "\n".join(out)

    def reader():
        batch = []
        file_list = list(parquet_files)
        if shuffle:
            random.shuffle(file_list)
        for pf in file_list:
            if stop.is_set():
                break
            try:
                table = pq.read_table(pf)
                cols = table.column_names
                pcol = "path" if "path" in cols else (
                    "max_stars_repo_path" if "max_stars_repo_path" in cols else None)
                ccol = table.column("content")
                idxs = list(range(len(table)))
                if shuffle:
                    random.shuffle(idxs)
                for i in idxs:
                    if stop.is_set():
                        break
                    try:
                        # Language filter: decide Python-ness from the path column only
                        if pcol is not None and not str(table.column(pcol)[i].as_py()).endswith(".py"):
                            continue
                        code = str(ccol[i].as_py())
                        # Strip the star_coder <reponame>/<gh_stars>/<filename> metadata prefixes
                        code = strip_meta(code)
                        if len(code) < min_chars:
                            continue
                        batch.append(code)
                        if len(batch) >= 128:
                            q.put(batch)
                            batch = []
                    except Exception:
                        continue
            except Exception as e:
                print(f"  [WARN]  {Path(pf).name}: {e}")
        if batch and not stop.is_set():
            q.put(batch)
        q.put(None)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    buffer = []
    total = 0

    def gen():
        nonlocal total
        pbar = tqdm(desc="LangOnly", unit="tok", unit_scale=True)
        while True:
            batch = q.get()
            if batch is None:
                break
            for enc in tokenizer.encode_batch(batch):
                buffer.extend(enc.ids)
                while len(buffer) >= seq_len + 1:
                    chunk = buffer[:seq_len + 1]
                    buffer[:] = buffer[seq_len:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    total += seq_len
                    pbar.update(seq_len)
                    yield x, y
        pbar.close()
        print(f"[DATA] lang-only online data stream finished: {total:,} tokens")

    return gen(), state.files_total


def create_offline_dataloader(bin_path, seq_len=1024, start_offset=0, max_tokens=None):
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f".bin not found: {bin_path}")

    file_gb = bin_path.stat().st_size / 1e9
    meta_path = bin_path.parent / f"{bin_path.stem}_meta.json"
    total_str = "?"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        total_str = f"{meta['total_tokens']/1e9:.2f}B"
    usable_chunks = (int(bin_path.stat().st_size / 2) - start_offset) // (seq_len + 1)
    print(f"[DATA] offline mode: {bin_path} ({file_gb:.2f}GB, {total_str} tokens)")
    print(f"[DATA] start_offset={start_offset:,}, max_tokens={max_tokens}, "
          f"usable chunks≈{usable_chunks:,}")

    data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
    chunk_size = seq_len + 1
    total_chunks = len(data) // chunk_size

    def gen():
        start_chunk = start_offset // seq_len
        yielded = 0
        pbar = tqdm(total=None, desc="Offline", unit="tok", unit_scale=True)
        for i in range(start_chunk, total_chunks):
            if max_tokens and yielded >= max_tokens: break
            offset = i * chunk_size
            chunk = data[offset:offset + chunk_size]
            x = torch.tensor(chunk[:seq_len], dtype=torch.long)
            y = torch.tensor(chunk[1:seq_len + 1], dtype=torch.long)
            yielded += seq_len
            pbar.update(seq_len)
            yield x, y
        pbar.close()
        print(f"[DATA] offline data stream finished: {yielded:,} tokens")

    return gen()


# ═══════════════════ SFT: loss mask / packing / epochs ═══════════════════
#
# Two loss-mask carriers (pick one; both must align token-by-token with the data stream):
#   A) separate mask file: uint8 (or any unsigned dtype), token-for-token length of the data bin
#      (mask[i]==0 → ignored, non-zero → counted in loss; values read as weights, constant scaling changes nothing)
#      naming convention: data/sft_train.bin ↔ data/sft_train_mask.bin (auto-discovered, or --mask-bin)
#   B) embedded bit15: the data bin is uint16, low 15 bits = token id (vocab=32000 < 32768),
#      high bit = loss flag (pass --loss-in-high-bit, or auto-detected when max>=vocab)
# Semantics: the mask marks *target* tokens -- within a chunk x=t[i:i+L], y=t[i+1:i+L+1],
#       so loss_mask = mask[i+1:i+L+1]; prompt span (signature) all 0, answer span and trailing <eos> all 1.
#
# packing: the data is a continuous stream of "sample → <eos> → next sample"; here we only do
#          fixed-length slicing, never padding, never reordering across samples; samples spanning chunks are safe too (the prompt span is masked anyway).

def _assert_bin_readable(bin_path, seq_len=1024, what="data"):
    """Raise a clear error for empty or half-written data files (build_sft_data.py may still be writing)."""
    p = Path(bin_path)
    if not p.exists():
        raise SystemExit(f"[DATA] [FAIL] {what} not found: {p}")
    n_bytes = p.stat().st_size
    if n_bytes < 2 * (seq_len + 1):
        raise SystemExit(
            f"[DATA] [FAIL] {what} too small/empty: {p} ({n_bytes} bytes < {2*(seq_len+1)} bytes for one chunk).\n"
            f"        If build_sft_data.py is still writing this file, wait for it to finish and retry.")


def resolve_loss_mask(mask_bin, bin_path, vocab_size, loss_in_high_bit=False,
                      no_loss_mask=False, for_val=False):
    """Decide the loss-mask carrier. Returns (kind, path); kind ∈ {"file","highbit","none"}."""
    tag = "val " if for_val else ""
    _assert_bin_readable(bin_path, what=f"{tag}data bin")
    if no_loss_mask:
        print(f"[MASK] [WARN]  explicit --no-loss-mask: {tag}loss covers every token "
              f"(degrades to plain next-token LM loss, not real SFT)")
        return "none", None
    if mask_bin:
        mp = Path(mask_bin)
        if not mp.exists():
            raise SystemExit(f"[MASK] [FAIL] mask file not found: {mp}")
        print(f"[MASK] {tag}using explicit mask file: {mp}")
        return "file", mp
    cand = Path(bin_path).parent / f"{Path(bin_path).stem}_mask.bin"
    if cand.exists():
        print(f"[MASK] {tag}auto-discovered companion mask file: {cand}")
        return "file", cand
    if loss_in_high_bit:
        print(f"[MASK] {tag}bit15 embedded-mask mode (--loss-in-high-bit)")
        return "highbit", None
    d = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
    probe = d[:min(len(d), 4_000_000)]
    mx = int(probe.max()) if len(probe) else 0
    if mx >= vocab_size:
        print(f"[MASK] [WARN]  {tag}no mask carrier specified; probed max(uint16)={mx} >= vocab({vocab_size}) "
              f"→ treating it as a bit15 embedded mask")
        return "highbit", None
    raise SystemExit(
        f"[MASK] [FAIL] no {tag}loss mask found (data max={mx} < vocab={vocab_size}).\n"
        f"        Pass --mask-bin <uint8 file>, drop a companion file {cand.name}, or add --loss-in-high-bit.\n"
        f"        Refusing to train silently with an all-ones mask: that counts loss on prompt spans too, i.e. 353M-style next-token FT.")


def masked_cross_entropy(logits, y, loss_mask, vocab_size):
    """Average only over target tokens with loss_mask==1 (prompt spans ignored).
    Within one optimizer step each micro-batch is normalized independently by its masked token count (standard masked-LM convention)."""
    per_tok = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="none")
    m = loss_mask.reshape(-1).float()
    denom = m.sum().clamp(min=1.0)
    return (per_tok * m).sum() / denom


def create_sft_dataloader(bin_path, mask_kind="file", mask_path=None, seq_len=1024,
                          epochs=1, start_epoch=0, start_offset_tokens=0,
                          replay_bin=None, replay_ratio=0.0, seed=42,
                          max_tokens=None, shuffle=False):
    """SFT data stream: each chunk yields (x, y, loss_mask, epoch).
    - Fixed-length chunk = seq_len+1 (one extra token for the y shift), never padded.
    - epochs: reread the same bin; only the first epoch starts at start_offset_tokens (resume).
    - replay_bin: optional pretraining stream (uint16, no mask) mixed in with probability replay_ratio → anti-forgetting;
      its loss_mask is all 1 (pure LM loss).
    """
    bin_path = Path(bin_path)
    _assert_bin_readable(bin_path, seq_len=seq_len, what="SFT data bin")
    meta_path = bin_path.parent / f"{bin_path.stem}_meta.json"
    total_str = "?"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        total_str = f"{meta.get('total_tokens', 0)/1e6:.2f}M"
    data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
    chunk_size = seq_len + 1
    total_chunks = len(data) // chunk_size
    n_tail = len(data) - total_chunks * chunk_size
    print(f"[DATA] SFT offline mode: {bin_path} ({bin_path.stat().st_size/1e9:.2f}GB, {total_str} tokens)")
    print(f"[DATA] chunks={total_chunks:,} × {seq_len} tok | epochs={epochs} | "
          f"dropped {n_tail} trailing tokens (partial chunk, never padded) | shuffle={shuffle}")
    if total_chunks == 0:
        raise ValueError(f"data shorter than one chunk: {len(data)} tokens < {chunk_size}")

    mask_arr = None
    if mask_kind == "file":
        mask_arr = np.memmap(str(mask_path), dtype=np.uint8, mode='r')
        if len(mask_arr) < len(data):
            raise ValueError(f"mask too short: mask={len(mask_arr):,} < data={len(data):,} ({mask_path})")
        print(f"[DATA] loss mask: {mask_path} (uint8, {len(mask_arr):,})")
    elif mask_kind == "highbit":
        print("[DATA] loss mask: embedded bit15 (high bit = loss flag)")
    else:
        print("[DATA] loss mask: all ones (--no-loss-mask)")

    replay_arr = None
    if replay_bin and replay_ratio and replay_ratio > 0:
        replay_arr = np.memmap(str(replay_bin), dtype=np.uint16, mode='r')
        r_chunks = len(replay_arr) // chunk_size
        print(f"[DATA] replay mix: {replay_bin} ({r_chunks:,} chunks), "
              f"mixed in with per-chunk probability {replay_ratio:.3f} (pure LM loss)")
        if r_chunks == 0:
            replay_arr = None
            print("[DATA] [WARN]  replay bin too small, ignored")

    rng = np.random.default_rng(seed)
    order = np.arange(total_chunks)
    if shuffle:
        rng.shuffle(order)
    per_epoch_tokens = total_chunks * seq_len
    ep0, off0 = int(start_epoch), int(start_offset_tokens)
    if off0 >= per_epoch_tokens:            # with epochs>1, token_offset runs past a whole epoch
        ep0 += off0 // per_epoch_tokens
        off0 = off0 % per_epoch_tokens
    start_chunk = off0 // seq_len
    if start_chunk >= total_chunks:
        start_chunk = 0

    def _take(src, i, with_mask):
        off = i * chunk_size
        raw_arr = np.asarray(src[off:off + chunk_size], dtype=np.uint16)
        if not with_mask or mask_kind == "none":
            return raw_arr.astype(np.int64), np.ones(chunk_size, dtype=np.int64)
        if mask_kind == "highbit":
            return (raw_arr & 0x7FFF).astype(np.int64), (raw_arr >> 15).astype(np.int64)
        return raw_arr.astype(np.int64), np.asarray(mask_arr[off:off + chunk_size], dtype=np.int64)

    def _make(src, i, with_mask):
        toks, mk = _take(src, i, with_mask)
        x = torch.from_numpy(np.ascontiguousarray(toks[:seq_len]))
        y = torch.from_numpy(np.ascontiguousarray(toks[1:seq_len + 1]))
        lm = torch.from_numpy(np.ascontiguousarray(mk[1:seq_len + 1])).float()   # target-aligned
        return x, y, lm

    def gen():
        yielded = 0
        stopped = False
        pbar = tqdm(total=total_chunks * epochs, desc="SFT", unit="chunk")
        for ep in range(ep0, int(epochs)):
            begin = start_chunk if ep == ep0 else 0
            for i in range(begin, total_chunks):
                if max_tokens and yielded >= max_tokens:
                    stopped = True
                    break
                if replay_arr is not None and rng.random() < replay_ratio:
                    x, y, lm = _make(replay_arr, int(rng.integers(0, len(replay_arr) // chunk_size)), False)
                else:
                    x, y, lm = _make(data, int(order[i]), True)
                yielded += seq_len
                pbar.update(1)
                yield x, y, lm, ep
            if stopped:
                break
        pbar.close()
        print(f"[DATA] SFT data stream finished: {yielded:,} tokens ({epochs} epoch)")

    return gen()


@torch.no_grad()
def evaluate_on_val_sft(model, val_bin_path, config, device, mask_kind="file",
                        mask_path=None, seq_len=1024, max_batches=50):
    """Masked val loss: same convention as training (loss on the answer span only)."""
    model.eval()
    if not Path(val_bin_path).exists():
        print(f"[VAL] [WARN]  no SFT val: {val_bin_path}")
        model.train()
        return None
    data = np.memmap(str(val_bin_path), dtype=np.uint16, mode='r')
    mask_arr = None
    if mask_kind == "file":
        mask_arr = np.memmap(str(mask_path), dtype=np.uint8, mode='r')
    chunk_size = seq_len + 1
    total_chunks = len(data) // chunk_size
    if total_chunks == 0:
        model.train()
        return None
    n_eval = min(total_chunks, max_batches)
    idx = np.random.choice(total_chunks, size=n_eval, replace=False)
    idx.sort()
    loss_sum, tok_sum = 0.0, 0.0
    for i in idx:
        off = int(i) * chunk_size
        raw_arr = np.asarray(data[off:off + chunk_size], dtype=np.uint16)
        if mask_kind == "highbit":
            toks = (raw_arr & 0x7FFF).astype(np.int64)
            mk = (raw_arr >> 15).astype(np.int64)
        else:
            toks = raw_arr.astype(np.int64)
            mk = (np.ones(chunk_size, dtype=np.int64) if mask_arr is None
                  else np.asarray(mask_arr[off:off + chunk_size], dtype=np.int64))
        x = torch.from_numpy(np.ascontiguousarray(toks[:seq_len])).unsqueeze(0).to(device)
        y = torch.from_numpy(np.ascontiguousarray(toks[1:seq_len + 1])).unsqueeze(0).to(device)
        lm = torch.from_numpy(np.ascontiguousarray(mk[1:seq_len + 1])).float().unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(x)
            per_tok = F.cross_entropy(logits.reshape(-1, config.vocab_size),
                                      y.reshape(-1), reduction="none")
        per_tok = per_tok.float()
        n = float(lm.sum().item())
        loss_sum += float((per_tok * lm.reshape(-1)).sum().item())
        tok_sum += n
    model.train()
    if tok_sum == 0:
        print("[VAL] [WARN]  SFT val mask is all 0, cannot compute masked loss")
        return None
    return loss_sum / tok_sum


def inspect_loss_mask(bin_path, mask_kind, mask_path, seq_len=1024, n_probe=64):
    """Startup self-check: sample chunks and report the fraction of 1s in the mask (0% or 100% are both bug signals)."""
    data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
    chunk_size = seq_len + 1
    total_chunks = len(data) // chunk_size
    n = min(total_chunks, n_probe)
    if n == 0:
        return
    mask_arr = None
    if mask_kind == "file":
        mask_arr = np.memmap(str(mask_path), dtype=np.uint8, mode='r')
    idx = np.linspace(0, total_chunks - 1, n).astype(np.int64)
    ones = zeros = 0
    for i in idx:
        off = int(i) * chunk_size
        if mask_kind == "highbit":
            raw_arr = np.asarray(data[off:off + chunk_size], dtype=np.uint16)
            mk = (raw_arr >> 15).astype(np.int64)[1:]
        elif mask_kind == "file":
            mk = np.asarray(mask_arr[off + 1:off + chunk_size], dtype=np.int64)
        else:
            mk = np.ones(seq_len, dtype=np.int64)
        ones += int(mk.sum())
        zeros += int((mk == 0).sum())
    tot = ones + zeros
    ratio = ones / max(tot, 1)
    print(f"[MASK] self-check on {n} chunks ({tot:,} target tokens total): "
          f"counted in loss {ones:,} ({ratio*100:.1f}%), ignored (prompt) {zeros:,} ({(1-ratio)*100:.1f}%)")
    if ones == 0:
        raise SystemExit("[MASK] [FAIL] no sampled token counts toward loss: mask and data are misaligned (training would learn nothing)")
    if zeros == 0:
        print("[MASK] [WARN]  no prompt span was masked in the sample (mask always 1?) -- confirm this is --no-loss-mask or pure LM data")


# ═══════════════════ Evaluation ═══════════════════

@torch.no_grad()
def evaluate_on_val(model, val_bin_path, config, device, seq_len=1024, max_batches=250, pad_id=0):
    model.eval()
    if not Path(val_bin_path).exists():
        model.train()
        return None

    data = np.memmap(str(val_bin_path), dtype=np.uint16, mode='r')
    chunk_size = seq_len + 1
    total_chunks = len(data) // chunk_size
    n_eval = min(total_chunks, max_batches)

    # Randomly sample 250 batches each time, covering the whole val bin (not fixed, to avoid systematic bias from a frozen sample)
    chunk_indices = np.random.choice(total_chunks, size=n_eval, replace=False)
    chunk_indices.sort()  # read in order, friendlier to the memmap cache

    total_loss = 0.0
    total_tokens = 0
    for i in chunk_indices:
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


# ═══════════════════ Training log ═══════════════════

def save_training_log(log_path, step, total_tokens, loss, lr, grad_norm,
                      elapsed, tok_per_sec, val_loss=None, data_pos="",
                      epoch=None, mask_ratio=None, pretrain_val_loss=None):
    log_path = Path(log_path)
    headers = ("step,epoch,total_tokens,loss,lr,grad_norm,val_loss,pretrain_val_loss,"
               "mask_ratio,elapsed_s,tok_per_sec,data_pos\n")
    if not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(headers)
    vl = f"{val_loss:.6f}" if val_loss is not None else ""
    pvl = f"{pretrain_val_loss:.6f}" if pretrain_val_loss is not None else ""
    mr = f"{mask_ratio:.4f}" if mask_ratio is not None else ""
    ep = "" if epoch is None else str(epoch)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{step},{ep},{total_tokens},{loss:.6f},{lr:.2e},{grad_norm:.4f},"
                f"{vl},{pvl},{mr},{elapsed:.1f},{tok_per_sec:.0f},{data_pos}\n")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════ Main training loop ═══════════════════

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"   BF16: {'[OK]' if torch.cuda.is_bf16_supported() else '[FAIL]'}")
        print(f"   PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 60)

    set_seed(args.seed)

    # ── Checkpoint & log dirs ──
    ckpt_base = Path(args.ckpt_dir) if args.ckpt_dir else DEFAULT_CKPT_DIR
    _guard_ckpt_dir(ckpt_base, allow_foreign=getattr(args, "allow_foreign_ckpt_dir", False))
    ckpt_base.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if hasattr(args, 'log_dir') and args.log_dir else DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_log.csv"
    print(f"[INIT] Checkpoint root: {ckpt_base}")
    print(f"[INIT] Log dir:         {log_dir}")
    print(f"[INIT] Training log:    {log_path}")

    # ── Model ──
    config = ModelConfig(d_model=args.d_model, num_layers=args.num_layers,
                         num_heads=args.num_heads, d_ff=args.d_ff)
    config.max_seq_len = args.seq_len
    config.dropout_rate = args.dropout

    model = CodeLLM(config).to(device)

    # ═══════════════════ Performance tuning (with capability checks) ═══════════════════

    cc_major = torch.cuda.get_device_capability(0)[0]

    # 1. TF32: supported on sm_80 (Ampere) and above
    #    sm_120 (RTX 5090) [OK] — fully supported on consumer Blackwell
    if cc_major >= 8:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        print(f"[PERF] TF32 enabled (sm_{cc_major}0, ~+10-15%)")
    else:
        print(f"[PERF] [WARN]  TF32 unavailable (needs sm_80+)")

    # 2. torch.compile: prefer Inductor (needs Triton), fall back to CUDA Graphs (no Triton needed)
    compile_ok = False
    compile_msg = ""
    if args.compile_mode == "none":
        compile_msg = "disabled (--compile-mode none, eager mode)"
    elif hasattr(torch, 'compile'):
        try:
            import triton
            # sm_120 (RTX 5090) workaround: BF16 autocast fusion bug (#191433)
            # max_fusion_size=1 is too conservative (no fusion = no speedup); 2-3 bypasses the three-way-fusion bug
            if cc_major >= 12:
                torch._inductor.config.max_fusion_size = 2
                print(f"[PERF] [WARN]  sm_120 workaround: max_fusion_size=2 (BF16 fusion bug, ~10-20%)")
            model = torch.compile(model, mode=args.compile_mode)
            compile_ok = True
            compile_msg = f"Inductor ({args.compile_mode}, ~+15-25%)"
        except ImportError:
            # No Triton: fall back to the CUDA Graphs backend
            try:
                model = torch.compile(model, backend="cudagraphs")
                compile_ok = True
                compile_msg = "CUDA Graphs (no Triton, ~+5-10%)"
            except Exception as e2:
                compile_msg = f"skipped: Triton unavailable, cudagraphs failed too ({e2})"
        except Exception as e:
            compile_msg = f"skipped: {e}"
    else:
        compile_msg = "PyTorch < 2.0"

    if compile_ok:
        print(f"[PERF] torch.compile enabled: {compile_msg} (sm_{cc_major}0)")
    else:
        print(f"[PERF] [WARN]  torch.compile {compile_msg}")
    compile_enabled = compile_ok  # for warmup check later

    n_params = sum(p.numel() for p in model.parameters())
    est_params = config.num_params
    print(f"\nArchitecture: {n_params:,} params ({n_params/1e6:.1f}M), estimated={est_params/1e6:.1f}M")
    print(f"   d_model={config.d_model}  layers={config.num_layers}  "
          f"heads={config.num_heads}  d_ff={config.d_ff}")
    print(f"   seq_len={config.max_seq_len}  vocab={config.vocab_size}  "
          f"dropout={config.dropout_rate}")

    eff_batch = args.batch_size * args.grad_accum
    per_step_tok = args.batch_size * args.seq_len
    per_optim_tok = per_step_tok * args.grad_accum
    print(f"\nTraining params:")
    print(f"   BS={args.batch_size} × GA={args.grad_accum} → effective batch={eff_batch}")
    print(f"   per forward: {per_step_tok:,} tok  per optimizer step: {per_optim_tok:,} tok")
    print(f"   LR={args.lr:.1e}  warmup={args.warmup_steps} steps  T_max={args.t_max}")
    print(f"   weight_decay={args.weight_decay}  grad_clip={args.grad_clip}")
    if args.max_tokens:
        est_steps = args.max_tokens // per_optim_tok
        print(f"   target {args.max_tokens/1e9:.1f}B → about {est_steps:,} steps")
        # Rough ETA based on RTX 5090 ~43K tok/s
        est_hrs = (args.max_tokens / 43000) / 3600
        print(f"   estimated {est_hrs:.0f} hours (~{est_hrs/24:.1f} days)")

    # ── Resume ──
    state = TrainingState()
    if args.max_tokens:
        state.max_tokens = args.max_tokens
    start_step = 0
    start_tokens = 0

    resume_path = None
    if args.auto_resume:
        latest = find_latest_ckpt(ckpt_base)
        if latest:
            resume_path = str(latest)
            print(f"[INIT] auto-resume: using {latest}")
        elif args.resume:
            # [WARN] 2026-09-12 fix: the old logic was `elif args.resume`, so passing --auto-resume together with --resume
            #    silently trained from scratch (Arm3 wasted 8 minutes locally: loss 5.5 / pretrain_val 6.6)
            resume_path = args.resume
            print(f"[INIT] [WARN] auto-resume found no ckpt → falling back to --resume {args.resume}")
        else:
            print("[INIT] no existing checkpoint, training from scratch")
    elif args.resume:
        resume_path = args.resume

    if resume_path:
        model, loaded_config, opt_state, sched_state, state = load_checkpoint(resume_path, device)
        model = model.to(device)  # make sure the model is on the right device

        # Architecture params: the checkpoint wins
        structural = ["d_model", "num_layers", "num_heads", "d_ff"]
        for k in structural:
            ckpt_val = getattr(loaded_config, k, None)
            cli_val = getattr(args, k, None)
            if cli_val is not None and ckpt_val is not None and cli_val != ckpt_val:
                print(f"[RESUME] [WARN]  --{k}={cli_val} != ckpt({ckpt_val}), using the checkpoint value")
                setattr(config, k, ckpt_val)
        config.max_seq_len = args.seq_len
        config.dropout_rate = args.dropout

        start_step = state.step
        start_tokens = state.total_tokens
        # max_tokens: a pretraining ckpt stores the "pretraining token budget"; SFT does not inherit it by default
        if state.max_tokens and not args.max_tokens and getattr(args, "inherit_max_tokens", False):
            args.max_tokens = state.max_tokens
            print(f"[RESUME] inheriting ckpt max_tokens={args.max_tokens:,}")
        elif state.max_tokens and not args.max_tokens:
            print(f"[SFT] ignoring ckpt max_tokens={state.max_tokens:,} "
                  f"(that is the pretraining budget; add --inherit-max-tokens if you need it)")
        print(f"[RESUME] step={start_step}, tokens={_fmt_tokens(start_tokens)}")
        dp = state.data_position
        print(f"[RESUME] data position: type={dp.source_type}, "
              f"file_idx={dp.file_index}/{dp.files_total}, "
              f"bin_path={dp.bin_path}, token_offset={dp.token_offset:,}")

    else:
        print("[INIT] training from scratch (no resume)")
        sched_state = None
        opt_state = None

    # ── SFT: take pretrained weights only; reset step/tokens/data position (a fresh FT round, warmup from the start) ──
    if resume_path and getattr(args, "reset_optimizer", False) and not getattr(args, "continue_step", False):
        print(f"[SFT] weights only: step {start_step}→0, tokens {_fmt_tokens(start_tokens)}→0, "
              f"epoch/data_position reset (disable with --continue-step)")
        start_step, start_tokens = 0, 0
        state.step, state.total_tokens = 0, 0
        state.epoch = 0
        state.round_start_tokens = 0
        state.data_position.token_offset = 0
        state.best_val_loss = float("inf")
        opt_state = None

    # ═══════════════ re-apply torch.compile (resume replaces the model with a fresh uncompiled object)═══════════════
    if compile_enabled and resume_path:
        try:
            import triton
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[PERF] torch.compile ({args.compile_mode}) re-applied to the resumed model")
        except ImportError:
            try:
                model = torch.compile(model, backend="cudagraphs")
                print("[PERF] torch.compile (cudagraphs) re-applied to the resumed model")
            except Exception as e:
                print(f"[PERF] [WARN] recompile after resume failed: {e}")
        except Exception as e:
            print(f"[PERF] [WARN] recompile after resume failed: {e}")

    # ── Optimizer (must be created after resume/compile so it references the final model's params) ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    if resume_path and opt_state is not None and not args.reset_optimizer:
        try:
            optimizer.load_state_dict(opt_state)
            print("[RESUME] optimizer state restored")
        except Exception as e:
            print(f"[RESUME] [WARN]  optimizer incompatible: {e}")
    elif resume_path and args.reset_optimizer:
        print("[RESUME] rebuilding the optimizer (not loading old state)")

    # ── Scheduler ──
    # Apply lr_override BEFORE scheduler creation so base_lrs are correct
    if args.lr_override is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr_override
        print(f"[RESUME] LR override: {args.lr_override:.2e}")

    # Use initial_lr (the stable peak, persisted in the checkpoint and untouched by cooldown) instead of the current lr:
    # on resume the current lr may already be post-cooldown/decay, which pollutes the cooldown ratio (final_lr/active_lr)
    # ── SFT: derive the total step count from epochs (determines the LR schedule span) ──
    planned_steps = None
    n_chunks_epoch = _count_chunks(args.bin_data, args.seq_len) if args.bin_data else 0
    if n_chunks_epoch:
        p_rep = min(max(getattr(args, "replay_ratio", 0.0) or 0.0, 0.0), 0.9)
        chunks_total = n_chunks_epoch * max(1, args.epochs) / (1.0 - p_rep)
        per_optim = max(1, args.batch_size * args.seq_len * args.grad_accum)
        planned_steps = max(1, int(chunks_total * args.seq_len) // per_optim)
        print(f"[INIT] plan: {n_chunks_epoch:,} chunks/epoch × {args.epochs} epoch "
              f"(replay {p_rep*100:.1f}%) → {planned_steps:,} optimizer steps "
              f"({planned_steps * per_optim / 1e6:.1f}M tokens)")

    active_lr = optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"])
    eta_min = active_lr * args.eta_min_factor

    if args.lr_schedule == "wsd":
        # WSD: Warmup-Stable-Decay
        # If max_tokens is set, compute total_steps from it; otherwise use t_max
        per_optim_tok = args.batch_size * args.seq_len * args.grad_accum
        if args.total_tokens:
            # Global total tokens (sum of the three datasets) → a consistent LR schedule span across rounds (continuity)
            total_steps = args.total_tokens // per_optim_tok
        elif args.max_tokens:
            total_steps = args.max_tokens // per_optim_tok
        elif args.max_steps:
            total_steps = args.max_steps
        elif planned_steps:
            total_steps = planned_steps
        else:
            total_steps = args.t_max

        warmup = args.warmup_steps
        stable_steps = int(total_steps * (1.0 - args.wsd_decay_fraction)) - warmup
        decay_steps = total_steps - warmup - stable_steps

        def wsd_lambda(step):
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            elif args.final_lr_steps > 0 and step >= total_steps - args.final_lr_steps:
                # Last final_lr_steps steps: LR drops to final_lr (cooldown)
                return args.final_lr / max(active_lr, 1e-12)
            elif step < warmup + stable_steps:
                return 1.0
            elif decay_steps <= 0:
                # [WARN] No-decay mode (wsd-decay-fraction=0.0): decay_steps=0, no decay phase.
                # Hold the LR constant once steps pass the end of the stable phase; otherwise progress explodes → negative LR → NaN
                # (2026-09-08 fix: with bs=12 total_steps was recomputed to 366210 and step 370000 overshot,
                #  progress=(370000-500-365710)/1=3790, λ=1-3790*0.9=-3410 → LR=-0.85 → NaN)
                return 1.0
            else:
                progress = (step - warmup - stable_steps) / decay_steps
                return 1.0 - progress * (1.0 - args.eta_min_factor)

        # Determine last_epoch:
        # - Reset optimizer → fresh WSD cycle from beginning
        # - Normal resume (no reset): load_state_dict will set it correctly
        # - Fallback: align with training step
        if args.reset_optimizer:
            init_epoch = warmup - 1  # fresh start
        elif resume_path and sched_state is not None:
            init_epoch = warmup - 1  # placeholder, load_state_dict overwrites
        else:
            # If start_step > warmup, we're in stable/decay phase
            init_epoch = max(warmup - 1, start_step)

        # PyTorch >= 2.x requires initial_lr in param_groups when last_epoch >= 0
        for pg in optimizer.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg['lr']

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, wsd_lambda, last_epoch=init_epoch)
        print(f"[INIT] WSD schedule: warmup={warmup}, stable={stable_steps}, "
              f"decay={decay_steps}, total={total_steps}")
        print(f"[INIT]   LR: {active_lr:.1e} → {eta_min:.1e} "
              f"(stable={active_lr:.1e} for {stable_steps} steps)")
        if args.final_lr_steps > 0:
            print(f"[INIT]   Cooldown: last {args.final_lr_steps} steps LR={args.final_lr:.1e} "
                  f"(from step {max(0, total_steps - args.final_lr_steps)})")

        if resume_path and not args.reset_optimizer and sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
                print("[RESUME] WSD scheduler restored")
            except Exception as e:
                print(f"[RESUME] [WARN]  WSD scheduler incompatible: {e}")
    else:
        # Cosine annealing
        t_max = args.t_max if (args.t_max and args.t_max > 0) else (planned_steps or 500)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min)
        print(f"[INIT] Cosine schedule: T_max={t_max}"
              f"{' (auto=planned steps)' if not args.t_max else ''}, "
              f"LR: {active_lr:.1e} → {eta_min:.1e}")

        if resume_path and not args.reset_optimizer and sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
                print("[RESUME] scheduler restored (LR curve continues)")
            except Exception as e:
                print(f"[RESUME] [WARN]  scheduler incompatible: {e}")

    # ── Dtype ──
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    amp_enabled = (dtype == torch.bfloat16)
    print(f"[INIT] dtype={dtype}, AMP={'[OK]' if amp_enabled else '[FAIL]'}")

    # ── Tokenizer ──
    tokenizer = load_tokenizer()
    pad_id = tokenizer.token_to_id("<pad>") or 0
    print(f"[INIT] pad_id={pad_id}, vocab={tokenizer.get_vocab_size()}")

    # ── Validation set ──
    if args.val_bin:
        print(f"[INIT] validation set (pretraining convention, anti-forgetting guardrail): {args.val_bin} "
              f"(exists={Path(args.val_bin).exists()})")
    args._sft_val_mask_kind, args._sft_val_mask_path = "none", None
    if getattr(args, "sft_val_bin", None):
        if Path(args.sft_val_bin).exists():
            _k, _p = resolve_loss_mask(args.val_mask_bin, args.sft_val_bin, config.vocab_size,
                                       loss_in_high_bit=args.loss_in_high_bit,
                                       no_loss_mask=args.no_loss_mask, for_val=True)
            args._sft_val_mask_kind, args._sft_val_mask_path = _k, _p
            print(f"[INIT] validation set (SFT masked convention): {args.sft_val_bin} (mask={_k})")
        else:
            print(f"[INIT] [WARN]  SFT val not found: {args.sft_val_bin}")
    if args.eval_every:
        print(f"[INIT] eval interval: every {args.eval_every} step")

    # ── Training data ──
    print(f"\n[INIT] mode: {args.mode}")
    if args.mode == "online":
        if getattr(args, "nofilter_lang_only", False):
            if not args.data_dir:
                print("[FAIL] online lang-only mode requires --data-dir (comma-separated dirs allowed)")
                return 1
            data_gen, n_files = create_lang_only_online_dataloader(
                data_dirs=args.data_dir, tokenizer=tokenizer,
                state=state.data_position, seq_len=args.seq_len)
            print("[DATA] language-only filter (keep Python); L1-L6 quality filters disabled")
        else:
            if not args.data_dir:
                print("[FAIL] online mode requires --data-dir")
                return 1
            data_gen, n_files = create_online_dataloader(
                data_dir=args.data_dir, tokenizer=tokenizer,
                state=state.data_position, seq_len=args.seq_len)
        state.data_position.source_type = "parquet"
        state.data_position.files_total = n_files
    elif args.mode == "offline":
        if not args.bin_data and resume_path and state.data_position.bin_path:
            args.bin_data = state.data_position.bin_path
            print(f"[INIT] restored bin path from checkpoint: {args.bin_data}")
        if not args.bin_data:
            print("[FAIL] offline mode requires --bin-data")
            return
        # Detect a data switch: if --bin-data changed, read from the start of the new file
        prev_bin = state.data_position.bin_path if state.data_position.source_type == "bin" else ""
        is_new_bin = prev_bin and Path(args.bin_data).resolve() != Path(prev_bin).resolve()
        if is_new_bin:
            print(f"[INIT] data switch: {prev_bin} → {args.bin_data}, token_offset/epoch reset")
            state.data_position.token_offset = 0
            state.epoch = 0
            state.round_start_tokens = start_tokens  # cumulative baseline for the new dataset
        start_off = state.data_position.token_offset if state.data_position.source_type == "bin" else 0
        mask_kind, mask_path = resolve_loss_mask(
            args.mask_bin, args.bin_data, config.vocab_size,
            loss_in_high_bit=args.loss_in_high_bit, no_loss_mask=args.no_loss_mask)
        inspect_loss_mask(args.bin_data, mask_kind, mask_path, seq_len=args.seq_len)
        data_gen = create_sft_dataloader(
            bin_path=args.bin_data, mask_kind=mask_kind, mask_path=mask_path,
            seq_len=args.seq_len, epochs=args.epochs,
            start_epoch=state.epoch, start_offset_tokens=start_off,
            replay_bin=args.replay_bin, replay_ratio=args.replay_ratio,
            seed=args.seed, max_tokens=args.max_tokens, shuffle=args.shuffle)
        print("[DATA] packing: fixed-length slicing, no padding (the model has no attention mask, so padding always leaks)")
        state.data_position.source_type = "bin"
        state.data_position.bin_path = args.bin_data

    # ═══════════════════ Training loop ═══════════════════

    # torch.compile warmup: trigger a full forward+backward compile
    # use the real batch shape to avoid a runtime recompile
    if compile_enabled:
        print(f"\n[PERF] torch.compile warming up (~30s)...")
        _t0 = time.time()
        dummy_x = torch.randint(0, config.vocab_size,
                                (args.batch_size, args.seq_len), device=device)
        dummy_y = torch.randint(0, config.vocab_size,
                                (args.batch_size, args.seq_len), device=device)
        torch.compiler.cudagraph_mark_step_begin()
        dummy_m = torch.ones_like(dummy_y, dtype=torch.float32)
        if amp_enabled:
            with torch.amp.autocast("cuda", dtype=dtype):
                logits = model(dummy_x)
                loss = masked_cross_entropy(logits, dummy_y, dummy_m, config.vocab_size)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum
        else:
            logits = model(dummy_x)
            loss = masked_cross_entropy(logits, dummy_y, dummy_m, config.vocab_size)
            if args.grad_accum > 1:
                loss = loss / args.grad_accum
        loss.backward()
        optimizer.zero_grad()
        print(f"[PERF] [OK] warmup done ({time.time()-_t0:.1f}s)")

    print(f"\n{'='*60}")
    print(f"training started")
    if args.max_tokens:
        print(f"   target: {args.max_tokens/1e9:.1f}B tokens")
    print(f"   save interval: {args.save_every} step  log interval: {args.log_every} step")
    print(f"{'='*60}\n")

    model.train()
    step = start_step
    total_tokens = start_tokens
    step_losses = []
    t_start = time.time()

    accum_loss = 0.0
    accum_count = 0
    cycle_tokens = 0
    batch_xs, batch_ys, batch_ms = [], [], []
    best_val_loss = state.best_val_loss
    best_pretrain_val = None
    recent_loss = 0.0
    mask_ratio_now = None
    pretrain_val_loss = None
    cur_epoch = -1
    optimizer.zero_grad()

    # first-step marker
    first_step_done = False

    try:
        for batch_item in data_gen:
            if len(batch_item) == 4:      # SFT: (x, y, loss_mask, epoch)
                x_seq, y_seq, m_seq, ep_idx = batch_item
            else:                          # support the 2-tuple from --mode online
                x_seq, y_seq = batch_item
                m_seq, ep_idx = torch.ones_like(y_seq, dtype=torch.float32), 0
            if ep_idx != cur_epoch:
                cur_epoch = ep_idx
                print(f"\n[{'='*58}]\n[SFT] Epoch {ep_idx + 1}/{args.epochs} "
                      f"(step {step}, {total_tokens/1e6:.1f}M tok)\n[{'='*58}]")
            batch_xs.append(x_seq)
            batch_ys.append(y_seq)
            batch_ms.append(m_seq)

            if len(batch_xs) < args.batch_size:
                continue

            x_batch = torch.stack(batch_xs).to(device)
            y_batch = torch.stack(batch_ys).to(device)
            m_batch = torch.stack(batch_ms).to(device)
            batch_xs, batch_ys, batch_ms = [], [], []

            # Warmup
            if step < args.warmup_steps:
                wlr = args.lr * (step + 1) / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = wlr

            # Forward + backward
            if compile_enabled:
                torch.compiler.cudagraph_mark_step_begin()
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits = model(x_batch)
                    loss = masked_cross_entropy(logits, y_batch, m_batch, config.vocab_size)
                    if args.grad_accum > 1:
                        loss = loss / args.grad_accum
            else:
                logits = model(x_batch)
                loss = masked_cross_entropy(logits, y_batch, m_batch, config.vocab_size)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum
            mask_ratio_now = float(m_batch.mean().item())

            loss.backward()
            accum_loss += loss.item() * (args.grad_accum if args.grad_accum > 1 else 1)
            accum_count += 1
            cycle_tokens += x_batch.shape[0] * x_batch.shape[1]

            # Optimizer step
            if accum_count >= args.grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                if step >= args.warmup_steps:
                    scheduler.step()

                step += 1
                total_tokens += cycle_tokens
                avg_loss = accum_loss / args.grad_accum
                step_losses.append(avg_loss)
                accum_loss, accum_count, cycle_tokens = 0.0, 0, 0

                if state.data_position.source_type == "bin":
                    state.data_position.token_offset = total_tokens - state.round_start_tokens

                # print confirmation after the first step
                if not first_step_done:
                    first_step_done = True
                    elapsed = time.time() - t_start
                    tok_per_sec = (total_tokens - start_tokens) / max(elapsed, 0.001)
                    print(f"[1ST] Step {step} | {total_tokens/1e6:.1f}M tok | "
                          f"Loss:{avg_loss:.4f} (answer span masked, {mask_ratio_now*100:.1f}% of tokens counted) | "
                          f"LR:{optimizer.param_groups[0]['lr']:.2e} | "
                          f"{tok_per_sec:,.0f} tok/s | batch_shape={x_batch.shape}")
                    if args.max_tokens:
                        new_tokens = total_tokens - state.round_start_tokens
                        eta_h = (args.max_tokens - new_tokens) / max(tok_per_sec, 1) / 3600
                        print(f"[1ST] ETA: {eta_h:.1f}h")

                # Log
                if step % args.log_every == 0:
                    elapsed = time.time() - t_start
                    tok_per_sec = (total_tokens - start_tokens) / max(elapsed, 0.001)
                    recent_loss = np.mean(step_losses[-args.log_every:])
                    lr_now = optimizer.param_groups[0]["lr"]
                    eta = ""
                    if args.max_tokens:
                        new_tokens = total_tokens - state.round_start_tokens
                        eta_h = (args.max_tokens - new_tokens) / max(tok_per_sec, 1) / 3600
                        eta = f" ETA:{eta_h:.1f}h"
                    print(f"Ep{cur_epoch+1}/{args.epochs} Step {step:6d} | "
                          f"{total_tokens/1e6:.1f}M tok | Loss:{recent_loss:.4f} | "
                          f"LR:{lr_now:.2e} | Grad:{grad_norm:.2f} | "
                          f"mask:{mask_ratio_now*100:.0f}% | {tok_per_sec:,.0f} tok/s{eta}")

                # Validation
                val_loss = None
                pretrain_val_loss = None
                if args.eval_every and step % args.eval_every == 0:
                    # Primary metric: SFT (prompt-masked) val; fall back to pretraining val if absent
                    if getattr(args, "sft_val_bin", None) and args._sft_val_mask_kind != "none":
                        val_loss = evaluate_on_val_sft(
                            model, args.sft_val_bin, config, device,
                            mask_kind=args._sft_val_mask_kind,
                            mask_path=args._sft_val_mask_path,
                            seq_len=args.seq_len, max_batches=args.val_batches)
                    elif args.val_bin:
                        val_loss = evaluate_on_val(
                            model, args.val_bin, config, device,
                            seq_len=args.seq_len, max_batches=args.val_batches, pad_id=pad_id)
                    # Anti-forgetting guardrail: also monitor pretraining val (unmasked convention, comparable to base)
                    if args.val_bin and getattr(args, "sft_val_bin", None):
                        pretrain_val_loss = evaluate_on_val(
                            model, args.val_bin, config, device,
                            seq_len=args.seq_len, max_batches=args.val_batches, pad_id=pad_id)
                        if pretrain_val_loss is not None:
                            if best_pretrain_val is None:
                                best_pretrain_val = pretrain_val_loss
                                print(f"  [anti-forget] pretraining val loss: {pretrain_val_loss:.4f} (baseline)")
                            else:
                                _d = pretrain_val_loss - best_pretrain_val
                                _f = "[OK]" if _d <= 0.05 else "[WARN] rose >0.05 (anti-forgetting red line)"
                                print(f"  [anti-forget] pretraining val loss: {pretrain_val_loss:.4f} "
                                      f"(Δ{_d:+.4f} vs best {best_pretrain_val:.4f}) {_f}")
                                best_pretrain_val = min(best_pretrain_val, pretrain_val_loss)
                    if val_loss is not None:
                        _tag = "[SFT masked] " if (getattr(args, "sft_val_bin", None)
                                                   and args._sft_val_mask_kind != "none") else ""
                        print(f"  {_tag}Val loss: {val_loss:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            state.best_val_loss = best_val_loss
                            best_path = ckpt_base / "best_val"
                            save_checkpoint(model, optimizer, scheduler, config, state, best_path)
                            _set_latest_link(ckpt_base, best_path.relative_to(ckpt_base).as_posix())
                            print(f"  * Best val: {val_loss:.4f}")

                # Save checkpoint — two-tier strategy
                #  MILESTONE: every --save-every steps (e.g. 5000), kept forever
                #  TEMP:      every --temp-every steps (e.g. 500), deleted at next milestone
                MILESTONE_EVERY = args.save_every
                TEMP_EVERY = getattr(args, 'temp_every', 500)

                if step % MILESTONE_EVERY == 0:
                    state.step = step
                    state.total_tokens = total_tokens
                    save_dir = ckpt_base / f"step_{step:06d}"
                    save_checkpoint(model, optimizer, scheduler, config, state, save_dir)
                    _set_latest_link(ckpt_base, save_dir.relative_to(ckpt_base).as_posix())

                    # Clean up TEMP checkpoints: delete only temporary ones older than the last 2 milestones,
                    # keeping TEMP from the last two 5000-step intervals (for merge_checkpoints.py fusion)
                    import shutil
                    keep_from = step - 2 * MILESTONE_EVERY
                    for d in sorted(ckpt_base.iterdir()):
                        if d.is_dir() and d.name.startswith("step_"):
                            try:
                                s = int(d.name.split("_")[1])
                                # only delete TEMP older than 2 milestones (milestones are kept forever)
                                if s < keep_from and s % MILESTONE_EVERY != 0:
                                    shutil.rmtree(d, ignore_errors=True)
                            except ValueError:
                                pass

                    elapsed = time.time() - t_start
                    tok_per_sec = (total_tokens - start_tokens) / max(elapsed, 0.001)
                    dp_info = (f"file#{state.data_position.file_index}"
                              if state.data_position.source_type == "parquet"
                              else f"offset={state.data_position.token_offset}")
                    save_training_log(log_path, step, total_tokens, recent_loss,
                                     lr_now, grad_norm, elapsed, tok_per_sec,
                                     val_loss=val_loss, data_pos=dp_info,
                                     epoch=cur_epoch, mask_ratio=mask_ratio_now,
                                     pretrain_val_loss=pretrain_val_loss)

                elif step % TEMP_EVERY == 0:
                    state.step = step
                    state.total_tokens = total_tokens
                    save_dir = ckpt_base / f"step_{step:06d}"
                    save_checkpoint(model, optimizer, scheduler, config, state, save_dir)
                    # No latest_link update for TEMP saves

                # Termination (uses tokens relative to this round)
                if args.max_tokens and (total_tokens - state.round_start_tokens) >= args.max_tokens:
                    print(f"\n[OK] target reached: {total_tokens/1e9:.2f}B tokens")
                    break
                if args.max_steps and step >= args.max_steps:
                    print(f"\n[OK] step limit reached: {step}")
                    break

    except KeyboardInterrupt:
        print("\ninterrupted manually")

    # ── Final save ──
    state.step = step
    state.total_tokens = total_tokens
    state.best_val_loss = best_val_loss
    state.epoch = max(cur_epoch, 0)
    final_path = ckpt_base / "final"
    save_checkpoint(model, optimizer, scheduler, config, state, final_path)
    _set_latest_link(ckpt_base, final_path.relative_to(ckpt_base).as_posix())

    elapsed = time.time() - t_start
    final_loss = np.mean(step_losses[-max(1, args.log_every):]) if step_losses else 0.0
    lr_final = optimizer.param_groups[0]["lr"]
    tok_per_sec = (total_tokens - start_tokens) / max(elapsed, 0.001)
    save_training_log(log_path, step, total_tokens, final_loss, lr_final,
                     0.0, elapsed, tok_per_sec, data_pos="final",
                     epoch=max(cur_epoch, 0), mask_ratio=mask_ratio_now)

    print(f"\n{'='*60}")
    print(f"[OK] training complete!")
    print(f"   Steps: {step}  Tokens: {_fmt_tokens(total_tokens)}  "
          f"Time: {elapsed/3600:.1f}h  Speed: {tok_per_sec:,.0f} tok/s")
    if best_val_loss < float("inf"):
        print(f"   Best val: {best_val_loss:.4f}")
    print(f"   Checkpoint: {final_path}")
    print(f"{'='*60}")


# ═══════════════════ CLI ═══════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CodeLM-435M SFT (prompt-span loss mask + packing + epochs)")

    g_mode = parser.add_argument_group("mode")
    g_mode.add_argument("--mode", choices=["online", "offline"], default="offline")
    g_mode.add_argument("--data-dir", type=str, default=None)
    g_mode.add_argument("--nofilter-lang-only", action="store_true",
                        help="language-only filter (keep .py) + streaming online training, no L1-L6; --data-dir accepts comma-separated dirs (fair nofilter baseline)")
    g_mode.add_argument("--bin-data", type=str, default=None)
    g_mode.add_argument("--val-bin", type=str, default=None,
                        help="pretraining val (unmasked convention; anti-forgetting guardrail, comparable to base)")

    g_sft = parser.add_argument_group("SFT: loss mask / packing / epochs")
    g_sft.add_argument("--epochs", type=int, default=1,
                       help="number of passes over the same flat .bin (multiple rounds on small datasets; default 1)")
    g_sft.add_argument("--mask-bin", type=str, default=None,
                       help="uint8 loss mask (token-aligned with --bin-data, 1 = counted in loss)")
    g_sft.add_argument("--loss-in-high-bit", action="store_true",
                       help="loss flag embedded in bit15 of the uint16 data (vocab 32000<32768, low 15 bits = token)")
    g_sft.add_argument("--no-loss-mask", action="store_true",
                       help="[WARN] disable the prompt mask (mask all 1) → degrades to plain next-token LM loss")
    g_sft.add_argument("--sft-val-bin", type=str, default=None,
                       help="held-out SFT val (.bin, same mask carrier), evaluated with masked loss")
    g_sft.add_argument("--val-mask-bin", type=str, default=None, help="uint8 mask for the SFT val")
    g_sft.add_argument("--replay-bin", type=str, default=None,
                       help="pretraining replay stream (uint16, no mask), mixed in per --replay-ratio for anti-forgetting")
    g_sft.add_argument("--replay-ratio", type=float, default=0.0,
                       help="probability that each chunk goes to replay (0.05-0.10 = 5-10%% replay mix)")
    g_sft.add_argument("--shuffle", action="store_true",
                       help="shuffle chunk order per epoch (default is sequential, friendlier to mmap)")
    g_sft.add_argument("--continue-step", action="store_true",
                       help="do not reset step/tokens on resume (by default SFT takes weights only and starts at step 0)")
    g_sft.add_argument("--inherit-max-tokens", action="store_true",
                       help="inherit max_tokens from the ckpt (ignored by default: that is the pretraining budget)")
    g_sft.add_argument("--allow-foreign-ckpt-dir", action="store_true",
                       help="allow --ckpt-dir outside fine-tuning/ (rejected by the isolation guard by default)")

    g_resume = parser.add_argument_group("Resume & optimizer")
    g_resume.add_argument("--resume", type=str, default=None)
    g_resume.add_argument("--auto-resume", action="store_true")
    g_resume.add_argument("--reset-optimizer", action=argparse.BooleanOptionalAction, default=True,
                          help="enabled by default for SFT (skip loading old optimizer state + reset step); "
                               "disable with --no-reset-optimizer")
    g_resume.add_argument("--lr-override", type=float, default=None)
    g_resume.add_argument("--ckpt-dir", type=str, default=None)
    g_resume.add_argument("--log-dir", type=str, default=None)

    g_train = parser.add_argument_group("Training hyperparameters")
    g_train.add_argument("--lr", type=float, default=5e-5)
    g_train.add_argument("--batch-size", type=int, default=16)
    g_train.add_argument("--grad-accum", type=int, default=2)
    g_train.add_argument("--max-tokens", type=int, default=None,
                         help="token cap for this round (current dataset), used as the termination condition")
    g_train.add_argument("--total-tokens", type=int, default=None,
                         help="global total tokens (sum of the three datasets) used for the LR schedule (decay/cooldown) span; must match across rounds")
    g_train.add_argument("--max-steps", type=int, default=None)
    g_train.add_argument("--warmup-steps", type=int, default=20)
    g_train.add_argument("--t-max", type=int, default=0,
                         help="cosine: cycle length (0 = auto-derive total steps from epochs); wsd: total training steps")
    g_train.add_argument("--eta-min-factor", type=float, default=0.1)
    g_train.add_argument("--lr-schedule", choices=["cosine", "wsd"], default="cosine",
                         help="LR schedule: cosine annealing (FT default) or warmup-stable-decay")
    g_train.add_argument("--wsd-decay-fraction", type=float, default=0.0,
                         help="WSM: 0.0 = no decay (warmup+stable); fusion is done by merge_checkpoints.py")
    g_train.add_argument("--final-lr-steps", type=int, default=0,
                         help="use --final-lr for the last N steps as cooldown (0 = off). Pass only on the last dataset round")
    g_train.add_argument("--final-lr", type=float, default=1e-4,
                         help="learning rate during the cooldown phase (used with --final-lr-steps)")
    g_train.add_argument("--compile-mode", choices=["reduce-overhead", "default", "none"],
                         default="default",
                         help="torch.compile mode: reduce-overhead (CUDA graphs, fastest) / default (no CUDA graphs, stable) / none (no compile, eager)")
    g_train.add_argument("--weight-decay", type=float, default=0.1)
    g_train.add_argument("--grad-clip", type=float, default=1.0)
    g_train.add_argument("--beta1", type=float, default=0.9)
    g_train.add_argument("--beta2", type=float, default=0.95)
    g_train.add_argument("--dropout", type=float, default=0.0)

    g_arch = parser.add_argument_group("Model architecture")
    g_arch.add_argument("--d_model", type=int, default=1024)
    g_arch.add_argument("--num-layers", type=int, default=22)
    g_arch.add_argument("--num-heads", type=int, default=16)
    g_arch.add_argument("--d_ff", type=int, default=4096)
    g_arch.add_argument("--seq-len", type=int, default=1024)

    g_log = parser.add_argument_group("Logging & saving")
    g_log.add_argument("--log-every", type=int, default=10)
    g_log.add_argument("--save-every", type=int, default=200)
    g_log.add_argument("--temp-every", type=int, default=200)
    g_log.add_argument("--eval-every", type=int, default=200)
    g_log.add_argument("--val-batches", type=int, default=250)
    g_log.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validation
    if args.mode == "online" and not args.data_dir:
        print("[FAIL] online mode requires --data-dir")
        return 1
    if args.mode == "offline" and not args.bin_data and not args.auto_resume and not args.resume:
        print("[FAIL] offline mode requires --bin-data (first training run)")
        return 1

    if args.lr_override and args.reset_optimizer:
        print("[WARN]  --lr-override + --reset-optimizer: optimizer was rebuilt, lr-override still applies")

    # Print non-default arguments
    defaults = {a.dest: a.default for a in parser._actions if a.dest != 'help'}
    overrides = {k: getattr(args, k) for k in defaults if getattr(args, k) != defaults.get(k)}
    if overrides:
        print(f"[ARGS] non-default args: {overrides}")

    run_training(args)
    return 0


if __name__ == "__main__":
    exit(main())
