#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════
CodeLM — unified pretraining script
══════════════════════════════════════════════════════════════════════

Two model architectures (parameterized via --num-layers/--d_ff):
  435M: --num-layers 22 --d_ff 4096   (default)
  353M: --num-layers 18 --d_ff 3840   (first generation, see archive/)

Three LR schedules (--lr-schedule):
  wsm    warmup+stable+explicit cooldown (default; as used for 435M-v2: --wsd-decay-fraction 0.0
         --final-lr-steps 30000 --final-lr 1e-4, tail merged by merge_checkpoints*.py)
  wsd    warmup+stable+automatic decay (--wsd-decay-fraction 0.2)
  cosine cosine annealing (used by the first-generation 353M)

Usage:
  python src/train.py --mode offline --bin-data data/all/train.bin --val-bin data/all/val.bin
  python src/train.py --mode offline --auto-resume
  python src/train.py --mode online --data-dir code_data
══════════════════════════════════════════════════════════════════════
"""

import os, sys, json, math, time, random, argparse, glob
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ═══════════════════ Dependency check (Windows compatible) ═══════════════════

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

print(f"[PATH] Script location:   {Path(__file__).resolve()}")
print(f"[PATH] Project dir:       {CLEAN_PRETRAIN_DIR}")


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
    """Load tokenizer: repo tokenizer/ dir (tokenizer_<arch>.json or default tokenizer.json)"""
    from tokenizers import Tokenizer
    # Repo layout: the top-level tokenizer/ holds tokenizer_353m.json / tokenizer_435m.json
    repo_tok_dir = Path(__file__).resolve().parent.parent / "tokenizer"  # repo root/tokenizer/
    candidates = [
        str(repo_tok_dir / "tokenizer_435m.json"),   # 435M (the main model in this repo)
        str(CLEAN_PRETRAIN_DIR / "tokenizer" / "tokenizer.json"),  # legacy layout fallback
        str(repo_tok_dir / "tokenizer_353m.json"),
    ]
    for i, tok_path in enumerate(candidates):
        tok = Path(tok_path)
        exists = tok.exists()
        print(f"[TOK] Candidate {i+1}: {tok} (exists={exists}, size={tok.stat().st_size if exists else 'N/A'})")
        if exists and tok.is_file():
            tokenizer = Tokenizer.from_file(tok_path)
            vocab = tokenizer.get_vocab_size()
            specials = {t: tokenizer.token_to_id(t) for t in ["<s>", "</s>", "<unk>", "<pad>", "<eos>"]}
            print(f"[TOK] [OK] Loaded: vocab={vocab}, specials={specials}")
            if tokenizer.token_to_id("<pad>") is None:
                tokenizer.add_special_tokens(["<pad>"])
                print(f"[TOK] [WARN]  added <pad> token")
            return tokenizer
    raise FileNotFoundError(f"Tokenizer not found. Searched: {candidates}")

# ═══════════════════ Constants ═══════════════════

DEFAULT_CKPT_DIR = CLEAN_PRETRAIN_DIR / "checkpoints_wsm"
IS_WINDOWS = sys.platform == "win32"
print(f"[ENV] Platform: {'Windows' if IS_WINDOWS else 'Linux/Mac'}, Python: {sys.version.split()[0]}")


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


# ═══════════════════ Training state (with data position) ═══════════════════

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
    # torch.compile prefixes state_dict keys with _orig_mod. on save; strip it so checkpoints stay clean and loadable
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
    print(f"[SAVE] Ckpt: {path.name} | step={state.step} | "
          f"tokens={state.total_tokens/1e9:.2f}B | pos={pos_info} | "
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
    print(f"[LOAD] Arch: d_model={config.d_model}, layers={config.num_layers}, "
          f"heads={config.num_heads}, d_ff={config.d_ff}")

    model = CodeLLM(config)
    sd = ckpt["model"]
    # Handle legacy checkpoints with an _orig_mod. prefix (produced by torch.compile saves)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)

    state = TrainingState.from_dict(ckpt.get("training_state", {}))
    print(f"[LOAD] Training state: step={state.step}, tokens={state.total_tokens/1e9:.2f}B, "
          f"best_val={state.best_val_loss:.4f}")
    return model, config, ckpt.get("optimizer"), ckpt.get("scheduler"), state


def find_latest_ckpt(base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_CKPT_DIR
    base_dir = Path(base_dir)
    print(f"[FIND] Searching checkpoint: {base_dir} (exists={base_dir.exists()})")
    if not base_dir.exists():
        print("[FIND] Directory does not exist, no checkpoint")
        return None

    target_name = _read_latest_link(base_dir)
    if target_name:
        target = base_dir / target_name
        print(f"[FIND] Pointer -> {target_name} (exists={target.exists()})")
        if target.exists() and (target / "checkpoint.pt").exists():
            print(f"[FIND] [OK] Found: {target}")
            return target

    # Collect every dir containing checkpoint.pt (including nested run_*/step_*)
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
        print(f"[FIND] [OK] scan found {len(candidates)}, latest: step_{best_step} ({best_dir})")
        return best_dir
    print("[FIND] [FAIL] No checkpoint found")
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
        raise FileNotFoundError(f"No parquet files: {data_dir}")
    print(f"[DATA] Online mode: {len(parquet_files)} files (of {total_files}, starting at #{state.file_index})")
    print(f"[DATA] Filtering: L1-L6 (same as pretrain_data.py)")

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
        print(f"[DATA] Filter stats: total={s['total']}, passed={s['passed']} "
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
        print(f"[DATA] Online stream finished: {total:,} tokens")

    return gen(), state.files_total


def create_lang_only_online_dataloader(data_dirs, tokenizer, state, seq_len=1024,
                                        min_chars=50, shuffle=True):
    """Language-only filtering (keeping .py) + multiple data dirs + shuffle + streaming online tokenize.
    No L1-L6 — used as a fair nofilter baseline: only other languages are filtered out,
    no other quality filtering is applied. Column names adapt to path / max_stars_repo_path.
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
        raise FileNotFoundError(f"No parquet files: {data_dirs}")
    print(f"[DATA] lang-only online mode: {len(parquet_files)}/{total_files} files, "
          f"language-only filtering (.py) + ≥{min_chars} chars, shuffle={shuffle}")

    q = queue.Queue(maxsize=3)
    stop = threading.Event()

    def strip_meta(code):
        """Strip star_coder's <reponame> / <gh_stars> / <filename> metadata prefixes"""
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
                        # Language filtering: use only the path column to decide whether a file is Python
                        if pcol is not None and not str(table.column(pcol)[i].as_py()).endswith(".py"):
                            continue
                        code = str(ccol[i].as_py())
                        # Strip star_coder's <reponame>/<gh_stars>/<filename> metadata prefixes
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
        print(f"[DATA] lang-only online stream finished: {total:,} tokens")

    return gen(), state.files_total


def create_offline_dataloader(bin_path, seq_len=1024, start_offset=0, max_tokens=None):
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f".bin does not exist: {bin_path}")

    file_gb = bin_path.stat().st_size / 1e9
    meta_path = bin_path.parent / f"{bin_path.stem}_meta.json"
    total_str = "?"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        total_str = f"{meta['total_tokens']/1e9:.2f}B"
    usable_chunks = (int(bin_path.stat().st_size / 2) - start_offset) // (seq_len + 1)
    print(f"[DATA] Offline mode: {bin_path} ({file_gb:.2f}GB, {total_str} tokens)")
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
        print(f"[DATA] Offline stream finished: {yielded:,} tokens")

    return gen()


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

    # Randomly sample 250 batches each time, covering the whole val bin (not fixed, to avoid bias from a fixed sample)
    chunk_indices = np.random.choice(total_chunks, size=n_eval, replace=False)
    chunk_indices.sort()  # Read in order, which is friendlier to the memmap cache

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
                      elapsed, tok_per_sec, val_loss=None, data_pos=""):
    log_path = Path(log_path)
    headers = "step,total_tokens,loss,lr,grad_norm,val_loss,elapsed_s,tok_per_sec,data_pos\n"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write(headers)
    vl = f"{val_loss:.6f}" if val_loss is not None else ""
    with open(log_path, "a") as f:
        f.write(f"{step},{total_tokens},{loss:.6f},{lr:.2e},{grad_norm:.4f},"
                f"{vl},{elapsed:.1f},{tok_per_sec:.0f},{data_pos}\n")


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
    print(f"[PC]  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"   BF16: {'[OK]' if torch.cuda.is_bf16_supported() else '[FAIL]'}")
        print(f"   PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 60)

    set_seed(args.seed)

    # ── Checkpoint & log directories ──
    ckpt_base = Path(args.ckpt_dir) if args.ckpt_dir else DEFAULT_CKPT_DIR
    ckpt_base.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if hasattr(args, 'log_dir') and args.log_dir else Path.cwd()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_log.csv"
    print(f"[INIT] Checkpoint root:  {ckpt_base}")
    print(f"[INIT] Log directory:    {log_dir}")
    print(f"[INIT] Training log:     {log_path}")

    # ── Model ──
    config = ModelConfig(d_model=args.d_model, num_layers=args.num_layers,
                         num_heads=args.num_heads, d_ff=args.d_ff)
    config.max_seq_len = args.seq_len
    config.dropout_rate = args.dropout

    model = CodeLLM(config).to(device)

    # ═══════════════════ Performance optimizations (with compatibility checks) ═══════════════════

    cc_major = torch.cuda.get_device_capability(0)[0]

    # 1. TF32: supported on sm_80 (Ampere) and newer
    #    sm_120 (RTX 5090) [OK] — fully supported by consumer Blackwell
    if cc_major >= 8:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        print(f"[PERF] [FIX] TF32 enabled (sm_{cc_major}0, ~+10-15%)")
    else:
        print(f"[PERF] [WARN]  TF32 unavailable (requires sm_80+)")

    # 2. torch.compile: prefer Inductor (needs Triton), fall back to CUDA Graphs (no Triton needed)
    compile_ok = False
    compile_msg = ""
    if args.compile_mode == "none":
        compile_msg = "disabled (--compile-mode none, eager mode)"
    elif hasattr(torch, 'compile'):
        try:
            import triton
            # sm_120 (RTX 5090) workaround: BF16 autocast fusion bug (#191433)
            # max_fusion_size=1 is too conservative (no fusion = no speedup); 2-3 works around the three-way fusion bug
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
                compile_msg = f"skipped: Triton unavailable, cudagraphs also failed ({e2})"
        except Exception as e:
            compile_msg = f"skipped: {e}"
    else:
        compile_msg = "PyTorch < 2.0"

    if compile_ok:
        print(f"[PERF] [FIX] torch.compile enabled: {compile_msg} (sm_{cc_major}0)")
    else:
        print(f"[PERF] [WARN]  torch.compile {compile_msg}")
    compile_enabled = compile_ok  # for warmup check later

    n_params = sum(p.numel() for p in model.parameters())
    est_params = config.num_params
    print(f"\n[CFG] Arch: {n_params:,} params ({n_params/1e6:.1f}M), estimated={est_params/1e6:.1f}M")
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
        print(f"   target {args.max_tokens/1e9:.1f}B → ~{est_steps:,} steps")
        # Rough ETA based on RTX 5090 ~43K tok/s
        est_hrs = (args.max_tokens / 43000) / 3600
        print(f"   estimated {est_hrs:.0f} h (~{est_hrs/24:.1f} days)")

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
        else:
            print("[INIT] No existing checkpoint, training from scratch")
    elif args.resume:
        resume_path = args.resume

    if resume_path:
        model, loaded_config, opt_state, sched_state, state = load_checkpoint(resume_path, device)
        model = model.to(device)  # Make sure the model is on the right device

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
        # Restore max_tokens from checkpoint if not overridden on CLI
        if state.max_tokens and not args.max_tokens:
            args.max_tokens = state.max_tokens
        print(f"[RESUME] step={start_step}, tokens={start_tokens/1e9:.2f}B")
        dp = state.data_position
        print(f"[RESUME] Data position: type={dp.source_type}, "
              f"file_idx={dp.file_index}/{dp.files_total}, "
              f"bin_path={dp.bin_path}, token_offset={dp.token_offset:,}")

    else:
        print("[INIT] Training from scratch (no resume)")
        sched_state = None
        opt_state = None

    # ═══════════════ torch.compile re-applied (resume replaces the model with a fresh, uncompiled object)═══════════════
    if compile_enabled and resume_path:
        try:
            import triton
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[PERF] [FIX] torch.compile ({args.compile_mode}) re-applied to the resumed model")
        except ImportError:
            try:
                model = torch.compile(model, backend="cudagraphs")
                print("[PERF] [FIX] torch.compile (cudagraphs) re-applied to the resumed model")
            except Exception as e:
                print(f"[PERF] [WARN] recompile after resume failed: {e}")
        except Exception as e:
            print(f"[PERF] [WARN] recompile after resume failed: {e}")

    # ── Optimizer (must be created after resume/compile, referencing the final model's params) ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    if resume_path and opt_state is not None and not args.reset_optimizer:
        try:
            optimizer.load_state_dict(opt_state)
            print("[RESUME] [CKPT] Optimizer state restored")
        except Exception as e:
            print(f"[RESUME] [WARN]  Optimizer incompatible: {e}")
    elif resume_path and args.reset_optimizer:
        print("[RESUME] [RUN] rebuilding optimizer (old state not loaded)")

    # ── Scheduler ──
    # Apply lr_override BEFORE scheduler creation so base_lrs are correct
    if args.lr_override is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr_override
        print(f"[RESUME] [FIX] LR override: {args.lr_override:.2e}")

    # Use initial_lr (the stable peak, persisted in the checkpoint and left alone by cooldown) rather than the current lr:
    # On resume the current lr may already be post-cooldown/decay, which would skew the cooldown ratio (final_lr/active_lr)
    active_lr = optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"])
    eta_min = active_lr * args.eta_min_factor

    if args.lr_schedule in ("wsd", "wsm"):
        # WSD/WSM: Warmup-Stable-Decay
        # wsd: --wsd-decay-fraction 0.2 auto-decays by 20% by default
        # wsm: --wsd-decay-fraction 0.0 (no decay, tail merged by merge_checkpoints.py)
        #      + --final-lr-steps 30000 explicit cooldown to --final-lr (the actual 435M-v2 config)
        per_optim_tok = args.batch_size * args.seq_len * args.grad_accum
        if args.total_tokens:
            # Global total tokens (sum of the three datasets) → the LR schedule span stays consistent across rounds (continuity)
            total_steps = args.total_tokens // per_optim_tok
        elif args.max_tokens:
            total_steps = args.max_tokens // per_optim_tok
        elif args.max_steps:
            total_steps = args.max_steps
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
            else:
                progress = (step - warmup - stable_steps) / max(decay_steps, 1)
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
        print(f"[INIT] [DOWN] WSD schedule: warmup={warmup}, stable={stable_steps}, "
              f"decay={decay_steps}, total={total_steps}")
        print(f"[INIT]   LR: {active_lr:.1e} → {eta_min:.1e} "
              f"(stable={active_lr:.1e} for {stable_steps} steps)")
        if args.final_lr_steps > 0:
            print(f"[INIT]   Cooldown: last {args.final_lr_steps} steps LR={args.final_lr:.1e} "
                  f"(from step {max(0, total_steps - args.final_lr_steps)} on)")

        if resume_path and not args.reset_optimizer and sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
                print("[RESUME] [UP] WSD scheduler restored")
            except Exception as e:
                print(f"[RESUME] [WARN]  WSD scheduler incompatible: {e}")
    else:
        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.t_max, eta_min=eta_min)
        print(f"[INIT] [DOWN] Cosine schedule: T_max={args.t_max}, "
              f"LR: {active_lr:.1e} → {eta_min:.1e}")

        if resume_path and not args.reset_optimizer and sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
                print("[RESUME] [UP] Scheduler restored (LR curve continues)")
            except Exception as e:
                print(f"[RESUME] [WARN]  Scheduler incompatible: {e}")

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
        val_exists = Path(args.val_bin).exists()
        print(f"[INIT] Val set: {args.val_bin} (exists={val_exists})")
        if val_exists:
            print(f"[INIT]   eval interval: every {args.eval_every} steps")

    # ── Training data ──
    print(f"\n[INIT] Mode: {args.mode}")
    if args.mode == "online":
        if getattr(args, "nofilter_lang_only", False):
            if not args.data_dir:
                print("[FAIL] online lang-only mode requires --data-dir (comma-separated dirs allowed)")
                return 1
            data_gen, n_files = create_lang_only_online_dataloader(
                data_dirs=args.data_dir, tokenizer=tokenizer,
                state=state.data_position, seq_len=args.seq_len)
            print("[DATA] language-only filtering (keeping Python), no L1-L6 quality filtering")
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
            print(f"[INIT] [CKPT] Restored bin path from checkpoint: {args.bin_data}")
        if not args.bin_data:
            print("[FAIL] offline mode requires --bin-data")
            return
        # Detect a data switch: if --bin-data changed, read from the start of the new file
        prev_bin = state.data_position.bin_path if state.data_position.source_type == "bin" else ""
        is_new_bin = prev_bin and Path(args.bin_data).resolve() != Path(prev_bin).resolve()
        if is_new_bin:
            print(f"[INIT] [RUN] Data switch: {prev_bin} → {args.bin_data}, token_offset reset")
            state.data_position.token_offset = 0
            state.round_start_tokens = start_tokens  # cumulative baseline for the new dataset
        start_off = state.data_position.token_offset if state.data_position.source_type == "bin" else 0
        data_gen = create_offline_dataloader(
            bin_path=args.bin_data, seq_len=args.seq_len,
            start_offset=start_off, max_tokens=args.max_tokens)
        state.data_position.source_type = "bin"
        state.data_position.bin_path = args.bin_data

    # ═══════════════════ Training loop ═══════════════════

    # torch.compile warmup: trigger a full forward+backward compile
    # use the real batch shape to avoid a runtime recompile
    if compile_enabled:
        print(f"\n[PERF] [HOT] torch.compile warming up (~30s)...")
        _t0 = time.time()
        dummy_x = torch.randint(0, config.vocab_size,
                                (args.batch_size, args.seq_len), device=device)
        dummy_y = torch.randint(0, config.vocab_size,
                                (args.batch_size, args.seq_len), device=device)
        torch.compiler.cudagraph_mark_step_begin()
        if amp_enabled:
            with torch.amp.autocast("cuda", dtype=dtype):
                logits = model(dummy_x)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size), dummy_y.view(-1),
                    ignore_index=pad_id)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum
        else:
            logits = model(dummy_x)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size), dummy_y.view(-1),
                ignore_index=pad_id)
            if args.grad_accum > 1:
                loss = loss / args.grad_accum
        loss.backward()
        optimizer.zero_grad()
        print(f"[PERF] [OK] warmup done ({time.time()-_t0:.1f}s)")

    print(f"\n{'='*60}")
    print(f"[START] Training started")
    if args.max_tokens:
        print(f"   target: {args.max_tokens/1e9:.1f}B tokens")
    print(f"   save interval: {args.save_every} steps  log interval: {args.log_every} steps")
    print(f"{'='*60}\n")

    model.train()
    step = start_step
    total_tokens = start_tokens
    step_losses = []
    t_start = time.time()

    accum_loss = 0.0
    accum_count = 0
    cycle_tokens = 0
    batch_xs, batch_ys = [], []
    best_val_loss = state.best_val_loss
    recent_loss = 0.0
    optimizer.zero_grad()

    # first-step flag
    first_step_done = False

    try:
        for x_seq, y_seq in data_gen:
            batch_xs.append(x_seq)
            batch_ys.append(y_seq)

            if len(batch_xs) < args.batch_size:
                continue

            x_batch = torch.stack(batch_xs).to(device)
            y_batch = torch.stack(batch_ys).to(device)
            batch_xs, batch_ys = [], []

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
                    loss = F.cross_entropy(
                        logits.view(-1, config.vocab_size),
                        y_batch.view(-1), ignore_index=pad_id)
                    if args.grad_accum > 1:
                        loss = loss / args.grad_accum
            else:
                logits = model(x_batch)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    y_batch.view(-1), ignore_index=pad_id)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum

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

                # print a confirmation after the first step
                if not first_step_done:
                    first_step_done = True
                    elapsed = time.time() - t_start
                    tok_per_sec = (total_tokens - start_tokens) / max(elapsed, 0.001)
                    print(f"[1ST] Step {step} | {total_tokens/1e6:.1f}M tok | "
                          f"Loss:{avg_loss:.4f} | LR:{optimizer.param_groups[0]['lr']:.2e} | "
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
                    print(f"Step {step:6d} | {total_tokens/1e9:.2f}B tok | "
                          f"Loss:{recent_loss:.4f} | LR:{lr_now:.2e} | "
                          f"Grad:{grad_norm:.2f} | {tok_per_sec:,.0f} tok/s{eta}")

                # Validation
                val_loss = None
                if args.val_bin and args.eval_every and step % args.eval_every == 0:
                    val_loss = evaluate_on_val(
                        model, args.val_bin, config, device,
                        seq_len=args.seq_len, max_batches=args.val_batches, pad_id=pad_id)
                    if val_loss is not None:
                        print(f"  [CHART] Val loss: {val_loss:.4f}")
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

                    # Clean up TEMP checkpoints: delete only temp checkpoints older than 2 milestones,
                    # keeping the TEMPs from the last two 5000-step windows (for merge_checkpoints.py)
                    import shutil
                    keep_from = step - 2 * MILESTONE_EVERY
                    for d in sorted(ckpt_base.iterdir()):
                        if d.is_dir() and d.name.startswith("step_"):
                            try:
                                s = int(d.name.split("_")[1])
                                # delete only TEMPs older than 2 milestones (milestones are kept forever)
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
                                     val_loss=val_loss, data_pos=dp_info)

                elif step % TEMP_EVERY == 0:
                    state.step = step
                    state.total_tokens = total_tokens
                    save_dir = ckpt_base / f"step_{step:06d}"
                    save_checkpoint(model, optimizer, scheduler, config, state, save_dir)
                    # No latest_link update for TEMP saves

                # Termination (using this round's relative tokens)
                if args.max_tokens and (total_tokens - state.round_start_tokens) >= args.max_tokens:
                    print(f"\n[OK] target reached: {total_tokens/1e9:.2f}B tokens")
                    break
                if args.max_steps and step >= args.max_steps:
                    print(f"\n[OK] step limit reached: {step}")
                    break

    except KeyboardInterrupt:
        print("\n⏸  Manually interrupted")

    # ── Final save ──
    state.step = step
    state.total_tokens = total_tokens
    state.best_val_loss = best_val_loss
    final_path = ckpt_base / "final"
    save_checkpoint(model, optimizer, scheduler, config, state, final_path)
    _set_latest_link(ckpt_base, final_path.relative_to(ckpt_base).as_posix())

    elapsed = time.time() - t_start
    final_loss = np.mean(step_losses[-max(1, args.log_every):]) if step_losses else 0.0
    lr_final = optimizer.param_groups[0]["lr"]
    tok_per_sec = (total_tokens - start_tokens) / max(elapsed, 0.001)
    save_training_log(log_path, step, total_tokens, final_loss, lr_final,
                     0.0, elapsed, tok_per_sec, data_pos="final")

    print(f"\n{'='*60}")
    print(f"[OK] Training complete!")
    print(f"   Steps: {step}  Tokens: {total_tokens/1e9:.2f}B  "
          f"Time: {elapsed/3600:.1f}h  Speed: {tok_per_sec:,.0f} tok/s")
    if best_val_loss < float("inf"):
        print(f"   Best val: {best_val_loss:.4f}")
    print(f"   Checkpoint: {final_path}")
    print(f"{'='*60}")


# ═══════════════════ CLI ═══════════════════

def main():
    parser = argparse.ArgumentParser(description="CodeLM-435M clean pretraining")

    g_mode = parser.add_argument_group("Mode")
    g_mode.add_argument("--mode", choices=["online", "offline"], default="offline")
    g_mode.add_argument("--data-dir", type=str, default=None)
    g_mode.add_argument("--nofilter-lang-only", action="store_true",
                        help="language-only filtering (keeping .py) + streaming online training, no L1-L6; --data-dir may hold comma-separated dirs (fair nofilter baseline)")
    g_mode.add_argument("--bin-data", type=str, default=None)
    g_mode.add_argument("--val-bin", type=str, default=None)

    g_resume = parser.add_argument_group("Resume & optimizer")
    g_resume.add_argument("--resume", type=str, default=None)
    g_resume.add_argument("--auto-resume", action="store_true")
    g_resume.add_argument("--reset-optimizer", action="store_true")
    g_resume.add_argument("--lr-override", type=float, default=None)
    g_resume.add_argument("--ckpt-dir", type=str, default=None)
    g_resume.add_argument("--log-dir", type=str, default=None)

    g_train = parser.add_argument_group("Training hyperparameters")
    g_train.add_argument("--lr", type=float, default=2.5e-4)
    g_train.add_argument("--batch-size", type=int, default=11)
    g_train.add_argument("--grad-accum", type=int, default=4)
    g_train.add_argument("--max-tokens", type=int, default=None,
                         help="token cap for this round (the current dataset), used as the termination condition")
    g_train.add_argument("--total-tokens", type=int, default=None,
                         help="global total tokens (sum of the three datasets), used to compute the LR schedule span (decay/cooldown); must match across rounds")
    g_train.add_argument("--max-steps", type=int, default=None)
    g_train.add_argument("--warmup-steps", type=int, default=500)
    g_train.add_argument("--t-max", type=int, default=50000,
                         help="cosine: cycle length; wsd: total training steps")
    g_train.add_argument("--eta-min-factor", type=float, default=0.1)
    g_train.add_argument("--lr-schedule", choices=["wsm", "wsd", "cosine"], default="wsm",
                         help="LR schedule: wsm=warmup+stable+explicit cooldown (435M-v2), wsd=warmup+stable+auto-decay, cosine=annealing (353M)")
    g_train.add_argument("--wsd-decay-fraction", type=float, default=0.0,
                         help="WSM: 0.0 = no decay (warmup+stable), merging is done by merge_checkpoints.py")
    g_train.add_argument("--final-lr-steps", type=int, default=0,
                         help="cooldown to --final-lr over the last N steps (0=off). Only pass it on the last dataset round")
    g_train.add_argument("--final-lr", type=float, default=1e-4,
                         help="learning rate during the cooldown phase (used with --final-lr-steps)")
    g_train.add_argument("--compile-mode", choices=["reduce-overhead", "default", "none"],
                         default="reduce-overhead",
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
    g_log.add_argument("--save-every", type=int, default=5000)
    g_log.add_argument("--temp-every", type=int, default=500)
    g_log.add_argument("--eval-every", type=int, default=500)
    g_log.add_argument("--val-batches", type=int, default=250)
    g_log.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate
    if args.mode == "online" and not args.data_dir:
        print("[FAIL] online mode requires --data-dir")
        return 1
    if args.mode == "offline" and not args.bin_data and not args.auto_resume and not args.resume:
        print("[FAIL] offline mode requires --bin-data (first run)")
        return 1

    if args.lr_override and args.reset_optimizer:
        print("[WARN]  --lr-override + --reset-optimizer: optimizer was rebuilt, lr-override still applied")

    # print non-default args
    defaults = {a.dest: a.default for a in parser._actions if a.dest != 'help'}
    overrides = {k: getattr(args, k) for k in defaults if getattr(args, k) != defaults.get(k)}
    if overrides:
        print(f"[ARGS] non-default args: {overrides}")

    run_training(args)
    return 0


if __name__ == "__main__":
    exit(main())
