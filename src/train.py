#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════
CodeLM — 统一预训练脚本
══════════════════════════════════════════════════════════════════════

两种模型架构(参数化, 用 --num-layers/--d_ff 指定):
  435M: --num-layers 22 --d_ff 4096   (默认)
  353M: --num-layers 18 --d_ff 3840   (第一代, 见 archive/)

三种 LR 调度(--lr-schedule):
  wsm    warmup+stable+显式cooldown(默认; 435M-v2 实际用: --wsd-decay-fraction 0.0
         --final-lr-steps 30000 --final-lr 1e-4, 尾部靠 merge_checkpoints*.py 融合)
  wsd    warmup+stable+自动衰减(--wsd-decay-fraction 0.2)
  cosine 余弦退火 (353M 第一代用)

用法:
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


# ═══════════════════ 依赖检查 (Windows 兼容) ═══════════════════

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
    print(f"[DEPS] 已加载: {', '.join(ok)}")

_check_deps()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ═══════════════════ 路径解析 ═══════════════════

CLEAN_PRETRAIN_DIR = Path(__file__).resolve().parent

print(f"[PATH] 脚本位置:   {Path(__file__).resolve()}")
print(f"[PATH] 本项目目录: {CLEAN_PRETRAIN_DIR}")


# ═══════════════════ 模型架构 (自包含) ═══════════════════

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
    """加载 tokenizer: 仓库 tokenizer/ 目录 (tokenizer_<arch>.json 或默认 tokenizer.json)"""
    from tokenizers import Tokenizer
    # 本仓库布局: 顶层 tokenizer/ 存有 tokenizer_353m.json / tokenizer_435m.json
    repo_tok_dir = Path(__file__).resolve().parent.parent / "tokenizer"  # repo根/tokenizer/
    candidates = [
        str(repo_tok_dir / "tokenizer_435m.json"),   # 435M (本仓库主力)
        str(CLEAN_PRETRAIN_DIR / "tokenizer" / "tokenizer.json"),  # 旧布局回退
        str(repo_tok_dir / "tokenizer_353m.json"),
    ]
    for i, tok_path in enumerate(candidates):
        tok = Path(tok_path)
        exists = tok.exists()
        print(f"[TOK] 候选{i+1}: {tok} (存在={exists}, 大小={tok.stat().st_size if exists else 'N/A'})")
        if exists and tok.is_file():
            tokenizer = Tokenizer.from_file(tok_path)
            vocab = tokenizer.get_vocab_size()
            specials = {t: tokenizer.token_to_id(t) for t in ["<s>", "</s>", "<unk>", "<pad>", "<eos>"]}
            print(f"[TOK] [OK] 加载: vocab={vocab}, specials={specials}")
            if tokenizer.token_to_id("<pad>") is None:
                tokenizer.add_special_tokens(["<pad>"])
                print(f"[TOK] [WARN]  已补加 <pad> token")
            return tokenizer
    raise FileNotFoundError(f"Tokenizer not found. Searched: {candidates}")

# ═══════════════════ 常量 ═══════════════════

DEFAULT_CKPT_DIR = CLEAN_PRETRAIN_DIR / "checkpoints_wsm"
IS_WINDOWS = sys.platform == "win32"
print(f"[ENV] 平台: {'Windows' if IS_WINDOWS else 'Linux/Mac'}, Python: {sys.version.split()[0]}")


# ═══════════════════ Windows 兼容: LATEST.txt 替代 symlink ═══════════════════

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


# ═══════════════════ 训练状态 (含数据位置) ═══════════════════

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
    # torch.compile 保存时 state_dict 的 key 带 _orig_mod. 前缀，剥离以保证 checkpoint 干净可加载
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
    print(f"[LOAD] 架构: d_model={config.d_model}, layers={config.num_layers}, "
          f"heads={config.num_heads}, d_ff={config.d_ff}")

    model = CodeLLM(config)
    sd = ckpt["model"]
    # 兼容旧的带 _orig_mod. 前缀的 checkpoint（torch.compile 保存时产生）
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)

    state = TrainingState.from_dict(ckpt.get("training_state", {}))
    print(f"[LOAD] 训练状态: step={state.step}, tokens={state.total_tokens/1e9:.2f}B, "
          f"best_val={state.best_val_loss:.4f}")
    return model, config, ckpt.get("optimizer"), ckpt.get("scheduler"), state


def find_latest_ckpt(base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_CKPT_DIR
    base_dir = Path(base_dir)
    print(f"[FIND] 搜索 checkpoint: {base_dir} (存在={base_dir.exists()})")
    if not base_dir.exists():
        print("[FIND] 目录不存在, 无 checkpoint")
        return None

    target_name = _read_latest_link(base_dir)
    if target_name:
        target = base_dir / target_name
        print(f"[FIND] 指针指向: {target_name} (存在={target.exists()})")
        if target.exists() and (target / "checkpoint.pt").exists():
            print(f"[FIND] [OK] 找到: {target}")
            return target

    # 收集所有有 checkpoint.pt 的目录（含 run_*/step_* 嵌套）
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
        print(f"[FIND] [OK] 扫描发现 {len(candidates)} 个, 最新: step_{best_step} ({best_dir})")
        return best_dir
    print("[FIND] [FAIL] 未找到 checkpoint")
    return None


# ═══════════════════ 数据加载器 ═══════════════════

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
        print(f"[DATA] ⏭  跳过前 {state.file_index}/{total_files} 个 parquet 文件")
        parquet_files = parquet_files[state.file_index:]
    state.files_total = total_files

    if not parquet_files:
        raise FileNotFoundError(f"无 parquet 文件: {data_dir}")
    print(f"[DATA] 在线模式: {len(parquet_files)} 个文件 (总共 {total_files}, 从 #{state.file_index} 开始)")
    print(f"[DATA] 过滤: L1-L6 (与 pretrain_data.py 一致)")

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
        print(f"[DATA] 过滤统计: total={s['total']}, passed={s['passed']} "
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
        print(f"[DATA] 在线数据流结束: {total:,} tokens")

    return gen(), state.files_total


def create_lang_only_online_dataloader(data_dirs, tokenizer, state, seq_len=1024,
                                        min_chars=50, shuffle=True):
    """仅语言过滤(保留 .py) + 多数据目录 + shuffle + 流式在线 tokenize。
    不做 L1-L6 —— 用于「公平的 nofilter 对比基线」：只剔除其它语言，
    其余质量过滤一律不加。列名自适应 path / max_stars_repo_path。
    data_dirs: 目录列表(或逗号分隔字符串)。"""
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
        raise FileNotFoundError(f"无 parquet 文件: {data_dirs}")
    print(f"[DATA] lang-only 在线模式: {len(parquet_files)}/{total_files} 文件, "
          f"仅语言过滤(.py) + ≥{min_chars}字符, shuffle={shuffle}")

    q = queue.Queue(maxsize=3)
    stop = threading.Event()

    def strip_meta(code):
        """剥离 star_coder 的 <reponame> / <gh_stars> / <filename> 元数据前缀"""
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
                        # 语言过滤：只用 path 列判断是否 Python
                        if pcol is not None and not str(table.column(pcol)[i].as_py()).endswith(".py"):
                            continue
                        code = str(ccol[i].as_py())
                        # 剥离 star_coder 的 <reponame>/<gh_stars>/<filename> 元数据前缀
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
        print(f"[DATA] lang-only 在线数据流结束: {total:,} tokens")

    return gen(), state.files_total


def create_offline_dataloader(bin_path, seq_len=1024, start_offset=0, max_tokens=None):
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f".bin 不存在: {bin_path}")

    file_gb = bin_path.stat().st_size / 1e9
    meta_path = bin_path.parent / f"{bin_path.stem}_meta.json"
    total_str = "?"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        total_str = f"{meta['total_tokens']/1e9:.2f}B"
    usable_chunks = (int(bin_path.stat().st_size / 2) - start_offset) // (seq_len + 1)
    print(f"[DATA] 离线模式: {bin_path} ({file_gb:.2f}GB, {total_str} tokens)")
    print(f"[DATA] start_offset={start_offset:,}, max_tokens={max_tokens}, "
          f"可用chunks≈{usable_chunks:,}")

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
        print(f"[DATA] 离线数据流结束: {yielded:,} tokens")

    return gen()


# ═══════════════════ 评估 ═══════════════════

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

    # 每次随机采样 250 批, 覆盖全 val bin (不固定, 避免固定样本的系统性偏差)
    chunk_indices = np.random.choice(total_chunks, size=n_eval, replace=False)
    chunk_indices.sort()  # 顺序读, 对 memmap 缓存友好

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


# ═══════════════════ 训练日志 ═══════════════════

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


# ═══════════════════ 主训练循环 ═══════════════════

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

    # ── Checkpoint & Log 目录 ──
    ckpt_base = Path(args.ckpt_dir) if args.ckpt_dir else DEFAULT_CKPT_DIR
    ckpt_base.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if hasattr(args, 'log_dir') and args.log_dir else Path.cwd()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_log.csv"
    print(f"[INIT] Checkpoint 根目录: {ckpt_base}")
    print(f"[INIT] 日志目录:         {log_dir}")
    print(f"[INIT] 训练日志:         {log_path}")

    # ── 模型 ──
    config = ModelConfig(d_model=args.d_model, num_layers=args.num_layers,
                         num_heads=args.num_heads, d_ff=args.d_ff)
    config.max_seq_len = args.seq_len
    config.dropout_rate = args.dropout

    model = CodeLLM(config).to(device)

    # ═══════════════════ 性能优化 (带兼容性检测) ═══════════════════

    cc_major = torch.cuda.get_device_capability(0)[0]

    # 1. TF32: sm_80 (Ampere) 及以上支持
    #    sm_120 (RTX 5090) [OK] — Blackwell 消费级完全支持
    if cc_major >= 8:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        print(f"[PERF] [FIX] TF32 已启用 (sm_{cc_major}0, ~+10-15%)")
    else:
        print(f"[PERF] [WARN]  TF32 不可用 (需要 sm_80+)")

    # 2. torch.compile: 优先 Inductor (需 Triton), 回退 CUDA Graphs (无需 Triton)
    compile_ok = False
    compile_msg = ""
    if args.compile_mode == "none":
        compile_msg = "已禁用 (--compile-mode none, eager 模式)"
    elif hasattr(torch, 'compile'):
        try:
            import triton
            # sm_120 (RTX 5090) workaround: BF16 autocast fusion bug (#191433)
            # max_fusion_size=1 太保守 (无融合=无加速), 2-3 绕过 bug 三合一
            if cc_major >= 12:
                torch._inductor.config.max_fusion_size = 2
                print(f"[PERF] [WARN]  sm_120 workaround: max_fusion_size=2 (BF16 fusion bug, ~10-20%)")
            model = torch.compile(model, mode=args.compile_mode)
            compile_ok = True
            compile_msg = f"Inductor ({args.compile_mode}, ~+15-25%)"
        except ImportError:
            # 无 Triton: 回退到 CUDA Graphs 后端
            try:
                model = torch.compile(model, backend="cudagraphs")
                compile_ok = True
                compile_msg = "CUDA Graphs (无Triton, ~+5-10%)"
            except Exception as e2:
                compile_msg = f"跳过: Triton不可用, cudagraphs也失败 ({e2})"
        except Exception as e:
            compile_msg = f"跳过: {e}"
    else:
        compile_msg = "PyTorch < 2.0"

    if compile_ok:
        print(f"[PERF] [FIX] torch.compile 已启用: {compile_msg} (sm_{cc_major}0)")
    else:
        print(f"[PERF] [WARN]  torch.compile {compile_msg}")
    compile_enabled = compile_ok  # for warmup check later

    n_params = sum(p.numel() for p in model.parameters())
    est_params = config.num_params
    print(f"\n[CFG] 架构: {n_params:,} params ({n_params/1e6:.1f}M), 估算={est_params/1e6:.1f}M")
    print(f"   d_model={config.d_model}  layers={config.num_layers}  "
          f"heads={config.num_heads}  d_ff={config.d_ff}")
    print(f"   seq_len={config.max_seq_len}  vocab={config.vocab_size}  "
          f"dropout={config.dropout_rate}")

    eff_batch = args.batch_size * args.grad_accum
    per_step_tok = args.batch_size * args.seq_len
    per_optim_tok = per_step_tok * args.grad_accum
    print(f"\n⚙  训练参数:")
    print(f"   BS={args.batch_size} × GA={args.grad_accum} → 有效批量={eff_batch}")
    print(f"   每forward: {per_step_tok:,} tok  每optimizer step: {per_optim_tok:,} tok")
    print(f"   LR={args.lr:.1e}  warmup={args.warmup_steps}步  T_max={args.t_max}")
    print(f"   weight_decay={args.weight_decay}  grad_clip={args.grad_clip}")
    if args.max_tokens:
        est_steps = args.max_tokens // per_optim_tok
        print(f"   目标 {args.max_tokens/1e9:.1f}B → 约 {est_steps:,} 步")
        # Rough ETA based on RTX 5090 ~43K tok/s
        est_hrs = (args.max_tokens / 43000) / 3600
        print(f"   预计 {est_hrs:.0f} 小时 (~{est_hrs/24:.1f} 天)")

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
            print("[INIT] 无已有 checkpoint，从头训练")
    elif args.resume:
        resume_path = args.resume

    if resume_path:
        model, loaded_config, opt_state, sched_state, state = load_checkpoint(resume_path, device)
        model = model.to(device)  # 确保模型在正确的设备上

        # 架构参数: checkpoint 为准
        structural = ["d_model", "num_layers", "num_heads", "d_ff"]
        for k in structural:
            ckpt_val = getattr(loaded_config, k, None)
            cli_val = getattr(args, k, None)
            if cli_val is not None and ckpt_val is not None and cli_val != ckpt_val:
                print(f"[RESUME] [WARN]  --{k}={cli_val} != ckpt({ckpt_val}), 使用 checkpoint 值")
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
        print(f"[RESUME] 数据位置: type={dp.source_type}, "
              f"file_idx={dp.file_index}/{dp.files_total}, "
              f"bin_path={dp.bin_path}, token_offset={dp.token_offset:,}")

    else:
        print("[INIT] 从头训练 (无 resume)")
        sched_state = None
        opt_state = None

    # ═══════════════ torch.compile 重新应用（resume 会替换 model 为未编译新对象）═══════════════
    if compile_enabled and resume_path:
        try:
            import triton
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[PERF] [FIX] torch.compile ({args.compile_mode}) 已重新应用到 resume 后的模型")
        except ImportError:
            try:
                model = torch.compile(model, backend="cudagraphs")
                print("[PERF] [FIX] torch.compile (cudagraphs) 已重新应用到 resume 后的模型")
            except Exception as e:
                print(f"[PERF] [WARN] resume 后重新编译失败: {e}")
        except Exception as e:
            print(f"[PERF] [WARN] resume 后重新编译失败: {e}")

    # ── Optimizer (必须在 resume/compile 之后创建，引用最终 model 的参数) ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    if resume_path and opt_state is not None and not args.reset_optimizer:
        try:
            optimizer.load_state_dict(opt_state)
            print("[RESUME] [CKPT] Optimizer 状态已恢复")
        except Exception as e:
            print(f"[RESUME] [WARN]  Optimizer 不兼容: {e}")
    elif resume_path and args.reset_optimizer:
        print("[RESUME] [RUN] 重建 optimizer (不加载旧状态)")

    # ── Scheduler ──
    # Apply lr_override BEFORE scheduler creation so base_lrs are correct
    if args.lr_override is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr_override
        print(f"[RESUME] [FIX] LR override: {args.lr_override:.2e}")

    # 用 initial_lr（稳定峰值，checkpoint 持久保存、cooldown 不会改）而非当前 lr：
    # resume 时当前 lr 可能已是 cooldown/decay 后的值，会污染 cooldown 倍率（final_lr/active_lr）
    active_lr = optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"])
    eta_min = active_lr * args.eta_min_factor

    if args.lr_schedule in ("wsd", "wsm"):
        # WSD/WSM: Warmup-Stable-Decay
        # wsd: --wsd-decay-fraction 0.2 默认自动衰减 20%
        # wsm: --wsd-decay-fraction 0.0 (无衰减, 尾部靠 merge_checkpoints.py 融合)
        #      + --final-lr-steps 30000 显式 cooldown 到 --final-lr (435M-v2 实际配置)
        per_optim_tok = args.batch_size * args.seq_len * args.grad_accum
        if args.total_tokens:
            # 全局总 tokens（三数据集总和）→ LR 调度区间跨轮一致（连贯性）
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
                # 最后 final_lr_steps 步：LR 降到 final_lr（cooldown）
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
            print(f"[INIT]   Cooldown: 最后 {args.final_lr_steps} 步 LR={args.final_lr:.1e} "
                  f"(从 step {max(0, total_steps - args.final_lr_steps)} 起)")

        if resume_path and not args.reset_optimizer and sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
                print("[RESUME] [UP] WSD scheduler 已恢复")
            except Exception as e:
                print(f"[RESUME] [WARN]  WSD scheduler 不兼容: {e}")
    else:
        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.t_max, eta_min=eta_min)
        print(f"[INIT] [DOWN] Cosine schedule: T_max={args.t_max}, "
              f"LR: {active_lr:.1e} → {eta_min:.1e}")

        if resume_path and not args.reset_optimizer and sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
                print("[RESUME] [UP] Scheduler 已恢复 (LR 曲线继续)")
            except Exception as e:
                print(f"[RESUME] [WARN]  Scheduler 不兼容: {e}")

    # ── Dtype ──
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    amp_enabled = (dtype == torch.bfloat16)
    print(f"[INIT] dtype={dtype}, AMP={'[OK]' if amp_enabled else '[FAIL]'}")

    # ── Tokenizer ──
    tokenizer = load_tokenizer()
    pad_id = tokenizer.token_to_id("<pad>") or 0
    print(f"[INIT] pad_id={pad_id}, vocab={tokenizer.get_vocab_size()}")

    # ── 验证集 ──
    if args.val_bin:
        val_exists = Path(args.val_bin).exists()
        print(f"[INIT] 验证集: {args.val_bin} (存在={val_exists})")
        if val_exists:
            print(f"[INIT]   eval 间隔: 每 {args.eval_every} step")

    # ── 训练数据 ──
    print(f"\n[INIT] 模式: {args.mode}")
    if args.mode == "online":
        if getattr(args, "nofilter_lang_only", False):
            if not args.data_dir:
                print("[FAIL] online lang-only 模式需要 --data-dir(可逗号分隔多目录)")
                return 1
            data_gen, n_files = create_lang_only_online_dataloader(
                data_dirs=args.data_dir, tokenizer=tokenizer,
                state=state.data_position, seq_len=args.seq_len)
            print("[DATA] 仅语言过滤(保留 Python)，不启用 L1-L6 质量过滤")
        else:
            if not args.data_dir:
                print("[FAIL] online 模式需要 --data-dir")
                return 1
            data_gen, n_files = create_online_dataloader(
                data_dir=args.data_dir, tokenizer=tokenizer,
                state=state.data_position, seq_len=args.seq_len)
        state.data_position.source_type = "parquet"
        state.data_position.files_total = n_files
    elif args.mode == "offline":
        if not args.bin_data and resume_path and state.data_position.bin_path:
            args.bin_data = state.data_position.bin_path
            print(f"[INIT] [CKPT] 从 checkpoint 恢复 bin 路径: {args.bin_data}")
        if not args.bin_data:
            print("[FAIL] offline 模式需要 --bin-data")
            return
        # 检测数据切换: 如果 --bin-data 变了, 从新文件开头读
        prev_bin = state.data_position.bin_path if state.data_position.source_type == "bin" else ""
        is_new_bin = prev_bin and Path(args.bin_data).resolve() != Path(prev_bin).resolve()
        if is_new_bin:
            print(f"[INIT] [RUN] 数据切换: {prev_bin} → {args.bin_data}, token_offset 清零")
            state.data_position.token_offset = 0
            state.round_start_tokens = start_tokens  # 新数据集的累计基线
        start_off = state.data_position.token_offset if state.data_position.source_type == "bin" else 0
        data_gen = create_offline_dataloader(
            bin_path=args.bin_data, seq_len=args.seq_len,
            start_offset=start_off, max_tokens=args.max_tokens)
        state.data_position.source_type = "bin"
        state.data_position.bin_path = args.bin_data

    # ═══════════════════ 训练循环 ═══════════════════

    # torch.compile 预热: 触发完整 forward+backward 编译
    # 用实际 batch shape 避免运行时 recompile
    if compile_enabled:
        print(f"\n[PERF] [HOT] torch.compile 预热中 (~30s)...")
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
        print(f"[PERF] [OK] 预热完成 ({time.time()-_t0:.1f}s)")

    print(f"\n{'='*60}")
    print(f"[START] 开始训练")
    if args.max_tokens:
        print(f"   目标: {args.max_tokens/1e9:.1f}B tokens")
    print(f"   保存间隔: {args.save_every} step  日志间隔: {args.log_every} step")
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

    # 首步标记
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

                # 第一步后打印确认
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

                    # 清理 TEMP checkpoint：只删 2 个 milestone 之前的临时检查点，
                    # 保留最后两个 5000-step 区间内的 TEMP（供 merge_checkpoints.py 融合）
                    import shutil
                    keep_from = step - 2 * MILESTONE_EVERY
                    for d in sorted(ckpt_base.iterdir()):
                        if d.is_dir() and d.name.startswith("step_"):
                            try:
                                s = int(d.name.split("_")[1])
                                # 只删 2 个 milestone 之前的 TEMP（milestone 永久保留）
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

                # 终止 (使用本轮相对 tokens)
                if args.max_tokens and (total_tokens - state.round_start_tokens) >= args.max_tokens:
                    print(f"\n[OK] 达到目标: {total_tokens/1e9:.2f}B tokens")
                    break
                if args.max_steps and step >= args.max_steps:
                    print(f"\n[OK] 达到步数上限: {step}")
                    break

    except KeyboardInterrupt:
        print("\n⏸  手动中断")

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
    print(f"[OK] 训练完成!")
    print(f"   Steps: {step}  Tokens: {total_tokens/1e9:.2f}B  "
          f"Time: {elapsed/3600:.1f}h  Speed: {tok_per_sec:,.0f} tok/s")
    if best_val_loss < float("inf"):
        print(f"   Best val: {best_val_loss:.4f}")
    print(f"   Checkpoint: {final_path}")
    print(f"{'='*60}")


# ═══════════════════ CLI ═══════════════════

def main():
    parser = argparse.ArgumentParser(description="CodeLM-435M 干净预训练")

    g_mode = parser.add_argument_group("模式")
    g_mode.add_argument("--mode", choices=["online", "offline"], default="offline")
    g_mode.add_argument("--data-dir", type=str, default=None)
    g_mode.add_argument("--nofilter-lang-only", action="store_true",
                        help="仅语言过滤(保留 .py) + 流式在线训练, 不含 L1-L6; --data-dir 可逗号分隔多目录 (nofilter 公平基线)")
    g_mode.add_argument("--bin-data", type=str, default=None)
    g_mode.add_argument("--val-bin", type=str, default=None)

    g_resume = parser.add_argument_group("恢复 & Optimizer")
    g_resume.add_argument("--resume", type=str, default=None)
    g_resume.add_argument("--auto-resume", action="store_true")
    g_resume.add_argument("--reset-optimizer", action="store_true")
    g_resume.add_argument("--lr-override", type=float, default=None)
    g_resume.add_argument("--ckpt-dir", type=str, default=None)
    g_resume.add_argument("--log-dir", type=str, default=None)

    g_train = parser.add_argument_group("训练超参数")
    g_train.add_argument("--lr", type=float, default=2.5e-4)
    g_train.add_argument("--batch-size", type=int, default=11)
    g_train.add_argument("--grad-accum", type=int, default=4)
    g_train.add_argument("--max-tokens", type=int, default=None,
                         help="本轮（当前数据集）token 上限，用于终止条件")
    g_train.add_argument("--total-tokens", type=int, default=None,
                         help="全局总 tokens（三数据集总和），用于 LR 调度（衰减/cooldown）区间计算，跨轮必须一致")
    g_train.add_argument("--max-steps", type=int, default=None)
    g_train.add_argument("--warmup-steps", type=int, default=500)
    g_train.add_argument("--t-max", type=int, default=50000,
                         help="cosine: cycle length; wsd: total training steps")
    g_train.add_argument("--eta-min-factor", type=float, default=0.1)
    g_train.add_argument("--lr-schedule", choices=["wsm", "wsd", "cosine"], default="wsm",
                         help="LR schedule: wsm=warmup+stable+explicit cooldown (435M-v2), wsd=warmup+stable+auto-decay, cosine=annealing (353M)")
    g_train.add_argument("--wsd-decay-fraction", type=float, default=0.0,
                         help="WSM: 0.0 = 无衰减 (warmup+stable), 融合靠 merge_checkpoints.py")
    g_train.add_argument("--final-lr-steps", type=int, default=0,
                         help="最后 N 步用 --final-lr 的 cooldown (0=关闭)。仅最后一个数据集轮次传入")
    g_train.add_argument("--final-lr", type=float, default=1e-4,
                         help="cooldown 阶段学习率 (配合 --final-lr-steps)")
    g_train.add_argument("--compile-mode", choices=["reduce-overhead", "default", "none"],
                         default="reduce-overhead",
                         help="torch.compile 模式: reduce-overhead(CUDA graphs,最快) / default(无CUDA graphs,稳) / none(不编译,eager)")
    g_train.add_argument("--weight-decay", type=float, default=0.1)
    g_train.add_argument("--grad-clip", type=float, default=1.0)
    g_train.add_argument("--beta1", type=float, default=0.9)
    g_train.add_argument("--beta2", type=float, default=0.95)
    g_train.add_argument("--dropout", type=float, default=0.0)

    g_arch = parser.add_argument_group("模型架构")
    g_arch.add_argument("--d_model", type=int, default=1024)
    g_arch.add_argument("--num-layers", type=int, default=22)
    g_arch.add_argument("--num-heads", type=int, default=16)
    g_arch.add_argument("--d_ff", type=int, default=4096)
    g_arch.add_argument("--seq-len", type=int, default=1024)

    g_log = parser.add_argument_group("日志 & 保存")
    g_log.add_argument("--log-every", type=int, default=10)
    g_log.add_argument("--save-every", type=int, default=5000)
    g_log.add_argument("--temp-every", type=int, default=500)
    g_log.add_argument("--eval-every", type=int, default=500)
    g_log.add_argument("--val-batches", type=int, default=250)
    g_log.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 验证
    if args.mode == "online" and not args.data_dir:
        print("[FAIL] online 模式需要 --data-dir")
        return 1
    if args.mode == "offline" and not args.bin_data and not args.auto_resume and not args.resume:
        print("[FAIL] offline 模式需要 --bin-data (首次训练)")
        return 1

    if args.lr_override and args.reset_optimizer:
        print("[WARN]  --lr-override + --reset-optimizer: optimizer 已重建, lr-override 仍应用")

    # 打印非默认参数
    defaults = {a.dest: a.default for a in parser._actions if a.dest != 'help'}
    overrides = {k: getattr(args, k) for k in defaults if getattr(args, k) != defaults.get(k)}
    if overrides:
        print(f"[ARGS] 非默认参数: {overrides}")

    run_training(args)
    return 0


if __name__ == "__main__":
    exit(main())
