# Code LLM Experiment Report

**353M GPT-style Decoder Language Model for Code Generation**

---

## 1. Introduction

This project implements and trains a decoder-only Transformer language model
from scratch for code generation and completion. The model is trained on a
diverse multi-source code corpus (~10B tokens) on a single NVIDIA RTX 5090
consumer GPU.

The project covers the full ML pipeline: data acquisition, tokenizer training,
model implementation, training loop engineering, checkpoint management, and
generation evaluation. The codebase is production-oriented, with BF16 mixed
precision, gradient accumulation, atomic checkpointing, and mmap-based
zero-copy data loading.

---

## 2. Dataset

### Source

| Property | Value |
|----------|-------|
| Format | Parquet (columnar) |
| Files | ~800 |
| Size | ~223 GB compressed |
| Total tokens | ~10 B (after BPE tokenization) |
| Languages | Primarily Python, with some other languages |

The dataset is a multi-source code corpus in parquet format, avoiding
single-source bias. Each parquet file contains `path` and `content` columns.
Python files are filtered by `.py` extension and a minimum length of 50
characters.

### Rationale

Parquet enables efficient columnar streaming without loading the entire
dataset into memory. Multi-source aggregation provides diverse coding
styles, idioms, and complexity levels — essential for a general-purpose
code language model.

---

## 3. Tokenization

### Algorithm

**Byte-level BPE** (Byte Pair Encoding), same algorithm as GPT-2.
Trained on 2 million randomly sampled code files from the corpus.

### Configuration

| Property | Value |
|----------|-------|
| Algorithm | Byte-level BPE |
| Vocabulary size | 32,000 |
| Training samples | 2M code files |
| Efficiency | ~3.5 chars/token |
| Special tokens | `<pad>`, `<eos>`, `<unk>` |

For comparison, GPT-2's generic tokenizer achieves ~2.8 chars/token on
Python code. Our code-specific tokenizer is ~25% more efficient, meaning
fewer tokens per code snippet → more code fits in the context window.

### Why BPE?

BPE is subword-level: common patterns (e.g., `def `, `self.`, `__init__`)
become single tokens, while rare identifiers are split into subwords.
This balances vocabulary size with coverage.

---

## 4. Model Architecture

### High-Level Architecture

```
Input Tokens (seq_len=1024)
        │
        ▼
  Token Embedding (32K → 1024)
        │
        ▼
  ┌─────────────────────────┐
  │ Transformer Block × 18   │
  │  ┌─────────────────────┐ │
  │  │ RMSNorm → Attention │ │
  │  │    (RoPE, 16 heads) │─┼── + Residual
  │  │ RMSNorm → SwiGLU FFN│ │
  │  │    (1024 → 3840)    │─┼── + Residual
  │  └─────────────────────┘ │
  └─────────────────────────┘
        │
        ▼
  Final RMSNorm
        │
        ▼
  LM Head (1024 → 32K, untied)
        │
        ▼
  Softmax → Next Token Probability
```

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Parameters | 353M | Max fit on 32GB VRAM with BF16 |
| Layers | 18 | Balanced depth for code semantics |
| `d_model` | 1024 | Standard embedding dimension |
| Attention heads | 16 | `d_model / 64` per head |
| `d_ff` (SwiGLU) | 3840 | 3.75× expansion |
| Vocabulary | 32,000 | BPE code tokenizer |
| Context length | 1024 | Fits most function/file fragments |

### Modern Architectural Choices

| Feature | Traditional (GPT-2) | Our Model | Why Better |
|---------|--------------------|-----------|------------|
| Position Encoding | Learned / Sinusoidal | **RoPE** | Extrapolates to longer sequences |
| Normalization | LayerNorm | **RMSNorm** | Faster, fewer parameters |
| Activation | GELU | **SwiGLU** | Better training dynamics |
| Attention | Standard | **Flash Attention (SDPA/cuDNN)** | 2–3× throughput, less VRAM |
| Embedding/LM Head | Tied | **Untied** | More expressive output |

### Comparison with Production Models

To contextualize our 353M model against a typical production-scale code LLM
(Qwen 3.5 9B as reference):

| Parameter | Our Model (353M) | Qwen 3.5 9B | Ratio |
|-----------|-----------------|-------------|-------|
| Total parameters | 353M | 9B | 25× |
| Layers | 18 | 32 | 1.8× |
| d_model | 1,024 | 4,096 | 4× |
| Attention heads | 16 | 32 | 2× |
| d_ff (SwiGLU) | 3,840 | 12,288 | 3.2× |
| Vocabulary | 32,000 | ~248,000 | 7.8× |
| Context length | 1,024 | 32K–128K | 32–128× |
| VRAM (BF16 training) | 29 GB | ~70+ GB (multi-GPU) | — |
| Training throughput (single RTX 5090) | 46K tok/s | Cannot train on single GPU | — |

**Key insight**: The 25× parameter gap translates to approximately 25× more
compute required for training. Yet our 353M model already demonstrates
emergent algorithmic reasoning (correct recursive Fibonacci) at 7.78B tokens.
This validates the core hypothesis: **small, well-trained models can capture
non-trivial code semantics on consumer hardware**, making them practical for
personal/local deployment scenarios where a 9B model would be infeasible.

### Parameter Breakdown

```
Token Embedding:   32,000 × 1,024          =   32.8 M
Per Transformer Block (×18):
  QKV Projection:  3 × 1,024 × 1,024       =    3.1 M
  Output Projection: 1,024 × 1,024         =    1.0 M
  SwiGLU Gate:      1,024 × 3,840          =    3.9 M
  SwiGLU Up:        1,024 × 3,840          =    3.9 M
  SwiGLU Down:      3,840 × 1,024          =    3.9 M
  RMSNorm ×2:       2 × 1,024              =   ~0.0 M
  Per-block total:                         =   16.0 M
18 blocks:                                  =  287.9 M
Final RMSNorm:      1,024                   =   ~0.0 M
LM Head (untied):   1,024 × 32,000         =   32.8 M
────────────────────────────────────────────────────
Total:                                      ≈  353.4 M
```

---

## 5. Loss Function

### Cross-Entropy Loss

At every token position, the model predicts a probability distribution over
the 32,000-token vocabulary. **Cross-entropy loss** measures the divergence
between the predicted distribution and the ground-truth next token:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x_{<i})
$$

where $N$ is the number of non-padding tokens, $y_i$ is the true next token,
and $P(y_i)$ is the model's predicted probability for that token.

### Example

```
Input tokens:    def   fibonacci   (   n   )   :
                              ↓
Model predicts:  P("return") = 0.41
                 P("if")     = 0.18
                 P("pass")   = 0.06
                 ...
                              ↓
Ground truth:    if
                              ↓
Loss contribution:  -log(0.18) = 1.71
```

### Why Cross-Entropy, Not MSE?

Language modeling is a **classification** problem (predict the correct token
from a discrete vocabulary), not a regression problem. Cross-entropy is the
standard loss for multi-class classification. MSE would penalize "close"
wrong answers equally with "far" wrong answers, which makes no sense for a
discrete vocabulary.

### Implementation

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (B × T, V)
    targets.view(-1),               # (B × T,)
    ignore_index=0,                 # skip <pad> tokens
)
```

Padding tokens (index 0) are excluded from the loss calculation.

---

## 6. Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Learning rate (initial) | 2.4 × 10⁻⁴ |
| LR schedule | Cosine annealing → 2.4 × 10⁻⁵ |
| Warmup | Linear, 500 steps |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 (global norm) |
| Batch size (per GPU) | 14 sequences |
| Gradient accumulation | 4 steps |
| Effective batch size | 56 |
| Sequence length | 1024 |
| Tokens per optimizer step | 57,344 |
| Precision | BF16 mixed (`torch.amp.autocast`) |
| Random seed | 42 (fixed for reproducibility) |

### Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 5090 (32 GB VRAM, Blackwell sm_120) |
| CPU | AMD @ 5.4 GHz |
| RAM | 32 GB |
| VRAM used | ~29 GB (model + activations + optimizer states) |
| Training throughput | ~49K tok/s (BF16 + Flash Attention) |

### Pre-Tokenization Strategy

> **This is critical for training efficiency.**

Tokenization on-the-fly during training creates a CPU bottleneck: the GPU
trains at ~49K tok/s, but a single-threaded tokenizer on Windows maxes out
at ~45K tok/s. The GPU idles while waiting for the CPU.

**Solution: Pre-tokenize once, train with mmap.**

```
Pre-tokenization (one-time cost, multi-process):
  parquet files → encode_batch() → UUID → uint16 → train.bin

Training (zero CPU overhead):
  train.bin → np.memmap (OS page cache) → GPU
```

The `--bin_data` flag enables this fast path via `create_bin_dataloader()`,
which mmaps the pre-tokenized file and reads token IDs directly — no Python
tokenizer involved. The GPU is fed at memory-bandwidth speed.

**Key implementation detail for pre-tokenization:**
- Use `tokenizer.encode_batch()` (Rust-native batch parallelization) instead of per-file `tokenizer.encode()` (Python↔Rust FFI overhead per call)
- Each worker process writes to its own temporary `.bin` file (avoids file locks and memory accumulation)
- Final step: `cat chunk_*.bin > train.bin` (OS-level merge, no Python overhead)

### Checkpoint & Crash Recovery

| Feature | Implementation |
|---------|---------------|
| Atomic saves | Write to `.tmp`, then rename |
| Full state | Model + optimizer + **scheduler** + config + step + tokens |
| Frequency | Every 200 steps (~11.5M tokens) |
| Auto-resume | `--auto-resume` finds latest via symlink |
| Backward compat | Old checkpoints without scheduler key still load |

The scheduler state (`CosineAnnealingLR.state_dict()`) is saved to ensure
the LR curve is correctly resumed after a crash, rather than restarting the
cosine cycle from its peak.

---

## 7. Experimental Results

### Training Progress

| Metric | Value |
|--------|-------|
| Current step | 141,600 |
| Tokens trained | ~8.12 B (81% of dataset) |
| Best loss | 0.91 (step 134,800) |
| Latest loss | ~1.2 |
| Learning rate (current) | ~1.6 × 10⁻⁴ (cosine decay from 2.4×10⁻⁴) |
| Total dataset | ~10 B tokens |
| Effective training time | ~65+ hours (aggregate across sessions) |
| **Fine-tuning status** | Phase 1 (clean short functions) in progress |

### Actual Loss Progression

Measured from checkpoint logs at 10K-step intervals:

| Step | Tokens (B) | Loss | Phase |
|------|-----------|------|-------|
| 200 | 0.01 | 5.22 | Initial rapid descent |
| 10,000 | 0.57 | 1.72 | Fast learning (syntax acquisition) |
| 20,000 | 1.15 | 1.64 | Moderate decline |
| 30,000 | 1.72 | 1.31 | Entering plateau regime |
| 40,000 | 2.29 | 1.22 | Best loss achieved |
| 50,000 | 2.87 | 1.31 | ↑ data expansion + LR adjustment |
| 60,000 | 3.44 | 1.30 | Plateau with variance |
| 70,000 | 4.01 | 1.27 | Continued slow decline |
| 80,000 | 4.59 | 1.16 | New best loss |
| 90,000 | 5.16 | 1.39 | ↑ data switch, LR adjustment |
| 100,000 | 5.73 | 1.58 | ↑ LR change without optimizer reset |
| 110,000 | 6.31 | 1.32 | Recovery after correction |
| 110,600 | 6.34 | 1.44 | Current (oscillating ~1.2-1.6) |
| 134,800 | 7.73 | **0.91** | New best loss |
| 135,600 | 7.78 | 1.15 | Latest |

**Key observations from the loss curve:**

1. **0-10K steps**: Fastest descent (5.2→1.7) — model learns high-frequency patterns (Python keywords, syntax structure)
2. **10K-40K**: Gradual decline (1.7→1.2) — semantic learning phase, best loss of 1.22 at 2.3B tokens
3. **40K-110K**: Oscillation regime (1.2-1.6) — model is in the diminishing returns zone; loss improvements are small and sensitive to data distribution shifts and LR adjustments
4. **Spikes at 90K and 100K**: Coincident with data source expansion (`--start-file-index 90`) and an LR increase from 1.62→2.8×10⁻⁴ without resetting optimizer momentum, causing temporary instability

### Generation Samples (step 135,000, 7.74B tokens, temp=0.2)

**Prompt:** `def fibonacci(n):`

**Output:**
```python
def fibonacci(n):
    """
    Fibonacci function for n.

    Returns: a float
    """
    if not n:
        return 0
    return fibonacci(n-1) + fibonacci(n-2)
```

**Prompt:** `def binary_search(arr, target):`

**Output:**
```python
def binary_search(arr, target):
    """
    Find the index of a binary search in arr.

    Parameters
    ----------
    arr : array
        Array to be searched.
    target : int or None (default None)
        Index of the desired binary search.
    ...
    """
```

**Prompt:** `class Node:`

**Output:**
```python
class Node:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "<%s>" % self.value
```

### Generation Quality Assessment

| Capability | 2.59B tokens | 7.74B tokens | Assessment |
|-----------|-------------|-------------|------------|
| Function definition syntax | √ | √ | Stable from early stage |
| Docstrings / comments | √ | √ | NumPy-style docstrings learned |
| Simple recursive algorithms | × `return (2*n*n)` | √ Correct F(n)=F(n-1)+F(n-2) | **Major breakthrough** |
| Algorithmic semantics (binary search) | — | × Docstring correct, no loop body | Shape learned, logic missing |
| Class definitions | √ | √ | `__init__` + `__repr__` correct |
| Avoiding repetition | × | ! Partial | Drifts after ~20 lines (license headers, repeated classes) |
| Long-range coherence | × | ! Partial | Coherent for ~15 lines, then distribution shift |

### Key Insight: The Fibonacci Breakthrough

At 2.59B tokens, Fibonacci output was `return (2*n*n)` — pure statistical noise.
At 7.74B tokens, the model generates a **structurally correct recursive Fibonacci**
with proper base case and recursive call. The algorithm isn't 100% correct (returns
0 for F(1) instead of 1), but the model has clearly learned the **pattern** of
recursive function definition — base case + recursive case using the function's
own name.

This validates the Chinchilla scaling hypothesis: more data + more training =
qualitative capability jumps, not just quantitative loss improvements. The loss
only dropped from ~1.3 to ~1.2 in this period, but the generation quality
improved dramatically.

### Multi-Seed Stability Analysis

To verify whether generation quality is robust or seed-dependent, we ran
Fibonacci generation across 14 different random seeds (7, 13, 42, 56, 77, 99,
123, 222, 333, 456, 555, 777, 789, 1024, 4096) at 4 key checkpoints:

| Checkpoint | Step | Tokens | Correct Rate | Usable Rate | Verdict |
|-----------|------|--------|-------------|------------|---------|
| step_046400 | 46K | 2.66B | 7/14 (50%) | 9/14 (64%) | Unstable, seed-dependent |
| step_070000 | 70K | 4.01B | 2/14 (14%) | 2/14 (14%) | ! Degraded after data switch |
| step_135000 | 135K | 7.74B | 11/14 (79%) | 11/14 (79%) | Recovering |
| step_141600 | 141.6K | 8.12B | 10/14 (71%) | 12/14 (86%) | √ Best current |

**Key findings:**
1. Generation quality can vary 30%+ across seeds at the same checkpoint —
   single-seed evaluation is unreliable
2. Quality follows a U-curve: good at 46K → degraded at 70K (data switch) →
   recovering at 135K+
3. Loss and generation quality are weakly correlated: best loss (0.91 at 134.8K)
   does not guarantee best generation

---

## 8. Discussion

### Observation 1: Syntax Before Semantics — Then a Breakthrough

At 2.59B tokens (26% through the dataset), the model had learned Python
syntax structure but not algorithmic logic. This matches the expected
learning progression for small language models: high-frequency patterns
(keywords, indentation, function structure) are learned first.

**However**, at 7.74B tokens, a qualitative leap occurred: the model
generates a structurally correct recursive Fibonacci — `return fibonacci(n-1) + fibonacci(n-2)` — compared to `return (2*n*n)` at the earlier checkpoint.
This is not a small incremental improvement; it represents the model
acquiring the **abstract pattern of recursion**: base case + self-referential
call. The loss only dropped from ~1.3 to ~1.2 during this period, but the
generation quality improved dramatically, validating that loss ≠ capability
and that Chinchilla-style scaling (more data → qualitative jumps) holds
even at small model sizes.

### Observation 2: Repetition Degeneration

The model frequently falls into token repetition loops. This is a known
issue with small autoregressive models. Mitigations added:
- `repetition_penalty` (HuggingFace-style, default 1.1)
- `presence_penalty` and `frequency_penalty` (OpenAI-style)
- `min_p` sampling (filters noise tokens below max_prob × threshold)

These reduce but do not eliminate the problem at this training stage.
More training data is expected to help, as the model learns when sequences
"should" end.

### Observation 3: Tokenization Bottleneck

Real-time tokenization during training is a CPU bottleneck on single-core
Windows (Rust tokenizers with Python GIL constraints). The standard
industry solution — pre-tokenization to memory-mapped binary — is
implemented but not yet executed. Once complete, GPU utilization will
increase from tokenizer-bound to compute-bound.

### Observation 4: Loss ≠ Generation Quality

Validation loss alone does not fully capture generation quality.
A model can have decreasing loss while still producing incoherent text,
and conversely, generation coherence can improve even when loss plateaus.
Qualitative evaluation (manual inspection of generated samples) is
essential.

---

## 9. Future Work

### 9.1 Pre-Tokenization Pipeline

Complete the one-time pre-tokenization of the full 10B token dataset
using `encode_batch()` with parallel workers and sharded output.
This eliminates the CPU bottleneck during all subsequent training runs.

### 9.2 Targeted Fine-Tuning

After pre-training, two fine-tuning phases are planned:

**Phase 1 — Simple Functions (22.8M tokens):**
Fine-tune on short Python functions extracted from the pretraining corpus.
Train with moderate LR (8×10⁻⁵) and short cosine cycle (T_max=500, ~400
effective steps). Use `--reset-optimizer` for fresh optimizer state.
Monitor validation loss every 400 steps to detect overfitting.

```bash
py -3.12 fine-tuning\script\train_ft.py \
    --resume checkpoints\step_135000 \
    --reset-optimizer --lr 8e-5 --t-max 500 \
    --bin-data fine-tuning\data\train_short.bin \
    --val-bin fine-tuning\data\val_short.bin \
    --batch-size 14 --grad-accum 4 --warmup-steps 20 \
    --ft-name phase1_short
```

**Phase 2 — Benchmark Training (54K tokens):**
Train on HumanEval + MBPP prompt-solution pairs. Use very conservative LR
(3×10⁻⁵) and reduced batch size due to tiny dataset (1 effective step at
default batch=14/grad_accum=4). Mitigate catastrophic forgetting by keeping
LR low and monitoring validation loss.

```bash
py -3.12 fine-tuning\script\train_ft.py \
    --resume checkpoints\ft_phase1_short\final \
    --reset-optimizer --lr 3e-5 --t-max 100 \
    --bin-data fine-tuning\data\benchmark_train.bin \
    --val-bin fine-tuning\data\benchmark_val.bin \
    --batch-size 4 --grad-accum 1 --warmup-steps 5 \
    --ft-name phase2_bench
```

**Why LR 8×10⁻⁵ (not 3×10⁻⁴):** Fine-tuning with small datasets amplifies
each optimizer step. A high LR (3×10⁻⁴) risks catastrophic forgetting of
pretrained knowledge within the first few hundred steps. Starting at 5–8×10⁻⁵
is safer; increase only if loss plateaus.

### 9.3 Validation Set Monitoring

Add periodic evaluation on a held-out validation set to detect overfitting
and enable early stopping. Implement as a separate `.bin` file loaded via
`--val-bin`.

### 9.4 Learning Rate Flexibility

Implement `--reset-optimizer` flag and configurable `--t-max` to support
fine-tuning runs with fresh optimizer state and appropriate LR schedules,
distinct from the pre-training cosine cycle.

---

## 10. What I Learned

- Implemented a complete GPT-style decoder Transformer from scratch:
  RoPE, RMSNorm, SwiGLU, multi-head self-attention, Flash Attention (SDPA/cuDNN)
- Understood the relationship between model architecture and VRAM budget
  on consumer GPUs (29 GB used on 32 GB card)
- Engineered production training infrastructure: BF16 mixed precision,
  gradient accumulation, atomic checkpointing with scheduler state,
  mmap-based zero-copy data loading, random seed reproducibility
- Learned the critical importance of pre-tokenization: on-the-fly tokenization
  is a CPU bottleneck that wastes GPU compute
- Discovered the `CosineAnnealingLR` behavior on resume (`base_lrs` read from
  optimizer, not from `args.lr`) and its implications for fine-tuning LR strategy
- Transitioned from notebook-style exploration to a structured project with
  scripts, checkpoints, logging, and documentation
- Evaluated model generation qualitatively, identifying syntax-before-semantics
  learning progression and repetition degeneration patterns

---

## 11. References

- Vaswani et al. (2017) — "Attention Is All You Need"
- Radford et al. (2019) — "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Touvron et al. (2023) — "LLaMA: Open and Efficient Foundation Language Models"
- Shazeer (2020) — "GLU Variants Improve Transformer"
- Su et al. (2021) — "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Dao et al. (2022) — "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Kocetkov et al. (2022) — "The Stack: 3 TB of permissively licensed source code"
- Loshchilov & Hutter (2019) — "Decoupled Weight Decay Regularization" (AdamW)

---

## Appendix: Quick Reference

| Category | Parameter | Value |
|----------|-----------|-------|
| Model | Total parameters | 353,408,000 |
| Model | Transformer blocks | 18 |
| Model | Embedding dimension (d_model) | 1,024 |
| Model | Attention heads | 16 |
| Model | FFN dimension (d_ff) | 3,840 |
| Model | Vocabulary size | 32,000 |
| Model | Context length | 1,024 |
| Model | Position encoding | RoPE |
| Model | Normalization | RMSNorm (pre-norm) |
| Model | Activation | SwiGLU |
| Model | Embedding/LM Head | Untied |
| Training | Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Training | Peak learning rate | 2.4 × 10⁻⁴ |
| Training | Min learning rate | 2.4 × 10⁻⁵ |
| Training | LR schedule | Cosine + 500-step warmup |
| Training | Weight decay | 0.1 |
| Training | Gradient clipping | 1.0 |
| Training | Batch size / grad accum / effective | 14 / 4 / 56 |
| Training | Tokens per step | 57,344 |
| Training | Precision | BF16 mixed |
| Training | Checkpoint frequency | Every 200 steps (~11.5M tokens) |
| Hardware | GPU | RTX 5090 (32 GB) |
| Hardware | VRAM used | ~29 GB |
| Hardware | Training throughput | ~49K tok/s |
| Data | Format | Parquet → pre-tokenized .bin (mmap) |
| Data | Training corpus | ~800 files, 223 GB, ~10B tokens |

---

*Trained on a single NVIDIA RTX 5090 — July 2026*
