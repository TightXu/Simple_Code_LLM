# 353M: pitfalls and lessons from the first generation

*Archive snapshot of the 353M project, July 2026.*

> The problems behind the 353M's design, in **symptom → root cause → fix → lesson** form. This is the generation that established the pipeline; the bugs the current version was built to fix are summarized in the root [`CHALLENGES.md`](../../CHALLENGES.md).

---

## Phase 1: choosing an environment — JAX → WSL2 → PyTorch on Windows

### 1.1 JAX has no GPU support on Windows

I started with JAX (I knew the Flax/Haiku ecosystem), but its CUDA backend is Linux-only, so Windows is CPU-only — unacceptably slow. Its own documentation says as much: "Windows support is experimental, CPU-only". **Lesson:** framework choice has to begin with confirming platform support.

### 1.2 WSL2: networking and GPU memory fragmentation

Inside WSL2 the network was dead, so HuggingFace downloads failed; with the GPU passed through, JAX hit a CUDA BFC (best-fit-with-coalescing) fragmentation bug and went OOM with plenty of VRAM apparently free; and cross-OS file access was slow. I dropped the JAX/WSL2 route and went back to native PyTorch on Windows. **Lesson:** WSL2 is not a production ML training platform — GPU passthrough works, memory management and networking are extra pitfalls.

### 1.3 PyTorch version hell: sm_120 + Python 3.14

The RTX 5090 is Blackwell (sm_120), which PyTorch 2.6 stable does not recognize, so it cannot compile CUDA kernels; sm_120 support existed only in the nightly cu128 build; and Python 3.14 is incompatible with that nightly, so I moved down to 3.12. Final environment: Python 3.12 + PyTorch 2.12 nightly + CUDA 12.8. **Lesson:** newest hardware plus newest software is compatibility hell; two or three months after a GPU launch, stable PyTorch has usually caught up.

---

## Phase 2: model scale and hyperparameter search

### 2.1 Scale: a 9B reference, a 353M model trained from scratch

The reference point was a Qwen 3.5 9B-class open code model — **25×** this model's parameter count. A 9B model does not fit on one RTX 5090 (BF16 training needs 70GB+, so several GPUs are mandatory), and even quantized inference was only barely feasible. The decision chain: 9B not viable → 400M-scale investigation showed VRAM headroom (a 200M model used 24GB) → 200M proved feasibility at 76K tok/s → 353M maximized single-card utilization at 46K tok/s using 29GB of 32GB.

| Parameter | 353M | Qwen 3.5 9B | Ratio |
|------|------------|-------------|------|
| Total params | 353M | 9B | 25× |
| Layers | 18 | 32 | 1.8× |
| d_model | 1,024 | 4,096 | 4× |
| Attention heads | 16 | 32 | 2× |
| d_ff (SwiGLU) | 3,840 | 12,288 | 3.2× |
| Vocab | 32K | ~248K | 7.8× |
| Training VRAM | 29 GB | ~70+ GB (multi-GPU) | — |

Scaling from 200M to 353M cost 1.76× the parameters for 0.60× the speed — better than linear. The core claim: across a 25× parameter gap, this 353M model had already emerged recursive reasoning by ~7.74B tokens (step 135K; the next saved checkpoint, step 135.6K, is logged at 7.78B), so small models on consumer hardware can capture non-trivial code semantics. **Lesson:** don't benchmark directly against large-model scale — validate the pipeline with the smallest usable model, then push the hardware to its limit step by step.

### 2.2 batch_size / grad_accum search

Too small a batch gives noisy gradients, too large OOMs. I first probed the batch ceiling at grad_accum=1 (14 was the largest that survived BF16), then used grad_accum=4 to reach an effective batch of 56, inside Chinchilla's suggested ~50-100 range: `--batch_size 14 --grad_accum 4` (57,344 tokens per step). **Lesson:** find the memory limit first, then reach the target effective batch through grad_accum — setting the batch first and tuning accumulation afterwards means repeated OOMs.

### 2.3 seq_len: 256 vs 512 vs 1024

Longer sequences cover more, but attention VRAM grows as O(seq_len²). I chose 1024: it is roughly 200-300 lines of Python, covering most function definitions and file fragments, where 512 often truncated function bodies and 256 saw only the signature; VRAM stayed around 29GB total; and 46K tok/s matched the pre-tokenization speed exactly, so there was no CPU bottleneck. **Lesson:** seq_len is task coverage ∩ VRAM ∩ speed.

---

## Phase 3: training — data and learning-rate pitfalls

### 3.1 The training data was reused over and over (the sneakiest pitfall)

I checked whether a resume actually reaches unseen data and how the consumed file position is recorded. It was not recorded at all: the slow path in `train_pt.py` has no data-level resume, so every resume starts again at `parquet_files[0]`. Of 1,126 files, the first 180 were trained repeatedly while the last 946 were never touched at all. The longest session worked through ≈87 files (22,239 s × 49K tok/s ≈ 1.09B tokens), which with a safety margin gives the fix I used: `--start-file-index 90`.

**Lesson:** data management matters more than architecture — prepare the complete dataset in one go before training; when downloading in batches, track which files the model has already seen; and resume logic has to record the data consumption position, not just model state.

### 3.2 Changing the learning rate without resetting optimizer momentum

Fibonacci generation became less reliable right after the data switch, when the LR was raised to 2.8e-4. Root cause: `CosineAnnealingLR` initializes `base_lrs = [group['lr'] for group in optimizer.param_groups]`, reading the LR from the **optimizer state dict**, not from `args.lr`; on resume the optimizer is restored with `param_groups[0]['lr'] = 1.62e-4` (an already-decayed value), so the new scheduler's base was 1.62e-4 rather than 2.4e-4. The LR was then forced to 2.8e-4 (+73%) with no optimizer reset, which amplified the influence of the historical gradients by the same 73% and pushed parameter updates off the current gradient direction. Loss jumped from 1.16 to 1.58 (step 80K→100K) and generation regressed until the dynamics recovered. Fix: an `--lr-override` argument (set to 2.2e-4) used together with `--reset-optimizer`. **Lesson:** momentum and second-moment estimates are calibrated to the old LR, so optimizer state is not a parameter that can be edited on its own.

### 3.3 Generation quality fluctuates while loss keeps falling

Training loss fell steadily (1.5 → 1.3 → 1.2) while generation quality was not monotonically improving, and one checkpoint with lower loss generated worse code (more repetition, less diversity). I added `repetition_penalty=1.1`, `min_p=0.0`, `temperature=0.2`, but those are inference-time corrections and do not fix the training problem: 353M capacity was the limit. **Lesson:** loss is a training metric, not a quality metric — sample and evaluate by hand regularly.

---

## Phase 4: fine-tuning, phase 1 — the data contamination disaster

**What was tried:** short Python functions extracted from the 800 parquet files (50-1000 chars, must contain a `def`, 3+ effective lines), noise-filtered for copyright/test/setup/shebang → 90,000 training + 10,000 validation entries, 22.8M tokens, trained at LR 8e-5 / t_max 500 to val_loss 1.3244 at step 300.

**What happened:** 20 generations (4 algorithms × 5 seeds), 0 correct. Every output jumped straight from the prompt signature into Django, then Plotly:

```
def some_func(x):          ← prompt
# -*- coding: utf-8 -*-    ← model immediately jumps to Django
from django.db import migrations
def forwards(apps, schema_editor):
    ...
class Migration(migrations.Migration):
    ...
import _plotly_utils.basevalidators   ← then switches to Plotly
class TicktextsrcValidator(...):
```

binary_search → Django ×5; two_sum → Django + Plotly ×5; is_palindrome → Django + Plotly ×5; inorder_traversal → all wrong (the un-tuned base model with seed 456 actually got it right).

**Root cause:** the extraction filter only blocked copyright, test, setup and shebang. It missed Django migrations (`migrations.RunPython`, `apps.get_model`, `class Migration`), Django models (`models.CharField`, `class Meta`, `verbose_name`), Plotly validators (`_plotly_utils.basevalidators`, `plotly_name=`), Python 2 leftovers (`# -*- coding:`, `from __future__ import`), files that are pure imports (>60% import lines) or pure class definitions (>70% class/decorator lines), and consecutive duplicate lines. This code is short, contains a `def` and carries no test/setup markers, so all of it passed the old filter; more than half of those 22.8M tokens may have been this kind of template that merely looks like a function, and the model learned the strongest pattern available: signature in, Django migration out.

**Fix:** 30+ exact contamination patterns (Django migrations/ORM/admin, Plotly validators, Python 2 leftovers, config boilerplate), filename filtering (`migrations/`, `tests/`, `setup.py`, `conftest.py`, `__init__.py`), five quality checks (import ratio >60%, class/decorator ratio >70%, consecutive duplicate lines, def-line ratio, presence of logic lines), and a full re-extraction of clean data from all 800 parquet files.

**Lessons:** data quality beats data volume (22.8M clean entries > 100M contaminated ones); loss is a dangerously misleading metric (val_loss 1.32 looked great while the model was learning to emit a Django template); cleaning rules must come from sampling the actual data; and the base model has to be validated before fine-tuning, or the regression goes unnoticed.

---

## Quick reference: bugs by category

| Category | Specific problem | One-line summary |
|---|---|---|
| Environment adaptation | JAX/WSL2/PyTorch | Newest hardware + newest software = compatibility hell |
| Scale decision | 9B→353M | Don't chase large models; maximize single-card utilization |
| Hyperparameters | batch/grad_accum/seq_len | The three constrain each other; search them jointly |
| Data management | resume doesn't track file position, so the first 180 files were trained repeatedly | the data pipeline affects results more than the model architecture |
| Optimizer | LR changed 1.62→2.8e-4 without an optimizer reset, loss 1.16→1.58 | optimizer state is not an independent variable |
| Evaluation | loss down ≠ quality up (at 6.3B tokens: syntactically right, semantically wrong) | generation tasks require manual sampling |
| Data contamination | Django/Plotly template contamination of the training data, model regressed after FT | insufficient data cleaning is poisoning by another name |
| Fine-tuning | fine-tuned on small data, quality worse than the base model | 22.8M garbage tokens < 0 fine-tuning |

---

## The Fibonacci breakthrough

- 2.59B tokens: `return (2*n*n)`, pure statistical noise
- 7.74B tokens (step 135,000): `return fibonacci(n-1) + fibonacci(n-2)`, correct recursive structure
- Multi-seed validation (14 seeds): 46K → 50%, 70K → 14% (broken by the data switch), 135K → 79%, 141.6K → 86% usable
- Loss moved only from ~1.3 to ~1.2 while capability made a qualitative jump; the best loss (0.91) is not the best generation
- Consistent with the project-wide finding that capability can jump while loss barely moves

## Additional observations

- Seed instability: the same checkpoint can differ by 30%+ in accuracy across seeds, so a single evaluation is unreliable.
- FT eval bug: `evaluate_loss` did not handle dict-format data; fixed.
- Epoch support: `train_ft.py` gained an `--epochs` argument for multi-epoch training on small datasets.
