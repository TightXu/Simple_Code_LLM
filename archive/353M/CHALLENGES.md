# Code LLM Project — Pitfalls & Lessons Learned

> Compiled from chat logs, training logs, checkpoint history, and memory fragments.
> Supplements the "challenges" section of TECHNICAL.md and the presentation.

---

## Phase 1: Environment Choice — JAX → WSL2 → PyTorch on Windows

### 1.1 JAX Doesn't Support GPU on Windows

- **Timeline**: Day 1-2
- **Problem**: The plan was to train with JAX (I knew the Flax/Haiku ecosystem), but JAX **does not support GPU acceleration on Windows** — its CUDA backend is Linux-only. On Windows you're stuck on CPU, and the training speed was unacceptable.
- **Lesson**: Framework choice has to start with confirming platform support. JAX's own docs say plainly, "Windows support is experimental, CPU-only".

### 1.2 WSL2: Networking + GPU Fragmentation, a Double Hit

- **Timeline**: Day 2
- **Problems**:
  - Networking inside WSL2 was dead, so HuggingFace data downloads didn't work
  - After passing the GPU through to WSL2, JAX hit a CUDA BFC (Best-Fit with Coalescing) memory fragmentation bug → OOM even though VRAM was clearly sufficient
  - WSL2 filesystem performance is poor (cross-OS file access is slow)
- **Decision**: Drop the JAX/WSL2 route and go back to native PyTorch on Windows
- **Lesson**: WSL2 is not suitable for production-grade ML training. GPU passthrough works, but memory management and networking are extra pitfalls.

### 1.3 PyTorch Version Hell: sm_120 + Python 3.14

- **Timeline**: Day 3, early morning
- **Problems**:
  - The RTX 5090 is Blackwell (sm_120), and **PyTorch 2.6 stable doesn't recognize the architecture**, so it can't compile CUDA kernels
  - sm_120 support only exists in the PyTorch nightly cu128 build
  - Python 3.14 is incompatible with PyTorch nightly → had to downgrade to 3.12
- **Final environment**: Python 3.12 + PyTorch 2.12 nightly + CUDA 12.8
- **Lesson**: Newest hardware (especially a new GPU architecture) + newest software = compatibility hell. Advice for future projects: wait 2-3 months after a GPU launch for PyTorch stable to catch up.

---

## Phase 2: Experimentation — Model Scale & Hyperparameter Search

### 2.1 Model Scale: 9B Reference → 353M Trained From Scratch

- **Background**: The starting reference was a Qwen 3.5 9B-class open code model (32 layers, d_model=4096, 32 heads, d_ff=12288, 248K vocab) — with **25×** our parameter count
- **Reality**: A 9B model simply doesn't fit on one RTX 5090 (32GB) — BF16 training needs 70GB+, so multiple GPUs are mandatory. Even quantized inference was only barely feasible.
- **Decision chain**:
  - 9B → not viable (doesn't fit on one card, training needs several)
  - 400M → investigation showed there was VRAM headroom (200M used only 24GB)
  - 200M (v1) → prove feasibility, 76K tok/s
  - 353M (v2) → maximize single-card utilization, 46K tok/s, VRAM 29GB/32GB
- **Key comparison**:

| Parameter | 353M (Ours) | Qwen 3.5 9B | Ratio |
|------|------------|-------------|------|
| Total params | 353M | 9B | 25× |
| Layers | 18 | 32 | 1.8× |
| d_model | 1,024 | 4,096 | 4× |
| Attention heads | 16 | 32 | 2× |
| d_ff (SwiGLU) | 3,840 | 12,288 | 3.2× |
| Vocab | 32K | ~248K | 7.8× |
| Training VRAM | 29 GB | ~70+ GB (multi-GPU) | — |

- **Key metrics**: 1.76× the parameters (v1→v2) for only 0.60× the speed — **better than linear scaling**
- **Core claim**: A 25× parameter gap, yet our 353M model had already emerged recursive reasoning after 7.78B tokens — small models on consumer hardware can still capture non-trivial code semantics
- **Lesson**: Don't benchmark directly against large-model scale. First validate the pipeline with the smallest usable model, then push the hardware to its limit step by step.

### 2.2 batch_size / grad_accum Combination Search

- **Problem**: batch_size too small → unstable training (noisy gradients); too large → OOM
- **Search process**:
  - Fixed grad_accum=1 first and probed the batch_size ceiling: 14 was the largest that didn't OOM under BF16
  - grad_accum=4 widened the effective batch to 56, which lines up with Chinchilla's suggested ~50-100 range
  - Final: `--batch_size 14 --grad_accum 4` (effective batch=56, 57,344 tokens per step)
- **Lesson**: Find the memory limit first, then hit your target effective batch via grad_accum. Not the other way round (setting the batch first and then tuning grad_accum leads to repeated OOMs).

### 2.3 seq_len Trade-off: 256 vs 512 vs 1024

- **Investigation**: Longer seq_len → more context, but VRAM grows as O(seq_len²) (the attention part)
- **Why 1024**:
  - 1024 tokens ≈ 200-300 lines of Python, covering the vast majority of function definitions and file fragments
  - 512 often truncated function bodies; 256 could only see the signature
  - VRAM cost stayed in an acceptable range (~29GB total)
  - 46K tok/s throughput matches tokenize speed exactly, so there's no CPU bottleneck
- **Lesson**: Choosing seq_len = task requirements (coverage) ∩ hardware constraints (VRAM) ∩ speed constraints (matching tokenize)

---

## Phase 3: Training — Data & Learning-Rate Pitfalls

### 3.1 Training Data Reused Over and Over (the Sneakiest Pitfall)

- **Source**: chat log 2026-07-17 02:54 (`session_20260717_025422_1a34af`)
- **How it surfaced**: The user asked directly: "After resuming from a checkpoint, will it train on new data that hasn't been used? How is that recorded?"
- **Root cause**: The slow path in `train_pt.py` does no data-level resume. Every resume starts reading from `parquet_files[0]` again.
- **Actual impact** (assistant's diagnosis):
  - "Files 1-300 may have been trained 3 times, files 301-400 once or twice, files 401-800 zero times"
  - The longest session ran 1.6B tokens → roughly 16% of the total. Of 1126 files, the first 180 were trained repeatedly and the last 946 were never touched.
- **User's fix**: Worked out from the training log that the longest session was 22239 seconds × 49K tok/s ≈ 1.09B tokens → about 87 files consumed, plus a safety margin → `--start-file-index 90`
- **Key exchange**:
  > User: "So it's always been training on repeated data? What's the point of the 800 files then?"
  > Assistant: "Every resume is re-chewing the first 43%"
  > User: "Don't take liberties, make the change the way I said. I only downloaded 800 files total"
- **Lesson**:
  - Data management matters more than model architecture. Prepare the complete dataset in one go before training
  - When downloading data in batches, you must track "which files the model has already seen"
  - Resume logic has to record the data consumption position, not just model state

### 3.2 Forcing a Learning-Rate Change and Ignoring Optimizer Momentum (the Biggest Cause of Quality Swings)

- **Source**: chat log 2026-07-17 (`session_20260717_025422_1a34af`)
- **User-reported regression** (line 4102):
  > "Fibonacci is even less reliable this time. It might also be because we switched to new data now, and I raised the learning rate to 2.8e-04"
- **Root-cause analysis**:
  1. On resume, CosineAnnealingLR reads `base_lrs` from the **optimizer's state_dict** (`[group['lr'] for group in optimizer.param_groups]`), **not** from `args.lr`
  2. Code verification: in `CosineAnnealingLR.__init__`, `self.base_lrs = [group['lr'] for group in optimizer.param_groups]`
  3. On resume the optimizer is restored from the checkpoint → `param_groups[0]['lr'] = 1.62e-04` (an already-decayed value); a new scheduler is built → `base_lrs = [1.62e-04]` — **not** 2.4e-04
  4. The user forced LR to 2.8e-04 (+73%) without resetting the optimizer:
     - Actual update = new LR × old momentum direction
     - The influence of the historical gradients was amplified by 73%
     - Parameter updates diverged from the correct gradient direction
  5. Direct consequence: loss jumped from 1.16 to 1.58 (step 80K→100K), and generation quality regressed (the Fibonacci output became "even less reliable")
- **Fix**: Implemented an `--lr-override` argument (user asked for 2.2e-04), to be used with `--reset-optimizer`
- **Key insight**: Changing the LR without resetting the optimizer = shifting gears in a moving car. Momentum and second-moment estimates are all context-dependent.
- **Lesson**: Optimizer state ≠ a parameter you can edit independently. Fine-tuning naturally needs a new LR → force an optimizer reset

### 3.3 Generation Quality Fluctuates Without Tracking Loss

- **Observations**:
  - Training loss kept falling (1.5 → 1.3 → 1.2), but generation quality was not monotonically improving
  - One checkpoint had lower loss yet generated worse code (more repetition, less diversity)
  - This may be related to the checkpoint being saved at a moment when it happened to be overfitting some batch pattern
- **Response**:
  - Added repetition_penalty=1.1, min_p=0.0, temperature=0.2 to suppress repetition
  - But these are **inference-time corrections**; they don't fix the training problem
  - The root cause was that the model capacity (353M) was still not enough for the complexity of code semantics
- **Lesson**: Loss is a training metric, not a quality metric. For generation tasks you **must sample and evaluate by hand regularly** — you can't go by the loss curve alone.

---

## Phase 4: Fine-tuning Phase 1 — Data Contamination Disaster (It Actually Happened)

### 4.1 Goal and Plan

- Extract short Python functions from the 800 parquet files (50-1000 chars, must have a def, 3+ effective lines)
- Filter noise: copyright, test, setup, shebang
- Result: 90,000 training + 10,000 validation entries, 22.8M tokens after pre-tokenization
- Training: LR=8e-5, t_max=500, ran to val_loss=1.3244 (step 300) → looked good

### 4.2 Catastrophic Result

- **20 generations (4 algorithms × 5 seeds), 0 correct**
- Every output followed one fixed pattern:
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

- binary_search → Django ×5
- two_sum → Django + Plotly ×5
- is_palindrome → Django + Plotly ×5
- inorder_traversal → all wrong (the base model with seed 456 actually got it right!)

### 4.3 Root Cause: Extraction Filter Was Badly Insufficient

- The old filter only blocked: copyright, test, setup, shebang
- **Categories it missed**:
  - Django migrations (`migrations.RunPython`, `apps.get_model`, `class Migration`)
  - Django models (`models.CharField`, `OneToOneField`, `class Meta`, `verbose_name`)
  - Plotly validators (`_plotly_utils.basevalidators`, `plotly_name=`, `edit_type=`)
  - Python 2 leftovers (`# -*- coding:`, `from __future__ import`)
  - Files that are pure imports (>60% of lines are imports)
  - Files that are pure class definitions (>70% class/decorator)
  - Consecutive duplicate lines (a signature of templates/config files)

- This code is short (<1000 chars), has a def, and carries no test/setup markers → it all passed the old filter
- More than half of those 22.8M tokens may have been this kind of garbage template that "looks like a function"
- The model learned the strongest pattern: "see a def signature → emit a Django migration"

### 4.4 Fix: Stronger Filters

Added 30+ precise contamination patterns + 5 quality checks:
- **Exact matches**: Django migrations, Django ORM, Django admin, Plotly validators, Python 2 leftovers, config/boilerplate
- **Filename filtering**: migrations/, tests/, setup.py, conftest.py, __init__.py
- **Quality checks**: discard if import ratio >60%, discard if class/decorator ratio >70%, consecutive duplicate line detection, def line ratio, check that logic lines exist
- **Full re-extraction from NAS**: re-extract clean data from all 800 parquet files

### 4.5 Lessons

1. **Data quality >> data volume**: 22.8M clean entries > 100M contaminated ones
2. **Loss is a dangerously misleading metric**: val_loss=1.32 looked great, but the model was actually learning to "emit a Django template"
3. **Cleaning rules must be based on sampling the actual data**: you can't just guess a few rules and call it done
4. **Validate the base model before fine-tuning**: compare base vs FT generation quality, otherwise you won't catch the regression

---

## Quick-Reference Table by Presentation Use

| Challenge category | Specific problem | One-line summary | Slide |
|---------|---------|-----------|-----------|
| Environment adaptation | JAX/WSL2/PyTorch | Newest hardware + newest software = compatibility hell | Slide 7 |
| Scale decision | 9B→353M | Don't chase large models; maximize single-card utilization | Slide 2/4 |
| Hyperparameters | batch/grad_accum/seq_len | The three parameters constrain each other; search them jointly | Slide 4 |
| Data management | Resume doesn't track file position → first 300 files trained repeatedly | The data pipeline affects results more than the model architecture | Slide 3/7 |
| Optimizer | change LR 1.62→2.8e-4 without reset → loss 1.16→1.58 | Optimizer state is not an independent variable | Slide 7 |
|| Evaluation | loss down ≠ quality up (at 6.3B tokens still syntactically right, semantically wrong) | Generation tasks require manual sampling | Slide 6 |
|| Data contamination | Django/Plotly template contamination of the training data → model regressed after FT | Insufficient data cleaning = poisoning by another name | Slide 8 |
|| Fine-tuning | Fine-tuned on small data, quality worse than the base model | 22.8M garbage tokens < 0 fine-tuning | Slide 8 |

---

## Suggested Narrative Arc for the Report

Organized along the causal chain:

```
I want to train a code LLM
  → Which framework? JAX (no Windows support) → WSL2 (networking + GPU bugs) → PyTorch native (chosen)
  → What scale? 9B (doesn't fit on one card) → 200M (proved feasibility) → 353M (pushes the hardware to the limit) (chosen)
  → Which hyperparameters? batch=14, grad_accum=4, seq_len=1024 (VRAM limit + speed balance) (chosen)
  → How to manage data? Batched downloads → files reused repeatedly (overfitting) → start-file-index to skip explicitly (chosen)
  → How to adjust the learning rate? Forced LR change → optimizer not reset → momentum scrambled → loss swings → generation regresses (chosen)
  → How to evaluate? loss ≠ quality, at 2.6B tokens output (2*n*n), at 7.7B tokens output F(n-1)+F(n-2) (chosen)
  → Fine-tuning: insufficient cleaning → Django/Plotly contamination → model regresses → base model beats FT (chosen)
  → Conclusion: data quality >> data volume, loss ≠ capability, cleaning rules must come from sampling the data
```

## Key Turning Point: the Fibonacci Breakthrough

- **2.59B tokens**: `return (2*n*n)` — pure statistical noise
- **7.74B tokens**: `return fibonacci(n-1) + fibonacci(n-2)` — correct recursive structure
- **Multi-seed validation (14 seeds)**: 46K→50%, 70K→14% (destroyed by the data switch), 135K→79%, 141.6K→86% usable
- Loss only went from ~1.3 to ~1.2, but capability made a qualitative jump
- **loss ≠ capability**: best loss 0.91 ≠ best generation
- Confirms the Chinchilla hypothesis still holds for small models

## Additional Findings

- **Seed instability**: The same checkpoint can differ by 30%+ in accuracy across seeds — a single evaluation is unreliable
- **FT eval bug**: `evaluate_loss` didn't handle dict-format data; fixed
- **Benchmark expansion**: added MBPP sanitized + CodeContests, currently downloading
- **Epoch support**: `train_ft.py` gained an `--epochs` argument for multi-epoch training on small datasets

---

*Last updated: 2026-07-17*
