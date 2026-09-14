# Lessons Learned — War Stories from Training the 435M

> A candid record of the failed first attempt at this model, the bugs that killed it, and the clean rebuild that followed. Format: **symptom → root cause → fix → lesson**.
>
> Context: this project's first attempt at the 435M ("v1") never learned — val loss stuck at 1.97 and generation collapsed. The root causes below are why the current clean rebuild ("v2", documented in `DESIGN.md`) exists. If you haven't read the README, start there.

---

## Stage 1: The v1 Disaster (a model that never learned)

### 1.1 Val Loss Stuck at 1.97, Generation Collapsed

v1 trained 21.5B tokens across star_coder and codeparrot rounds. Val loss never went below 1.97. Fibonacci output: `return n`. Quicksort output: unrelated code. Every generation configuration (temperature, penalties) was tested — nothing helped.

It took two independent root causes stacking to produce this:

### 1.2 Root Cause #1: The Optimizer Referenced a Dead Model (the worst bug of the project)

**Symptom**: after resume, training runs normally, loss logs look plausible, but the model never learns. No error, no warning, nothing.

**Root cause**: `load_checkpoint()` returns a **brand-new** model object (`model = CodeLLM(config)`), but the optimizer was created *before* resume and still held references to the **old** model's parameters. `optimizer.step()` faithfully updated the old, discarded model. The new model was frozen forever.

**Why 353M never hit this**: its `restore_checkpoint()` loads weights *in-place* (`model.load_state_dict(...)`) — the optimizer always referenced the same living model. v1's code created a new object. One line of design difference, completely different behavior.

**Fix**: create the optimizer *after* resume (2026-08-14, both WSD and WSM scripts), verified with a minimal reproduction (old code: weights unchanged after step; fixed: weights update).

**Lesson**: audit every resume path for **object identity** — "new object vs in-place load" is the difference between training and shadowboxing. A silent bug that produces healthy logs is worth ten loud crashes.

### 1.3 Root Cause #2: Tokenizer–Data Mismatch

v1's tokenizer was a **multilingual** BPE 32K — its vocabulary is full of `public`/`void`/`String`/`Controller`/`document`/`window`/`css`/`php` — while the training data was pure Python. Worse: it had no `<gh_stars>` token, and 20% of star_coder samples literally begin with `<gh_stars>N`. Only 65.7% of the vocabularies overlap between v1 and v2 tokenizers, and 20,851 shared token IDs map to *different* strings.

**Fix**: retrain the tokenizer on a uniform, shuffled sample of all three *filtered* datasets (3.54 → 3.64 chars/token).

**Lesson**: tokenizer and training data must come from the same distribution. Sample uniformly from everything you will train on — not "the first few files".

### 1.4 The Contributing Factor: Cumulative-Token Arithmetic

`total_tokens` accumulated across training rounds. Every consumer of it assumed per-round values:
- stop condition `total_tokens >= max_tokens` → switching to a smaller dataset stopped after one step
- resume offset → out-of-bounds reads
- ETA → negative numbers

**Fix**: track `round_start_tokens` and compute per-round values as deltas.

---

## Stage 2: Rebuilding Clean (v2 design decisions, each mapping to a v1 failure)

| v1 failure | v2 design response |
|---|---|
| mismatched tokenizer | retrained on uniform sample of the filtered corpus |
| 20% metadata-prefixed samples | `strip_metadata.py`, residual = 0 (3.35M train rows verified) |
| star_coder trained first | codeparrot first (matches tokenizer), then the_stack, then star_coder |
| resume re-read → repeated data | one merged 62GB train.bin, single continuous run, no resume-by-dataset |
| LR bump without momentum reset → 1.16→1.58 spike | WSM: constant LR + one final cooldown, no mid-flight changes |
| per-dataset LR schedules | global schedule over the merged stream |

---

## Stage 3: v2 Training — New Bugs, New Lessons

### 3.1 The active_lr Bug (cooldown silently disabled)

**Symptom**: the final 30K-step cooldown (LR → 1e-4) triggered at step ~601K, but by step 612,711 the LR was back at 2.5e-4. The model trained ~24K steps longer at full LR than intended.

**Root cause**: the scheduler computed `final_lr / active_lr` as a scaling ratio, where `active_lr = optimizer.param_groups[0]["lr"]` reads the *current* LR. On a fresh (non-resume) run that's the peak LR — fine. After a resume during cooldown, it's already 1e-4, so the ratio = 1.0 and the LR was multiplied back *up* to 2.5e-4.

**Fix**: read `.get("initial_lr", ...)` instead of the current LR. Discovered by diffing the expected vs actual LR trajectory — a habit learned from the v1 audit.

**Lesson**: LR schedule bugs are silent by nature. Plot the actual LR curve and diff it against the plan; never trust "cooldown triggered" log lines.

### 3.2 torch.compile: `reduce-overhead` Is Slower Than `default` on sm_120

**Symptom**: `reduce-overhead` gave 43K tok/s; `default` gave 52K. Also `accessing tensor output of CUDAGraphs that has been overwritten` errors.

**Root cause**: CUDA graphs pay off for many-small-kernel workloads; a 435M model has large kernels. PyTorch 2.10+ `cudagraph_trees.py` regressions made it worse (GitHub #171672, #174575).

**Lesson**: on Blackwell (sm_120) + current PyTorch, use `--compile-mode default`. Measure; don't assume the "faster" mode is faster.

### 3.6 eval.py: `tokenizers.Encoding` is not iterable + model not moved to device

**Symptom**: `src/eval/eval.py` crashed on first run — `TypeError: 'tokenizers.Encoding' object is not iterable`, then after fixing that, `RuntimeError: mat1 and mat2 have different dtype` (weights on CPU, ids on GPU).

**Root cause**:
- `tokenizer.encode(prompt)` returns a `tokenizers.Encoding` object, not a list — the code did `[min(i, ...) for i in input_ids]` expecting iteration. Fix: `.ids`.
- `load_checkpoint(model_path, device)` uses `map_location=device` (loads ckpt tensors to that device) but **never calls `model.to(device)`** — the model object stays on CPU while generation ids go to GPU.
- `model.to(torch.bfloat16)` then breaks: `train.py` forward has `x = self.embed(input_ids) * math.sqrt(self.config.d_model)` — the scalar multiply promotes to float32, so bf16 weights + float32 activations mismatch. Fix: keep float32 eval.

**Lesson**: a released eval script must be runnable; bugs that only appear when actually invoked (device/dtype/object-type) need smoke-testing before publishing.

### 3.7 eval.py reference-pass false negatives: docstring placeholders, trailing junk, missing `def`

**Symptom**: 435M v2 (clean data) generated *logically correct* implementations, yet `reference_pass` = 0/50 — why?

**Root cause** (chain):
1. `check_implementation` ran on the **generation only** (not `prompt + gen`), so `required: ["def mergesort"]` failed — the `def` line is in the prompt, not the generation.
2. Models trained on real GitHub Python write a **docstring first** (`"""Binary search..."""`), consuming generation budget. The rest of the function body gets truncated or the docstring is followed by garbage (`# Driver Code`, `if __name__`, `import numpy...`).
3. `judge_reference` executed the *full* generation (docstring + garbage) — a docstring is a valid expression, but the function body often ends up incomplete/unparseable.

**Fix**: add `extract_function_body(full_code)`: take the code (prompt+gen), find the first `def ` line, collect the function body (post-docstring, post-garbage), stop at the first top-level (non-indented) line. Run `check_implementation` and `run_script` on the **extracted core only**. `judge_reference` now takes `alg["prompt"] + gen`.

**Verification**: v2 merged went 0/50 → 7/50 (fibonacci 3/10, binary_search 3/10, two_sum 1/10; quicksort/mergesort stayed 0 — they genuinely lack `merge`/partition helpers, so the strictness holds). No false positives: all AST-unparseable samples still fail; no empty-function-body passes.

**Lesson**: a docstring-writing model is not a broken model — the prompt style and the judging must both be compatible with the model distribution. Extract the function body before judging; don't judge raw continuation text.

### 3.3 Windows Commit-Memory OOM (a full day of misdiagnosis)

**Symptom**: `CUDA out of memory. Tried to allocate X MiB ... Y GiB free` — huge free VRAM, tiny allocation fails. Reproduced across compile modes and model versions, so it looked like a code/driver bug.

**Root cause**: **Windows commit limit**, not VRAM. WDDM charges GPU allocations against system commit; with the pagefile accidentally disabled (moved to another partition, pagefile.sys never created), the commit ceiling was ~38GB. Hitting 38/38 makes `cudaMalloc` refuse.

**Fix**: fixed-size pagefile on a reliable drive + reboot (`Get-CimInstance Win32_PageFileUsage` to verify).

**Lesson**: "CUDA OOM with plenty of free VRAM" on Windows → check Task Manager → Performance → Memory → **Committed** before touching anything else.

### 3.4 GBK Encoding Crash (2026-09-02, killed a run at 31%)

**Symptom**: the 18B no-filter ablation run died at step ~128K with a decode error reading `train_meta.json`. The JSON was pure ASCII when written; Windows Python defaults to GBK and choked on a UTF-8 byte.

**Fix**: explicit `encoding="utf-8"` on every `open()`.

**Lesson**: on Windows, never rely on the default encoding — even for files you wrote yourself as ASCII.

### 3.5 Compile State Must Survive Resume

`torch.compile` applied before a resume is discarded when the resume replaces the model object → 15-25% throughput loss after every resume. Saved checkpoints also carry `_orig_mod.` prefixes that break loading into an uncompiled model. Both fixed: re-compile after resume, strip the prefix on save/load.

---

## Stage 4: Evaluation Bugs (or: how we learned to stop trusting pass-rates)

1. **Cheating**: `return arr.sort()` passes an output-based quicksort test. Fix: implementation checks + human grading.
2. **False negatives**: logically correct solutions failed on tab/space mixing (TabError). Fix: human review is primary.
3. **Docstring blindness**: from-scratch models treat docstrings as the end of the function — standard HumanEval scores 0/164 on *every* checkpoint. Fix: docstring-free prompts; HumanEval kept only with this caveat.
4. **Batch generation OOM**: per-sample generation (≈1GB peak) instead of batched attention-mask generation (5× estimated peak).

---

## The TL;DR

1. **Resume paths need an object-identity audit** — a new model object plus an old optimizer equals a frozen model with healthy logs.
2. **Silent bugs cost more than crashes** — the optimizer bug and the active_lr bug together wasted more compute than every crash combined.
3. **Distribution match everywhere**: tokenizer↔data, finetune↔eval, filter↔corpus.
4. **One merged stream** eliminates the entire class of multi-round resume/cumulative-token bugs.
5. **Windows training**: fixed pagefile, explicit UTF-8 everywhere, `--compile-mode default` on sm_120.
6. **Human grading is the metric**; script pass-rates are evidence, not verdicts.

*(The first generation has its own war stories: JAX→PyTorch platform wars, the Django contamination disaster, and the resume re-read bug — see [archive/353M/CHALLENGES.md](archive/353M/CHALLENGES.md).)*
