# Lessons from training the 435M

> What broke, why, and what the design does about it — **symptom → root cause → fix → lesson**, one short entry each.
>
> Naming: **435M v1** is this project's first attempt at this model (summarized in §1), **v2** is the clean rebuild documented in [`DESIGN.md`](DESIGN.md), and the earlier generation is always **353M** ([`archive/353M/`](archive/353M/)). New here? Start with the [`README`](README.md).

---

## 1. The failed first attempt (435M v1)

v1 scaled the 353M up to 22 layers / d_ff 4096 and trained 21.5B tokens from two sources (star_coder 8.64B, then codeparrot_clean 12.86B) in sequential dataset rounds. It never learned: val loss sat at 1.97 for the entire run and generation collapsed (fibonacci → `return n`, quicksort → unrelated code), under every temperature and penalty setting I tried. Three problems came out of it, and each one became a design decision in v2:

1. **The optimizer pointed at a dead model.** `load_checkpoint()` returns a **new** model object (`model = CodeLLM(config)`), but the optimizer had been created *before* the resume, so `optimizer.step()` updated the discarded model: the logs looked plausible and nothing was learned. Fix: create the optimizer *after* resume (both WSD and WSM scripts), verified with a minimal reproduction (old code: weights unchanged after a step; fixed: weights update). The 353M never hit this, because its `restore_checkpoint()` loads weights *in place* (`model.load_state_dict(...)`). This is the primary explanation for the frozen 1.97, not a data law.
2. **Tokenizer–data mismatch.** v1 reused the 353M-era multilingual 32K BPE on a pure-Python corpus: a vocabulary full of `public`/`void`/`String`/`Controller`/`document`/`window`/`css`/`php` (65.7% of the two vocabularies overlap, and 20,851 shared token IDs map to different strings). It also had no `<gh_stars>` token while 20% of star_coder samples begin with that prefix. Fix: retrain on a uniform, shuffled sample of all three *filtered* datasets (3.54 → 3.64 chars/token). A supporting factor rather than the primary culprit — the frozen training is what killed it.
3. **Cumulative-token arithmetic across rounds.** `total_tokens` accumulated across rounds while every consumer assumed a per-round value: the stop condition `total_tokens >= max_tokens` fired after one step when a smaller dataset was switched in, resume offsets read out of bounds, and ETAs went negative. Fix: track `round_start_tokens` and compute per-round values as deltas.

v1 also trained star_coder first, with its metadata still attached, and kept the sequential multi-dataset rounds (`--auto-resume`) that made the 353M's repeat-training bug possible in the first place.

---

## 2. Rebuilding clean (every v1 failure, and the response to it)

| v1 failure | v2 design response |
|---|---|
| mismatched tokenizer | retrained on uniform sample of the filtered corpus |
| 20% metadata-prefixed samples | `strip_metadata.py`, residual = 0 (3.35M train rows verified) |
| star_coder trained first | codeparrot first (matches tokenizer), then the_stack, then star_coder |
| resume re-read → repeated data | one merged 62GB train.bin, single continuous run, no resume-by-dataset |
| LR bump without momentum reset → 1.16→1.58 spike | WSM: constant LR + one final cooldown, no mid-flight changes |
| per-dataset LR schedules | global schedule over the merged stream |

---

## 3. v2 training: bugs in a run that worked

### 3.1 Cooldown silently disabled (the `active_lr` bug)

The final 30K-step cooldown (LR → 1e-4) started at step ~601K, yet by step 612,711 the LR was back at 2.5e-4: the run spent ~24K extra steps (~1.1B tokens) at full LR. Cause: the scheduler scaled by `final_lr / active_lr`, where `active_lr = optimizer.param_groups[0]["lr"]` reads the *current* LR — on a fresh run that is the peak LR, but on a resume during cooldown it is already 1e-4, so the ratio was 1.0 and the LR was multiplied back *up*. Fix: read `.get("initial_lr", ...)`, found by diffing the actual LR trajectory against the plan. **Lesson:** LR schedules fail silently; plot the curve and diff it against the plan, and never trust a "cooldown triggered" log line.

### 3.2 `reduce-overhead` is slower than `default` on sm_120

43K tok/s against 52K, plus `accessing tensor output of CUDAGraphs that has been overwritten` errors. CUDA graphs pay off for many-small-kernel workloads and a 435M model has large kernels; PyTorch 2.10+ `cudagraph_trees.py` regressions made it worse (GitHub #171672, #174575). **Lesson:** on Blackwell (sm_120) use `--compile-mode default` — measure, don't assume the "faster" mode is faster.

### 3.3 The released eval script did not run

`src/eval/eval.py` crashed on `TypeError: 'tokenizers.Encoding' object is not iterable` (`tokenizer.encode` returns an Encoding object, not a list → `.ids`), then on `RuntimeError: mat1 and mat2 have different dtype`. In `load_checkpoint(model_path, device)` the checkpoint tensors were loaded with `map_location=device` but `model.to(device)` was never called, so the model object stayed on the CPU while generation ids went to the GPU; casting the model to bf16 fails separately, because the forward's `x = self.embed(input_ids) * math.sqrt(self.config.d_model)` promotes to float32. **Lesson:** a script that ships with the code has to be run as a script before it ships — device, dtype and object-type bugs only show up when it is actually invoked.

### 3.4 Reference-pass false negatives: a docstring-writing model

v2 generated logically correct implementations and `reference_pass` still read 0/50. `check_implementation` ran on the generation only, so `required: ["def mergesort"]` failed (the `def` line lives in the prompt); a model trained on real GitHub Python writes a docstring first, spends the budget on it and follows it with junk (`# Driver Code`, `import numpy...`); and `judge_reference` executed that raw continuation. Fix: `extract_function_body(prompt + gen)` takes the first `def`, keeps its post-docstring body and stops at the first top-level line — only that core is checked and executed. Verdict: 0/50 → 7/50 (fibonacci 3/10, binary_search 3/10, two_sum 1/10; quicksort and mergesort stayed 0 because the model genuinely lacks `merge`/partition helpers, so the strictness holds), and no false positives: AST-unparseable samples still fail, no empty body passes. **Lesson:** the judging has to be compatible with the model distribution, not just the prompt style.

### 3.5 CUDA OOM with plenty of free VRAM: the Windows commit limit

WDDM charges GPU allocations against system commit. With the pagefile accidentally disabled (moved to another partition, `pagefile.sys` never created) the ceiling was ~38GB, and hitting 38/38 makes `cudaMalloc` refuse even for a tiny allocation. Fix: a fixed-size pagefile on a reliable drive, then reboot, verified with `Get-CimInstance Win32_PageFileUsage`. **Lesson:** check Task Manager → Performance → Memory → **Committed** before touching anything else.

### 3.6 State that has to survive a Windows run

Two failures of the same kind, both silent. The 18B no-filter run died 31% of the way through (step ~128K) with a decode error reading `train_meta.json`, a file written as pure ASCII, because Windows Python reads with the GBK default and choked on a UTF-8 byte; every `open()` now passes `encoding="utf-8"`. A resume replaces the model object and discards `torch.compile` (15-25% of throughput lost after every resume), and saved checkpoints carry `_orig_mod.` prefixes that break loading into an uncompiled model; both are fixed by recompiling after resume and stripping the prefix on save/load. **Lesson:** never rely on the Windows default encoding, even for files written by your own code, and treat compile state as part of the checkpoint contract.

---

## 4. Evaluation failures

1. **Scripts cheat**: `return arr.sort()` passes an output-based quicksort test → implementation checks plus human grading.
2. **Scripts false-negative**: logically correct solutions failed on tab/space mixing (TabError) → human review as the primary metric.
3. **Base models do not read docstrings**: a from-scratch model treats a docstring as the end of the function, so standard HumanEval prompts score 0/164 problems on every checkpoint, the shipped one included. Docstring-free prompts give the headline numbers; HumanEval stays in the harness with this caveat.
4. **Batch generation OOMs**: per-sample generation (≈1GB peak) instead of batched attention-mask generation (5× estimated peak).

---

## 5. What I take from it

1. **Audit resume paths for object identity** — a new model object plus an old optimizer is a frozen model with healthy logs.
2. **Silent bugs cost more than crashes** — the optimizer bug wasted the entire 21.5B-token v1 run, and the active_lr bug added ~24K steps at full LR (~1.1B tokens) on top of v2's cooldown.
3. **Match the distribution everywhere**: tokenizer↔data, fine-tuning data↔eval task, filter↔corpus.
4. **One merged stream** removes the whole class of multi-round resume and cumulative-token bugs.
5. **Windows training**: fixed pagefile, explicit UTF-8, `--compile-mode default` on sm_120.
6. **Human grading is the metric**; script pass-rates are evidence, not verdicts.

*(The first generation has its own war stories — the JAX→WSL2→PyTorch platform search, the Django/Plotly contamination disaster and the resume re-read bug: [`archive/353M/CHALLENGES.md`](archive/353M/CHALLENGES.md).)*
