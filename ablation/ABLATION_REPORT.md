# Ablation Report: Data Quality at 18B Tokens
## Filtered vs. no-filter pretraining — a matched-scale comparison

**Date**: 2026-09-10
**Models compared**: `filtered v2 @18B` (step_375000, 18.004B tokens) vs `no-filter @18B` (step_380000, 18.0018B tokens)
**Author**: TightXu
**Repo**: single-repository layout (`src/train.py`, `src/eval/eval.py`, `ablation/`)

---

## 1. Summary

The same 435M architecture was trained twice on equal token budgets (18B) with two data pipelines that differ in exactly one dimension, quality filtering, and both runs were evaluated on val loss, a five-algorithm generative suite (10 seeds each), and HumanEval (163 of the 164 problems in the task set × 10 seeds = 1,630 generations per model).

| Metric | no-filter @18B | filtered v2 @18B | Winner |
|---|---|---|---|
| Val loss @18.00B (log, matched) | 1.4315 | 1.4103 | filtered, −0.021 |
| Val loss, token-aligned average (73 matched points) | — | — | filtered, −0.118 |
| Best val in 15–18.7B window | 1.4133 | 1.3173 | filtered |
| Five-algorithm suite (5 × 10 seeds) | 0/50 | 0/50 | tie (both below threshold) |
| — `arr.sort()` cheating detected | 0/50 | 13/50 | no-filter "tries harder" |
| HumanEval (163 × 10 seeds = 1,630/model) | 0/1,630 | 4/1,630 (1 problem) | filtered |

**Conclusions**

1. **Filtering still wins at 18B, but the val-loss gap narrows**: from ~1B pilot (Δ −0.20) to ~−0.12 average across matched points, and −0.02 at the exact 18B mark. The gap is real and consistently in the same direction, but it is shrinking, not holding.
2. **Neither model has learned to implement basic algorithms at 18B**: 0/50 on the five-algorithm suite for both. The same filtered pipeline reaches 66/100 at 31B. **The 18B→31B jump, not 8B→18B, is where implementation ability appears.**
3. **The qualitative failure modes differ sharply**: the filtered model writes GitHub-style code (docstrings, edge-case guards) and resorts to builtins (`arr.sort()`, caught by the anti-cheat check in 13/50 cases); the no-filter model never cheats, always attempts a real algorithm, and fails by misunderstanding the problem or producing structurally wrong code.
4. **The only correct HumanEval implementation in the entire 3,260-generation HumanEval run came from the filtered model** (`get_positive`, solved correctly across 4 seeds). The no-filter model produced comment walls / import dumps in all 1,630 attempts.

---

## 2. Experimental design

### 2.1 Shared configuration (both runs)

| Component | Value |
|---|---|
| Architecture | 435M — d_model 1024, 22 layers, 16 heads, d_ff 4096, seq_len 1024, vocab 32,000 |
| Trainer | `src/train.py` — WSM schedule, constant LR + one final cooldown |
| Batch | BS 11 → 12 (changed mid-run), GA 4 |
| LR | 2.5e-4, 500-step warmup |
| Val set | `data/all/val.bin` (0.65B tokens) — identical for both runs |
| Eval cadence | every 5,000 steps |
| Precision | BF16 AMP, `torch.compile(mode="default")` |

### 2.2 Data recipes (this is the variable under test)

| | filtered v2 | no-filter |
|---|---|---|
| File | `data/all/train.bin` (62 GB) | `ablation/data_nofilter_v2/train.bin` (36 GB) |
| Composition (order) | codeparrot_clean 4.13B → the_stack 18.5B → star_coder 8.4B | codeparrot 9B → star_coder 9B (50/50) |
| Quality filtering (L1–L6) | applied (framework, tests, py2, configs, autogen, quality metrics) | none |
| Metadata stripping | applied (residual 0.00%) | applied — *after the fix described in §6* |
| Language filtering (`.py` only) | applied | applied |
| Deduplication | applied | none |
| EOS token injected | yes — `<eos>` (id 4), 0.062% of tokens | no — 0.00005% |
| Tokens in the compared checkpoint | 18.004B (step 375,000) | 18.0018B (step 380,000) |

Note on comparability: the two runs are matched on *tokens consumed* and on architecture/trainer/val-set, but not on dataset composition: the filtered run contains the_stack (18.5B in the full corpus) while the no-filter run does not, and the filtered run's codeparrot is deduplicated to 4.13B. The no-filter run was designed as "the same data sources only lighter-filtered", but the 50/50 codeparrot/star_coder split (a deliberate decision to keep the no-filter run tractable and language-clean) makes the recipe itself a second difference. See §5.3.

### 2.3 Checkpoint selection

Both checkpoints are the saved node closest to 18B tokens in their respective runs, with no merging, no best-of selection and no cooldown applied to either (both were still in the constant-LR phase at 18B):

- filtered v2: `checkpoints_wsm/step_375000` — 18,004,492,288 tokens
- no-filter: `checkpoints_wsm_nofilter_full/step_380000` — 18,001,768,448 tokens

---

## 3. Results

### 3.1 Val loss — token-aligned comparison

Val-loss points were extracted from both training logs and aligned on the token axis (nearest-neighbour within 60M tokens). 73 points matched across the common range (0.23B → 18.74B).

| Window | matched points | mean Δ (v2 − no-filter) | median |
|---|---|---|---|
| Full range 0.23–18.74B | 73 | −0.118 | −0.104 |
| Early (0.23–2B) | 12 | −0.130 | — |
| Middle (6.8–12.4B) | 24 | −0.140 | — |
| Late (15.0–18.7B) | 16 | −0.067 | — |

Selected matched points:

| tokens | no-filter | filtered v2 | Δ |
|---|---|---|---|
| 0.90B | 1.9158 | 1.7111 | −0.205 |
| 4.06B | 1.5532 | 1.5340 | −0.019 |
| 8.17B | 1.6175 | 1.3503 | −0.267 |
| 12.10B | 1.4847 | 1.3939 | −0.091 |
| 15.30B | 1.4341 | 1.3839 | −0.050 |
| 18.00B | 1.4315 | 1.4103 | −0.021 |

The filtered run is lower at 72 of 73 matched points. The one inversion (−0.013, at 18.25B) is within single-eval noise.

Trend: the gap is large and stable through mid-training (−0.13 to −0.14) and **narrows in the final window** (−0.067), consistent with the no-filter run slowly catching up as it consumes more unique tokens. At the 18B mark itself the two runs are within 0.021 of each other. In plain terms: filtering still helps at 18B, but the advantage compresses as the no-filter run sees more data.

> Data-quality note. Val losses above come from the training logs (same val set, eval every 5K steps, one shared evaluation path). The `best_val_loss` field stored inside each checkpoint records a run-level running minimum and does not match the log for the filtered run (1.2842 vs. the log's window minimum of 1.3173); the log values are used throughout this report because they are the reproducible, matched comparison.

### 3.2 Five-algorithm generative suite (10 seeds each)

Protocol: prompt = function signature + indent (`def fibonacci(n):\n    `), temperature 0.2, top-k 50, repetition penalty 1.2, min-p 0.05, ≤256 new tokens; judge = implementation check (banned/required patterns) + execution against reference asserts, on the extracted function body.

| Algorithm | no-filter pass | filtered pass | Dominant failure (no-filter) | Dominant failure (filtered) |
|---|---|---|---|---|
| fibonacci | 0/10 | 0/10 | exec (10) | missing structure (9) |
| quicksort | 0/10 | 0/10 | exec (10) | banned `arr.sort()` (4), exec (5) |
| binary_search | 0/10 | 0/10 | missing `mid` (8) | exec (6) |
| two_sum | 0/10 | 0/10 | exec (7) | exec (7) |
| mergesort | 0/10 | 0/10 | missing `mid` (9) | banned `arr.sort()`/`sorted()` (8) |
| **Total** | 0/50 | 0/50 | exec 30 / missing 20 / banned 0 | exec 19 / missing 18 / banned 13 |

**Neither model reaches a single passing implementation at 18B.** What the suite *does* separate is behaviour. Representative generations:

- fibonacci, no-filter (seed 0): `if n == 0: return 1 / else: return fibonacci(n-1) + fibonacci(n-2)`, correct recursion shape, wrong base case (returns 1 for n=0), which makes `fibonacci(1)` recurse forever → timeout.
- fibonacci, filtered (seed 0): `return (1 + 2 * n) / 3`, no recursion at all; the shape is wrong even though the style (clean one-liner) looks plausible.
- mergesort, filtered (seeds 0/42/999): `arr.sort() / return arr`, cheating, caught by the anti-cheat check; the model knows the *idiom* for "sort an array in Python" but not the algorithm.
- two_sum, no-filter (seed 0): comments describe "find the maximum value common to two arrays"; the problem is misread, not merely mis-implemented.
- two_sum, filtered (seed 42): `if len(nums) == 0: return 0 / if target < nums[0]: return target ...`, a wrong answer, but the *edge-case-guarding style* of real code.

### 3.3 HumanEval (163 problems × 10 seeds = 1,630 generations per model)

Standard HumanEval prompts (docstring + signature), docstring prompt style, same generation parameters. The shipped task set `ablation/HumanEval.jsonl` holds 164 problems; this run's records cover 163 of them (`HumanEval/0` has no generations), so every denominator below is 1,630 per model, 3,260 across both.

| Model | reference pass | tasks with ≥1 passing seed | human verification |
|---|---|---|---|
| no-filter @18B | 0/1,630 | 0/163 | all outputs are comment walls / import dumps, so the model never engages with the task |
| filtered v2 @18B | **4/1,630 (0.2%)** | 1/163 | verified correct: `HumanEval/30 get_positive` implemented as `[i for i in l if i > 0]`, reproduced across seeds 999, 2024, 12, 55 |

The four passing generations were audited by hand: the extracted function body is a correct, complete implementation of `get_positive`, and the reference asserts execute cleanly against it. This is the **first non-zero HumanEval pass in the project's history**: previous checkpoints (353M @8B/16B, 435M v1, 435M v2 @31B under standard prompts) passed no problem at all (0/164 problems).

It is still one easy problem out of the 163 scored. This is a direction signal, not a capability claim.

### 3.4 What the evaluations agree on

- The filtered model writes code that looks like code (docstrings, guards, idiomatic constructs). The no-filter model writes code that looks like a *model of code* (misread problems, broken structure, no idiom).
- The filtered model knows shortcuts (`arr.sort()`) that the no-filter model has never seen. Its corpus contained the idiom; the no-filter corpus apparently did not surface it as a pattern. This is a double-edged result: a cheat detector *and* evidence of exposure to real-world Python.
- At 18B neither can implement; at 31B the filtered model can (66/100 human-graded). **Implementation ability arrives with scale, and only the filtered pipeline has ever shown it.**

---

## 4. Analysis

### 4.1 Why the val gap narrows

The no-filter run's first 9B tokens are codeparrot (raw), against the filtered run's 4.13B deduplicated codeparrot + 13.9B the_stack at the same point. Early on, the filtered run's data is denser and cleaner per token, hence the ~0.13–0.14 gap. As training proceeds the no-filter run keeps consuming *unique* tokens from its star_coder half (which is language-filtered and metadata-clean), so its loss catches up. **Quality filtering buys its advantage mostly in the low-token regime; at 18B with unique data the penalty for skipping it is small in loss terms.**

### 4.2 Why the loss gap is the wrong headline anyway

Loss says 0.02–0.12. Generation says: **one model can be taught to read a docstring and produce a correct implementation, the other cannot produce anything but noise**: 4/1,630 vs 0/1,630, with the passing generations human-verified. The loss number cannot express this difference; the failure-mode table (misread problems, missing `mid`, no idiom) can. This matches the project's standing methodology: **loss orders models coarsely; human review of generations decides.**

### 4.3 The scale story

| | 1B | 18B | 31B |
|---|---|---|---|
| filtered val (training log) | 1.71 | 1.41 | 1.27 at the 31B mark; merged checkpoint measures 1.2363 |
| no-filter val | 1.91 | 1.43 | — |
| filtered, algorithm suite | — | 0/50 | 66/100 |
| filtered, HumanEval (standard prompts) | 0/164 problems | 4/1,630 generations (1 problem) | 0/164 problems |

Recursion on fibonacci emerges between 10B and 31B (0/5 → 2/5 → 5/5 across the scaling suite); full algorithm implementation appears only at 31B. **18B is an intermediate checkpoint on the way to capability, not a capability threshold**, which is exactly why the 18B comparison is interesting: it isolates the data-quality effect *before* the scale effect dominates.

---

## 5. Threats to validity

### 5.1 Evaluation judging changed during this work (disclosed)

The judge used here is not the judge used in older runs in this repo:

- `judge_reference` now extracts the **function body** (`extract_function_body`) from `prompt + generation` before executing, so a model that writes a docstring first and garbage afterwards is judged on what it actually implemented; previously it was judged on the raw continuation (all-zero false negatives).
- The implementation check now runs on `prompt + gen` (the `def` line lives in the prompt).
- Both 18B models were judged with the **same** updated judge, so the comparison between them is fair; comparisons to *older numbers in this repo* (e.g. the 353M 6/100) use a different judge and should be read as approximate.

Verification performed on the updated judge: no false positives (all AST-unparseable generations still fail; no empty-body passes); v2 @31B moved 0/50 → 7/50 with every passing sample hand-checked.

### 5.2 EOS asymmetry (disclosed, judged immaterial)

The filtered pipeline injects `<eos>` (id 4) at sample boundaries; the no-filter pipeline does not (see §6.3). In principle this could make the filtered model more likely to stop generating. In practice: **neither model emits EOS in any of the 3,260 generations** (nor does v2@31B), because at 0.062% of tokens the signal is too weak to teach stopping. The generation stop condition in the harness is `</s>` (id 1), which neither model ever produces; both therefore generate to the 256-token cap. **The asymmetry, if anything, penalises the filtered model** (its data carries an additional undocumented token), and the human review only reads the correct portion of each generation in any case.

### 5.3 Data recipes are not identical (§2.2)

The comparison isolates *filtering* but not *dataset composition*: the filtered run's 18B includes 13.9B of the_stack, which the no-filter run does not contain at all, and the two runs' codeparrot components are filtered vs raw at different sizes (4.13B deduplicated vs 9B raw). I constrained the no-filter run (codeparrot/star_coder 50:50, no the_stack) to keep it tractable and language-clean. Results should be read as "filtered pipeline as designed" vs "the no-filter alternative as constrained", not as a single-variable causal claim at 18B.

### 5.4 Single-problem HumanEval signal

With 4/1,630 the signal is one easy problem, so it stays a direction indicator rather than a capability result.

### 5.5 Mid-run data and schedule fixes affect the no-filter run's history

The no-filter run crashed twice and was resumed twice (§6). Its checkpoint at 18B is valid (data verified byte-identical where shared, metadata residual 0.00% after the fix), but its history is less "clean-room" than the filtered run's.

Disclosed in detail (found in 2026-09 by auditing the trainer's own counters):

- The run began (09-01) on the first no-filter bin (`ablation/data_nofilter/train.bin`, 30B tokens) and consumed **5.970B** tokens before the crash.
- When the run switched to the regenerated bin (`ablation/data_nofilter_v2/train.bin`, 18B tokens — **whose first 5.97B bytes are identical to the old bin**, which is what made the resume seamless), the trainer **reset the bin cursor to 0** and rebased its per-round token counter to the cumulative count at the switch (`round_start_tokens = 5.970B`; the equivalent code now lives in `src/train.py`).
- Consequence: the first 5.97B tokens read from the new bin are the **same tokens already trained on in the first phase**, so the no-filter arm's 18.0B node contains a **repeated 5.97B block** (≈33% of its token-steps are a second pass over content it had already seen), and only ≈12.03B of its 18.0B are distinct new content (5.97B of pre-fix codeparrot + 6.06B of newly fixed data). The filtered arm has no such repetition (its training log shows `data_pos offset ≡ total_tokens` throughout).
- The same accounting applied `--max-tokens 18B` **per round**, so after the switch the run's actual target was ≈23.97B cumulative tokens (the run was stopped by hand at 18.98B). The 18B node used for this comparison (`step_380000`, cumulative 18.002B) is unaffected.
- Read together with §5.3, the 18B contrast is "filtered pipeline as designed (18B distinct tokens)" vs "the no-filter alternative as constrained (≈12.03B distinct tokens, including a repeated block and 5.97B of pre-fix preprocessing)". The direction of the result is unchanged; the comparison is disclosed as non-clean-room.

---

## 6. Infrastructure fixes made during this run (context for the numbers)

These are recorded because they affect the provenance of the no-filter checkpoint.

### 6.1 Metadata-strip bug in star_coder preprocessing (the crash root cause)

Symptom: no-filter training val collapsed to 2.5–3.2 at the 9B data boundary (twice, reproducibly).

Root cause: star_coder's `content` field chains tags on a single line (`<reponame>foo/bar<gh_stars>12\n…`). The first strip implementation removed only the leading tag, leaving `foo/bar<gh_stars>12` in the text, so **10.58% of samples** carried this residue, and the model learned it as content right at the dataset boundary.

Fix: regex rewritten to strip all chained tags in one pass. Verified: residue **10.58% → 0.00%**; the regenerated bin is byte-identical to the old one for the first 9B tokens (codeparrot segment), so the run resumed seamlessly.

Evidence it was the root cause: with the fixed bin, the same training continued across the same boundary with val 1.42–1.43 (vs 2.5+ before), and completed 18B.

### 6.2 WSD scheduler bug: batch-size change + autoresume → negative LR

Symptom: LR went negative, loss went NaN, at step ~370K of a run whose stable phase was supposed to last 399K steps.

Root cause: BS 11→12 changed tokens/step (45,056 → 49,152), so on autoresume `total_steps` was recomputed (399,502 → 366,210) and the stable-phase boundary moved *below* the current step. The decay branch then computed `progress = (step − 500 − stable)/decay_steps` with `decay_steps = 0` → LR = 1 − progress·0.9 → −3408 → NaN.

Fix: guard `decay_steps <= 0` → return constant LR. Verified: the repaired run continued past the old crash point at constant 2.5e-4 with no NaN.

**Lesson**: a batch-size change mid-run is not inherently fatal, but *batch-size change + autoresume + a scheduler whose phase boundaries are recomputed from total_steps* is a compound hazard.

### 6.3 EOS investigation

Question raised during evaluation: "is EOS in the data or not?"

Findings (script + binary evidence):
- `<eos>` (id 4) is injected, but only in the `pre_tokenize_*` (bin-generation) path used by the filtered corpus: `data/all/train.bin` measures 0.062% id-4 (≈1 per sample, matching "text + `<eos>`"). The v1 pipeline also had a `has_eos` monitor enforcing this.
- The no-filter path (`prepare_nofilter_*.py`) does not inject it: `ablation/data_nofilter_v2/train.bin` measures 0.00005%.
- `</s>` (id 1) was never injected in any pipeline (all binaries ≈0).
- Practical impact: none measurable (§5.2).

### 6.4 Eval harness fixes

Three bugs found when the released harness was first actually run, plus one judging improvement (§5.1 documentation of provenance):

1. `tokenizer.encode(prompt)` returns a `tokenizers.Encoding`, not a list, so `.ids` is required.
2. `load_checkpoint` loads with `map_location=device` but never `model.to(device)` → CPU weights + CUDA inputs; also must not cast to bf16 (the model's `x * math.sqrt(d_model)` promotes to float32, so bf16 weights mismatch).
3. `--algorithms-file` added so custom task sets (e.g. HumanEval) can be run through the same harness; HumanEval-format tasks are auto-converted.
4. `extract_function_body` judging (§5.1).

---

## 7. Reproduce

```bash
# 1. the two checkpoints (the WSM trainer is now src/train.py; the original script is not shipped)
#    filtered: data/all/train.bin (31B, codeparrot_clean + the_stack + star_coder)
#    no-filter: ablation/data_nofilter_v2/train.bin (18B, raw codeparrot + star_coder(.py))
py -3.12 src/train.py --mode offline \
    --bin-data <bin> --val-bin data/all/val.bin \
    --max-tokens <n> --ckpt-dir <dir> --log-dir <dir> \
    --compile-mode default --lr 2.5e-4 --batch-size 12 --grad-accum 4 \
    --warmup-steps 500 --lr-schedule wsm --save-every 5000 --eval-every 5000

# 2. evaluation (both models in one invocation; 10 seeds; HumanEval via task file)
py -3.12 src/eval/eval.py \
    --models "checkpoints_wsm/step_375000,checkpoints_wsm_nofilter_full/step_380000" \
    --seeds 0,42,123,999,2024,7,12,55,101,313 \
    --algorithms-file ablation/HumanEval.jsonl \
    --prompt-style docstring \
    --output-dir ablation/eval_pair/eval_pair_humaneval
```

Artifacts from this run:
- Five-algorithm generations: `ablation/eval_pair/eval_pair_publish_18b_v2/` (100 records)
- HumanEval generations: `ablation/eval_pair/eval_pair_humaneval_18b_v2/` (3,260 records)
- Training logs: `ablation/nofilter_logs_full/training_log.csv` (no-filter), `training_log.csv` (filtered)

---

## 8. Conclusions

1. **Data filtering remains beneficial at 18B, with a narrowing margin in loss** (−0.12 average across matched points; −0.02 at the exact 18B mark; 72/73 points favouring the filtered run).
2. **Neither 18B model can implement the five reference algorithms** (0/50 each). The filtered pipeline reaches 66/100 at 31B: **implementation ability is a scale phenomenon that the filtered pipeline reaches and, so far, only that pipeline has reached.**
3. **Failure modes differ qualitatively**: the filtered model mis-implements but writes idiomatic, docstring-bearing, edge-case-guarded code (and resorts to builtin sorting in 13 of the 50 five-algorithm generations, 26%); the no-filter model mis-reads problems and produces structurally broken code, but never cheats.
4. **The only correct HumanEval implementation across the 3,260 HumanEval generations belongs to the filtered model** (1 problem, 4 seeds, hand-verified), a direction signal rather than a capability result.
5. **Practical implication for this project**: with filtering, budget should go to *more tokens* (18B → 31B is where capability appears), not to further filter refinement at fixed scale. Without filtering, no amount of the 18B-scale training produced a working implementation.

---

## Appendix A — Artifact index

| Artifact | Path |
|---|---|
| Filtered @18B checkpoint | `checkpoints_wsm/step_375000/checkpoint.pt` (18.004B) |
| No-filter @18B checkpoint | `checkpoints_wsm_nofilter_full/step_380000/checkpoint.pt` (18.0018B) |
| Filtered training log | `training_log.csv` |
| No-filter training log | `ablation/nofilter_logs_full/training_log.csv` |
| Five-algorithm generations | `ablation/eval_pair/eval_pair_publish_18b_v2/generations.jsonl` |
| HumanEval generations | `ablation/eval_pair/eval_pair_humaneval_18b_v2/generations.jsonl` |
| HumanEval task set | `ablation/HumanEval.jsonl` (164 problems; 163 scored in this run) |
| Eval harness | `src/eval/eval.py` |
| Summarizer | `ablation/summarize_humaneval.py` |

## Appendix B — HumanEval/30: the one correct generation (verbatim extract)

```python
def get_positive(l: list):
    """Return only positive numbers in the list.
    >>> get_positive([-1, 2, -4, 5, 6])
    [2, 5, 6]
    >>> get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    [5, 3, 2, 3, 9, 123, 1]
    """
    return [i for i in l if i > 0]
```

Reference asserts (all pass): `get_positive([-1,-2,4,5,6]) == [4,5,6]`; `get_positive([5,3,-5,2,3,3,9,0,123,1,-10]) == [5,3,2,3,3,9,123,1]`; `get_positive([-1,-2]) == []`; `get_positive([]) == []`.

Model: `checkpoints_wsm_v2_18b` (filtered, step_375000). Seeds: 999, 2024, 12, 55.
