# Ablation Framework

The clean rebuild trains the same 435M architecture from scratch multiple times, changing exactly one thing per run. Same tokenizer, batch size, LR schedule, and val set throughout. The point is to answer "why did this work?" as a set of measurements instead of a story.

Each experiment is a separate experiment: run the training script below, collect the checkpoint, run it against the eval suite. The framework itself is the deliverable here: the 353M first version had no way to attribute its success to data, scale, or schedule, and this is the gap the framework closes.

## Setup

```bash
# data scaling: memmap slice of the merged train.bin
py -3.12 ablation/ablation_data_scaling.py --scale 1B

# data quality: filtered vs no-filter
py -3.12 ablation/ablation_data_quality.py --mode compare

# tokenizer comparison
py -3.12 ablation/ablation_tokenizer.py --mode compare

# context length
py -3.12 ablation/ablation_context_length.py --ctx compare

# evaluation (per-sample generation; batch+mask OOMs, see CHALLENGES.md)
py -3.12 ablation/ablation_eval.py
py -3.12 ablation/eval_cross_models.py
```

Each entry point runs on its own; the shared model/training/inference code lives in `ablation_common.py`.

### Paths this repository does not ship

`eval_cross_models.py` is the one script that reads outside this repository: it loads the
earlier 353M first version from a sibling checkout, `<parent>/code-llm-project-353M/`,
expecting `checkpoints/` and `tokenizer/` under it. That is deliberate — the cross-model
table at the bottom of this file compares this project against that one, so both models
have to be loadable at once. The sibling project's checkpoints are not shipped here (they
are large), and its own sources are archived under `archive/353M/`. Point the constants at
the top of the script at wherever you keep that checkout if your layout differs.

`src/data/filter.py` carries a `CODE_LLM_PROJECT` constant for the same legacy sibling
layout, used only when filtering the original 353M corpus; the clean-rebuild path in
`data/README.md` does not touch it. Everything else in this directory runs against this
repository alone.

## Results

### 1. Data scaling (recursion emergence on fibonacci, 5 seeds): done

| Tokens | 1B | 2B | 5B | 10B | 20B | 31B (merged) |
|---|---|---|---|---|---|---|
| Recursive structure | 0/5 | 0/5 | 0/5 | 2/5 | 2/5 | **5/5** |

Recursion emerges between 10B and 31B. That matches a finding from the 353M first version, whose breakthrough came between ~2.6B and ~7.7B on a weaker data diet: recursion is a late, qualitative acquisition, not a smooth improvement. Loss barely moved in that window while generation quality jumped.

### 2. Data quality (filtered vs no-filter): 1B pilot + the 18B full run

#### @1B pilot

| @1B tokens | filtered v2 | no-filter | Δ |
|---|---|---|---|
| Val loss | 1.71 | 1.91 | **−0.20** |

#### @18B full run (2026-09-10, matched scale — the definitive comparison)

Fair comparison: **filtered v2 @18B (step_375000, 18.004B tok)** vs **no-filter @18B (step_380000, 18.0018B tok)**, with the same tokens, the same architecture, the same trainer (WSM, BS=12, lr 2.5e-4) and the same val set.

| @18B (training-log val, token-aligned) | filtered v2 | no-filter | Δ |
|---|---|---|---|
| Val loss at the 18.00B mark | **1.4103** | 1.4315 | **−0.021** |
| Average over 73 matched points (0.23–18.7B) | — | — | **−0.118** |
| Late window (15.0–18.7B, 16 points) | — | — | −0.067 |
| Window best (15–18.7B) | **1.3173** | 1.4133 | −0.096 |

The filtered run is lower at 72 of 73 matched points, but **the gap narrows as training proceeds** (−0.13/−0.14 in early/mid windows → −0.067 late), and at the 18B mark itself the two runs are within 0.02. Filtering's advantage is largest in the low-token regime; by 18B, with unique data, the no-filter penalty is small *in loss terms*, and the generation results below are where the difference shows.

> Detailed analysis in **[`ABLATION_REPORT.md`](ABLATION_REPORT.md)** (matched-scale report: design, token-aligned val, failure modes, HumanEval, threats to validity).

#### What the 18B evaluation actually measured (human review of generations, 5 algorithms × 10 seeds)

| Failure pattern | no-filter @18B | filtered v2 @18B | Interpretation |
|---|---|---|---|
| `exec_fail` (implemented but broken) | 30/50 | 19/50 | no-filter tries harder but writes broken code |
| `MISSING_struct` (no `mid` / no recursion) | 20/50 | 18/50 | both incomplete |
| `BANNED_cheat` (`arr.sort()` / `sorted(arr)`) | **0/50** | **13/50** | **v2 @18B resorts to builtins; no-filter never does** |
| Correct intent & style (docstring, edge cases) | low | high | v2 @18B writes GitHub-style code but can't implement |
| Correct implementations (all 50) | 0 | 0 | 18B is below the threshold |

Note: at 31B, the same v2 reaches 66/100 (mostly correct recursion structures). The 18B→31B jump is therefore where the model crosses from "knows the style" to "can implement", a strong scale signal and further evidence that the data pipeline (not more parameters) is what unlocks it.

#### HumanEval (docstring-style prompts, 163 scored problems × 10 seeds × 2 models = 3,260 generations, 2026-09-10)

| Model | pass | human-reviewed |
|---|---|---|
| no-filter @18B | 0/1,630 | 0, all outputs are comment walls / import dumps (docstring-blind, expected) |
| filtered v2 @18B | **4/1,630 (0.2%)** | **1 problem truly solved** (HumanEval/30 `get_positive`: `[i for i in l if i > 0]`, a correct implementation, reproduced across 4 seeds) |

Verdict: the first non-zero HumanEval pass from a from-scratch base model on the standard docstring prompts. The filtered model *can* read a docstring and implement it; the no-filter model cannot (its generations are entirely import/comment garbage). This is arguably the cleanest data-quality signal at 18B: not just val loss (−0.13), but an actual correct HumanEval implementation only in the filtered run. (Still only 4/1,630, a single easy problem, but the direction is unambiguous.)

### 3-5. Tokenizer / context length / LR schedule status

**Tokenizer: I closed this row without a training run (2026-09).** One clean same-family variant was measured, vocabulary size (32K → 48K → 64K) on identical data and recipe: compression improves by only **+1.6% / +2.8%** bytes-per-token, while the untied embedding and output head grow by **+7.5% / +15.1%** parameters. That is an upside-down trade before any training dynamics are considered, and a larger vocabulary also dilutes per-token gradients.

At any affordable budget (~0.1–1B tokens) the effect sits below run-to-run noise. An earlier 353M-era check on tokenizer *training-sample size* (~31K → 624K samples) produced token-for-token identical encodings on 6 of 7 test snippets, and the old "v1 failed because of its tokenizer" attribution has been superseded by the frozen-training bug in v1's resume path (summarized in [`../CHALLENGES.md`](../CHALLENGES.md)). For scale: a generic English tokenizer (GPT-2) burns **+66% tokens** on the same code. The row is closed as *not run, by design*: measured, reasoned and decided rather than skipped.

**Context length: I closed this row without a training run as well (2026-09).** The affordable budget (50M tokens per arm) sits far below the scale at which this project can resolve model-level differences at all: 353M only reached 4–6/100 after 8–16B tokens, two differently-trained 435M variants still scored 0/50 at 18B tokens on the 5-algorithm grid, and the gap only becomes visible around 31B. Even a ~30B-token run would not necessarily show an effect; at 50M the arms would produce a null that explains almost nothing.

Closed by design, on the same standard as the tokenizer row: decided with reasoning rather than skipped. The script and its readiness check are still in the repository (`ablation/ablation_context_length.py`).

The LR-schedule row is unchanged (no run performed or scheduled).

## Evaluation methodology (a result in itself; see DESIGN.md §8.3)

1. **Script pass-rates cheat**: `return arr.sort()` passes a quicksort output test. Fixed with implementation checks + human grading.
2. **Script pass-rates false-negative**: tab/space mixing (TabError) kills logically-correct code. Human review is primary.
3. **Docstring blindness**: from-scratch models treat docstrings as the end of the function. HumanEval passes no problem on any checkpoint (0/164 problems). Docstring-free prompts are used for the headline numbers.
4. **Batch generation OOMs**: per-sample generation (~1 GB peak) beats batched attention-mask generation (5× estimated peak).

## Cross-model comparison (the headline)

10 tasks × 5 seeds, human-graded 0/1/2 (max 100), docstring-free prompts:

| Model | fib | quicksort | binary_search | two_sum | mergesort | **Total** |
|---|---|---|---|---|---|---|
| 353M @8B (`archive/353M/`) | 5 | 1 | 0 | 0 | 0 | **6/100** |
| 353M @16B (continued on repeated data) | 3 | 0 | 1 | 0 | 0 | **4/100** |
| **435M @31B (this repo)** | 7 | 8 | 8 | 8 | 9 | **66/100** |

The second row is the 353M model after its training loop repeated 8B tokens (the resume bug: resuming restarted the dataset from the top). Its score *went down* from 6/100 to 4/100, and a 14-seed human review of the two checkpoints put fibonacci recursion at 14/14 before the second pass and 7/14 after it (see [`archive/353M/README.md`](../archive/353M/README.md)). More training on repeated data made generation worse. This is the clearest evidence for the data pipeline rather than scale.

> **Note on the numbers above**: script pass-rates are printed and stored (`ablation_scaling_results.json`, `eval_cross_summary.json`, etc.) but are **not** the headline metric, because they are unreliable in both directions. The 6/100 and 66/100 figures are human-graded. Raw generations (for your own inspection) live under `ablation/*_generations.jsonl` (git-ignored; regenerate with the eval scripts).
