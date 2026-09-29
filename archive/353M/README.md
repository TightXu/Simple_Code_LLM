# 353M: the first generation (archived)

This directory holds the complete record of the first version: the model that proved I could train a code LLM from scratch on a single consumer GPU. It is archived, not deleted — it is the baseline the current version is measured against, and most of the lessons in the root [`CHALLENGES.md`](../../CHALLENGES.md) come from it.

| | Details |
|---|---|
| Architecture | 353M — 18 layers / d_model 1024 / d_ff 3840 (SwiGLU) / 16 heads / 32K BPE |
| Data | codeparrot/github-code Python, loose filter (`.py` + length), ~10.6 B unique |
| Actual training | 16.09 B cumulative (8 B duplicated by a resume bug, ~2.5 B tail never seen) |
| LR schedule | cosine annealing, 2.4e-4 → 2.4e-5, 500-step warmup |
| Best/final loss | 0.91 @ step 134,800 / 1.21 @ step 280,591 |
| Emergence | recursive fibonacci structure at ~7.74 B tokens (11/14 seeds at step 135K) |
| Human-graded | **6/100** (10 tasks × 5 seeds) |
| FT experiments | contaminated FT regressed (56→28%); clean same-domain FT +21pp |

## Why it scored 6/100

The three known defects (also recorded in the root [`CHALLENGES.md`](../../CHALLENGES.md)):

1. **Resume re-read from file 0**: `--auto-resume` restarted the dataset from the top, so ~8 B tokens got trained twice while ~2.5 B never appeared. The "16.09 B trained" figure is ~2× inflated in coverage.
2. **Loose filtering**: `.py` + length only. No quality gates, no contamination removal, no metadata strip (the current pipeline in `src/data/filter.py` fixes this with strict filtering + copyright + content checks).
3. **Mid-training LR bump without optimizer reset**: loss spiked 1.16 → 1.58 and generation regressed until the optimizer dynamics recovered.

On the same 10-task human-graded benchmark, the current rebuild (435M) scores **66/100**. The gap between the two numbers is why this rebuild exists.

## The continuation run (8B → 16B)

The model was later continued to 16.09B cumulative tokens on the same corpus, with the resume bug still active (the first ~8B got trained twice). A 14-seed human review of both checkpoints, eight tasks each (2026-08-31), measured:

| Measure (14 seeds) | 8B checkpoint | 16B checkpoint |
|---|---|---|
| fibonacci: recursive structure | 14/14 | 7/14 |
| fibonacci: correct base case | 0/14 | 0/14 |
| quicksort: an actual implementation (`arr.sort()` counts as functional, not as quicksort) | 5/14 | 0/14 |
| script pass rate (fibonacci / quicksort) | 0/14 / 3/14 | 0/14 / 0/14 |

The verdict of that review was that the continuation did not improve on the 8B checkpoint and degraded it, which is what the resume bug predicts: the second pass was over data the model had already seen.

These numbers measure *recursive structure*, not correctness. `TECHNICAL.md`'s own 14-seed table (steps 46.4K / 70K / 135K / 141.6K) records correct and usable rates for the pretraining checkpoints, which is a different measure on the same model.

## Files

| File | What it is |
|---|---|
| `README.md` | the original project landing page |
| `TECHNICAL.md` | the full technical report (617 lines) |
| `CHALLENGES.md` | the original war stories (including the JAX→WSL2→PyTorch platform wars) |
| `loss_curve.png` | training loss curve |

The unified pipeline in `src/` covers the same ground (with `--num-layers 18 --d_ff 3840 --lr-schedule cosine` reproducing this generation's architecture and schedule). The original 353M scripts (train.py written on JAX, later train_pt.py, plus the 6 eval variants) were intentionally folded into the unified `src/train.py` and `src/eval/eval.py` — that folding is the point of the rebuild, so they're not carried over as duplicates.
