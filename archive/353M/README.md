# 353M — The First Generation (Archived)

This directory holds the complete record of the first version: the model that proved a code LLM can be trained from scratch on a single consumer GPU. It's archived, not deleted. It's the baseline the current version is measured against, and it's where most of the lessons in the root `CHALLENGES.md` come from.

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

The three known defects (the root `CHALLENGES.md` covers them in detail):

1. **Resume re-read from file 0** — `--auto-resume` restarted the dataset from the top, so ~8 B tokens got trained twice while ~2.5 B never appeared. The "16.09 B trained" figure is ~2× inflated in coverage.
2. **Loose filtering** — `.py` + length only. No quality gates, no contamination removal, no metadata strip (the current pipeline in `src/data/filter.py` fixes this with strict filtering + copyright + content checks).
3. **Mid-training LR bump without optimizer reset** — loss spiked 1.16 → 1.58 and generation regressed until the optimizer dynamics recovered.

On the same 10-task human-graded benchmark, the current rebuild (435M) scores **66/100**. The gap between the two numbers is why this rebuild exists.

## Files

| File | What it is |
|---|---|
| `README.md` | the original project landing page |
| `TECHNICAL.md` | the full technical report (617 lines) |
| `CHALLENGES.md` | the original war stories (including the JAX→WSL2→PyTorch platform wars) |
| `loss_curve.png` | training loss curve |

The unified pipeline in `src/` covers the same ground (with `--num-layers 18 --d_ff 3840 --lr-schedule cosine` reproducing this generation's architecture and schedule). The original 353M scripts (train.py written on JAX, later train_pt.py, plus the 6 eval variants) were intentionally folded into the unified `src/train.py` and `src/eval/eval.py` — that folding is the point of the rebuild, so they're not carried over as duplicates.
