# Fine-tuning a 435M code model: what worked and what did not

*Post-training experiment log for a from-scratch code LLM. One RTX 5090, ~30 training runs, 2026-09.*

> In this repo: the project overview is in [`../README.md`](../README.md), pretraining design decisions are in [`../DESIGN.md`](../DESIGN.md), bug stories in [`../CHALLENGES.md`](../CHALLENGES.md). The scripts that produced everything below are in [`scripts/`](scripts/), and every raw output is in [`results/`](results/) — indexed by model and line range in [`results/RAW-INDEX.md`](results/RAW-INDEX.md).

## 1. Context

The model in question is a 435M-parameter code LLM that I pretrained from scratch on ~31B tokens of Python and mixed code data. Pretraining left it able to continue code, but unable to follow instructions. The post-training goal was narrow and concrete: it should read a plain-English request ("write a function that checks whether a number is prime"), *and* complete a bare function signature, and produce code that runs.

Everything below ran on a single RTX 5090 (32 GB). Training runs were 12 minutes to 2 hours each; ~1.9 TB of intermediate checkpoints were produced and archived.

One thing about how the work was organised, because it shaped the results. The implementation — training harness, data pipeline, evaluation harness, most scripts — was developed with AI assistance. Every finding below is backed by a reproducible artefact rather than a summary: the two most valuable ones (a silent bug in the weight-interpolation code, and a "champion" model that did not generalise) surfaced only after re-running the code that produced them.

## 2. Before optimising anything, the measurement had to be fixed

Three problems made the first few weeks of comparisons meaningless.

**Single runs are not comparable.** Same data, same hyperparameters, only the random seed changed: pass rates ranged from 65% to 82%. Any two runs can differ by 6 points with no real difference in quality. Every early claim of the form "model A beats model B by 3 points" was noise.

**Validation loss does not select models.** This came up three separate times, twice in controlled comparisons where the checkpoint with the *better* validation loss was worse on tasks — once by 17 points, from the same training run, on the same data. Loss on a per-arm validation set is also not comparable across arms, since each arm's validation set comes from its own data distribution. Model selection has to be execution-based.

**The metric itself was contaminated.** The signature-column score was partly measuring output discipline rather than correctness. Correct implementations were marked wrong for printing a usage example after the function, or for sorting a list in place and returning `None` instead of a new list. A hand audit of raw generations showed that on one task, three models had all written a correct merge sort; the only difference was whether they appended `return arr` or an example — 19/20, 4/20 and 0/20 pass rates from that alone.

The fix was a protocol, not a patch: 20 attempts per task per prompt style (which cuts the uncertainty on a comparison from ±7 points to ±1.7), per-sample hand audits of raw output, a second "lenient" metric reported alongside the strict one (allowing in-place mutation conventions), and a task set that was held out of all selection. Fixing the metric changed more conclusions than any recipe change did, and it made the numbers worse.

## 3. What was tried

| # | Approach | What it was | Result |
|---|---|---|---|
| 1 | Signature data only | 29.4M tokens of `def f(...)` → body continuation | Completion works, instructions do not: 22% on the natural-language column, worse than the un-tuned base model |
| 2 | Instruction data mixed in | Adding 3M–15M tokens of public instruction data at 10:1, 3.3:1, 2:1, 1.2:1 | The first real gain: natural-language column 22% → 42% → 71%. Signature column drops as the instruction share rises |
| 3 | "Gold standard" corpus | Keep only instruction examples whose reference solution passes its unit tests (62M → 147M → 226M tokens) | The best single decision. Teaches from verified solutions instead of hoping the model will produce them: natural-language column 42% → 78% on the same protocol |
| 4 | Ratio sweep | Gold : signature at 1:1, 1.5:1, 2:1, 3:1, 5:1, 8:1 | Only re-allocates between the two columns (natural-language up monotonically, signature down); the total stays inside ±6-point seed noise from 1:1 to 8:1 (65–74% mean pass rate) |
| 5 | Two-stage training | Teach instructions first, then continue on signature data only | Failed, and instructively: the natural-language column fell from 78% to 60%. The second stage overwrote the first (catastrophic forgetting). Mixed in one pass beats staged |
| 6 | Self-distillation | Generate many candidates, keep only those that pass the test suite, train on those | Pipeline works but the yield is too low (19% of candidates usable, ~150 useful tokens each). Shelved |
| 7 | Weight averaging ("soups") | Average the weights of two trained models, no training | 17 blends, none better than its best parent; the blends sit between the parents and closer to the weaker one |
| 8 | Multi-seed averaging | Same recipe, three seeds, averaged weights | Better than the mean of its members (+1.7), worse than the best member (81 vs 78 in one case). Buys stability, not a new record |

A note on #7: the first pass at it was wrong in a way that produced plausible numbers. The interpolation code used `v.to(torch.float32)` to get a copy for arithmetic — but when the tensor is already float32, `.to()` returns the *same* tensor, so the in-place arithmetic overwrote the parent model. Each successive blend was interpolating with a corrupted parent, compounding the error. A least-squares check against the nominal weights exposed it: files labelled 0.65 / 0.50 / 0.35 / 0.20 actually contained 0.52 / 0.26 / 0.09 / 0.02. All α-sweep results had to be withdrawn and rebuilt, and the rebuilt code now refuses to write a blend whose measured deviation exceeds tolerance.

## 4. The trap: a champion that did not generalise

By this point one model led the board: 81.6% (mean pass rate across both prompt styles, 20 attempts per task), well clear of the field. The plan was to ship it.

Before doing so I built a held-out set: 12 new tasks that had never been used for any selection decision, checked first with reference implementations (all 12 pass under the strict metric; two deliberately in-place implementations fail strict and pass lenient, which confirms the lenient metric does what it claims).

On those 12 tasks the leader finished **last** — 55.8%, against 59–65% for the rest. The model that looked mid-pack on the tuning set was the best generaliser (64.6%).

The explanation is mundane and worth stating plainly: the "champion" had been selected on a fixed set of 11 tasks over many rounds of comparison. Part of its advantage was a stable formatting habit that happened to match what those tasks rewarded — and that habit paid nothing on new tasks. This is the standard failure mode of selecting on the same set repeatedly, and it produced the single largest correction in the project: a 26-point swing between the leaderboard and an honest evaluation.

## 5. What ships, and why

**The shipped model is the mid-pack one** (74.8% on the tuning set, 64.6% on the held-out set — first of seven candidates on unseen tasks, second of eleven on the tuning set). Recipe: 435M base, 147M tokens of gold-standard instruction data mixed with 29.4M tokens of signature data at 5:1, one epoch, LR 2e-5 cosine schedule, 32k tokens per step, ~90 minutes on one RTX 5090. Training about 30 models, all evaluated on both prompt styles with hand-audited outputs.

## 6. Limits

Three caveats apply to every number above.

1. **The benchmark is 23 short algorithmic tasks.** That is a proxy for "writes small functions correctly"; it says nothing about debugging, library use, or multi-file changes. Every number here is bounded by that.
2. **Seed variance is the largest effect measured in this project, and the shipped model rests on a single seed.** The honest interval around its true quality is roughly ±6 points. The recipe differences chased here (±3) are smaller than the noise that could not be removed.
3. **The held-out set was used once (correct) but is only 12 tasks (thin).** It rules out the specific failure it was built to catch; it does not prove broad generalisation.

## 7. What I would do next

- **Multi-seed search with selection and confirmation sets kept separate.** With ±6 seed noise, picking the best of 10 seeds on a selection set and confirming on unused tasks is the only lever identified in this project that converts luck into a reliable gain.
- **Preference learning from execution feedback.** The sandbox already produces pass/fail signals; turning them into training signal is the one approach that raises the whole distribution rather than re-allocating between the two columns.
- **Ruled out**: more ratio tuning, more staged schedules, or more weight averaging. All three are measured and capped.

## 8. Reproducibility

Every number in this report traces back to a raw generation file (one JSON record per attempt: prompt, completion, pass/fail). [`results/RAW-INDEX.md`](results/RAW-INDEX.md) maps each model to the file and line range holding its outputs; `results/tables/` holds the per-model summary files; `results/generations/` holds the outputs themselves. The replayable entry points are the scripts in [`scripts/`](scripts/): `build_raw_index.py` (regenerates the index from whatever is present), `rejudge_lenient.py` (re-derives the lenient column on CPU only, seconds), `eval_heldout.py` (the 12-task held-out set).

Results that were superseded are kept in the report as correction records rather than deleted, including the ones that made this project look worse.
