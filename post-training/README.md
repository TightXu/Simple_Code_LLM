# Post-training (supervised fine-tuning)

The model in `../DESIGN.md` is a code **completion** model: it continues whatever it is given, and it does not read instructions. This directory is the second half of the project — about 30 supervised fine-tuning runs on that baseline, plus the measurement work needed to tell whether any of them actually helped.

**Start here:** [`POST_TRAINING.md`](POST_TRAINING.md) — what was tried, what moved the needle, and the two results that had to be withdrawn.

## Layout

| Path | What it is |
|---|---|
| `POST_TRAINING.md` | the experiment log: eight approaches, their scores, and the corrections |
| `scripts/` | data preparation, SFT training, evaluation, weight averaging, output indexing |
| `results/tables/` | per-model summaries (both metric columns, 20-seed runs, held-out set, lenient re-judge) |
| `results/generations/` | every generation — one JSON record per attempt: `prompt`, `completion`, `passed` |
| `results/RAW-INDEX.md` | model → file → line range → pass counts; regenerate with `scripts/build_raw_index.py` |

## The pipeline as it was run

1. **Signature corpus** — `scripts/build_sft_data.py`: `def f(...)` → body pairs drawn from the pretraining corpus, minus a leak blacklist of the evaluation tasks.
2. **Instruction data** — `scripts/build_instr_data.py`: public instruction datasets (OpenCodeInstruct shards, self-oss-instruct) converted to the same contract, optionally keeping only examples whose reference solution passes its own unit tests (the "gold standard" corpus).
3. **Mixing** — `scripts/mix_bins.py`: signature and instruction bins interleaved at a chosen ratio into one training file.
4. **Contract check** — `scripts/verify_instr_contract.py`: uint16 token ids + uint8 loss mask, chunk 1025, loss only on the solution segment.
5. **Training** — `scripts/sft_train.py`: 1 epoch, LR 2e-5 cosine → 10%, warmup 50, 8 × 4 accumulation = 32k tokens/step, seed parameterized.
6. **Evaluation** — two prompt styles, repeated over seeds, every attempt kept:
   - `scripts/eval_sft.py` — signature prompts (`def f(...):` + indent)
   - `scripts/eval_nl.py` — natural-language requests (`### Task … ### Code`)
7. **Judging correction** — `scripts/rejudge_lenient.py`: re-derives the lenient column on CPU only (it allows in-place mutation conventions, which the strict metric marks wrong).
8. **Held-out validation** — `scripts/eval_heldout.py`: scores candidates on 12 tasks that were never used for any selection decision.
9. **Weight averaging** — `scripts/soup_models.py` (builds interpolations, with an exact-average self-check before writing anything) and `scripts/soup_sweep.py`.

Orchestration examples live in `scripts/` too: `chain_g_ext.sh` (gold-standard ratio sweep), `chain_night_quality.sh` (seed-noise quantification), `chain_recipe_soup.sh`, `chain_seed_20seeds.sh`, `chain_g9_soup_final.sh` (20-seed precision runs and weight soups). They call the tools above in sequence and log to `logs/`; everything ran serially on a single GPU.

## Reproducing a number

```bash
# find where a model's raw outputs live
grep -n 'g9s46_sig' results/RAW-INDEX.md          # → file + line range

# read them (one JSON record per attempt)
sed -n '1,220p' results/generations/generations_noise20_g9s46_sig.jsonl

# re-derive the lenient column (CPU only, seconds, no GPU)
py -3.12 scripts/rejudge_lenient.py
```

## Caveats worth knowing before trusting a number

- **Seed variance is ±6 points.** Same data, same hyperparameters, different random seed. Single-run comparisons are not evidence — this is the finding that reshaped the whole project.
- **Validation loss does not select models.** Demonstrated three times here, twice in controlled comparisons where the better loss belonged to the worse model.
- **The strict metric under-counts.** A correct implementation that sorts a list in place and returns `None`, or appends a usage example after the function, is marked wrong. Both columns are reported for that reason.
- **5-seed and 20-seed numbers are different protocols** (different seed sets) and must not be compared with each other.
- **The held-out set was used once.** It is 12 tasks: it rules out the failure it was built to catch, it does not prove generalisation.
- **Checkpoint paths in the shipped tables are repo-relative** (`fine-tuning/ckpt_g9_s46/best_val`); the original runs used local directories on the machine that trained them.
- The scripts are the working versions from the experiments (paths parameterized, comments in English); the checkpoints they produced are not in the repo — the evidence in `results/` is.
