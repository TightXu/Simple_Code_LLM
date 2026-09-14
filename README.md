# A Code Language Model From Scratch on a Single RTX 5090

What happens when you train a code-generating language model from the ground up — data, tokenizer, architecture, training loop, evaluation — on one consumer graphics card? This repo is that question, tried twice.

The first version (`archive/353M/`) showed the question itself has a yes: a single GPU can run the whole pipeline, and even at a fraction of the size of a frontier model the result learns the pattern of recursion. It also showed three problems: weakly filtered data, about eight billion tokens quietly trained twice by a resume bug, and a mid-training learning-rate change that wasn't planned for. So the second version, the one this repo documents, rebuilt everything on a clean foundation and turned those problems into a controlled set of experiments.

The headline is the difference between the two. Nearly the same parameter count, the same single 32 GB RTX 5090, the same 32K-token code vocabulary. Yet on a human-graded 10-task benchmark the first version scores 6/100 and the rebuild scores 66/100. The architecture did not change. What changed was the data: a tokenizer retrained on a sample drawn from the same distribution the model is expected to learn, metadata stripped from a fifth of the corpus, and a training loop that actually trains.

## The project in two versions

| | First version (archived) | Clean rebuild (this repo) |
|---|---|---|
| Architecture | 353M — 18 layers / 3840 FFN | 435M — 22 layers / 4096 FFN |
| Data | codeparrot Python, loose filter, ~10.6 B unique | 3 sources, strict filtering (copyright, metadata strip, quality gates), 31.0 B |
| LR schedule | cosine annealing | WSM: stable + explicit 30K-step cooldown + checkpoint merge |
| Tokens run | 16.09 B cumulative (8 B duplicated) | 31.0 B, single continuous stream |
| Val loss | 1.21 | 1.2363 (merged, measured) |
| **Human-graded (10 tasks × 5 seeds)** | **6/100** | **66/100** |
| Full story | [`archive/353M/`](archive/353M/) | [`DESIGN.md`](DESIGN.md) |

A note on the loss numbers. The two sides use different LR schedules: the first run anneals with cosine (loss glides downward the whole run), while the clean rebuild runs WSM — a constant stable LR with a short cooldown at the end, because its final model is an average of the last checkpoints rather than a single late checkpoint. Constant-LR runs end with a naturally higher loss than cosine runs; that's an artifact of the schedule, not of model quality. The difference to pay attention to is the human-graded column: 6/100 vs 66/100, same architecture, same GPU, different data pipeline.

If you read nothing else, read why the scores differ. The two models are the same kind of creature; one outputs comments and unrelated code, the other outputs working implementations. The difference is where the effort went — data engineering, not architecture — and the ablation framework in `ablation/` is how that effort is measured rather than asserted.

## What's in this repo

```
├── README.md            # you are here
├── DESIGN.md            # design record: decisions, rationale, results (this rebuild)
├── CHALLENGES.md        # every bug / contaminant / lesson — both versions
├── LICENSE              # MIT
│
├── src/                 # all runnable code
│   ├── train.py         #   unified training (parameterized arch: --num-layers/--d_ff;
│   │                    #   LR schedules: wsm / wsd / cosine, incl. 353M & 435M both)
│   ├── merge_checkpoints.py           #   uniform checkpoint averaging
│   ├── merge_checkpoints_weighted.py  #   LR-weighted averaging (used for the 435M merged model)
│   ├── data/            #   filter.py (strict filtering + copyright), prepare_stream.py
│   ├── tokenizer/       #   train_tokenizer.py, strip_metadata.py, retokenize.py
│   └── eval/            #   eval.py — unified multi-model × multi-algorithm × multi-seed
│                        #   (CLI by default, --interactive for guided mode)
│
├── ablation/            # the ablation framework (scaling / quality / tokenizer / context)
│   ├── ablation_common.py, ablation_*.py, eval_cross_models.py, HumanEval.jsonl
│   └── README.md        # result table + status
│
├── post-training/       # supervised fine-tuning on the merged baseline
│   ├── POST_TRAINING.md #   experiment log: what was tried, what moved the needle
│   ├── scripts/         #   data prep / SFT training / evaluation / weight averaging
│   └── results/         #   summary tables + every generation, indexed (RAW-INDEX.md)
│
├── archive/353M/        # first version: README / TECHNICAL / CHALLENGES / scripts
├── results/             # pretraining loss curves + cross-model comparison figure
├── data/README.md       # dataset sources + pipeline reproduction
├── tokenizer/           # tokenizer_353m.json, tokenizer_435m.json
└── experiments/         # .gitignored — checkpoints / eval outputs (not in git)
```

## How to run

Python 3.12 and one CUDA GPU. `pip install -r requirements.txt` (install a CUDA build of PyTorch for your GPU first) — everything below was run on a single RTX 5090.

```bash
# 1. data — filter the raw corpus and pack it into one continuous token stream
#    (sources and the per-layer filter rules: data/README.md)
py -3.12 src/data/prepare_stream.py --data-dir <raw_parquet_dir> --output-dir data/all

# 2. tokenizer — 32K byte-level BPE over a uniform sample of all filtered sources
py -3.12 src/tokenizer/train_tokenizer.py

# 3. pretrain — 435M config (22 layers / d_ff 4096) with the WSM schedule
py -3.12 src/train.py --mode offline \
    --bin-data data/all/train.bin --val-bin data/all/val.bin \
    --num-layers 22 --d_ff 4096 --lr-schedule wsm \
    --warmup-steps 500 --final-lr-steps 30000 --compile-mode default

# 4. merge — the final model is a weighted average of the last checkpoints, not the last step
py -3.12 src/merge_checkpoints_weighted.py --ckpt-dir experiments/ckpt_435m \
    --window-steps 10000 --weight-cooldown 0.30 --out experiments/ckpt_435m/merged

# 5. evaluate — multi-model × multi-task × multi-seed, human-reviewable output
py -3.12 src/eval/eval.py --interactive
```

`--mode online --data-dir <dir>` streams and filters on the fly instead of using a packed `.bin`. Post-training is scripted under `post-training/`; `post-training/POST_TRAINING.md` says what each run was and how it scored.

## Key decisions that matter

- **One merged 62 GB `train.bin`, one continuous run.** The three datasets are concatenated into a single file and trained without resetting anything between them. No dataset boundaries, no optimizer resets mid-flight. This is because the first version's bug — resuming a run restarted the dataset from the top, so some data got trained twice and some never at all — is a class of bug a single continuous stream cannot have. It's not "we were careful so it didn't happen"; the design makes it impossible.
- **WSM + explicit cooldown + checkpoint merge.** Run training at constant learning rate, then lower the rate for the last 30K steps to recover tail quality, then average the last 10K checkpoints into the final model. (There's also a WSD variant that decays automatically over the last 20% of training; it's kept as a reference schedule, not what the final model used.)
- **Evaluation is a research problem, not a checkbox.** Autograder pass-rates cheat (a `return arr.sort()` passes a quicksort output test) and false-negative on tab/space mixing. From-scratch models don't read docstrings, so standard HumanEval prompts score 0/164 on every checkpoint. The honest metric at this scale is human-graded review of fixed docstring-free prompts across seeds, with a reference pass-rate computed alongside. Every generation is kept in the repo (`post-training/results/`, one JSON record per attempt, indexed by model and line range in `RAW-INDEX.md`) so the evidence stays inspectable.

## Post-training

The baseline above is a code *completion* model: it continues whatever it is given, and it does not read instructions. The second half of the project is supervised fine-tuning — about 30 runs on the same GPU, asking what turns it into a model that follows a request. Short version: what worked was training on instruction data whose reference solutions pass their own tests; the gold:signature ratio only traded between the two evaluation columns; random-seed variance (±6 points on identical settings) was larger than any recipe difference; weight averaging never beat its best parent; and the leader on the tuned-on benchmark finished **last** on 12 unseen tasks. (These are pass rates over repeated attempts — a different measure from the 0/1/2-graded benchmark above, and the two should not be compared.) The shipped model is the one that held up on the held-out set.

Full log, corrections included: [`post-training/POST_TRAINING.md`](post-training/POST_TRAINING.md). Per-attempt generations: [`post-training/results/RAW-INDEX.md`](post-training/results/RAW-INDEX.md).

## Why this is here

Most "train your own LLM" material stops at a toy model with 10M parameters and no real evaluation. This one goes somewhere less comfortable: a real dataset, real training infrastructure, and a hard look at what a few-hundred-million-parameter model actually learns — including the parts that go wrong. Data contamination, silent training bugs, evaluation that lies. If you want to understand what sits between `pip install transformers` and a model that writes working code, this is for you.

Both runs trained on a single NVIDIA RTX 5090 (32 GB, Blackwell sm_120), July–September 2026, and neither run ever left one machine.
