# A Code Language Model From Scratch on a Single RTX 5090
> 30-second version: [TLDR.md](TLDR.md)

This repo documents my three training runs and two complete versions for a code LLM built from scratch, end to end, on one consumer GPU: data, tokenizer, architecture, training loop and evaluation.

The first version, the 353M model ([`archive/353M/`](archive/353M/)), showed that a single GPU can run the whole pipeline, and that even at a fraction of the size of a frontier model the result learns the pattern of recursion. It also showed three problems: weakly filtered data, about eight billion tokens quietly trained twice by a resume bug, and a mid-training learning-rate change that wasn't planned for. An intermediate 435M attempt then failed outright (val loss frozen at 1.97 across 21.5B tokens, no usable generation), and what it exposed is summarized in [`CHALLENGES.md`](CHALLENGES.md). So the 435M documented here starts from a clean foundation and turns all of those problems into a controlled set of experiments.

The headline is the difference between the two finished versions. Nearly the same parameter count, the same single 32 GB RTX 5090, the same 32K-token code vocabulary. Yet on a human-graded 10-task benchmark the first version scores 6/100 and the rebuild scores 66/100. The architecture did not change. What I changed was the data: a tokenizer retrained on a sample drawn from the same distribution the model is expected to learn, metadata stripped from a fifth of the corpus, and a training loop that actually trains.

## The project in two versions

| Version | 353M first version (archived) | 435M v2 (this repo) |
|---|---|---|
| Architecture | 353M — 18 layers / 3840 FFN | 435M — 22 layers / 4096 FFN |
| Data | codeparrot Python, loose filter, ~10.6 B unique | 3 sources, strict filtering (copyright, metadata strip, quality gates), 31.0 B |
| LR schedule | cosine annealing | WSM: stable + explicit 30K-step cooldown + checkpoint merge |
| Tokens run | 16.09 B cumulative (8 B duplicated) | 31.0 B, single continuous stream |
| Val loss | 1.21 | 1.2363 (merged, measured) |
| **Human-graded (10 tasks × 5 seeds)** | **6/100** | **66/100** |
| Full story | [`archive/353M/`](archive/353M/) | [`DESIGN.md`](DESIGN.md) |

The two sides use different LR schedules: the first run anneals with cosine (loss glides downward the whole run), while the clean rebuild runs WSM (a constant stable LR with a short cooldown at the end), because its final model is an average of the last checkpoints rather than a single late checkpoint. Constant-LR runs end with a naturally higher loss than cosine runs; that's an artifact of the schedule, not of model quality. The column that matters is the human-graded one: 6/100 vs 66/100, same architecture, same GPU, different data pipeline.

The reason the scores differ is worth stating plainly: these are the same kind of model, and one outputs comments and unrelated code where the other outputs working implementations. The difference is where the effort went: data engineering, not architecture. The ablation framework in `ablation/` is how I measure that rather than assert it.

## What's in this repo

```
├── README.md            # you are here
├── DESIGN.md            # design record: decisions, rationale, results (this rebuild)
├── CHALLENGES.md        # every bug / contaminant / lesson — both versions
├── LICENSE              # MIT
├── requirements.txt     # Python dependencies (install a CUDA build of PyTorch separately)
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
│   ├── POST_TRAINING.md #   experiment log: what was tried, what worked and what did not
│   ├── scripts/         #   data prep / SFT training / evaluation / weight averaging
│   └── results/         #   summary tables + every generation, indexed (RAW-INDEX.md)
│
├── archive/353M/        # first version: README / TECHNICAL / CHALLENGES / loss curve
├── results/             # pretraining loss curves + cross-model comparison figure
├── data/README.md       # dataset sources + pipeline reproduction
├── tokenizer/           # tokenizer_353m.json, tokenizer_435m.json
└── experiments/         # .gitignored — checkpoints / eval outputs (not in git)
```

## How to run

Python 3.12 and one CUDA GPU. `pip install -r requirements.txt` (install a CUDA build of PyTorch for your GPU first). Everything below was run on a single RTX 5090.

```bash
# 1. data: filter the raw corpus and pack it into one continuous token stream
#    (sources and the per-layer filter rules: data/README.md)
py -3.12 src/data/prepare_stream.py --data-dir <raw_parquet_dir> --output-dir data/all

# 2. tokenizer: 32K byte-level BPE over a uniform sample of all filtered sources
py -3.12 src/tokenizer/train_tokenizer.py

# 3. pretrain: 435M config (22 layers / d_ff 4096) with the WSM schedule
py -3.12 src/train.py --mode offline \
    --bin-data data/all/train.bin --val-bin data/all/val.bin \
    --num-layers 22 --d_ff 4096 --lr-schedule wsm \
    --warmup-steps 500 --final-lr-steps 30000 --compile-mode default

# 4. merge: the final model is a weighted average of the last checkpoints, not the last step
py -3.12 src/merge_checkpoints_weighted.py --ckpt-dir experiments/ckpt_435m \
    --window-steps 10000 --weight-cooldown 0.30 --out experiments/ckpt_435m/merged

# 5. evaluate: multi-model × multi-task × multi-seed, human-reviewable output
py -3.12 src/eval/eval.py --interactive
```

`--mode online --data-dir <dir>` streams and filters on the fly instead of using a packed `.bin`. Post-training is scripted under `post-training/`; `post-training/POST_TRAINING.md` says what each run was and how it scored.

## The decisions that shaped it

- **One merged 62 GB `train.bin`, one continuous run.** The three datasets are concatenated into a single file and trained without resetting anything between them: no dataset boundaries, no optimizer resets mid-flight. The first version's bug, where resuming a run restarted the dataset from the top so that some data got trained twice and some never at all, is a class of bug a single continuous stream cannot have. The design makes that failure class impossible rather than merely avoided.
- **WSM + explicit cooldown + checkpoint merge.** Training runs at constant learning rate, then the rate is lowered for the last 30K steps to recover tail quality, then the last 10K checkpoints are averaged into the final model. (There's also a WSD variant that decays automatically over the last 20% of training; it's kept as a reference schedule, not what the final model used.)
- **Evaluation is a research problem, not a checkbox.** Autograder pass-rates cheat (a `return arr.sort()` passes a quicksort output test) and false-negative on tab/space mixing. From-scratch models don't read docstrings, so standard HumanEval prompts score 0/164 problems on every checkpoint. The honest metric at this scale is human-graded review of fixed docstring-free prompts across seeds, with a reference pass-rate computed alongside, which is what I use. Every generation is kept in the repo (`post-training/results/`, one JSON record per attempt, indexed by model and line range in `RAW-INDEX.md`) so the evidence stays inspectable.

## Post-training

The baseline above is a code *completion* model: it continues whatever it is given, and it does not read instructions. The second half of the project is supervised fine-tuning, about 30 runs on the same GPU, asking what turns it into a model that follows a request. Short version: what worked was training on instruction data whose reference solutions pass their own tests; the gold:signature ratio only traded between the two evaluation columns; random-seed variance (±6 points on identical settings) was larger than any recipe difference; weight averaging never beat its best parent; and the leader on the tuned-on benchmark finished **last** on 12 unseen tasks. (These are pass rates over repeated attempts — a different measure from the 0/1/2-graded benchmark above, and the two should not be compared.) The shipped model is the one that held up on the held-out set.

The implementation throughout (training harness, data pipeline, evaluation code) was developed with AI assistance. [`post-training/POST_TRAINING.md`](post-training/POST_TRAINING.md) describes how that work was organised, corrections included. Per-attempt generations: [`post-training/results/RAW-INDEX.md`](post-training/results/RAW-INDEX.md).

## Why this is here

Most "train your own LLM" material stops at a toy model with 10M parameters and no real evaluation. This one goes somewhere less comfortable: a real dataset, real training infrastructure, and a hard look at what a few-hundred-million-parameter model actually learns, including the parts that go wrong. It covers data contamination, silent training bugs and evaluation failures, with the raw generations kept in the repository for inspection, through to a model that writes working code.

Both runs trained on one machine: a single NVIDIA RTX 5090 (32 GB, Blackwell sm_120), July–September 2026.

## Start here

A code LLM trained from scratch on one RTX 5090: an archived 353M and a 435M rebuild trained on 31B tokens of filtered Python, then fine-tuned (about 30 runs).

1. [DESIGN.md](DESIGN.md): how the 435M was designed and what it scores (66/100 vs the 353M's 6/100, human-graded).
2. [ablation/README.md](ablation/README.md): why the data pipeline mattered, measured rather than asserted (a 14-seed human review: fibonacci recursion 14/14 before, 7/14 after training on repeated data).
3. [post-training/POST_TRAINING.md](post-training/POST_TRAINING.md): how the shipped model was picked (the tuning-set leader finished last on 12 unseen tasks: 55.8% vs the shipped 64.6%).

Check it yourself: open [ablation/README.md](ablation/README.md#cross-model-comparison-the-headline), the cross-model table (6/100 vs 66/100).

Limits: the headline benchmark is 10 tasks; tokenizer, context-length and learning-rate-schedule rows were closed without training runs.

Summary: [TLDR.md](TLDR.md).
