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
| Full story | [`archive/353M/`](archive/353M/) | [`TECHNICAL.md`](TECHNICAL.md) |

A note on the loss numbers. The two sides use different LR schedules: the first run anneals with cosine (loss glides downward the whole run), while the clean rebuild runs WSM — a constant stable LR with a short cooldown at the end, because its final model is an average of the last checkpoints rather than a single late checkpoint. Constant-LR runs end with a naturally higher loss than cosine runs; that's an artifact of the schedule, not of model quality. The difference to pay attention to is the human-graded column: 6/100 vs 66/100, same architecture, same GPU, different data pipeline.

If you read nothing else, read why the scores differ. The two models are the same kind of creature; one outputs comments and unrelated code, the other outputs working implementations. The difference is where the effort went — data engineering, not architecture — and the ablation framework in `ablation/` is how that effort is measured rather than asserted.

## What's in this repo

```
├── README.md            # you are here
├── TECHNICAL.md         # technical report (the clean rebuild, with 353M as reference)
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
├── archive/353M/        # first version: README / TECHNICAL / CHALLENGES / scripts
├── results/             # loss curves + cross-model comparison figure
├── data/README.md       # dataset sources + pipeline reproduction
├── tokenizer/           # tokenizer_353m.json, tokenizer_435m.json
└── experiments/         # .gitignored — checkpoints / eval outputs (not in git)
```

## Key decisions that matter

- **One merged 62 GB `train.bin`, one continuous run.** The three datasets are concatenated into a single file and trained without resetting anything between them. No dataset boundaries, no optimizer resets mid-flight. This is because the first version's bug — resuming a run restarted the dataset from the top, so some data got trained twice and some never at all — is a class of bug a single continuous stream cannot have. It's not "we were careful so it didn't happen"; the design makes it impossible.
- **WSM + explicit cooldown + checkpoint merge.** Run training at constant learning rate, then lower the rate for the last 30K steps to recover tail quality, then average the last 10K checkpoints into the final model. (There's also a WSD variant that decays automatically over the last 20% of training; it's kept as a reference schedule, not what the final model used.)
- **Evaluation is a research problem, not a checkbox.** Autograder pass-rates cheat (a `return arr.sort()` passes a quicksort output test) and false-negative on tab/space mixing. From-scratch models don't read docstrings, so standard HumanEval prompts score 0/164 on every checkpoint. The honest metric at this scale is human-graded review of fixed docstring-free prompts across seeds, with a reference pass-rate computed alongside. Generations are kept in `results/` so the evidence stays inspectable.

## Why this is here

Most "train your own LLM" material stops at a toy model with 10M parameters and no real evaluation. This one goes somewhere less comfortable: a real dataset, real training infrastructure, and a hard look at what a few-hundred-million-parameter model actually learns — including the parts that go wrong. Data contamination, silent training bugs, evaluation that lies. If you want to understand what sits between `pip install transformers` and a model that writes working code, this is for you.

Both runs trained on a single NVIDIA RTX 5090 (32 GB, Blackwell sm_120), July–September 2026, and neither run ever left one machine.
