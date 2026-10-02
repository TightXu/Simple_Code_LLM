# Simple_Code_LLM — the 30-second version

A code LLM trained from scratch on one RTX 5090: an archived 353M first version and a 435M rebuild trained on 31B tokens of filtered Python, then fine-tuned. The 435M rebuild is what ships; the 353M is archived.

## What it shows
- Same architecture, same GPU, same 32K vocabulary: the archived 353M scores 6/100 on a 10-task human-graded benchmark; the 435M rebuild scores 66/100.
- The change was the data pipeline, not the architecture: 31B tokens of strictly filtered Python merged into one continuous stream, with no dataset boundaries or optimizer resets.
- Continuing the 353M on repeated data made it worse: a 14-seed review put fibonacci recursion at 14/14 before and 7/14 after.
- An intermediate 435M attempt failed outright: validation loss frozen at 1.97 across 21.5B tokens, no usable generation.
- Post-training was about 30 SFT runs; the tuning-set leader finished last on 12 unseen tasks, 55.8% against the shipped model's 64.6%.

## Where to start
- [README.md](README.md), section `## Start here`
- [DESIGN.md](DESIGN.md) and [post-training/POST_TRAINING.md](post-training/POST_TRAINING.md)

## What it does not claim
- The headline benchmark is 10 tasks x 5 seeds, human-graded; it says nothing beyond small algorithm problems.
- The tokenizer, context-length and learning-rate-schedule ablations were closed by reasoning, without training runs.
