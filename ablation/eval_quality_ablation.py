#!/usr/bin/env python3
"""
Data-quality ablation comparison: v2 filtered @1B vs nofilter @1B
================================================================
Matched comparison: same training tokens (1B), same val.bin, same architecture/hyperparameters.
Verdicts come from human review of the saved outputs (not script pass-rates).

Usage:
  py -3.12 ablation/eval_quality_ablation.py
"""
import sys
import json
import time
import gc
from pathlib import Path

import torch

ABLATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ABLATION_DIR))

from eval_cross_models import (  # noqa: E402
    ModelConfig, CodeLLM, load_tokenizer, load_model, generate_one,
    check_algo, check_humaneval, ALGORITHMS, SIMPLE_TASKS, load_humaneval,
)

PROJECT_ROOT = ABLATION_DIR.parent
HUMANEVAL_PATH = ABLATION_DIR / "HumanEval.jsonl"

MODELS = [
    ("filtered-v2-1B",
     PROJECT_ROOT / "checkpoints_wsm_v2_1B",
     PROJECT_ROOT / "tokenizer" / "tokenizer_435m.json"),
    ("nofilter-1B",
     PROJECT_ROOT / "checkpoints_wsm_nofilter" / "final",
     PROJECT_ROOT / "tokenizer" / "tokenizer_435m.json"),
]
SEEDS = [0, 42, 123, 999, 2024]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = dict(max_new=256, temperature=0.2, top_k=50,
               repetition_penalty=1.2, min_p=0.05, device=device)
    he_problems = load_humaneval(HUMANEVAL_PATH)
    print("[DEV] %s | HumanEval %d problems | seeds=%s" % (device, len(he_problems), SEEDS))

    all_samples = []
    summaries = []

    for name, ckpt_dir, tok_path in MODELS:
        print("\n" + "=" * 60)
        print("model: %s (%s)" % (name, ckpt_dir))
        print("=" * 60)
        t0 = time.time()
        tokenizer = load_tokenizer(tok_path)
        model = load_model(ckpt_dir, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print("  params: %.0fM" % n_params)

        task_pass = {}
        for algo in ALGORITHMS + SIMPLE_TASKS:
            p = 0
            for seed in SEEDS:
                comp = generate_one(model, tokenizer, algo["prompt"], seed, **gen)
                ok = check_algo(algo["prompt"], comp, algo["entry"], algo["tests"])
                p += (1 if ok else 0)
                all_samples.append({"model": name, "type": "task", "name": algo["name"],
                                    "seed": seed, "prompt": algo["prompt"],
                                    "completion": comp, "passed": ok})
            task_pass[algo["name"]] = "%d/%d" % (p, len(SEEDS))
            print("  %-16s %s/%d" % (algo["name"], p, len(SEEDS)))

        he_pass = 0
        for prob in he_problems:
            comp = generate_one(model, tokenizer, prob["prompt"], SEEDS[0], **gen)
            ok = check_humaneval(prob["prompt"], comp, prob["entry_point"], prob["test"])
            he_pass += (1 if ok else 0)
            all_samples.append({"model": name, "type": "humaneval",
                                "name": prob["task_id"], "seed": SEEDS[0],
                                "entry_point": prob["entry_point"],
                                "prompt": prob["prompt"],
                                "completion": comp, "passed": ok})
        print("  HumanEval pass@1: %d/%d (%.2f%%)" % (he_pass, len(he_problems),
                                                      100.0 * he_pass / len(he_problems)))

        summaries.append({"model": name, "params_m": round(n_params, 1),
                          "tasks": task_pass,
                          "humaneval": "%d/%d" % (he_pass, len(he_problems)),
                          "elapsed_s": round(time.time() - t0, 1)})
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(ABLATION_DIR / "eval_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    with open(ABLATION_DIR / "eval_quality_generations.jsonl", "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("\n[SAVE] eval_quality_summary.json + eval_quality_generations.jsonl (%d rows)" % len(all_samples))


if __name__ == "__main__":
    main()
