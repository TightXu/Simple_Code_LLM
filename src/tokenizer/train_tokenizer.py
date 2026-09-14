#!/usr/bin/env python3
"""Sample uniformly from the three clean JSONL files and train a pure-Python 32K BPE tokenizer.
Note: memory limit ~4GB, keep the per-dataset sample at 80K rows."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "tokenizer" / "tokenizer.json"
PER_DS = 80_000  # uniform sample per dataset (4GB memory cap)


def count_lines(p):
    n = 0
    with open(p, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def sample(p, target):
    total = count_lines(p)
    step = max(1, total // target)
    texts = []
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % step == 0:
                try:
                    texts.append(json.loads(line)["text"])
                except Exception:
                    pass
    return texts


def main():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    datasets = ["codeparrot_clean", "the_stack_v1", "star_coder_data"]
    all_texts = []
    for ds in datasets:
        t = sample(DATA / ds / "train.jsonl", PER_DS)
        print("sample", ds, len(t))
        all_texts.extend(t)
    print("total", len(all_texts), "lines, training ...")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(all_texts, trainer)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT))
    print("DONE vocab", tokenizer.get_vocab_size(), "saved to", OUT)

    test = "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n"
    enc = tokenizer.encode(test)
    print("test tokens", len(enc.tokens), "chars/token", round(len(test) / len(enc.tokens), 1))


if __name__ == "__main__":
    main()
