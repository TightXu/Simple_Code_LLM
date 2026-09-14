#!/usr/bin/env python3
"""Strip StarCoderData metadata prefixes (\<gh_stars\>/\<reponame\>/\<filename\>) from star_coder JSONL."""
import json, re
from pathlib import Path

META = re.compile(r"^(?:<(?:gh_stars|reponame|filename)>[^\n]*\n)+")


def strip_file(path):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    stripped = kept = 0
    with open(p, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            text = item.get("text", "")
            new = META.sub("", text)
            if new != text:
                stripped += 1
            if new:
                item["text"] = new
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
    tmp.replace(p)
    return stripped, kept


if __name__ == "__main__":
    base = Path(__file__).resolve().parent / "data" / "star_coder_data"
    for split in ("train", "val"):
        s, k = strip_file(base / (split + ".jsonl"))
        print("DONE", split, "stripped", s, "kept", k)
