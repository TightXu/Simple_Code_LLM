# Post-training audit results (2026-10)

Raw per-item human scores for the supervised fine-tuning work described in
[`DESIGN.md` §7.4 and §12](../../DESIGN.md). Every `.jsonl` row is one generated
sample, scored **0 / 1 / 2** by hand with a one-line reason:

- `2` = a correct implementation
- `1` = the idea is right but a detail is wrong (base case inverted, off-by-one,
  `None` vs `-1`, pivot dropped, a required import missing, function renamed)
- `0` = wrong, gibberish, comments only, or a docstring with no body

Script pass/fail is recorded in `script_passed` but **is not** the metric. In this
data set the test harness under-scores correct answers heavily (41 of 66 correct
signature-column answers on one 1B arm), so every number below comes from the
hand scores.

## Scale

Each checkpoint is evaluated on **11 tasks × 5 seeds**; with the 0/1/2 rubric that
is **110 points per column** (signature / instruction) per variant. Files ending
`-A` / `-B` are the two seed shards (`0,42,123` and `999,2024`) merged by
`(model, task, seed)`.

## 1B (1,096M parameters)

| Arm (all on the `merged` base, training seed 42) | Signature /110 | Instruction /110 |
|---|---|---|
| base, no SFT | 61 | 83 |
| gold-only instruction data, 62.4M tokens | 65 / 67 | 95 / 97 |
| **5:1 gold:signature mix, 91.8M tokens** | **85 / 91** | **98 / 97** |
| gold-only, 91.8M tokens (equal-token control) | 70 / 68 | 94 / 103 |

| Base checkpoint (no SFT) | Signature /110 | Instruction /110 |
|---|---|---|
| merged | 61 | 83 |
| final | 46 | 76 |
| best_val | 41 | 78 |

Attribution (equal-token control, both arms 91.8M tokens, same base, same seed):

- composition effect (**mix − gold** at equal tokens): **signature +38 of 220**
  (−17.3 points for gold; 10/10 variant-seed cells negative), instruction +2 (n.s.)
- data-volume effect (**gold ×1.47 − gold ×1**): signature +6 of 220 (inside seed
  noise), instruction +5 (n.s.)

## 435M (434.7M parameters)

| Arm (`G9` = the 5:1 mix that shipped; 3 bases × {final, best_val}) | Signature /110 | Instruction /110 |
|---|---|---|
| merged-final / merged-best_val | 83 / 81 | 87 / 87 |
| final-final / final-best_val | 88 / 83 | 87 / 100 |
| best_val-final / best_val-best_val | 82 / 79 | 87 / 92 |
| gold-only arm (G1, 62.4M tokens) | 69 / 67 | 91 / 93 |

| Base checkpoint (no SFT) | Signature /110 | Instruction /110 |
|---|---|---|
| merged | 57 | 65 |
| final | 40 | 38 |
| best_val | 18 | 66 |

Including the earlier 435M base audits, this directory holds **2,475 hand-scored
rows** across both models, both columns and all five seeds.

## What the numbers say

1. SFT is a net gain at both scales, and the gain is stable across seeds on the
   instruction column (6/6 arms × 5/5 seeds non-negative).
2. The choice of base checkpoint stops mattering after SFT. 435M's signature
   column separates its three bases by 39 points before SFT and its six arms by
   9 points after.
3. Per-seed ordering of the bases is not stable on the 1B: `merged > final >
   best_val` holds on 2 of 5 seeds. A difference smaller than the per-seed spread
   is not a result — see DESIGN.md §8.3.
4. The signature-column improvement comes from **data composition**, not from
   training longer. The mechanism (docstring-first output colliding with
   continuation prompts) is written up in DESIGN.md §7.4.

## Caveats

- These are hand scores by one reviewer against a published rubric, not
  inter-annotator agreement; the ambiguous cases are flagged `uncertain` in the
  files (typically 0–26 per shard).
- `best_val` is chosen on the SFT validation loss, which is **not** comparable
  across data pools (each pool has its own validation split).
- The 435M arms were re-evaluated with 5 seeds in 2026-10; the **shipped**
  checkpoint (`G9` at training seed 46) was not part of that re-audit.
