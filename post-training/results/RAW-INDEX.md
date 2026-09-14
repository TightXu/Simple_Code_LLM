# Raw output index (arm → file → line range)

generated: 2026-09-14 15:35 · script `fine-tuning/build_raw_index.py` (re-runnable, read-only)

**How to use**: when a report shows a score for some arm, find it in the table below → open the
matching file → `sed -n 'A,Bp'` to pull that arm's full raw generations (one JSON per line: `prompt` / `completion` / `passed`).
Line numbers are **1-based physical lines**, because one file may mix two checkpoints of the same arm (`final` / `bestval`), so the in-line `model` field wins.

## 5-run pass (11 tasks × 5 seeds)

| arm (in-line model field) | column | file | line range | n | passed | lenient | seeds / tasks |
|---|---|---|---|---|---|---|---|
| `A2-lr5e5-bestval` | signature column | `generations/eval_sft_generations_6models.jsonl` | 221–275 | 55 | 22 | — | 5 / 11 |
| `A3-big-bestval` | signature column | `generations/eval_sft_generations_6models.jsonl` | 166–220 | 55 | 28 | — | 5 / 11 |
| `A4-instr-bestval` | signature column | `generations/eval_sft_generations_6models.jsonl` | 276–330 | 55 | 23 | — | 5 / 11 |
| `A4-instr-final` | signature column | `generations/eval_sft_generations_a4final.jsonl` | 1–55 | 55 | 33 | — | 5 / 11 |
| `A5-bigmix-bestval` | NL column | `generations/generations_a5_nl.jsonl` | 56–110 | 55 | 6 | — | 5 / 11 |
| `A5-bigmix-bestval` | signature column | `generations/generations_a5_sig.jsonl` | 56–110 | 55 | 24 | — | 5 / 11 |
| `A5-bigmix-final` | NL column | `generations/generations_a5_nl.jsonl` | 1–55 | 55 | 27 | — | 5 / 11 |
| `A5-bigmix-final` | signature column | `generations/generations_a5_sig.jsonl` | 1–55 | 55 | 33 | — | 5 / 11 |
| `A6-2ep-bestval` | NL column | `generations/generations_a6_nl.jsonl` | 56–110 | 55 | 5 | — | 5 / 11 |
| `A6-2ep-bestval` | signature column | `generations/generations_a6_sig.jsonl` | 56–110 | 55 | 21 | — | 5 / 11 |
| `A6-2ep-final` | NL column | `generations/generations_a6_nl.jsonl` | 1–55 | 55 | 25 | — | 5 / 11 |
| `A6-2ep-final` | signature column | `generations/generations_a6_sig.jsonl` | 1–55 | 55 | 27 | — | 5 / 11 |
| `A7-bigseed-bestval` | NL column | `generations/generations_a7_nl.jsonl` | 56–110 | 55 | 34 | — | 5 / 11 |
| `A7-bigseed-bestval` | signature column | `generations/generations_a7_sig.jsonl` | 56–110 | 55 | 25 | — | 5 / 11 |
| `A7-bigseed-final` | NL column | `generations/generations_a7_nl.jsonl` | 1–55 | 55 | 28 | — | 5 / 11 |
| `A7-bigseed-final` | signature column | `generations/generations_a7_sig.jsonl` | 1–55 | 55 | 36 | — | 5 / 11 |
| `A8-2to1-bestval` | NL column | `generations/generations_a8_nl.jsonl` | 56–110 | 55 | 39 | — | 5 / 11 |
| `A8-2to1-bestval` | signature column | `generations/generations_a8_sig.jsonl` | 56–110 | 55 | 22 | — | 5 / 11 |
| `A8-2to1-final` | NL column | `generations/generations_a8_nl.jsonl` | 1–55 | 55 | 31 | — | 5 / 11 |
| `A8-2to1-final` | signature column | `generations/generations_a8_sig.jsonl` | 1–55 | 55 | 27 | — | 5 / 11 |
| `A9-1to1-bestval` | NL column | `generations/generations_a9_nl.jsonl` | 56–110 | 55 | 35 | — | 5 / 11 |
| `A9-1to1-bestval` | signature column | `generations/generations_a9_sig.jsonl` | 56–110 | 55 | 31 | — | 5 / 11 |
| `A9-1to1-final` | NL column | `generations/generations_a9_nl.jsonl` | 1–55 | 55 | 24 | — | 5 / 11 |
| `A9-1to1-final` | signature column | `generations/generations_a9_sig.jsonl` | 1–55 | 55 | 19 | — | 5 / 11 |
| `BASE-best_val` | signature column | `generations/eval_cross_generations.jsonl` | 56–110 | 55 | 6 | — | 5 / 11 |
| `BASE-final` | signature column | `generations/eval_cross_generations.jsonl` | 1–55 | 55 | 5 | — | 5 / 11 |
| `BASE-merged` | signature column | `generations/eval_cross_generations.jsonl` | 111–165 | 55 | 4 | — | 5 / 11 |
| `BASE-merged` | signature column | `generations/eval_sft_generations_6models.jsonl` | 1–55 | 55 | 4 | — | 5 / 11 |
| `G1-gold-bestval` | NL column | `generations/generations_g1_nl.jsonl` | 56–110 | 55 | 43 | — | 5 / 11 |
| `G1-gold-bestval` | signature column | `generations/generations_g1_sig.jsonl` | 56–110 | 55 | 13 | — | 5 / 11 |
| `G1-gold-final` | NL column | `generations/generations_g1_nl.jsonl` | 1–55 | 55 | 41 | — | 5 / 11 |
| `G1-gold-final` | signature column | `generations/generations_g1_sig.jsonl` | 1–55 | 55 | 11 | — | 5 / 11 |
| `G10-g8to1-bestval` | NL column | `generations/generations_g10_nl.jsonl` | 56–110 | 55 | 46 | — | 5 / 11 |
| `G10-g8to1-bestval` | signature column | `generations/generations_g10_sig.jsonl` | 56–110 | 55 | 26 | — | 5 / 11 |
| `G10-g8to1-final` | NL column | `generations/generations_g10_nl.jsonl` | 1–55 | 55 | 47 | — | 5 / 11 |
| `G10-g8to1-final` | signature column | `generations/generations_g10_sig.jsonl` | 1–55 | 55 | 31 | — | 5 / 11 |
| `G2-goldsig-bestval` | NL column | `generations/generations_g2_nl.jsonl` | 56–110 | 55 | 32 | — | 5 / 11 |
| `G2-goldsig-bestval` | signature column | `generations/generations_g2_sig.jsonl` | 56–110 | 55 | 8 | — | 5 / 11 |
| `G2-goldsig-final` | NL column | `generations/generations_g2_nl.jsonl` | 1–55 | 55 | 35 | — | 5 / 11 |
| `G2-goldsig-final` | signature column | `generations/generations_g2_sig.jsonl` | 1–55 | 55 | 22 | — | 5 / 11 |
| `G3-g11-bestval` | NL column | `generations/generations_g3_nl.jsonl` | 56–110 | 55 | 40 | — | 5 / 11 |
| `G3-g11-bestval` | signature column | `generations/generations_g3_sig.jsonl` | 56–110 | 55 | 32 | — | 5 / 11 |
| `G3-g11-bestval` | NL column | `generations/generations_g6_nl.jsonl` | 56–110 | 55 | 40 | — | 5 / 11 |
| `G3-g11-bestval` | signature column | `generations/generations_g6_sig.jsonl` | 56–110 | 55 | 32 | — | 5 / 11 |
| `G3-g11-final` | NL column | `generations/generations_g3_nl.jsonl` | 1–55 | 55 | 32 | — | 5 / 11 |
| `G3-g11-final` | signature column | `generations/generations_g3_sig.jsonl` | 1–55 | 55 | 31 | — | 5 / 11 |
| `G3-g11-final` | NL column | `generations/generations_g6_nl.jsonl` | 1–55 | 55 | 32 | — | 5 / 11 |
| `G3-g11-final` | signature column | `generations/generations_g6_sig.jsonl` | 1–55 | 55 | 31 | — | 5 / 11 |
| `G4-2stage-bestval` | NL column | `generations/generations_g4_nl.jsonl` | 56–110 | 55 | 33 | — | 5 / 11 |
| `G4-2stage-bestval` | signature column | `generations/generations_g4_sig.jsonl` | 56–110 | 55 | 19 | — | 5 / 11 |
| `G4-2stage-final` | NL column | `generations/generations_g4_nl.jsonl` | 1–55 | 55 | 32 | — | 5 / 11 |
| `G4-2stage-final` | signature column | `generations/generations_g4_sig.jsonl` | 1–55 | 55 | 18 | — | 5 / 11 |
| `G5-g15-bestval` | NL column | `generations/generations_g5_nl.jsonl` | 56–110 | 55 | 31 | — | 5 / 11 |
| `G5-g15-bestval` | signature column | `generations/generations_g5_sig.jsonl` | 56–110 | 55 | 30 | — | 5 / 11 |
| `G5-g15-final` | NL column | `generations/generations_g5_nl.jsonl` | 1–55 | 55 | 43 | — | 5 / 11 |
| `G5-g15-final` | signature column | `generations/generations_g5_sig.jsonl` | 1–55 | 55 | 36 | — | 5 / 11 |
| `G5-g15-s43-bestval` | NL column | `generations/generations_g5s43_nl.jsonl` | 56–110 | 55 | 34 | — | 5 / 11 |
| `G5-g15-s43-bestval` | signature column | `generations/generations_g5s43_sig.jsonl` | 56–110 | 55 | 28 | — | 5 / 11 |
| `G5-g15-s43-final` | NL column | `generations/generations_g5s43_nl.jsonl` | 1–55 | 55 | 41 | — | 5 / 11 |
| `G5-g15-s43-final` | signature column | `generations/generations_g5s43_sig.jsonl` | 1–55 | 55 | 36 | — | 5 / 11 |
| `G5-g15-s44-bestval` | NL column | `generations/generations_g5s44_nl.jsonl` | 56–110 | 55 | 35 | — | 5 / 11 |
| `G5-g15-s44-bestval` | signature column | `generations/generations_g5s44_sig.jsonl` | 56–110 | 55 | 26 | — | 5 / 11 |
| `G5-g15-s44-final` | NL column | `generations/generations_g5s44_nl.jsonl` | 1–55 | 55 | 42 | — | 5 / 11 |
| `G5-g15-s44-final` | signature column | `generations/generations_g5s44_sig.jsonl` | 1–55 | 55 | 37 | — | 5 / 11 |
| `G7-g2to1-bestval` | NL column | `generations/generations_g7_nl.jsonl` | 56–110 | 55 | 43 | — | 5 / 11 |
| `G7-g2to1-bestval` | signature column | `generations/generations_g7_sig.jsonl` | 56–110 | 55 | 33 | — | 5 / 11 |
| `G7-g2to1-final` | NL column | `generations/generations_g7_nl.jsonl` | 1–55 | 55 | 36 | — | 5 / 11 |
| `G7-g2to1-final` | signature column | `generations/generations_g7_sig.jsonl` | 1–55 | 55 | 38 | — | 5 / 11 |
| `G8-g3to1-bestval` | NL column | `generations/generations_g8_nl.jsonl` | 56–110 | 55 | 47 | — | 5 / 11 |
| `G8-g3to1-bestval` | signature column | `generations/generations_g8_sig.jsonl` | 56–110 | 55 | 34 | — | 5 / 11 |
| `G8-g3to1-final` | NL column | `generations/generations_g8_nl.jsonl` | 1–55 | 55 | 45 | — | 5 / 11 |
| `G8-g3to1-final` | signature column | `generations/generations_g8_sig.jsonl` | 1–55 | 55 | 30 | — | 5 / 11 |
| `G8-g3to1-s43-bestval` | NL column | `generations/generations_g8s43_nl.jsonl` | 56–110 | 55 | 45 | — | 5 / 11 |
| `G8-g3to1-s43-bestval` | signature column | `generations/generations_g8s43_sig.jsonl` | 56–110 | 55 | 30 | — | 5 / 11 |
| `G8-g3to1-s43-final` | NL column | `generations/generations_g8s43_nl.jsonl` | 1–55 | 55 | 42 | — | 5 / 11 |
| `G8-g3to1-s43-final` | signature column | `generations/generations_g8s43_sig.jsonl` | 1–55 | 55 | 29 | — | 5 / 11 |
| `G8-g3to1-s44-bestval` | NL column | `generations/generations_g8s44_nl.jsonl` | 56–110 | 55 | 44 | — | 5 / 11 |
| `G8-g3to1-s44-bestval` | signature column | `generations/generations_g8s44_sig.jsonl` | 56–110 | 55 | 29 | — | 5 / 11 |
| `G8-g3to1-s44-final` | NL column | `generations/generations_g8s44_nl.jsonl` | 1–55 | 55 | 45 | — | 5 / 11 |
| `G8-g3to1-s44-final` | signature column | `generations/generations_g8s44_sig.jsonl` | 1–55 | 55 | 30 | — | 5 / 11 |
| `G9-g5to1-bestval` | NL column | `generations/generations_g9_nl.jsonl` | 56–110 | 55 | 50 | — | 5 / 11 |
| `G9-g5to1-bestval` | signature column | `generations/generations_g9_sig.jsonl` | 56–110 | 55 | 28 | — | 5 / 11 |
| `G9-g5to1-final` | NL column | `generations/generations_g9_nl.jsonl` | 1–55 | 55 | 35 | — | 5 / 11 |
| `G9-g5to1-final` | signature column | `generations/generations_g9_sig.jsonl` | 1–55 | 55 | 28 | — | 5 / 11 |
| `G9-g5to1-s43-bestval` | NL column | `generations/generations_g9s43_nl.jsonl` | 56–110 | 55 | 44 | — | 5 / 11 |
| `G9-g5to1-s43-bestval` | signature column | `generations/generations_g9s43_sig.jsonl` | 56–110 | 55 | 27 | — | 5 / 11 |
| `G9-g5to1-s43-final` | NL column | `generations/generations_g9s43_nl.jsonl` | 1–55 | 55 | 38 | — | 5 / 11 |
| `G9-g5to1-s43-final` | signature column | `generations/generations_g9s43_sig.jsonl` | 1–55 | 55 | 29 | — | 5 / 11 |
| `G9-g5to1-s44-bestval` | NL column | `generations/generations_g9s44_nl.jsonl` | 56–110 | 55 | 48 | — | 5 / 11 |
| `G9-g5to1-s44-bestval` | signature column | `generations/generations_g9s44_sig.jsonl` | 56–110 | 55 | 31 | — | 5 / 11 |
| `G9-g5to1-s44-final` | NL column | `generations/generations_g9s44_nl.jsonl` | 1–55 | 55 | 34 | — | 5 / 11 |
| `G9-g5to1-s44-final` | signature column | `generations/generations_g9s44_sig.jsonl` | 1–55 | 55 | 28 | — | 5 / 11 |
| `SFT-bestval` | signature column | `generations/eval_sft_generations_6models.jsonl` | 111–165 | 55 | 24 | — | 5 / 11 |
| `SFT-final` | signature column | `generations/eval_sft_generations_6models.jsonl` | 56–110 | 55 | 23 | — | 5 / 11 |
| `SOUP-A7G8-020` | NL column | `generations/generations_soup_nl.jsonl` | 1–55 | 55 | 36 | — | 5 / 11 |
| `SOUP-A7G8-020` | signature column | `generations/generations_soup_sig.jsonl` | 1–55 | 55 | 36 | — | 5 / 11 |
| `SOUP-A7G8-035` | NL column | `generations/generations_soup_nl.jsonl` | 56–110 | 55 | 36 | — | 5 / 11 |
| `SOUP-A7G8-035` | signature column | `generations/generations_soup_sig.jsonl` | 331–385 | 55 | 31 | — | 5 / 11 |
| `SOUP-A7G8-050` | NL column | `generations/generations_soup_nl.jsonl` | 111–165 | 55 | 38 | — | 5 / 11 |
| `SOUP-A7G8-050` | signature column | `generations/generations_soup_sig.jsonl` | 386–440 | 55 | 29 | — | 5 / 11 |
| `SOUP-A7G8-065` | NL column | `generations/generations_soup_nl.jsonl` | 166–220 | 55 | 35 | — | 5 / 11 |
| `SOUP-A7G8-065` | signature column | `generations/generations_soup_sig.jsonl` | 441–495 | 55 | 33 | — | 5 / 11 |
| `SOUP-A7G8-080` | NL column | `generations/generations_soup_nl.jsonl` | 221–275 | 55 | 39 | — | 5 / 11 |
| `SOUP-A7G8-080` | signature column | `generations/generations_soup_sig.jsonl` | 496–550 | 55 | 35 | — | 5 / 11 |
| `SOUP-G10G5-050` | NL column | `generations/generations_soup_nl.jsonl` | 606–660 | 55 | 42 | — | 5 / 11 |
| `SOUP-G10G5-050` | signature column | `generations/generations_soup_sig.jsonl` | 276–330 | 55 | 33 | — | 5 / 11 |
| `SOUP-G5G8-050` | NL column | `generations/generations_soup_nl.jsonl` | 276–330 | 55 | 43 | — | 5 / 11 |
| `SOUP-G5G8-050` | signature column | `generations/generations_soup_sig.jsonl` | 551–605 | 55 | 36 | — | 5 / 11 |
| `SOUP-G8self-050` | NL column | `generations/generations_soup_nl.jsonl` | 331–385 | 55 | 44 | — | 5 / 11 |
| `SOUP-G8self-050` | signature column | `generations/generations_soup_sig.jsonl` | 606–660 | 55 | 32 | — | 5 / 11 |
| `SOUP-G8x3-bestval` | NL column | `generations/generations_g8x3_nl.jsonl` | 1–55 | 55 | 47 | — | 5 / 11 |
| `SOUP-G8x3-bestval` | signature column | `generations/generations_g8x3_sig.jsonl` | 1–55 | 55 | 31 | — | 5 / 11 |
| `SOUP-G8x3-final` | NL column | `generations/generations_g8x3_nl.jsonl` | 56–110 | 55 | 43 | — | 5 / 11 |
| `SOUP-G8x3-final` | signature column | `generations/generations_g8x3_sig.jsonl` | 56–110 | 55 | 31 | — | 5 / 11 |
| `SOUP-G9G5-030` | NL column | `generations/generations_soup_nl.jsonl` | 386–440 | 55 | 48 | — | 5 / 11 |
| `SOUP-G9G5-030` | signature column | `generations/generations_soup_sig.jsonl` | 56–110 | 55 | 32 | — | 5 / 11 |
| `SOUP-G9G5-050` | NL column | `generations/generations_soup_nl.jsonl` | 441–495 | 55 | 47 | — | 5 / 11 |
| `SOUP-G9G5-050` | signature column | `generations/generations_soup_sig.jsonl` | 111–165 | 55 | 31 | — | 5 / 11 |
| `SOUP-G9G5-070` | NL column | `generations/generations_soup_nl.jsonl` | 496–550 | 55 | 45 | — | 5 / 11 |
| `SOUP-G9G5-070` | signature column | `generations/generations_soup_sig.jsonl` | 166–220 | 55 | 35 | — | 5 / 11 |
| `SOUP-G9G8-050` | NL column | `generations/generations_soup_nl.jsonl` | 551–605 | 55 | 47 | — | 5 / 11 |
| `SOUP-G9G8-050` | signature column | `generations/generations_soup_sig.jsonl` | 221–275 | 55 | 33 | — | 5 / 11 |
| `SOUP-G9x5-bestval` | signature column | `generations/eval_sft_generations.jsonl` | 1–220 | 220 | 129 | — | 20 / 11 |
| `SOUP-G9x5-bestval` | NL column | `generations/nl_eval_generations.jsonl` | 1–220 | 220 | 186 | — | 20 / 11 |

## 20-run precision pass (11 tasks × 20 seeds, isolated run)

| arm (in-line model field) | column | file | line range | n | passed | lenient | seeds / tasks |
|---|---|---|---|---|---|---|---|
| `G8-g3to1-bestval` | NL column | `generations/generations_noise20_g8bv_nl.jsonl` | 1–220 | 220 | 175 | — | 20 / 11 |
| `G8-g3to1-bestval` | signature column | `generations/generations_noise20_g8bv_sig.jsonl` | 1–220 | 220 | 129 | — | 20 / 11 |
| `G8-g3to1-s43-bestval` | NL column | `generations/generations_noise20_g8s43_nl.jsonl` | 1–220 | 220 | 169 | — | 20 / 11 |
| `G8-g3to1-s43-bestval` | signature column | `generations/generations_noise20_g8s43_sig.jsonl` | 1–220 | 220 | 119 | — | 20 / 11 |
| `G8-g3to1-s44-bestval` | NL column | `generations/generations_noise20_g8s44_nl.jsonl` | 1–220 | 220 | 174 | — | 20 / 11 |
| `G8-g3to1-s44-bestval` | signature column | `generations/generations_noise20_g8s44_sig.jsonl` | 1–220 | 220 | 109 | — | 20 / 11 |
| `G9-g5to1-bestval` | NL column | `generations/generations_noise20_g9bv_nl.jsonl` | 1–220 | 220 | 201 | — | 20 / 11 |
| `G9-g5to1-bestval` | signature column | `generations/generations_noise20_g9bv_sig.jsonl` | 1–220 | 220 | 120 | — | 20 / 11 |
| `G9-g5to1-s43-bestval` | NL column | `generations/generations_noise20_g9s43_nl.jsonl` | 1–220 | 220 | 178 | — | 20 / 11 |
| `G9-g5to1-s43-bestval` | signature column | `generations/generations_noise20_g9s43_sig.jsonl` | 1–220 | 220 | 110 | — | 20 / 11 |
| `G9-g5to1-s44-bestval` | NL column | `generations/generations_noise20_g9s44_nl.jsonl` | 1–220 | 220 | 193 | — | 20 / 11 |
| `G9-g5to1-s44-bestval` | signature column | `generations/generations_noise20_g9s44_sig.jsonl` | 1–220 | 220 | 131 | — | 20 / 11 |
| `G9-g5to1-s45-bestval` | NL column | `generations/generations_noise20_g9s45_nl.jsonl` | 1–220 | 220 | 197 | — | 20 / 11 |
| `G9-g5to1-s45-bestval` | signature column | `generations/generations_noise20_g9s45_sig.jsonl` | 1–220 | 220 | 162 | — | 20 / 11 |
| `G9-g5to1-s46-bestval` | NL column | `generations/generations_noise20_g9s46_nl.jsonl` | 1–220 | 220 | 199 | — | 20 / 11 |
| `G9-g5to1-s46-bestval` | signature column | `generations/generations_noise20_g9s46_sig.jsonl` | 1–220 | 220 | 130 | — | 20 / 11 |
| `SOUP-G5x3-final` | NL column | `generations/generations_noise20_g5x3fi_nl.jsonl` | 1–220 | 220 | 163 | — | 20 / 11 |
| `SOUP-G5x3-final` | signature column | `generations/generations_noise20_g5x3fi_sig.jsonl` | 1–220 | 220 | 139 | — | 20 / 11 |
| `SOUP-G8x3-bestval` | NL column | `generations/generations_noise20_soup3_nl.jsonl` | 1–220 | 220 | 196 | — | 20 / 11 |
| `SOUP-G8x3-bestval` | signature column | `generations/generations_noise20_soup3_sig.jsonl` | 1–220 | 220 | 123 | — | 20 / 11 |
| `SOUP-G9G5-030` | NL column | `generations/generations_noise20_g9g5_030_nl.jsonl` | 1–220 | 220 | 188 | — | 20 / 11 |
| `SOUP-G9G5-030` | signature column | `generations/generations_noise20_g9g5_030_sig.jsonl` | 1–220 | 220 | 129 | — | 20 / 11 |
| `SOUP-G9top2-bestval` | NL column | `generations/generations_noise20_g9top2_nl.jsonl` | 1–220 | 220 | 191 | — | 20 / 11 |
| `SOUP-G9top2-bestval` | signature column | `generations/generations_noise20_g9top2_sig.jsonl` | 1–220 | 220 | 130 | — | 20 / 11 |
| `SOUP-G9x3-bestval` | NL column | `generations/generations_noise20_g9x3bv_nl.jsonl` | 1–220 | 220 | 195 | — | 20 / 11 |
| `SOUP-G9x3-bestval` | signature column | `generations/generations_noise20_g9x3bv_sig.jsonl` | 1–220 | 220 | 121 | — | 20 / 11 |
| `SOUP-G9x5-bestval` | NL column | `generations/generations_noise20_g9x5_nl.jsonl` | 1–220 | 220 | 186 | — | 20 / 11 |
| `SOUP-G9x5-bestval` | signature column | `generations/generations_noise20_g9x5_sig.jsonl` | 1–220 | 220 | 129 | — | 20 / 11 |

## held-out set (12 new tasks × 10 seeds)

| arm (in-line model field) | column | file | line range | n | passed | lenient | seeds / tasks |
|---|---|---|---|---|---|---|---|
| `G8-g3to1-bestval` | NL column | `generations/heldout_generations.jsonl` | 1561–1680 | 120 | 91 | 91 | 10 / 12 |
| `G8-g3to1-bestval` | signature column | `generations/heldout_generations.jsonl` | 1441–1560 | 120 | 58 | 68 | 10 / 12 |
| `G9-g5to1-bestval` | NL column | `generations/heldout_generations.jsonl` | 841–960 | 120 | 97 | 101 | 10 / 12 |
| `G9-g5to1-bestval` | signature column | `generations/heldout_generations.jsonl` | 721–840 | 120 | 56 | 66 | 10 / 12 |
| `G9-g5to1-s44-bestval` | NL column | `generations/heldout_generations.jsonl` | 601–720 | 120 | 91 | 91 | 10 / 12 |
| `G9-g5to1-s44-bestval` | signature column | `generations/heldout_generations.jsonl` | 481–600 | 120 | 52 | 63 | 10 / 12 |
| `G9-g5to1-s45-bestval` | NL column | `generations/heldout_generations.jsonl` | 121–240 | 120 | 80 | 80 | 10 / 12 |
| `G9-g5to1-s45-bestval` | signature column | `generations/heldout_generations.jsonl` | 1–120 | 120 | 54 | 64 | 10 / 12 |
| `G9-g5to1-s46-bestval` | NL column | `generations/heldout_generations.jsonl` | 361–480 | 120 | 101 | 101 | 10 / 12 |
| `G9-g5to1-s46-bestval` | signature column | `generations/heldout_generations.jsonl` | 241–360 | 120 | 54 | 66 | 10 / 12 |
| `SOUP-G8x3-bestval` | NL column | `generations/heldout_generations.jsonl` | 1321–1440 | 120 | 91 | 91 | 10 / 12 |
| `SOUP-G8x3-bestval` | signature column | `generations/heldout_generations.jsonl` | 1201–1320 | 120 | 60 | 70 | 10 / 12 |
| `SOUP-G9top2-bestval` | NL column | `generations/heldout_generations.jsonl` | 1081–1200 | 120 | 93 | 93 | 10 / 12 |
| `SOUP-G9top2-bestval` | signature column | `generations/heldout_generations.jsonl` | 961–1080 | 120 | 55 | 66 | 10 / 12 |

## Lenient pass (re-judged to allow the "sort in place" convention)

The strict column requires the function to return a result, so a correct implementation that mutates the list in place and returns None is scored as failed.
The lenient column re-judges **failed samples** only (when the sole argument is a list, in-place mutation also passes); the input is the same batch of generations files.

| file | contents |
|---|---|
| `eval/lenient_results.json` | strict/lenient score per model (55 points per column) + source file name, produced by `rejudge_lenient.py` (CPU-only, re-runnable) |
| the "lenient" column in the held-out section table | strict and lenient are both judged at generation time (field `passed_lenient`) |

## Summary file reference

| file | contents |
|---|---|
| `eval_sft_summary.json` | signature-column summary (per-model per-task `x/5`) |
| `nl_eval_summary.json` | NL-column summary |
| `heldout_summary.json` / `heldout_results.json` | held-out set: strict/lenient for both columns, per model |
| `noise20_*_{sig,nl}.json` | per-model summary of the 20-run precision pass |
| `soup_summary.md` | weight soup candidate table |

166 records (arm × file × column) covering 69 distinct model names.
