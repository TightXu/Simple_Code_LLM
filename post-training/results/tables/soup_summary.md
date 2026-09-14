# Weight-soup candidates (5-run protocol, 11 tasks × 5 seeds per column)

Produced by `scripts/soup_sweep.py`; each row is a weight interpolation of two trained arms, evaluated with the same protocol as the single arms.
Reference rows at the bottom are the interpolated models' parents.

| model | NL | signature | total (/110) |
|---|---|---|---|
| `SOUP-A7G8-020` | 36 | 36 | 72 |
| `SOUP-A7G8-035` | 36 | 31 | 67 |
| `SOUP-A7G8-050` | 38 | 29 | 67 |
| `SOUP-A7G8-065` | 35 | 33 | 68 |
| `SOUP-A7G8-080` | 39 | 35 | 74 |
| `SOUP-G10G5-050` | 42 | 33 | 75 |
| `SOUP-G5G8-050` | 43 | 36 | 79 |
| `SOUP-G8self-050` | 44 | 32 | 76 |
| `SOUP-G8x3-bestval` | 47 | 31 | 78 |
| `SOUP-G8x3-final` | 43 | 31 | 74 |
| `SOUP-G9G5-030` | 48 | 32 | 80 |
| `SOUP-G9G5-050` | 47 | 31 | 78 |
| `SOUP-G9G5-070` | 45 | 35 | 80 |
| `SOUP-G9G8-050` | 47 | 33 | 80 |

| reference arm | NL | signature | total (/110) |
|---|---|---|---|
| `G8-g3to1-bestval` | 47 | 34 | 81 |
| `G9-g5to1-bestval` | 50 | 28 | 78 |
| `G5-g15-final` | 43 | 36 | 79 |
| `G10-g8to1-final` | 47 | 31 | 78 |
| `G7-g2to1-final` | 36 | 38 | 74 |
| `A7-bigseed-final` | 28 | 36 | 64 |
| `A9-1to1-bestval` | 35 | 31 | 66 |
