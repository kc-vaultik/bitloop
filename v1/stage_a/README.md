# Stage A — exploratory results (2026-11-20/21)

Everything here is one seed on the validation split. It supports no confirmatory claim; its job is to shape Stage B.

## 6x6 Sudoku, v0 code, ARM CPU (exact accuracy at K = 4)

| Arm | Width | Best lr | 1,500 steps | 6,000 steps |
| --- | --- | --- | --- | --- |
| fp | 64 | 8e-3 | 93.4 | 98.8 |
| tern | 64 | 2e-2 | 59.0 | 70.5 |
| tern | 128 | 8e-3 / 2e-2 | 76.6 | 90.3 |
| tern | 256 | 2e-2 | 81.9 | - |
| spike | 64 | 8e-3 | 23.9 | - |
| spike | 128 | 2e-2 | 31.0 | - |

(Per-run JSON for this sweep is still on the server; only the summary above has been retrieved.)

## 9x9 Sudoku-Extreme, rating-0 puzzles, width 192, 8,000 steps (Kaggle 2 x T4, batch 256, 548k puzzles)

Per-run JSON and logs: `kaggle_session1/`.

| Arm | lr | K = 1 | K = 2 | K = 4 | K = 16 |
| --- | --- | --- | --- | --- | --- |
| fp | 3e-3 | 15.5 | 31.6 | 71.6 | 72.7 |
| fp | 1e-2 | - | - | 65.3 | 66.7 |
| tern | 3e-3 / 1e-2 / 3e-2 | 15.5 | 15.5 | 15.5 | 15.5 |
| spike | 3e-3 | 13.7 | 14.7 | 14.9 | 14.9 |

Rating 1-9 puzzles: fp 0.3 %, tern 0.0 % at this budget (floor).

## Observation that motivates Stage A2

The ternary core does not refine: its accuracy is identical at every recursion depth and at learning rates a decade
apart, equal to what full precision reaches after a single step. Stage A2 changes one thing at a time against the
all-ternary core (full-precision cell-mixing or channel layers, a learned per-output gain, separate latent and answer
networks, depth, width) to find which part of the loop needs precision.

## Stage A2 — first results (laptop GTX 1650 Ti, batch 64, 22,649 rating-0 puzzles, width 192, 8,000 steps)

Per-run JSON and logs: `laptop_a2/` (mixed-precision arms) and the table below for the two reference arms from the
earlier laptop pilot. Exact accuracy (%) on 2,000 validation puzzles by number of recursion steps at test time.

| Arm | Full-precision share of core weights | K = 1 | K = 2 | K = 4 | K = 8 | K = 16 |
| --- | --- | --- | --- | --- | --- | --- |
| fp (reference) | 100 % | 16.0 | 21.9 | 41.6 | 42.55 | 42.6 |
| tern (reference) | 0 % | 15.9 | 16.0 | 16.0 | 16.0 | 16.0 |
| tern, full-precision cell-mixing layers | 15 % | 16.05 | 16.05 | 16.05 | 16.05 | 16.05 |
| tern, full-precision channel layers | 85 % | 16.0 | 17.55 | 20.1 | 20.4 | 20.4 |

One seed each. Neither part restores refinement on its own: with 85 % of the core weights in full precision and only
the cell-mixing layers ternary, recursion adds 4.4 points where the full-precision model adds 26.6. The remaining
arms (per-output gain, split latent/answer networks, depth, width) are still running here and at batch 256 on Kaggle.
