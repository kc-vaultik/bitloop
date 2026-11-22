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

Per-run JSON and logs: `laptop/` (`M_*` are the mixed-precision arms, `R0_*` the two reference arms from the earlier
laptop pilot). Exact accuracy (%) on 2,000 validation puzzles by number of recursion steps at test time.

| Arm | Full-precision share of core weights | K = 1 | K = 2 | K = 4 | K = 8 | K = 16 |
| --- | --- | --- | --- | --- | --- | --- |
| fp (reference) | 100 % | 16.0 | 21.9 | 41.6 | 42.55 | 42.6 |
| tern (reference) | 0 % | 15.9 | 16.0 | 16.0 | 16.0 | 16.0 |
| tern, full-precision cell-mixing layers | 15 % | 16.05 | 16.05 | 16.05 | 16.05 | 16.05 |
| tern, full-precision channel layers | 85 % | 16.0 | 17.55 | 20.1 | 20.4 | 20.4 |

One seed each. Neither part restores refinement on its own: with 85 % of the core weights in full precision and only
the cell-mixing layers ternary, recursion adds 4.4 points where the full-precision model adds 26.6. The remaining
arms (per-output gain, split latent/answer networks, depth, width) are still running here and at batch 256 on Kaggle.

## Stage A2 — 6x6 Sudoku, mixed precision (ARM CPU, width 64, 1,500 steps, batch 128, seed 1, validation)

Ternary arms at lr 2e-2, full-precision arm at lr 8e-3. Exact accuracy (%) by recursion steps at test time.
(Per-run JSON is still on the server; only this summary has been retrieved.)

| Arm | Flags | K = 1 | K = 2 | K = 4 | K = 16 |
| --- | --- | --- | --- | --- | --- |
| tern | | 33.6 | 52.3 | 55.7 | 56.1 |
| tern, per-output gain | `--wscale row` | 39.7 | 60.8 | 63.8 | 63.9 |
| tern, fp channel layers | `--fp_parts ch` | 37.8 | 74.4 | 73.6 | 73.4 |
| tern, fp cell-mixing layers | `--fp_parts tok` | 54.8 | 75.9 | 78.3 | 78.4 |
| tern, gain + fp cell-mixing | `--wscale row --fp_parts tok` | 57.0 | 78.9 | 81.8 | 81.7 |
| spike, fp cell-mixing | `--mode spike --fp_parts tok` | 33.5 | 48.7 | 52.8 | 52.9 |
| split nets, both ternary | `--split 1` | 35.5 | 65.7 | 64.6 | 64.2 |
| split nets, fp answer net | `--split 1 --fp_net y` | 46.5 | 49.4 | 50.9 | 51.0 |
| split nets, fp latent net | `--split 1 --fp_net z` | 59.5 | 87.0 | 88.0 | 88.1 |
| split nets, both fp | `--mode fp --split 1` | 68.8 | 89.2 | 88.6 | 88.4 |

A full-precision latent network with a ternary answer network is within a point of the all-full-precision split
model; the reverse is worse than all-ternary. One seed on the easy task, and 6x6 and 9x9 already disagree on
cell-mixing precision (+22.6 points here, nothing on the 9x9 laptop run above), so this is a lead, not a result.

## Stage A2 — 9x9 rating-0 puzzles on Kaggle (2 x T4, batch 256, 548,009 puzzles, width 192, 8,000 steps, seed 1)

Per-run JSON and logs: `kaggle_session2/`. Exact accuracy (%) on 5,000 validation puzzles by recursion steps at test
time. Reference rows are from `kaggle_session1/`.

| Arm | Full-precision share of core weights | K = 1 | K = 2 | K = 4 | K = 16 |
| --- | --- | --- | --- | --- | --- |
| fp (reference) | 100 % | 15.5 | 31.6 | 71.6 | 72.7 |
| tern (reference) | 0 % | 15.5 | 15.5 | 15.5 | 15.5 |
| tern, 8 supervised steps, 3 latent updates | 0 % | 15.6 | 16.2 | 16.5 | 16.5 |
| tern, per-output gain | 0 % | 15.3 | 16.6 | 17.9 | 18.0 |
| tern, width 384 (3.5x the weights) | 0 % | 15.5 | 19.4 | 25.8 | 26.1 |
| tern, fp cell-mixing layers | 15 % | 15.5 | 19.9 | 25.0 | 25.1 |
| tern, gain + fp cell-mixing | 15 % | 15.5 | 21.3 | 28.2 | 28.7 |
| tern, fp channel layers | 85 % | 15.5 | 21.3 | 34.9 | 35.6 |
| split nets, both ternary | 0 % | 15.4 | 15.4 | 15.4 | 15.4 |
| split nets, fp answer net | 50 % | 15.4 | 16.4 | 18.1 | 17.5 |
| split nets, fp latent net | 50 % | 15.5 | 19.4 | 31.0 | 31.3 |
| split nets, both fp | 100 % | 15.6 | 25.8 | 52.0 | 52.2 |

Reading (one seed, exploratory):
- Every arm starts at the same one-step accuracy (15.3-15.6 %). The whole gap is in refinement.
- Precision in the latent network matters more than in the answer network (31.0 against 18.1), the same direction as
  on 6x6. Unlike 6x6 it does not close the gap to the full-precision split model (52.0).
- Splitting the networks costs the full-precision model 20 points (71.6 to 52.0), so the split design is a
  diagnostic, not a candidate architecture.
- An all-ternary core refines once it is wide enough: width 384 reaches 25.8 % with no full-precision core weights.
- No mixed-precision arm comes within 35 points of full precision at this budget.
- The laptop run above (batch 64, 22,649 puzzles) showed no gain from full-precision cell-mixing; here it adds 9.5
  points. The laptop runs are under-trained and are not used for conclusions.

Deviation: `A_R0_fp_deep_lr3e-3` could not resume. Its session-1 checkpoint holds quantized-weight buffers that the
revised model no longer registers, so loading failed; the job has no result from this session.
