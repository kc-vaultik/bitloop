# BitLoop v0 — pre-registration

Written 2026-11-20, after budget pilots on the full-precision arm only (seed 0, validation split)
and before any ternary or spiking model was trained. Nothing below is changed after results.

## Question
Can a tiny weight-shared recursive network keep its accuracy when its core weights are ternary
{-1, 0, +1} and its activations are sparse integer spike counts, and can extra recursion buy back
whatever precision costs?

## Task and data
6x6 Sudoku (2x3 boxes), locally minimal unique-solution puzzles (8-13 clues, mean 10.1).
22,984 distinct solution grids, split by grid: 19,984 train / 1,000 validation / 2,000 test.
Train-time augmentation: digit relabeling only. Test set is touched once, by the final evaluation.

## Model (identical across arms except quantization)
Recursive core: 2 MLP-mixer blocks, width 64, shared across all recursion steps.
One supervision step = 2 latent updates z <- f(x+y+z), then 1 answer update y <- f(y+z).
Deep supervision over K_train = 4 steps, state detached between steps.
Embeddings, RMS norms and the 6-way output head are full precision in all arms.

## Arms
- fp: float weights, GELU.
- tern: ternary core weights (absmean scale, straight-through estimator), GELU.
- spike: ternary core weights + integer spike-count activations in [0, 4], adaptive per-token
  threshold (mean absolute pre-activation), straight-through estimator.

## Budget (fixed from the fp pilot, same for every arm)
1,500 steps, batch 128, AdamW lr 2e-3 one-cycle, weight decay 0.01, grad clip 1.0.
Seeds 1, 2, 3 per arm. No per-arm tuning.

## Metrics
Primary: exact-puzzle accuracy on the 2,000 test puzzles at K = 4, mean over seeds.
Secondary: accuracy at K in {1, 2, 8, 16}; per-puzzle core operation counts (float MACs for fp,
float adds through nonzero weights for tern, integer synaptic events for spike); weight and
activation sparsity.

## Hypotheses and decision rule
- H1: tern is within 5 points of fp at K = 4.
- H2: spike is within 5 points of fp at K = 4, with activation sparsity >= 50%.
- H3 (tested only if H1 or H2 fails): the failing arm, retrained with K_train = 8 under the same
  step budget, comes within 5 points of fp at K = 4.
- Kill: H1 fails and H3 fails for tern.

## Known limitations, stated up front
Toy scale; one task; 3 seeds; op counts are a proxy, not measured energy; norms and per-token
scales still need float arithmetic; fp baseline is not tuned to saturation.
