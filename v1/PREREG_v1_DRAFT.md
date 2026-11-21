# BitLoop v1 — Stage B pre-registration (DRAFT, not yet frozen)

Status: draft written 2026-11-20. Items marked [A] are filled from the exploratory Stage A pilot
(validation split only) and then this file is frozen and hashed before any Stage B run.

## Revision 2026-11-21 (still a draft; nothing frozen)

Stage A (exploratory, one seed, validation only) changed what Stage B should test.

Observed:
- 9x9, rating-0 puzzles: tern scores 15.5 % at K = 1, 2, 4 and 16 and at three learning rates a decade apart;
  fp goes 15.5 -> 31.6 -> 71.6 % over the same steps. The ternary core does not refine across recursion steps.
- 6x6: tern at width 128 and 6,000 steps reaches 90.3 % against fp 98.8 % (width 64, same steps); spike trails
  tern by 35-45 points at matched width, so "spiking is nearly free" (old H3) is withdrawn as a hypothesis.
- Rating 1-9 and full difficulty are at floor for every arm at this budget and are dropped from Stage B.

Stage B is therefore re-centred on one question: which part of the recursive loop needs precision for
refinement to occur. Candidate confirmatory hypotheses, to be chosen and frozen after Stage A2:
- M1: keeping only the cell-mixing layers in full precision (about 15 % of core weights) restores refinement,
      defined as accuracy at K = 4 exceeding accuracy at K = 1 by at least 20 points.
- M2: a learned per-output gain on ternary layers restores refinement without any full-precision matrix.
- M3: with separate latent and answer networks, precision matters in one of them and not the other.
- M4: refinement gain (accuracy at K = 4 minus K = 1) is the mediator: across arms and seeds it predicts
      final accuracy better than core bit count does.
Primary metric stays exact-puzzle accuracy; refinement gain is added as a pre-specified secondary metric.
Stage A2 runs these arms once each on rating-0 9x9 (laptop GPU) and on 6x6 (ARM CPU) to pick which to confirm.

## Working title
Bits for loops: ternary, event-driven recursive reasoning cores, and what precision costs under weight-tied recursion.

## Claims the paper is allowed to make (decided before the data)
C1. Iso-accuracy cost: the number of ternary weights (width multiplier) a recursive core needs to match
    a full-precision core of width W, and the integer-operation count at that match point.
C2. Spiking is (or is not) free once weights are ternary: accuracy difference between tern and spike
    at matched width, with activation sparsity and event counts.
C3. Recursion as a substitute for precision: whether training with more supervised recursion steps
    closes the precision gap at fixed width (v0 showed test-time-only recursion does not).
C4. Scale granularity: per-token vs per-group vs per-tensor spike thresholds under recursion, to test
    whether the drift mechanism reported for post-training Int4 (arXiv 2607.16237) also governs
    natively trained spiking cores.
C5. Measured cost: wall-clock and energy per puzzle of an integer-only kernel for the spike core on a
    laptop CPU (and one microcontroller if available), replacing the operation-count proxy.

## Task and data
Sudoku-Extreme (sapientinc/sudoku-extreme). Data-rich regime: 1,000,000 training puzzles sampled with a
fixed seed from the 3.8M-puzzle train file; 5,000 validation puzzles from the same file, disjoint;
20,000 test puzzles sampled with a fixed seed from the test file. Any train or validation puzzle that
also appears in the test sample is dropped. This is NOT the 1,000-example protocol of HRM/TRM and no
number here is comparable with theirs; one confirmatory full-protocol run per arm is planned only if
compute allows and is reported separately.
Secondary task (generality): Maze-Hard 30x30, if Stage B on Sudoku completes within the compute plan.

## Arms and architecture
As v0 (2 MLP-mixer blocks, shared across recursion; fp / tern / spike), width W = 192 unless [A] says
otherwise. Embeddings, norms and head full precision in all arms.

## Experiments
B1 main comparison: fp(W), tern(W), spike(W), each at its own learning rate chosen in Stage A [A],
   5 seeds, [A] steps. Primary metric: exact-puzzle accuracy on test at K = K_train.
B2 width scaling: tern and spike at W, 2W, 4W; 3 seeds.
B3 train-time recursion: K_train in {2, 4, 8} for each arm at W; 3 seeds.
B4 threshold granularity: spike with token / group-32 / tensor thresholds; 3 seeds.
B5 iso-accuracy read-out: smallest ternary width whose 95% bootstrap interval overlaps fp(W).

## Statistics
Per-arm mean and standard deviation over seeds, plus a 95% bootstrap interval over test puzzles
pooled across seeds for every headline difference. A difference is called "matched" only if the
interval for (fp - arm) includes values within +-2 points. No test-set-based selection of anything.
All runs, including diverged ones, are reported.

## Hypotheses
H1: tern(W) is more than 2 points below fp(W) (v0 suggests a large gap).
H2: some width in {2W, 4W} brings tern within 2 points of fp(W).
H3: spike(W) is within 2 points of tern(W), with activation sparsity >= 50%.
H4: increasing K_train narrows the fp - tern gap (difference-in-differences > 0).
H5: tensor-level thresholds are worse than token-level by more than 2 points; group-32 is within 2.

## Compute plan [A]
Fixed after Stage A calibration: seconds per step per arm on the Kaggle T4, total GPU-hours for
B1-B4, and the number of weekly sessions needed. If B1-B4 exceed 4 weeks of free quota, B3 drops to
K_train in {4, 8} and B2 to {W, 2W}; that reduction is decided here, before running, not after.

## Energy reporting
Operation counts as in v0 plus the analytical SNN energy model of Lemaire et al. (2023). Proxy numbers
are labelled as proxies everywhere; C5 measurements are reported separately.

## Stage C (standard protocol, for comparability with published numbers) — planned, not yet specified
The 1,000-example Sudoku-Extreme protocol of HRM/TRM, using the fast recipe of Dillon (arXiv 2601.19085:
2-layer MLP-mixer, 512 hidden, about 7M parameters, Muon + AdamW, 8k steps at batch 384, 48 minutes on one
RTX 5090, 85.5% +- 1.3% reported). Estimated 6-10 hours per run on a Kaggle T4, so 3 arms x 3 seeds is
about two weeks of free quota. Requires a clean-room reimplementation (the reference code states no
license) and a reproduction of the full-precision number within its reported spread before any
quantized arm is run. Specified and frozen only after Stage B.
