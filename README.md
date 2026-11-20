# BitLoop

Ternary and event-driven (spiking) recursive reasoning cores: what precision costs under weight-tied recursion.

**Status (2026-11-20): early research, private working repository. Nothing here is a published result.**

## What is here

| Path | Contents |
| --- | --- |
| `v0/` | Pre-registered pilot on 6x6 Sudoku, CPU only. Frozen pre-registration (`PREREG.md`), code and data hashes (`MANIFEST.sha256`), every run's JSON and log, analysis script and figure. |
| `v1/` | Code for the follow-up on Sudoku-Extreme (difficulty ladder), tests, the Stage A notebook, and the **draft** Stage B pre-registration. |
| `prereg/` | Index of pre-registrations and the commit at which each was frozen. |

## v0 result (pre-registered, negative)

A 46k-parameter recursive solver, 3 seeds per arm, 2,000 held-out test puzzles, exact-puzzle accuracy at K = 4:

| Arm | Accuracy | Core operations per puzzle |
| --- | --- | --- |
| Full precision | 73.9 +- 2.9 % | 22.1M float multiply-accumulates |
| Ternary weights | 12.7 +- 2.3 % | 14.7M float adds |
| Ternary + integer spikes | 14.5 +- 1.2 % | 6.2M integer accumulates |

- H1 (ternary within 5 points of full precision): **fails**.
- H2 (spiking within 5 points, >= 50 % activation sparsity): **fails on accuracy**, passes on sparsity (63.5 %).
- H3 (retraining with 8 supervised recursion steps closes the gap): **fails** for ternary (11.6 +- 0.6 %) and spiking (14.3 +- 1.4 % at K = 8).
- The pre-registered kill rule is met: a same-size ternary swap at this budget does not recover through recursion.

Deviations from the plan are listed in `v0/DEVIATIONS.md`. Operation counts are a proxy, not measured energy.

## What v1 asks

A reformulated question under a new pre-registration, after the failed v0 test: how many ternary weights and how much
trained recursion it takes to match a full-precision recursive core, whether integer spiking stays nearly free on top of
ternary weights, and whether the cost grows with the reasoning depth a puzzle needs. Stage A (exploratory, validation
split only) is in progress; Stage B is not frozen yet.

## Reproducing v0

```
cd v0
python3 gen_data.py 23000 data.npz        # deterministic; the committed data.npz is the same file
python3 bitloop.py --mode fp --seed 1 --d 64 --nb 2 --steps 1500 --split te --out runs/fp_1.json
python3 analyze.py
```

## Method note

Experiments, code and analysis were produced with AI assistance (Claude) under the direction of the author, who is
responsible for the claims. No licence has been chosen yet; all rights reserved until one is added.
