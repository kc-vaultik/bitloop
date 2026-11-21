# Stage C — anchoring to a published baseline

Goal: before any quantized result on the standard Sudoku-Extreme protocol (1,000 training puzzles), reproduce the
full-precision baseline of *Speed is Confidence* (Dillon, arXiv 2601.19085) within its reported spread
(85.5 % +- 1.3 puzzle accuracy; the authors' seed-42 log ends at 87.09 %).

We use the authors' released code, Apache-2.0, https://github.com/rekursiv-ai/sic, pinned to commit
`164f6776593516bb436aa8154799fcde1b97e163`, experiment `x182m` (8,000 steps, batch 384, 2-layer MLP-mixer, 512 hidden).
We do not copy that code into this repository.

Status (2026-11-20): **not yet run.** `BitLoop_stageC_probe.ipynb` is a 40-minute probe that measures speed and memory
on a free Kaggle T4 and compares early accuracy with the authors' log. It has been checked statically (cells parse,
the log parser is validated against the authors' published `x182m.log`) but never executed.

Known risks the probe is meant to settle:
- The recipe keeps the whole model in bfloat16. A T4 has no native bfloat16, so speed there is unknown.
- The authors used 16 GiB at batch 384; a T4 has 15 GiB.
- On the authors' RTX 5090, 500 training steps take 175 s and one 38,400-puzzle evaluation takes 304 s. A full run
  on a T4 must evaluate rarely to fit a 12-hour session.

Two changes to the third-party code, made by the notebook at run time: the dataset builder's random generator is
seeded (20261120) so the 1,000-puzzle subset is reproducible, and two short experiment files are added that subclass
the authors' `x182m` experiment.
