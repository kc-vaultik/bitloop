# Stage C — anchoring to a published baseline

Goal: before any quantized result on the standard Sudoku-Extreme protocol (1,000 training puzzles), reproduce the
full-precision baseline of *Speed is Confidence* (Dillon, arXiv 2601.19085) within its reported spread
(85.5 % +- 1.3 puzzle accuracy; the authors' seed-42 log ends at 87.09 %).

We use the authors' released code, Apache-2.0, https://github.com/rekursiv-ai/sic, pinned to commit
`164f6776593516bb436aa8154799fcde1b97e163`, experiment `x182m` (8,000 steps, batch 384, 2-layer MLP-mixer, 512 hidden).
We do not copy that code into this repository.

Status (2026-11-22): **probe running (third attempt), no accuracy or speed result yet.**

- First execution (Kaggle, 2 x T4) stopped in setup after 4 minutes. The authors' package is named `code`; when
  PyTorch is imported with the repository root on the import path, Python 3.13's debugger picks up that package
  instead of the standard-library module of the same name and fails. Our check ran from the repository root. The
  notebook now runs its import check from a neutral directory, and falls back to a preload wrapper for the dataset
  builder if the authors' own `python -m code...` invocation fails for the same reason.
- Second execution stopped at the import check: the training code imports `pytest` at module level, and we had
  installed without the authors' development tools. The notebook now installs that one package at the version their
  lock file pins. Third execution launched 2026-11-22.
- One measurement survived both runs. Time per 31,104 x 512 by 512 x 2,048 matrix product plus SiLU on a T4:
  float32 19.4 / 19.3 ms, bfloat16 21.6 / 22.2 ms, float16 3.0 / 2.8 ms (first / second run). The published all-bfloat16 recipe gets no speed-up on this GPU.
- The probe therefore has three arms: the published recipe (bfloat16, 40 minutes of training), and float16 and
  float32 variants (20 minutes each). float16 is a deviation from the published recipe and would need its own
  validation against the reported accuracy before it could serve as the baseline.

Known risks the probe is meant to settle:
- Speed of the published recipe on a T4 (see the micro-benchmark above).
- The authors used 16 GiB at batch 384; a T4 has 15 GiB.
- On the authors' RTX 5090, 500 training steps take 175 s and one 38,400-puzzle evaluation takes 304 s. A full run
  on a T4 must evaluate rarely to fit a 12-hour session.

Changes to the third-party code, made by the notebook at run time: the dataset builder's random generator is
seeded (20261120) so the 1,000-puzzle subset is reproducible, and three short experiment files are added that
subclass the authors' `x182m` experiment. Third-party files are kept in `/tmp`, outside the notebook output.
