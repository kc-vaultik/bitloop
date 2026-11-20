# Running BitLoop v1 Stage A on a free GPU

Recommended: Kaggle Notebooks (about 30 GPU-hours per week, 2 x T4, sessions up to 12 h, runs unattended).

## One-time (about 5 minutes, needs a computer or the Kaggle site on your phone)
1. Sign in at kaggle.com (Google sign-in works) and verify your phone number in Settings.
   Phone verification is what unlocks GPU and Internet in notebooks.

## Each run
1. kaggle.com -> Create -> New Notebook -> File -> Import Notebook -> upload `BitLoop_v1_stageA.ipynb`.
2. Right-hand panel, Session options: Accelerator = GPU T4 x2, Internet = On.
3. Top right: Save Version -> "Save & Run All (Commit)" -> Save. You can close the tab.
4. When the version finishes (about 8 h), open it -> Output -> download `bitloop_results.zip`
   and attach it in this chat.

## Continuing an unfinished queue in a later session
Open the notebook -> Add Input -> Notebook Output Files -> pick the previous version of this notebook,
then Save & Run All again. Finished jobs are skipped and half-finished ones resume from checkpoint.

## Colab fallback
Open the .ipynb in Colab, Runtime -> Change runtime type -> T4 GPU, set HOURS = 3 in the first code
cell, Run all, keep the tab open, download `bitloop_results.zip` at the end. Colab does not keep files
after the session ends.

## What Stage A does
Checks the data pipeline, measures seconds per step for each arm, then runs 15 validation-only pilot
jobs (learning-rate sweep per arm, one deeper fp run, one wide ternary run, two spike-threshold variants).
It never reads the test split. Its output fixes the budget and learning rates for the pre-registered Stage B.
