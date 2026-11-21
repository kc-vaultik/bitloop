"""BitLoop v1 Stage A2 on a local GPU (exploratory, validation split only): which part of the loop needs precision?

Rating-0 Sudoku-Extreme puzzles, same configuration as the Stage A-lite pilot (width 192, K_train 4, 8,000 steps,
batch from the earlier calibration). Each job changes one thing relative to the all-ternary arm.
Resumable: run again and finished jobs are skipped.
"""
import ctypes, json, os, subprocess, sys
import torch

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
if os.name == "nt":  # keep Windows awake while this process lives; reverts automatically on exit
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
assert torch.cuda.is_available(), "CUDA not available"
bs = json.load(open("results/_calibration.json"))["bs"]
R0 = ["--max_rating", "0", "--lr", "3e-3"]
def J(name, *extra): return {"name": name, "args": [*R0, *extra]}
jobs = [J("M_tern_fptok", "--mode", "tern", "--fp_parts", "tok"),          # cell-mixing MLPs full precision (15% of core weights)
        J("M_tern_fpch", "--mode", "tern", "--fp_parts", "ch"),            # channel MLPs full precision (85%)
        J("M_tern_row", "--mode", "tern", "--wscale", "row"),              # all ternary + one learned gain per output unit
        J("M_tern_split_fpy", "--mode", "tern", "--split", "1", "--fp_net", "y"),   # separate answer network, full precision
        J("M_tern_split_fpz", "--mode", "tern", "--split", "1", "--fp_net", "z"),   # separate latent network, full precision
        J("M_tern_split", "--mode", "tern", "--split", "1"),               # control: split, all ternary
        J("M_fp_split", "--mode", "fp", "--split", "1"),                   # control: split, all full precision
        J("M_tern_deep", "--mode", "tern", "--ktrain", "8", "--n_lat", "3"),
        J("M_tern_row_fptok", "--mode", "tern", "--wscale", "row", "--fp_parts", "tok"),
        J("M_spike_fptok", "--mode", "spike", "--fp_parts", "tok")]
json.dump(jobs, open("jobs_mixed.json", "w"), indent=1)
common = (f"--task sudoku9 --train_csv data/train_head.csv --test_csv data/test_head.csv --n_train 200000 --n_val 2000 "
          f"--n_test 2000 --d 192 --nb 2 --n_lat 2 --ktrain 4 --bs {bs} --steps 8000 --seed 1 --eval_split va")
sys.exit(subprocess.call([sys.executable, "run_queue.py", "jobs_mixed.json", "--hours", str(HOURS), "--outdir", "results",
                          "--common", common]))
