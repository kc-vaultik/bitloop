"""Build BitLoop_v1_stageA.ipynb (Kaggle / Colab) from the source files in this directory."""
import json, hashlib

src = {f: open(f).read() for f in ["bitloop_v1.py", "run_queue.py"]}
sha = {f: hashlib.sha256(s.encode()).hexdigest()[:16] for f, s in src.items()}

COMMON = ("--task sudoku9 --train_csv data/train.csv --test_csv data/test.csv --n_train 1000000 --n_val 5000 "
          "--n_test 20000 --d 192 --nb 2 --n_lat 2 --ktrain 4 --bs 256 --steps 8000 --seed 1 --eval_split va")
R0 = ["--max_rating", "0"]
def J(name, *extra): return {"name": name, "args": [*R0, "--lr", "3e-3", *extra]}
jobs = [J("A_R0_fp_deep_lr3e-3", "--mode", "fp", "--ktrain", "8", "--n_lat", "3"),   # resumes the session-1 checkpoint
        J("A2_tern_fptok", "--mode", "tern", "--fp_parts", "tok"),
        J("A2_tern_fpch", "--mode", "tern", "--fp_parts", "ch"),
        J("A2_tern_row", "--mode", "tern", "--wscale", "row"),
        J("A2_tern_split_fpy", "--mode", "tern", "--split", "1", "--fp_net", "y"),
        J("A2_tern_split_fpz", "--mode", "tern", "--split", "1", "--fp_net", "z"),
        J("A2_tern_split", "--mode", "tern", "--split", "1"),
        J("A2_fp_split", "--mode", "fp", "--split", "1"),
        J("A2_tern_deep", "--mode", "tern", "--ktrain", "8", "--n_lat", "3"),
        J("A2_tern_row_fptok", "--mode", "tern", "--wscale", "row", "--fp_parts", "tok"),
        J("A2_tern_wide", "--mode", "tern", "--d", "384")]


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = [
    md("# BitLoop v1 — Stage A2 (exploratory): which part of the loop needs precision?\n\n"
       "Rating-0 Sudoku-Extreme puzzles. Each job changes one thing against the all-ternary core: full-precision "
       "cell-mixing or channel layers, a learned gain per output unit, separate latent and answer networks, depth, width.\n\n"
       "**Stage A is exploratory.** It measures speed, sweeps learning rates per arm on the *validation* split, "
       "and never reads the test split. Its only job is to fix the budget and per-arm learning rates for the "
       "pre-registered Stage B.\n\n"
       "**Kaggle:** Settings → Accelerator **GPU T4 x2**, Internet **On**, then *Save Version → Save & Run All (Commit)*. "
       "It runs unattended and stops itself after `HOURS`. Run it again later to continue the queue: add the previous "
       "version's output as an input dataset, or simply rerun and it will redo unfinished jobs only if `results/` is restored.\n\n"
       "**Colab:** Runtime → Change runtime type → T4 GPU, set `HOURS = 3`, Run all, keep the tab open, download "
       "`bitloop_results.zip` at the end.\n\n"
       f"Source hashes: `bitloop_v1.py` {sha['bitloop_v1.py']}, `run_queue.py` {sha['run_queue.py']}."),
    code("HOURS = 7.5   # wall-clock budget for the job queue; keep below the platform's session limit\n"
         "import torch, subprocess, os, glob, shutil\n"
         "print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
         "assert torch.cuda.is_available(), 'No GPU: turn on the GPU accelerator and rerun.'\n"
         "os.makedirs('results', exist_ok=True); os.makedirs('data', exist_ok=True)\n"
         "# Restore results from an earlier session if its output was attached as an input (Kaggle)\n"
         "for f in glob.glob('/kaggle/input/**/results/*', recursive=True):\n"
         "    if not os.path.exists(os.path.join('results', os.path.basename(f))): shutil.copy(f, 'results')\n"
         "print('restored files:', len(os.listdir('results')))"),
    code("%%writefile bitloop_v1.py\n" + src["bitloop_v1.py"]),
    code("%%writefile run_queue.py\n" + src["run_queue.py"]),
    code("# Sudoku-Extreme (Wang et al. 2025), public on Hugging Face: about 800 MB\n"
         "base = 'https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/'\n"
         "for f in ['train.csv', 'test.csv']:\n"
         "    if not os.path.exists('data/' + f):\n"
         "        subprocess.check_call(['wget', '-q', '-c', base + f, '-O', 'data/' + f])\n"
         "    print(f, round(os.path.getsize('data/' + f) / 1e6), 'MB')"),
    code("# Sanity checks on the data pipeline before spending GPU hours\n"
         "import bitloop_v1 as B, torch\n"
         "P, S, R = B.read_csv('data/test.csv', limit=20000)\n"
         "Pt, St = torch.tensor(P).long().cuda(), torch.tensor(S).long().cuda()\n"
         "assert ((Pt == 0) | (Pt == St)).all() and B.valid_solution(St, 9).all()\n"
         "g = torch.Generator(device='cuda').manual_seed(0); p2, s2 = B.augment(Pt, St, 9, g)\n"
         "assert B.valid_solution(s2, 9).all() and ((p2 == 0) | (p2 == s2)).all()\n"
         "print('data and augmentation checks passed; mean clues', float((Pt > 0).sum(1).float().mean()))"),
    code("# Calibration: seconds per training step for each arm at the Stage A configuration\n"
         "import time, json, torch.nn.functional as F\n"
         "calib = {}\n"
         "for mode in ['fp', 'tern', 'spike']:\n"
         "    m = B.BitLoop(81, 10, 192, mode, 2, 2, (4, 'token', 32)).cuda(); opt = torch.optim.AdamW(m.parameters(), 1e-3)\n"
         "    for i in range(25):\n"
         "        if i == 5: torch.cuda.synchronize(); t = time.time()\n"
         "        outs = m(Pt[:256], 4); loss = sum(F.cross_entropy(o.reshape(-1, 9), (St[:256] - 1).reshape(-1)) for o in outs) / 4\n"
         "        opt.zero_grad(); loss.backward(); opt.step()\n"
         "    torch.cuda.synchronize(); calib[mode] = (time.time() - t) / 20\n"
         "    print(f'{mode:6s} {calib[mode]:.3f} s/step  ->  8000 steps = {calib[mode] * 8000 / 3600:.2f} h, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')\n"
         "json.dump({'s_per_step': calib, 'gpu': torch.cuda.get_device_name(0), 'n_gpu': torch.cuda.device_count()}, open('results/_calibration.json', 'w'))\n"
         "del m, opt, Pt, St, p2, s2; torch.cuda.empty_cache()"),
    code("jobs = " + json.dumps(jobs, indent=1) + "\n"
         "json.dump(jobs, open('jobs_stageA2.json', 'w'))\n"
         f"COMMON = '{COMMON}'\n"
         "print(len(jobs), 'jobs')"),
    code("!python3 run_queue.py jobs_stageA2.json --hours $HOURS --outdir results --common \"$COMMON\""),
    code("# Summary table (validation split) and a zip of everything to send back\n"
         "import json, glob, os\n"
         "rows = []\n"
         "for f in sorted(glob.glob('results/A*.json')):\n"
         "    d = json.load(open(f)); a = d['args']\n"
         "    if 'final' not in d: rows.append((os.path.basename(f), 'DIVERGED')); continue\n"
         "    k = str(a['ktrain']); r = d['final'][k]\n"
         "    rows.append((os.path.basename(f)[:-5], f\"exact {r['exact']*100:5.1f}%  cell {r['cell']*100:5.1f}%  K16 {d['final']['16']['exact']*100:5.1f}%  \"\n"
         "                 f\"train {d['train_s']/3600:.2f} h  by_rating {[round(v[1]*100,1) if v[1] is not None else None for v in r['by_rating'].values()]}\"))\n"
         "for n, s in rows: print(f'{n:26s} {s}')\n"
         "print('unfinished:', [os.path.basename(f) for f in glob.glob('results/*.ckpt')])\n"
         "!zip -qr bitloop_results.zip results -x 'results/*.pt' && ls -la bitloop_results.zip"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open("BitLoop_v1_stageA2.ipynb", "w"), indent=1)
print("wrote notebook:", len(cells), "cells;", len(jobs), "jobs;", sha)
