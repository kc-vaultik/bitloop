"""Build BitLoop_stageC_probe.ipynb: a short, measured run of the published fast TRM baseline on a free GPU.

The notebook runs third-party code (Apache-2.0) pinned to one commit. It changes nothing in that code except
(a) seeding the dataset builder so the 1,000-puzzle training subset is reproducible, and (b) adding two small
experiment files that subclass the authors' own baseline experiment with a shorter run.
"""
import json

REPO = "https://github.com/rekursiv-ai/sic"
COMMIT = "164f6776593516bb436aa8154799fcde1b97e163"
DATA_SEED = 20261120

PROBE = '''"""BitLoop probe of x182m (authors' fast TRM baseline, seed 42): same schedule, stopped early by time."""
from experiment import main
from x182m import Experiment as Base


class Experiment(Base):
    data_dir: str = "DATA_DIR"
    eval_every_steps: int = 100          # the training loop only prints timing at evals
    max_eval_samples: int = 1_152        # authors use 38,400; kept small so the probe is cheap
    max_eval_samples_train: int = 0
    max_train_sec: float = 60 * 40       # 40 minutes of training time, evals excluded
    checkpoint_steps: list[int] = []     # noqa: RUF012


if __name__ == "__main__":
    main(Experiment())
'''

PROBE_DT = '''"""BitLoop probe: x182m with DTYPE parameters, to see what this precision costs on this GPU."""
from experiment import main
from model import TRM3, TRM3ConfigProtocol
from x182m import Experiment as Base

import torch


class Experiment(Base):
    data_dir: str = "DATA_DIR"
    eval_every_steps: int = 100
    max_eval_samples: int = 1_152
    max_eval_samples_train: int = 0
    max_train_sec: float = 60 * 20       # 20 minutes of training time, evals excluded
    checkpoint_steps: list[int] = []     # noqa: RUF012

    config: TRM3ConfigProtocol = TRM3.Config(
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        z_L_init_svd=True,
        z_L_random_init=False,
        dtype=torch.DTYPE,
    )


if __name__ == "__main__":
    main(Experiment())
'''


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = [
    md("# BitLoop Stage C probe — can a free T4 reproduce the published baseline?\n\n"
       "Runs the authors' own code for *Speed is Confidence* (Dillon 2026, Apache-2.0), experiment `x182m`: the fast "
       "Tiny Recursive Model baseline on Sudoku-Extreme (1,000 training puzzles, 8k steps at batch 384, reported "
       "85.5% ± 1.3). This probe trains the published recipe (all bfloat16) for at most 40 minutes on one GPU, and "
       "float16 and float32 variants for 20 minutes each on the other, to measure speed and memory and to compare "
       "early accuracy with the authors' published log. It decides whether a full reproduction fits a free session.\n\n"
       f"Third-party code: `{REPO}` pinned to commit `{COMMIT[:12]}`. Two changes only: the dataset builder is seeded "
       f"(`{DATA_SEED}`) so the training subset is reproducible, and three short experiment files are added. Everything third-party lives in `/tmp`, so the notebook output holds only `probe_out/`.\n\n"
       "**Kaggle:** Accelerator **GPU T4 x2**, Internet **On**, then *Save Version → Save & Run All (Commit)*. "
       "About 1.5 hours. Download `stageC_probe.zip` from the Output tab afterwards."),
    code("import torch, time, json, subprocess, os\n"
         "n = torch.cuda.device_count(); assert n >= 1, 'No GPU: turn on the accelerator and rerun.'\n"
         "print('torch', torch.__version__, '| GPUs:', [torch.cuda.get_device_name(i) for i in range(n)])\n"
         "print('bf16 supported (reported):', torch.cuda.is_bf16_supported())\n"
         "# Micro-benchmark: the published recipe keeps the whole model in bfloat16, which older GPUs lack natively\n"
         "bench = {}\n"
         "for name, dt in [('float32', torch.float32), ('float16', torch.float16), ('bfloat16', torch.bfloat16)]:\n"
         "    try:\n"
         "        a = torch.randn(384 * 81, 512, device='cuda', dtype=dt); w = torch.randn(512, 2048, device='cuda', dtype=dt)\n"
         "        for i in range(13):\n"
         "            if i == 3: torch.cuda.synchronize(); t = time.time()\n"
         "            b = torch.nn.functional.silu(a @ w)\n"
         "        torch.cuda.synchronize(); bench[name] = round((time.time() - t) / 10 * 1000, 2)\n"
         "    except Exception as e:\n"
         "        bench[name] = 'ERROR ' + repr(e)[:120]\n"
         "print('ms per 31104x512 @ 512x2048 matmul + SiLU:', bench)\n"
         "os.makedirs('probe_out', exist_ok=True); json.dump({'gpus': [torch.cuda.get_device_name(i) for i in range(n)], 'bench_ms': bench, 'torch': torch.__version__}, open('probe_out/gpu.json', 'w'))"),
    code("%%bash\nset -e\n"
         "curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1\n"
         "export PATH=$HOME/.local/bin:$PATH\n"
         "mkdir -p /tmp/bitloop_probe && cd /tmp/bitloop_probe\n"
         f"rm -rf sic && git clone -q {REPO} sic && cd sic && git checkout -q {COMMIT}\n"
         f"test \"$(git rev-parse HEAD)\" = \"{COMMIT}\" && echo \"pinned commit OK: $(git log -1 --format='%h %ad' --date=short)\"\n"
         "head -3 LICENSE | tail -2\n"
         "uv sync --no-dev --frozen 2>&1 | tail -3\n"
         "# The training code imports pytest at module level (the authors' default `uv sync` installs their dev tools);\n"
         "# install just that one package, at the version their lock file pins.\n"
         "uv pip install -q --python .venv/bin/python pytest==9.0.2 2>&1 | tail -1\n"
         "# The repository's package is named `code`, like a standard-library module that Python 3.13's debugger needs\n"
         "# at import time. Run the check from a directory where that name is not shadowed.\n"
         "cd /tmp && /tmp/bitloop_probe/sic/.venv/bin/python -c \"import sys, torch, experiment, model; "
         "print('venv python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| experiment from', experiment.__file__)\""),
    code("%%bash\nset -e\nOUT=$PWD/probe_out\ncd /tmp/bitloop_probe/sic\n"
         "# Seed the builder so the 1,000-puzzle training subset and its augmentations are reproducible\n"
         f"sed -i 's/^rng = np.random.default_rng()$/rng = np.random.default_rng({DATA_SEED})/' code/tools/dataset/build_sudoku_dataset.py\n"
         f"grep -n 'default_rng({DATA_SEED})' code/tools/dataset/build_sudoku_dataset.py\n"
         "export HF_HUB_DISABLE_PROGRESS_BARS=1\n"
         "D=/tmp/bitloop_probe/data/sudoku-extreme-1k-aug-1000\n"
         "ARGS=\"--output-dir $D --subsample-size 1000 --num-aug 1000\"\n"
         "# First the authors' own invocation. If the `code` name clash breaks it on this Python, load the debugger's\n"
         "# dependency first and then run the same module unchanged.\n"
         "if .venv/bin/python -m code.tools.dataset.build_sudoku_dataset $ARGS > $OUT/build.log 2>&1; then echo 'builder: plain invocation'\n"
         "else rm -rf $D; echo 'builder: plain invocation failed, using the preload wrapper'; tail -2 $OUT/build.log\n"
         "  .venv/bin/python -P -c \"import pdb, sys, runpy; sys.modules.pop('code', None); sys.path.insert(0, '$PWD'); "
         "sys.argv = ['build_sudoku_dataset'] + '$ARGS'.split(); runpy.run_module('code.tools.dataset.build_sudoku_dataset', run_name='__main__')\" > $OUT/build.log 2>&1\n"
         "fi\n"
         "tail -3 $OUT/build.log\n"
         "cd $D && find . -type f | sort | xargs sha256sum | tee $OUT/dataset_sha256.txt | cut -c1-16,65- && du -sh ."),
    code("import os\n"
         "data_dir = '/tmp/bitloop_probe/data/sudoku-extreme-1k-aug-1000'; exp = '/tmp/bitloop_probe/sic/code/sudoku/'\n"
         "open(exp + 'zz_probe.py', 'w').write(" + json.dumps(PROBE) + ".replace('DATA_DIR', data_dir))\n"
         "for dt in ['float16', 'float32']:\n"
         "    open(exp + f'zz_probe_{dt}.py', 'w').write(" + json.dumps(PROBE_DT) + ".replace('DATA_DIR', data_dir).replace('DTYPE', dt))\n"
         "print('probe experiments written; data at', data_dir, '|', sorted(f for f in os.listdir(exp) if f.startswith('zz_')))"),
    code("%%bash\nOUT=$PWD/probe_out\ncd /tmp/bitloop_probe/sic\n"
         "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 20 > $OUT/mem.log 2>&1 &\nMEM=$!\n"
         "# GPU 0: the published recipe as is (bfloat16), 40 min. GPU 1 (if present): float16 then float32, 20 min each.\n"
         "CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/sudoku/zz_probe.py > $OUT/run_bf16.stdout 2>&1 &\nP0=$!\n"
         "if [ \"$(nvidia-smi -L | wc -l)\" -ge 2 ]; then\n"
         "  ( CUDA_VISIBLE_DEVICES=1 .venv/bin/python code/sudoku/zz_probe_float16.py > $OUT/run_fp16.stdout 2>&1; echo \"fp16 run exit code: $?\"\n"
         "    CUDA_VISIBLE_DEVICES=1 .venv/bin/python code/sudoku/zz_probe_float32.py > $OUT/run_fp32.stdout 2>&1; echo \"fp32 run exit code: $?\" ) &\n"
         "  P1=$!\nfi\n"
         "wait $P0; echo \"bf16 run exit code: $?\"\n"
         "if [ -n \"$P1\" ]; then wait $P1; fi\n"
         "kill $MEM 2>/dev/null\n"
         "cp code/sudoku/zz_probe*.log $OUT/ 2>/dev/null; true"),
    code("import re, glob, json\n"
         "ref = {100: None, 500: (60.72, 3.80), 1000: (67.05, 13.26)}   # authors' x182m log: cell %, puzzle % (38,400 test puzzles)\n"
         "for tag in ['bf16', 'fp16', 'fp32']:\n"
         "    try: txt = open(f'probe_out/run_{tag}.stdout', errors='replace').read()\n"
         "    except FileNotFoundError: continue\n"
         "    steps = re.findall(r'^Step\\s*(\\d+)\\s+Test\\s+([\\d.]+)% /\\s*([\\d.]+)%.*?\\(([\\d.]+)s /\\s*([\\d.]+)s\\)', txt, flags=re.M)\n"
         "    print(f'=== {tag}: {len(steps)} evals')\n"
         "    if not steps: print(txt[-1500:]); continue\n"
         "    for s, cell, puz, ev, tr in steps[:3] + steps[-3:]:\n"
         "        print(f'  step {int(s):5d}  cell {cell}%  puzzle {puz}%   eval {float(ev):6.1f}s   train-since-last-eval {float(tr):6.1f}s')\n"
         "    tr = [float(x[4]) for x in steps[1:]]   # skip the first interval: it includes compilation\n"
         "    if tr:\n"
         "        sps = sum(tr) / (100 * len(tr)); print(f'  -> {sps:.2f} s/step after warm-up; 8,000 steps = {sps * 8000 / 3600:.1f} h of training')\n"
         "        ev = [float(x[3]) for x in steps]; print(f'  -> eval of 1,152 puzzles: {sum(ev)/len(ev):.0f}s; a full 38,400-puzzle eval = {sum(ev)/len(ev)*38400/1152/60:.0f} min')\n"
         "mem = {}\n"
         "for line in open('probe_out/mem.log'):\n"
         "    p = line.replace('MiB', '').split(',')\n"
         "    if len(p) == 2 and p[1].strip().isdigit(): mem[p[0].strip()] = max(mem.get(p[0].strip(), 0), int(p[1]))\n"
         "print('peak GPU memory (MiB) by GPU index (GPU 1 is the larger of float16 and float32):', mem)\n"
         "print('authors (RTX 5090, bf16): 0.35 s/step, 304 s per 38,400-puzzle eval; step 500 = 60.72% cell / 3.80% puzzle')"),
    code("!zip -qr stageC_probe.zip probe_out && ls -la stageC_probe.zip"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open("BitLoop_stageC_probe.ipynb", "w"), indent=1)
print("wrote BitLoop_stageC_probe.ipynb:", len(cells), "cells")
