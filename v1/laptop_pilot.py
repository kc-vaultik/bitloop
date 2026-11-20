"""BitLoop v1 Stage A-lite on a local GPU (exploratory, validation split only).

A difficulty ladder on the 176k-puzzle head of the Sudoku-Extreme train file: rating-0 puzzles first,
then rating 1-9, then full difficulty, with a batch size chosen to fit this GPU's memory.
Resumable: run it again and finished jobs are skipped, half-finished ones continue from checkpoint.
"""
import ctypes, json, os, subprocess, sys, time
import torch, torch.nn.functional as F
import bitloop_v1 as B

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
if os.name == "nt":  # keep Windows awake while this process lives; reverts automatically on exit
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
os.makedirs("results", exist_ok=True)
assert torch.cuda.is_available(), "CUDA not available"
torch.cuda.set_per_process_memory_fraction(0.85)  # fail fast with OOM instead of spilling into slow shared memory
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)

cal_path = "results/_calibration.json"
if os.path.exists(cal_path):
    cal = json.load(open(cal_path))
else:
    P, S, _ = B.read_csv("data/test_head.csv", limit=4000)
    Pt, St = torch.tensor(P).long().cuda(), torch.tensor(S).long().cuda()
    assert ((Pt == 0) | (Pt == St)).all() and B.valid_solution(St, 9).all()
    g = torch.Generator(device="cuda").manual_seed(0); p2, s2 = B.augment(Pt, St, 9, g)
    assert B.valid_solution(s2, 9).all() and ((p2 == 0) | (p2 == s2)).all()
    print("data and augmentation checks passed", flush=True)

    def bench(mode, bs, d=192, n=12):
        m = B.BitLoop(81, 10, d, mode, 2, 2, (4, "token", 32)).cuda(); opt = torch.optim.AdamW(m.parameters(), 1e-3)
        torch.cuda.reset_peak_memory_stats()
        for i in range(n):
            if i == 3: torch.cuda.synchronize(); t = time.time()
            outs = m(Pt[:bs], 4); loss = sum(F.cross_entropy(o.reshape(-1, 9), (St[:bs] - 1).reshape(-1)) for o in outs) / 4
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        return (time.time() - t) / (n - 3), torch.cuda.max_memory_allocated() / 2 ** 30

    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    bs_ok = None
    for bs in [256, 128, 64]:
        try:
            print("probing batch size", bs, flush=True)
            sps, mem = bench("spike", bs, n=6)
            print(f"  ok: {sps:.2f} s/step, peak {mem:.2f} GB", flush=True)
            bs_ok = bs; break   # fits under the 85% memory cap set above
        except torch.OutOfMemoryError:
            pass
        import gc; gc.collect(); torch.cuda.empty_cache()
    assert bs_ok, "no batch size fits"
    cal = {"gpu": torch.cuda.get_device_name(0), "vram_gb": total, "bs": bs_ok, "s_per_step": {}, "peak_gb": {}}
    for mode in ["fp", "tern", "spike"]:
        cal["s_per_step"][mode], cal["peak_gb"][mode] = bench(mode, bs_ok)
        print(f"{mode:6s} bs {bs_ok}: {cal['s_per_step'][mode]:.3f} s/step, peak {cal['peak_gb'][mode]:.2f} GB "
              f"-> 8000 steps = {cal['s_per_step'][mode] * 8000 / 3600:.2f} h", flush=True)
    json.dump(cal, open(cal_path, "w"), indent=1)
    del Pt, St, p2, s2; torch.cuda.empty_cache()

FULL = ["--n_train", "170000", "--n_val", "5000", "--n_test", "20000"]          # full-difficulty jobs keep their original split
R0 = ["--max_rating", "0"]                                 # rating 0: solvable without backtracking
R9 = ["--min_rating", "1", "--max_rating", "9"]            # rating 1-9: a few backtracks
def J(name, band, mode, lr, *extra): return {"name": name, "args": [*band, "--mode", mode, "--lr", lr, *extra]}
jobs = [J("R0_fp_lr3e-3", R0, "fp", "3e-3"), J("R0_fp_lr1e-2", R0, "fp", "1e-2"),
        J("R0_tern_lr1e-2", R0, "tern", "1e-2"), J("R0_spike_lr1e-2", R0, "spike", "1e-2"),
        J("R0_tern_lr3e-2", R0, "tern", "3e-2"), J("R0_spike_lr3e-2", R0, "spike", "3e-2"),
        J("R0_tern_lr3e-3", R0, "tern", "3e-3"), J("R0_spike_lr3e-3", R0, "spike", "3e-3"),
        J("R9_fp_lr3e-3", R9, "fp", "3e-3"), J("R9_fp_lr1e-2", R9, "fp", "1e-2"),
        J("R0_fp_deep_lr3e-3", R0, "fp", "3e-3", "--ktrain", "8", "--n_lat", "3"),
        J("R9_tern_lr1e-2", R9, "tern", "1e-2"), J("R9_spike_lr1e-2", R9, "spike", "1e-2"),
        J("R0_spike_group_lr1e-2", R0, "spike", "1e-2", "--spike_scale", "group"),
        J("R0_spike_tensor_lr1e-2", R0, "spike", "1e-2", "--spike_scale", "tensor")]
for lr in ["3e-3", "1e-2"]:
    for mode in ["fp", "tern", "spike"]:
        jobs.append(J(f"L_{mode}_lr{lr}", FULL, mode, lr))
json.dump(jobs, open("jobs_laptop.json", "w"), indent=1)
common = (f"--task sudoku9 --train_csv data/train_head.csv --test_csv data/test_head.csv --n_train 200000 --n_val 2000 "
          f"--n_test 2000 --d 192 --nb 2 --n_lat 2 --ktrain 4 --bs {cal['bs']} --steps 8000 --seed 1 --eval_split va")
sys.exit(subprocess.call([sys.executable, "run_queue.py", "jobs_laptop.json", "--hours", str(HOURS), "--outdir", "results",
                          "--common", common]))
