"""Run a list of BitLoop jobs across the available GPUs, resumably, inside a wall-clock budget.

  python3 run_queue.py jobs.json --hours 8.5 --outdir results

jobs.json: [{"name": "fp_lr3e-3_s1", "args": ["--mode", "fp", "--lr", "3e-3", ...]}, ...]
A job is skipped if results/<name>.json exists and resumed if results/<name>.ckpt exists,
so the same command can be re-run in a fresh session until the queue is empty.
A job may list files it depends on ("needs": ["results/base.json"]); it starts only once they all exist.
"""
import argparse, json, os, subprocess, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("jobs"); ap.add_argument("--hours", type=float, default=8.5)
ap.add_argument("--outdir", default="results"); ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--common", default="", help="args shared by every job, space separated; a job's own args come later and override them")
a = ap.parse_args()

try:
    import torch
    ngpu = torch.cuda.device_count()
except Exception:
    ngpu = 0
workers = a.workers or max(ngpu, 1)
os.makedirs(a.outdir, exist_ok=True)
t0 = time.time(); deadline = t0 + a.hours * 3600
jobs = json.load(open(a.jobs))
pending = [j for j in jobs if not os.path.exists(os.path.join(a.outdir, j["name"] + ".json"))]
print(f"{len(pending)} of {len(jobs)} jobs to run on {workers} worker(s), {ngpu} GPU(s), budget {a.hours} h", flush=True)
status = {}; lock = threading.Lock(); running = set()


def take():
    """Next job whose dependencies exist; "wait" if only blocked jobs remain and something is still running."""
    with lock:
        for i, j in enumerate(pending):
            if all(os.path.exists(n) for n in j.get("needs", [])):
                running.add(j["name"]); return pending.pop(i)
        if pending and running:
            return "wait"
        for j in pending:
            status[j["name"]] = "BLOCKED: missing " + ", ".join(n for n in j.get("needs", []) if not os.path.exists(n))
        pending.clear()
        return None


def work(w):
    env = dict(os.environ)
    if ngpu:
        env["CUDA_VISIBLE_DEVICES"] = str(w % ngpu)
    while True:
        left = (deadline - time.time()) / 60
        if left < 3:
            return
        j = take()
        if j is None:
            return
        if j == "wait":
            time.sleep(20); continue
        out = os.path.join(a.outdir, j["name"])
        cmd = [sys.executable, "bitloop_v1.py", *a.common.split(), *j["args"], "--out", out, "--max_minutes", f"{left - 2:.1f}"]
        with open(out + ".log", "a") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        status[j["name"]] = {0: "done", 3: "paused at time budget"}.get(rc, f"FAILED rc={rc}")
        with lock:
            running.discard(j["name"])
        print(f"[{(time.time() - t0) / 60:6.1f} min] worker {w}: {j['name']} -> {status[j['name']]}", flush=True)


ts = [threading.Thread(target=work, args=(w,)) for w in range(workers)]
[t.start() for t in ts]; [t.join() for t in ts]
done = sum(os.path.exists(os.path.join(a.outdir, j["name"] + ".json")) for j in jobs)
print(f"queue state: {done}/{len(jobs)} complete. Re-run this cell in a new session to continue." if done < len(jobs) else "ALL JOBS COMPLETE")
json.dump(status, open(os.path.join(a.outdir, "_last_session.json"), "w"), indent=1)
