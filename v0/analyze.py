import json, glob, numpy as np, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

Ks = ["1", "2", "4", "8", "16"]
R = {}
for f in sorted(glob.glob("runs/*.json")):
    d = json.load(open(f)); a = d["args"]
    name = a["mode"] + (f"_k{a['ktrain']}" if a["ktrain"] != 4 else "")
    R.setdefault(name, []).append(d)

# Energy proxy, 45 nm (Horowitz 2014): 32-bit float mult 3.7 pJ, float add 0.9 pJ, 32-bit int add 0.1 pJ
E_MAC, E_FADD, E_IADD = 4.6, 0.9, 0.1
out = {}
print(f"{'arm':10s} n  " + "  ".join(f"K={k:>2s}" for k in Ks) + "   (exact-puzzle accuracy, mean +- sd over seeds)")
for name, runs in R.items():
    acc = np.array([[r["final"][k]["exact"] for k in Ks] for r in runs])
    f4 = [r["final"]["4"] for r in runs]
    ops = dict(mac=np.mean([x["mac"] for x in f4]), add=np.mean([x["add"] for x in f4]), syn=np.mean([x["syn"] for x in f4]))
    pj = ops["mac"] * E_MAC + ops["add"] * E_FADD + ops["syn"] * E_IADD
    asp = [x["act_sparsity"] for x in f4 if x["act_sparsity"] is not None]
    wsp = [x["w_sparsity"] for x in f4 if x["w_sparsity"] is not None]
    out[name] = dict(n=len(runs), mean=acc.mean(0).tolist(), sd=acc.std(0, ddof=1).tolist() if len(runs) > 1 else None,
                     per_seed_k4=acc[:, 2].tolist(), ops_k4=ops, uJ_k4=pj / 1e6,
                     act_sparsity=float(np.mean(asp)) if asp else None, w_sparsity=float(np.mean(wsp)) if wsp else None,
                     cell_k4=float(np.mean([x["cell"] for x in f4])), core_params=runs[0]["core_params"], params=runs[0]["params"])
    sd = acc.std(0, ddof=1) if len(runs) > 1 else np.zeros(5)
    print(f"{name:10s} {len(runs)}  " + "  ".join(f"{m*100:4.1f}+-{s*100:3.1f}" for m, s in zip(acc.mean(0), sd)))
print()
for name, o in out.items():
    print(name, "K=4 per-seed", [round(x * 100, 1) for x in o["per_seed_k4"]], "cell", round(o["cell_k4"] * 100, 1),
          "| ops", {k: f"{v:.3g}" for k, v in o["ops_k4"].items()}, "| proxy uJ", round(o["uJ_k4"], 3),
          "| act sparsity", o["act_sparsity"], "| w sparsity", o["w_sparsity"])
json.dump(out, open("results.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
col = {"fp": "#444", "tern": "#1f77b4", "spike": "#d62728", "tern_k8": "#17becf", "spike_k8": "#ff7f0e"}
for name, o in out.items():
    m = np.array(o["mean"]) * 100; s = np.array(o["sd"] or [0] * 5) * 100
    ax[0].errorbar([1, 2, 4, 8, 16], m, yerr=s, marker="o", label=name, color=col.get(name), capsize=3)
    r = R[name]
    e = [np.mean([x["final"][k]["mac"] * E_MAC + x["final"][k]["add"] * E_FADD + x["final"][k]["syn"] * E_IADD for x in r]) / 1e6 for k in Ks]
    ax[1].plot(e, m, marker="o", label=name, color=col.get(name))
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("recursion steps K at test time"); ax[0].set_ylabel("exact-puzzle accuracy (%)"); ax[0].legend(); ax[0].set_title("Accuracy vs recursion")
ax[1].set_xscale("log"); ax[1].set_xlabel("core energy proxy per puzzle (uJ, 45 nm op costs)"); ax[1].set_ylabel("exact-puzzle accuracy (%)"); ax[1].set_title("Accuracy vs energy proxy")
plt.tight_layout(); plt.savefig("fig1.png", dpi=150)
