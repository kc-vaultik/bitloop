"""BitLoop v1: recursive reasoning core with full-precision / ternary / ternary+spiking arms.

Runs on CPU or one CUDA GPU. Resumable (checkpoint every --ckpt_every steps), so it survives
free-tier session limits. One invocation = one run; run_queue.py drives many.

Tasks
  sudoku6 : 6x6 Sudoku from a .npz made by gen_data.py (v0 data)
  sudoku9 : Sudoku-Extreme CSVs (source,question,answer,rating) from HF sapientinc/sudoku-extreme,
            used in a data-rich regime (large train subsample), NOT the 1k-example protocol.

Arms
  fp    : float weights, GELU
  tern  : ternary core weights {-1,0,+1} x scale (absmean, STE), GELU
  spike : ternary core weights + integer spike counts in [0, smax]; threshold granularity
          --spike_scale token | group | tensor
Embeddings, RMS norms and the output head are full precision in every arm.
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


# ----------------------------------------------------------------------------- op counting
class Counter:
    def __init__(self):
        self.on = False; self.reset()

    def reset(self):
        self.mac = self.add = self.syn = 0.0
        self.act_n = self.act_nz = self.w_n = self.w_nz = 0.0


CNT = Counter()


# ----------------------------------------------------------------------------- layers
class QLinear(nn.Module):
    def __init__(self, i, o, mode):
        super().__init__()
        self.w = nn.Parameter(torch.empty(o, i)); nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        self.mode = mode; self.wq = None

    def refresh(self):
        """Quantize once per optimizer step; the same tensor is reused by every recursion pass."""
        if self.mode == "fp":
            self.wq = self.w
        else:
            s = self.w.abs().mean().clamp(min=1e-5)
            q = (self.w / s).round().clamp(-1, 1) * s
            self.wq = self.w + (q - self.w).detach()

    def forward(self, x, x_is_spike=False, counts=None):
        w = self.wq
        if CNT.on:
            B = x.shape[0]; ntok = x[0].numel() / x.shape[-1]
            if self.mode == "fp":
                CNT.mac += ntok * w.numel()
            else:
                nzw = (w != 0).float()
                CNT.w_n += w.numel(); CNT.w_nz += nzw.sum().item()
                if x_is_spike:
                    c = (x if counts is None else counts).detach().float()   # events are integer spike counts
                    CNT.syn += (c * nzw.sum(0)).sum().item() / B
                    CNT.act_n += c[0].numel(); CNT.act_nz += (c != 0).float().sum().item() / B
                else:
                    CNT.add += ntok * nzw.sum().item()
        return F.linear(x, w)


def spike(x, smax, scale, group):
    """Integer spike counts in [0, smax] with an adaptive threshold; returns (counts, threshold)."""
    a = x.abs()
    if scale == "token":
        th = a.mean(-1, keepdim=True)
    elif scale == "tensor":
        th = a.mean(tuple(range(1, x.dim())), keepdim=True)
    else:  # group: blocks of `group` channels share a threshold
        n = x.shape[-1]; g = min(group, n); pad = (-n) % g
        ap = F.pad(a, (0, pad)) if pad else a
        th = ap.reshape(*x.shape[:-1], -1, g).mean(-1, keepdim=True).expand(*x.shape[:-1], -1, g).reshape(*x.shape[:-1], -1)[..., :n]
        if pad:  # correct the mean of the last, partially filled block
            th = th.clone(); th[..., n - (g - pad):] = th[..., n - (g - pad):] * g / (g - pad)
    th = th.clamp(min=1e-5)
    u = (F.relu(x) / th).clamp(max=smax)
    return u + (u.round() - u).detach(), th


class MLP(nn.Module):
    def __init__(self, d, h, mode, cfg):
        super().__init__()
        self.a, self.b, self.mode, self.cfg = QLinear(d, h, mode), QLinear(h, d, mode), mode, cfg

    def _lin(self, lin, x):
        c, th = spike(x, *self.cfg)
        if th.shape[-1] == 1:          # one threshold per token or tensor: factor it out of the accumulation
            return lin(c, True) * th
        return lin(c * th, True, counts=c)  # per-group thresholds act as per-input synaptic scales

    def forward(self, x):
        if self.mode == "spike":
            return self._lin(self.b, self._lin(self.a, x))
        return self.b(F.gelu(self.a(x)))


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Block(nn.Module):
    def __init__(self, d, L, mode, cfg):
        super().__init__()
        self.tok = MLP(L, 2 * L, mode, cfg)   # mixes across cells, per channel
        self.ch = MLP(d, 2 * d, mode, cfg)    # mixes across channels, per cell

    def forward(self, x):
        x = x + self.tok(rms(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ch(rms(x))


class BitLoop(nn.Module):
    def __init__(self, L, V, d, mode, nb, n_lat, cfg):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.y0 = nn.Parameter(torch.randn(d) * 0.02); self.z0 = nn.Parameter(torch.randn(d) * 0.02)
        self.net = nn.Sequential(*[Block(d, L, mode, cfg) for _ in range(nb)])
        self.head = nn.Linear(d, V - 1)
        self.n_lat = n_lat

    def forward(self, puz, K):
        for m in self.modules():
            if isinstance(m, QLinear):
                m.refresh()
        x = self.emb(puz) + self.pos
        y = self.y0.expand_as(x); z = self.z0.expand_as(x)
        outs = []
        for _ in range(K):
            for _ in range(self.n_lat):
                z = self.net(x + y + z)
            y = self.net(y + z)
            outs.append(self.head(rms(y)))
            y, z = y.detach(), z.detach()   # deep supervision: no gradient across steps
        return outs


# ----------------------------------------------------------------------------- data
def load_sudoku6(path):
    D = np.load(path)
    g = lambda k: torch.tensor(D[k]).long()
    return dict(side=6, tr=(g("Ptr"), g("Str")), va=(g("Pva"), g("Sva")), te=(g("Pte"), g("Ste")))


def read_csv(path, limit=None):
    """Sudoku-Extreme CSV -> (puzzles, solutions) uint8 [N, 81]; '.' or '0' = blank. Skips malformed rows."""
    P, S, R = [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split(",")
            if len(c) < 3 or len(c[1]) != 81 or len(c[2]) != 81:
                continue
            P.append(c[1]); S.append(c[2]); R.append(int(c[3]) if len(c) > 3 and c[3].strip().lstrip("-").isdigit() else -1)
            if limit and len(P) >= limit:
                break
    conv = lambda rows: (np.frombuffer("".join(rows).replace(".", "0").encode(), dtype=np.uint8) - 48).reshape(-1, 81)
    return conv(P), conv(S), np.array(R, dtype=np.int64)


def load_sudoku9(train_csv, test_csv, n_train, n_val, n_test, data_seed=0, min_rating=0, max_rating=10 ** 9):
    Ptr, Str, Rtr = read_csv(train_csv); Pte, Ste, Rte = read_csv(test_csv)
    ktr = (Rtr >= min_rating) & (Rtr <= max_rating); kte = (Rte >= min_rating) & (Rte <= max_rating)   # difficulty band
    Ptr, Str, Rtr, Pte, Ste, Rte = Ptr[ktr], Str[ktr], Rtr[ktr], Pte[kte], Ste[kte], Rte[kte]
    r = np.random.RandomState(data_seed)
    i = r.permutation(len(Ptr)); j = r.permutation(len(Pte))[:n_test]
    n_train = min(n_train, len(Ptr) - n_val)
    tr, va = i[:n_train], i[n_train:n_train + n_val]
    # leakage check on puzzles: drop any train/val puzzle that also appears in the test subsample
    te_keys = {Pte[k].tobytes() for k in j}
    keep = lambda idx: np.array([k for k in idx if Ptr[k].tobytes() not in te_keys])
    tr2, va2 = keep(tr), keep(va)
    t = lambda a: torch.tensor(a).long()
    return dict(side=9, tr=(t(Ptr[tr2]), t(Str[tr2])), va=(t(Ptr[va2]), t(Str[va2])), te=(t(Pte[j]), t(Ste[j])), va_r=t(Rtr[va2]), te_r=t(Rte[j]),
                dropped=int(len(tr) - len(tr2) + len(va) - len(va2)), n_train_pool=int(len(Ptr)), n_test_pool=int(len(Pte)))


def _line_perm(B, br, nband, g, dev):
    """Random validity-preserving permutation of 9 (or 6) lines: shuffle bands, then lines inside each band."""
    band = torch.rand(B, nband, generator=g, device=dev).argsort(1)                 # [B, nband]
    inner = torch.rand(B, nband, br, generator=g, device=dev).argsort(2)            # [B, nband, br]
    return (band.unsqueeze(2) * br + inner).reshape(B, nband * br)


def augment(p, s, side, g, geometric=True):
    """Digit relabeling (+ band/stack line shuffles and transpose on square-box grids). Train batches only."""
    B, dev = len(p), p.device
    perm = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev),
                      torch.rand(B, side, generator=g, device=dev).argsort(1) + 1], 1)
    p, s = perm.gather(1, p), perm.gather(1, s)
    if geometric and side == 9:
        rp = _line_perm(B, 3, 3, g, dev); cp = _line_perm(B, 3, 3, g, dev)
        idx = (rp.unsqueeze(2) * 9 + cp.unsqueeze(1)).reshape(B, 81)
        p, s = p.gather(1, idx), s.gather(1, idx)
        tflip = torch.rand(B, generator=g, device=dev) < 0.5
        tr = torch.arange(81, device=dev).reshape(9, 9).t().reshape(81)
        p = torch.where(tflip.unsqueeze(1), p[:, tr], p); s = torch.where(tflip.unsqueeze(1), s[:, tr], s)
    return p, s


def valid_solution(s, side):
    """True per row if every row, column and box of the grid holds each digit once (used by tests)."""
    br, bc = (3, 3) if side == 9 else (2, 3)
    g = s.reshape(-1, side, side); want = torch.arange(1, side + 1, device=s.device)
    ok = (g.sort(2).values == want).all(2).all(1) & (g.sort(1).values == want.unsqueeze(1)).all(1).all(1)
    boxes = g.reshape(-1, side // br, br, side // bc, bc).permute(0, 1, 3, 2, 4).reshape(-1, side, side)
    return ok & (boxes.sort(2).values == want).all(2).all(1)


# ----------------------------------------------------------------------------- eval
RBINS = [(0, 0), (1, 9), (10, 49), (50, 10 ** 9)]


@torch.no_grad()
def evaluate(model, P, S, Ks, dev, count=False, bs=1000, ratings=None):
    model.eval(); res = {}
    for K in Ks:
        CNT.reset(); ex = cell = n = 0; solved = []
        for i in range(0, len(P), bs):
            p, s = P[i:i + bs].to(dev), S[i:i + bs].to(dev)
            CNT.on = count and i == 0          # ops are counted on the first batch
            pred = model(p, K)[-1].argmax(-1) + 1
            CNT.on = False
            blank = p == 0
            okp = ((pred == s) | ~blank).all(1); solved.append(okp.cpu()); ex += okp.sum().item()
            cell += (pred == s)[blank].float().sum().item(); n += blank.sum().item()
        r = {"exact": ex / len(P), "cell": cell / n}
        if ratings is not None:
            sv = torch.cat(solved).float()
            r["by_rating"] = {f"{lo}-{hi if hi < 10 ** 9 else 'up'}": [int(((ratings >= lo) & (ratings <= hi)).sum()),
                              float(sv[(ratings >= lo) & (ratings <= hi)].mean()) if ((ratings >= lo) & (ratings <= hi)).any() else None]
                              for lo, hi in RBINS}
        if count:
            r.update(mac=CNT.mac, add=CNT.add, syn=CNT.syn,
                     act_sparsity=1 - CNT.act_nz / CNT.act_n if CNT.act_n else None,
                     w_sparsity=1 - CNT.w_nz / CNT.w_n if CNT.w_n else None)
        res[K] = r
    return res


class EMA:
    def __init__(self, model, decay):
        self.decay = decay; self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def swap(self, model):
        cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow); return cur


# ----------------------------------------------------------------------------- main
def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sudoku9"); ap.add_argument("--data", default="data.npz")
    ap.add_argument("--train_csv", default="train.csv"); ap.add_argument("--test_csv", default="test.csv")
    ap.add_argument("--n_train", type=int, default=1000000); ap.add_argument("--n_val", type=int, default=5000)
    ap.add_argument("--n_test", type=int, default=20000)
    ap.add_argument("--min_rating", type=int, default=0); ap.add_argument("--max_rating", type=int, default=10 ** 9)
    ap.add_argument("--mode", default="fp"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--d", type=int, default=192); ap.add_argument("--nb", type=int, default=2)
    ap.add_argument("--n_lat", type=int, default=2); ap.add_argument("--ktrain", type=int, default=4)
    ap.add_argument("--smax", type=int, default=4); ap.add_argument("--spike_scale", default="token")
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--eval_split", default="va"); ap.add_argument("--eval_ks", default="1,2,4,8,16")
    ap.add_argument("--log_every", type=int, default=500); ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--out", default="run"); ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--max_minutes", type=float, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    a = get_args(argv)
    t_start = time.time()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    done_path, ck_path = a.out + ".json", a.out + ".ckpt"
    if os.path.exists(done_path):
        print("already done:", done_path); return 0
    D = load_sudoku6(a.data) if a.task == "sudoku6" else load_sudoku9(a.train_csv, a.test_csv, a.n_train, a.n_val, a.n_test, min_rating=a.min_rating, max_rating=a.max_rating)
    side = D["side"]; L, V = side * side, side + 1
    Ptr, Str = D["tr"]; Pva, Sva = D["va"]; Pev, Sev = D[a.eval_split]
    torch.manual_seed(a.seed)
    model = BitLoop(L, V, a.d, a.mode, a.nb, a.n_lat, (a.smax, a.spike_scale, a.group)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    ema = EMA(model, a.ema) if a.ema > 0 else None
    g_cpu = torch.Generator().manual_seed(a.seed); g_dev = torch.Generator(device=dev).manual_seed(a.seed)
    step, log, elapsed = 0, [], 0.0
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        g_cpu.set_state(ck["g_cpu"].cpu()); g_dev.set_state(ck["g_dev"].cpu())   # generator states are CPU byte tensors, even for a CUDA generator
        step, log, elapsed = ck["step"], ck["log"], ck["elapsed"]
        if ema: ema.shadow = ck["ema"]
        print(f"resumed {a.out} at step {step}", flush=True)

    def save():
        tmp = ck_path + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), g_cpu=g_cpu.get_state(),
                        g_dev=g_dev.get_state(), step=step, log=log, elapsed=elapsed + time.time() - t_start,
                        ema=ema.shadow if ema else None), tmp)
        os.replace(tmp, ck_path)

    Ks = [int(k) for k in a.eval_ks.split(",")]
    while step < a.steps:
        model.train(); step += 1
        idx = torch.randint(0, len(Ptr), (a.bs,), generator=g_cpu)
        p, s = augment(Ptr[idx].to(dev), Str[idx].to(dev), side, g_dev)
        outs = model(p, a.ktrain)
        loss = sum(F.cross_entropy(o.reshape(-1, V - 1), (s - 1).reshape(-1)) for o in outs) / len(outs)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if ema: ema.update(model)
        if not math.isfinite(loss.item()):
            json.dump(dict(args=vars(a), diverged_at=step, log=log), open(done_path, "w")); print("diverged"); return 0
        if step % a.log_every == 0 or step == a.steps:
            cur = ema.swap(model) if ema else None
            r = evaluate(model, Pva[:1000], Sva[:1000], [a.ktrain], dev)[a.ktrain]
            if cur: model.load_state_dict(cur)
            t = elapsed + time.time() - t_start
            log.append((step, loss.item(), r["exact"], r["cell"], t))
            print(f"{a.mode} s{a.seed} {step} loss {loss.item():.3f} val exact {r['exact']:.3f} cell {r['cell']:.3f} t {t:.0f}s", flush=True)
        if step % a.ckpt_every == 0 and step < a.steps:
            save()
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                print("time budget reached; checkpointed"); return 3
    if ema: ema.swap(model)
    final = evaluate(model, Pev, Sev, Ks, dev, count=True, ratings=D.get(a.eval_split + "_r"))
    out = dict(args=vars(a), device=str(dev), params=sum(p.numel() for p in model.parameters()),
               core_params=sum(p.numel() for p in model.net.parameters()), n_train=len(Ptr), n_eval=len(Pev),
               dropped_for_leakage=D.get("dropped", 0), log=log, final=final, train_s=elapsed + time.time() - t_start,
               torch=torch.__version__)
    torch.save(model.state_dict(), a.out + ".pt")
    json.dump(out, open(done_path, "w"))
    if os.path.exists(ck_path): os.remove(ck_path)
    print(json.dumps({k: round(v["exact"], 4) for k, v in final.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
