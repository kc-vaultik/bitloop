"""BitLoop v0: tiny recursive model with full-precision / ternary / ternary+spiking arms.

Arms
  fp     : float weights, float activations (GELU)
  tern   : ternary weights {-1,0,+1}*scale (absmean, STE), float activations (GELU)
  spike  : ternary weights + integer spike-count activations in [0, SMAX]
           (adaptive threshold = mean |pre-activation| per token, STE)
Embeddings, norms and the output head stay full precision in every arm (stated limitation).
"""
import argparse, json, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

L, V = 36, 7
SMAX = 4


class Counter:
    """Accumulates operation counts per sample for one forward pass."""
    def __init__(self):
        self.on = False; self.reset()

    def reset(self):
        self.mac = 0.0      # float multiply-accumulates (fp arm)
        self.add = 0.0      # float add/sub through nonzero ternary weights, dense activations
        self.syn = 0.0      # integer synaptic events: spikes x nonzero fan-out
        self.act_n = 0.0; self.act_nz = 0.0; self.w_n = 0.0; self.w_nz = 0.0


CNT = Counter()


class QLinear(nn.Module):
    def __init__(self, i, o, mode):
        super().__init__()
        self.w = nn.Parameter(torch.empty(o, i)); nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        self.mode = mode

    def qweight(self):
        if self.mode == "fp":
            return self.w
        s = self.w.abs().mean().clamp(min=1e-5)
        q = (self.w / s).round().clamp(-1, 1) * s
        return self.w + (q - self.w).detach()

    def forward(self, x, x_is_spike=False):
        w = self.qweight()
        if CNT.on:
            B = x.shape[0]
            ntok = x[0].numel() / x.shape[-1]
            if self.mode == "fp":
                CNT.mac += ntok * w.numel()
            else:
                nzw = (w != 0).float()
                CNT.w_n += w.numel(); CNT.w_nz += nzw.sum().item()
                if x_is_spike:
                    fan = nzw.sum(0)  # nonzero fan-out of each input unit
                    CNT.syn += (x.detach() * fan).sum().item() / B
                    CNT.act_n += x[0].numel(); CNT.act_nz += (x != 0).float().sum().item() / B
                else:
                    CNT.add += ntok * nzw.sum().item()
        return F.linear(x, w)


def spike(x):
    """Adaptive-threshold integer spike counts in [0, SMAX]; returns (counts, per-token scale)."""
    th = x.abs().mean(-1, keepdim=True).clamp(min=1e-5)
    u = F.relu(x) / th
    c = u.round().clamp(0, SMAX)
    c = u.clamp(0, SMAX) + (c - u.clamp(0, SMAX)).detach()  # STE
    return c, th


class MLP(nn.Module):
    def __init__(self, d, h, mode):
        super().__init__()
        self.a, self.b, self.mode = QLinear(d, h, mode), QLinear(h, d, mode), mode

    def forward(self, x):
        if self.mode == "spike":
            c, th = spike(x); hmid = self.a(c, True) * th
            c2, th2 = spike(hmid); return self.b(c2, True) * th2
        return self.b(F.gelu(self.a(x)))


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Block(nn.Module):
    def __init__(self, d, mode):
        super().__init__()
        self.tok = MLP(L, 2 * L, mode)   # mixes across the 36 cells, per channel
        self.ch = MLP(d, 2 * d, mode)    # mixes across channels, per cell

    def forward(self, x):
        x = x + self.tok(rms(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ch(rms(x))


class BitLoop(nn.Module):
    def __init__(self, d=96, mode="fp", n_lat=2, nb=2):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.y0 = nn.Parameter(torch.randn(d) * 0.02); self.z0 = nn.Parameter(torch.randn(d) * 0.02)
        self.net = nn.Sequential(*[Block(d, mode) for _ in range(nb)])
        self.head = nn.Linear(d, V - 1)
        self.n_lat = n_lat

    def forward(self, puz, K):
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


def augment(p, s, g):
    """Digit relabeling: a validity-preserving symmetry, applied to train batches only."""
    perm = torch.stack([torch.cat([torch.zeros(1, dtype=torch.long), torch.randperm(6, generator=g) + 1]) for _ in range(len(p))])
    return perm.gather(1, p), perm.gather(1, s)


@torch.no_grad()
def evaluate(model, P, S, Ks, count=False):
    model.eval(); res = {}
    for K in Ks:
        CNT.reset(); CNT.on = count
        ex = cell = n = 0
        for i in range(0, len(P), 500):
            p, s = P[i:i + 500], S[i:i + 500]
            if count and i > 0:
                CNT.on = False  # ops are counted on the first 500 test puzzles
            pred = model(p, K)[-1].argmax(-1) + 1
            blank = p == 0
            ok = (pred == s) | ~blank
            ex += ok.all(1).sum().item(); cell += (pred == s)[blank].float().sum().item(); n += blank.sum().item()
        r = {"exact": ex / len(P), "cell": cell / n}
        if count:
            r.update(mac=CNT.mac, add=CNT.add, syn=CNT.syn,
                     act_sparsity=1 - CNT.act_nz / CNT.act_n if CNT.act_n else None,
                     w_sparsity=1 - CNT.w_nz / CNT.w_n if CNT.w_n else None)
        CNT.on = False
        res[K] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fp"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--d", type=int, default=96); ap.add_argument("--ktrain", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--nb", type=int, default=1); ap.add_argument("--split", default="va"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); g = torch.Generator().manual_seed(a.seed)
    D = np.load("data.npz")
    Ptr, Str = torch.tensor(D["Ptr"]).long(), torch.tensor(D["Str"]).long()
    Pev, Sev = torch.tensor(D["P" + a.split]).long(), torch.tensor(D["S" + a.split]).long()
    Pva, Sva = torch.tensor(D["Pva"]).long()[:500], torch.tensor(D["Sva"]).long()[:500]  # progress log only
    model = BitLoop(a.d, a.mode, nb=a.nb)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    t0 = time.time(); log = []
    for step in range(1, a.steps + 1):
        model.train()
        idx = torch.randint(0, len(Ptr), (a.bs,), generator=g)
        p, s = augment(Ptr[idx], Str[idx], g)
        outs = model(p, a.ktrain)
        loss = sum(F.cross_entropy(o.reshape(-1, V - 1), (s - 1).reshape(-1)) for o in outs) / len(outs)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps:
            r = evaluate(model, Pva, Sva, [a.ktrain])[a.ktrain]
            log.append((step, loss.item(), r["exact"], time.time() - t0))
            print(a.mode, a.seed, step, f"loss {loss.item():.3f} exact {r['exact']:.3f} cell {r['cell']:.3f} t {time.time()-t0:.0f}s", flush=True)
    final = evaluate(model, Pev, Sev, [1, 2, 4, 8, 16], count=True)
    out = dict(args=vars(a), params=sum(p.numel() for p in model.parameters()),
               core_params=sum(p.numel() for p in model.net.parameters()), log=log, final=final, train_s=time.time() - t0)
    if a.out:
        json.dump(out, open(a.out, "w")); torch.save(model.state_dict(), a.out.replace(".json", ".pt"))
    print(json.dumps(final))


if __name__ == "__main__":
    main()
