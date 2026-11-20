"""Correctness tests for bitloop_v1. Run: python3 test_v1.py <train_head.csv> <v0 bitloop dir>"""
import sys, os, json, subprocess, numpy as np, torch
import bitloop_v1 as B

csv, v0dir = sys.argv[1], sys.argv[2]
torch.manual_seed(0)

# 1. CSV loader: shapes, value range, puzzles agree with solutions on clues, solutions are valid grids
P, S, R = B.read_csv(csv, limit=5000)
assert len(R) == 5000 and R.min() >= 0
assert P.shape == (5000, 81) and S.shape == (5000, 81) and P.max() <= 9 and S.min() >= 1 and S.max() <= 9
assert ((P == 0) | (P == S)).all()
Pt, St = torch.tensor(P).long(), torch.tensor(S).long()
assert B.valid_solution(St, 9).all()
D0 = B.load_sudoku9(csv, csv, 3000, 500, 500, max_rating=0)
assert (D0["va_r"] == 0).all() and (D0["te_r"] == 0).all() and len(D0["tr"][0]) > 1000
D1 = B.load_sudoku9(csv, csv, 3000, 500, 500, min_rating=1, max_rating=9)
assert (D1["va_r"] >= 1).all() and (D1["va_r"] <= 9).all()
te = {x.numpy().tobytes() for x in D0["te"][0]}; assert not any(x.numpy().tobytes() in te for x in D0["tr"][0])  # same file both sides: leak guard must hold
print("1 loader ok")

# 2. Augmentation preserves validity, clue consistency and clue count; and actually changes the grids
g = torch.Generator().manual_seed(1)
p2, s2 = B.augment(Pt, St, 9, g)
assert B.valid_solution(s2, 9).all() and ((p2 == 0) | (p2 == s2)).all()
assert ((p2 == 0).sum(1) == (Pt == 0).sum(1)).all() and (s2 != St).any(1).float().mean() > 0.99
D6 = B.load_sudoku6(os.path.join(v0dir, "data.npz"))
p6, s6 = B.augment(*D6["va"], 6, g)
assert B.valid_solution(s6, 6).all() and ((p6 == 0) | (p6 == s6)).all()
print("2 augmentation ok")

# 3. v1 forward equals v0 forward given identical weights (all three arms), token-scaled spikes
sys.path.insert(0, v0dir); import bitloop as V0
for mode in ["fp", "tern", "spike"]:
    torch.manual_seed(3)
    m0 = V0.BitLoop(64, mode, nb=2)
    m1 = B.BitLoop(36, 7, 64, mode, 2, 2, (4, "token", 32)); m1.load_state_dict(m0.state_dict())
    x = D6["va"][0][:64]
    o0, o1 = m0(x, 4)[-1], m1(x, 4)[-1]
    assert torch.allclose(o0, o1, atol=1e-5), (mode, (o0 - o1).abs().max())
    # gradients agree too
    l0, l1 = o0.square().mean(), o1.square().mean(); l0.backward(); l1.backward()
    g0 = m0.net[0].ch.a.w.grad; g1 = m1.net[0].ch.a.w.grad
    assert torch.allclose(g0, g1, atol=1e-6, rtol=1e-3), (mode, (g0 - g1).abs().max())
print("3 v0/v1 equivalence ok")

# 4. spike(): integer counts in range for every granularity; group handles a ragged last block
for scale in ["token", "group", "tensor"]:
    x = torch.randn(4, 9, 70)
    c, th = B.spike(x, 4, scale, 32)
    assert (c == c.round()).all() and c.min() >= 0 and c.max() <= 4 and th.shape[-1] in (1, 70)
x = torch.randn(2, 3, 70); _, th = B.spike(x, 4, "group", 32)
assert torch.allclose(th[..., 64:], x.abs()[..., 64:].mean(-1, keepdim=True).expand(-1, -1, 6), atol=1e-6)
for scale in ["token", "group", "tensor"]:
    m = B.BitLoop(81, 10, 48, "spike", 2, 2, (4, scale, 32)); o = m(Pt[:8], 2)[-1]
    assert o.shape == (8, 81, 9) and torch.isfinite(o).all()
    r = B.evaluate(m, Pt[:50], St[:50], [2], torch.device("cpu"), count=True)[2]
    assert r["syn"] > 0 and 0 < r["act_sparsity"] < 1 and r["mac"] == 0 and r["add"] == 0
print("4 spike ok")

# 5. Op counts match closed forms for the fp arm
m = B.BitLoop(36, 7, 64, "fp", 2, 2, (4, "token", 32))
r = B.evaluate(m, *[t[:100] for t in D6["va"]], [4], torch.device("cpu"), count=True)[4]
per_pass = 2 * (64 * (36 * 72 * 2) + 36 * (64 * 128 * 2))
assert r["mac"] == per_pass * 3 * 4, (r["mac"], per_pass * 12)
print("5 op count ok")

# 6. Checkpoint/resume reproduces an uninterrupted run exactly
def run(out, extra=()):
    return subprocess.run([sys.executable, "bitloop_v1.py", "--task", "sudoku6", "--data", os.path.join(v0dir, "data.npz"),
                           "--d", "32", "--steps", "40", "--bs", "32", "--log_every", "20", "--ckpt_every", "20",
                           "--threads", "1", "--device", "cpu", "--mode", "spike", "--out", out, *extra], capture_output=True, text=True)
for f in ["/tmp/ra.json", "/tmp/rb.json", "/tmp/rb.ckpt", "/tmp/ra.pt", "/tmp/rb.pt"]:
    if os.path.exists(f): os.remove(f)
a = run("/tmp/ra"); assert a.returncode == 0, a.stderr[-800:]
b1 = run("/tmp/rb", ["--max_minutes", "0.000001"]); assert b1.returncode == 3, (b1.returncode, b1.stderr[-800:])
b2 = run("/tmp/rb"); assert b2.returncode == 0 and "resumed" in b2.stdout, b2.stderr[-800:]
ja, jb = json.load(open("/tmp/ra.json")), json.load(open("/tmp/rb.json"))
assert ja["final"] == jb["final"] and [l[:4] for l in ja["log"]] == [l[:4] for l in jb["log"]], "resume differs"
sa, sb = torch.load("/tmp/ra.pt"), torch.load("/tmp/rb.pt")
assert all(torch.equal(sa[k], sb[k]) for k in sa)
print("6 resume ok")
print("ALL TESTS PASSED")
