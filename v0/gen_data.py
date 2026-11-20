"""6x6 Sudoku generator (2x3 boxes). Unique-solution puzzles, split by solution grid."""
import random, sys, numpy as np
from multiprocessing import Pool

N = 6
BR, BC = 2, 3  # box rows, box cols
PEERS = []
for i in range(36):
    r, c = divmod(i, N)
    p = set()
    for k in range(N):
        p.add(r * N + k); p.add(k * N + c)
    br, bc = r // BR * BR, c // BC * BC
    for a in range(BR):
        for b in range(BC):
            p.add((br + a) * N + bc + b)
    p.discard(i)
    PEERS.append(tuple(p))


def cands(g, i):
    used = {g[j] for j in PEERS[i]}
    return [d for d in range(1, N + 1) if d not in used]


def fill(g, rng):
    # most-constrained-cell backtracking with random digit order
    best, bc = -1, None
    for i in range(36):
        if g[i] == 0:
            c = cands(g, i)
            if bc is None or len(c) < len(bc):
                best, bc = i, c
                if len(c) <= 1:
                    break
    if best < 0:
        return True
    rng.shuffle(bc)
    for d in bc:
        g[best] = d
        if fill(g, rng):
            return True
    g[best] = 0
    return False


def count(g, limit=2):
    best, bc = -1, None
    for i in range(36):
        if g[i] == 0:
            c = cands(g, i)
            if bc is None or len(c) < len(bc):
                best, bc = i, c
                if len(c) <= 1:
                    break
    if best < 0:
        return 1
    n = 0
    for d in bc:
        g[best] = d
        n += count(g, limit - n)
        if n >= limit:
            break
    g[best] = 0
    return n


def make(seed):
    rng = random.Random(seed)
    sol = [0] * 36
    fill(sol, rng)
    puz = sol[:]
    order = list(range(36)); rng.shuffle(order)
    for i in order:  # dig to a locally minimal unique puzzle
        keep = puz[i]; puz[i] = 0
        if count(puz[:]) != 1:
            puz[i] = keep
    return puz, sol


if __name__ == "__main__":
    n = int(sys.argv[1]); out = sys.argv[2]
    with Pool(2) as p:
        res = p.map(make, range(n), chunksize=64)
    # dedupe by solution grid, then split by solution so no grid crosses splits
    seen, P, S = set(), [], []
    for puz, sol in res:
        k = tuple(sol)
        if k in seen:
            continue
        seen.add(k); P.append(puz); S.append(sol)
    P, S = np.array(P, dtype=np.int8), np.array(S, dtype=np.int8)
    n = len(P); nte = 2000; nva = 1000
    np.savez(out, Ptr=P[: n - nte - nva], Str=S[: n - nte - nva],
             Pva=P[n - nte - nva: n - nte], Sva=S[n - nte - nva: n - nte],
             Pte=P[n - nte:], Ste=S[n - nte:])
    print("unique grids", n, "mean clues", (P > 0).sum(1).mean(), "min", (P > 0).sum(1).min(), "max", (P > 0).sum(1).max())
