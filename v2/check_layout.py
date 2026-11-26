"""Is the KV region organised in 8-page (32 KB) chunks? Uses a full-page trace (old slicer format, idx/off).
For each alignment offset o, chunk completeness = fraction of accessed pages whose whole chunk (p-o)//8 was accessed
in the same slice. A clear winner near 1.0 = chunks of 8 contiguous pages accessed together."""
import sys, numpy as np
d = np.load(sys.argv[1]); off, idx = d["off"], d["idx"]
n = len(off) - 1
sel = np.linspace(0, n - 1, min(n, 200)).astype(int)
for size in (2, 4, 8, 16, 32):
    best = []
    for o in range(size):
        comp = []
        for i in sel:
            p = idx[off[i]:off[i + 1]]
            if p.size < 100: continue
            g = (p.astype(np.int64) - o) // size
            u, c = np.unique(g, return_counts=True)
            comp.append((c == size).sum() * size / p.size)
        best.append(np.mean(comp))
    print(f"chunk size {size:2d}: completeness by offset {np.round(best, 3).tolist()[:8]}{'...' if size > 8 else ''} max {max(best):.3f} at offset {int(np.argmax(best))}")
