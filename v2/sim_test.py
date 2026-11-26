"""Synthetic check of analyze2.py: build replay/trace files whose ground truth is known, then analyze them.
  sim 'shared'  : a request's prefill touches its app's system-prompt pages + its user's history pages
                  (+ fresh pages); decode touches the same pages. Expect same-user and same-app AUROC >> 0.5,
                  next-turn accuracy >> chance.
  sim 'random'  : every request touches fresh random pages. Expect AUROC ~ 0.5, next-turn ~ chance.
  Network: event sizes depend on app in both sims (unpadded app AUROC > 0.5); long/short from token count.
  Probes: hit -> lower TTFT in 'shared', identical distribution in 'random'.
Usage: python sim_test.py <out_dir>
"""
import json, os, sys
import numpy as np

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
KV = 50000
APPS = ["tax", "hr", "clinic", "legal"]


def make(mode, seed=0):
    r = np.random.default_rng(seed)
    app_pages = {a: r.choice(KV, 300, replace=False) for a in APPS}
    reqs, t = [], 0.0
    sess = []
    for s in range(80):
        app = APPS[s % 4]; user = f"{app}-u{s}"
        turns = 1 + r.geometric(0.45)
        sess.append((user, app, min(turns, 6), r.uniform(0, 600)))
    hist = {}
    for user, app, turns, start in sess:
        tt = start
        for k in range(turns):
            ttft = r.uniform(0.2, 1.5); nout = int(r.integers(20, 400)); itl = 0.05
            new = r.choice(KV, 100, replace=False)
            if mode == "shared":
                pages = np.unique(np.r_[app_pages[app], hist.get(user, np.zeros(0, int)), new])
                hist[user] = np.unique(np.r_[hist.get(user, np.zeros(0, int)), new])
            else:
                pages = r.choice(KV, 400 + 100 * k, replace=False)
            evt = ttft + np.arange(nout + 2) * itl
            base = 120 + 8 * APPS.index(app)
            evb = list((base + r.integers(0, 6, nout + 2)).astype(int))
            reqs.append(dict(kind="workload", user=user, app=app, turn=k, sid=user, t_send=tt, t_first=tt + ttft,
                             t_end=tt + evt[-1], req_bytes=int(2000 + 300 * k + 400 * APPS.index(app)),
                             prompt_tokens=500, cached_tokens=300 if mode == "shared" and k else 0, out_tokens=nout,
                             ev_t=list(evt), ev_bytes=evb, _pages=pages))
            tt = tt + evt[-1] + r.uniform(10, 60)
    for k in range(24):                                   # probes
        for at in ("same_org", "cross_org"):
            ts = r.uniform(50, 550)
            for truth in (True, False):
                hit = truth and (mode == "shared") and at == "same_org"
                ttft = r.uniform(0.1, 0.2) if hit else r.uniform(0.3, 0.5)
                reqs.append(dict(kind="probe_attack", attacker_type=at, truth_hit=truth, user="x", app="tax",
                                 t_send=ts, t_first=ts + ttft, t_end=ts + ttft + 0.4, req_bytes=3000,
                                 prompt_tokens=600, cached_tokens=550 if hit else 0, out_tokens=8,
                                 ev_t=list(ttft + np.arange(10) * 0.05), ev_bytes=[130] * 10, _pages=r.choice(KV, 50)))
                ts += 2
    # slices: 50 ms cycle; slice holds pages of requests in prefill or decode at that time (prefill: all pages,
    # decode: all pages too - like real attention reads)
    T = max(q["t_end"] for q in reqs) + 1
    c0 = np.arange(0, T, 0.05); c1 = c0 + 0.05
    buckets = [[] for _ in c0]
    for q in reqs:
        end = q["t_first"] if os.environ.get("SIM_PREFILL_ONLY") else q["t_end"]
        a, b = int(q["t_send"] / 0.05), int(end / 0.05) + 1
        for i in range(a, min(b, len(c0))):
            buckets[i].append(q["_pages"])
    bits = np.zeros((len(c0), KV), bool)
    for i, bk in enumerate(buckets):
        if bk: bits[i, np.concatenate(bk)] = True
    bits = np.packbits(bits, axis=1)
    parts = np.column_stack([c0, c1, np.zeros_like(c0), np.zeros_like(c0), np.zeros_like(c0)])
    for q in reqs:
        del q["_pages"]
    reqs.sort(key=lambda q: q["t_send"])
    json.dump({"setting": mode, "requests": reqs}, open(f"{OUT}/replay-{'shared' if mode=='shared' else 'nocache'}-L0.5.json", "w"), default=lambda o: o.item())
    np.savez_compressed(f"{OUT}/trace-{'shared' if mode=='shared' else 'nocache'}-L0.5.npz", t0=c0, t1=c1, mon=np.arange(KV), bits=bits, parts=parts, kv_pages=KV, cost=np.zeros(len(c0)))


make("shared"); make("random", 1)
print("sim files written to", OUT)
