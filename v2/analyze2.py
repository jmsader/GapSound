"""GapSound v2 analysis: every adversary against every defense combination, AUROC only.

Input: a run directory with, per (setting, load), replay-<setting>-L<load>.json (replay.py) and
trace-<setting>-L<load>.npz (slicer.py). Padding (response obfuscation) is applied here, post hoc, to the recorded
wire view, so each server run yields both the padded and the unpadded network view.

Adversaries and the questions each one asks
  Host (emulated hypervisor; sees KV-page accesses per time slice, request arrival/first-response times, and the
  encrypted traffic at the VM's NIC):
    same-user  : are two requests from the same user?             (pairs)
    same-app   : are two requests from different users of the same app (same system prompt)?  (pairs)
    next-turn  : which earlier request is this message's previous turn?  (top-1 accuracy among candidates)
    Page attribution: a request owns the pages seen in slices overlapping its prefill window [arrival, first token].
    A slice's pages are weighted 1/k (k = requests whose windows overlap it); for a pair, slices both requests
    overlap are excluded. Score = sum_p min(wA(p), wB(p)) / min(sum wA, sum wB).
    Baselines without page identity: length (request size), timing (time to first token / inter-token gap),
    timing proximity (gap between the two requests); logistic regression on baselines, with and without overlap.
  Other user (PromptPeek precondition): is a probed prefix in the cache? AUROC of fast time-to-first-token.
  Network (Whisper Leak-style; sees one user's encrypted response stream: event sizes and times):
    app classification (4 apps, macro one-vs-rest AUROC) and long vs short response (> median output tokens),
    gradient-boosted trees on aggregate stream features, 5-fold CV grouped by user; with and without padding
    (each streamed event padded by Unif{0..48} bytes, Wei et al. 2024 D=48; OpenAI/Azure/Cloudflare add a
    random-length field per chunk).
  Cost: uncached prompt tokens computed, time to first token, inter-token latency.
Usage: python analyze2.py <run_dir> [--merge 1,4] [--quick]
"""
import glob, json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RUN = sys.argv[1]
MERGES = [int(x) for x in (sys.argv[sys.argv.index("--merge") + 1].split(",") if "--merge" in sys.argv else ["1", "4"])]
QUICK = "--quick" in sys.argv
B_BOOT = 100 if QUICK else 500
PAD_D = 48
BG_SLICES = 2           # background-subtraction variant: slices before arrival treated as in-flight work
NEXT_TURN_HORIZON_S = 900.0
MAX_POS, NEG_PER_POS, MAX_NEG = 5000, 4, 20000
SETTINGS = ["shared", "orgsalt", "salted", "nocache"]


# ---------------------------------------------------------------- statistics
def auc_ci(y, s, seed=0):
    y, s = np.asarray(y, int), np.asarray(s, float)
    if y.size == 0 or y.min() == y.max():
        return float("nan"), (float("nan"), float("nan"))
    a = roc_auc_score(y, s); r = np.random.default_rng(seed); bs = []
    for _ in range(B_BOOT):
        k = r.integers(0, y.size, y.size)
        if y[k].min() != y[k].max():
            bs.append(roc_auc_score(y[k], s[k]))
    return float(a), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def cv_auc_lr(X, y):
    y = np.asarray(y, int)
    if y.min() == y.max():
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


# ---------------------------------------------------------------- host view
class HostTrace:
    """Slices (merged by `merge`), per-request page weights, pair scores."""

    def __init__(self, npz, reqs, merge=1, background=0):
        d = np.load(npz)
        if "parts" in d and len(d["parts"]):
            c0, c1 = d["parts"][:, 0], d["parts"][:, 1]           # effective window: mark start .. read end
        else:
            c0, c1 = d["t0"], d["t1"]
        n = len(c0) // merge * merge
        if "bits" in d:
            bits, mon = d["bits"][:n], d["mon"]
        else:                                                      # old full-page format (idx/off), test runs only
            off, idx, kvp = d["off"], d["idx"], int(d["kv_pages"])
            mon = np.arange(kvp); bits = np.zeros((n, (kvp + 7) // 8), np.uint8)
            for i in range(n):
                b = np.zeros(kvp, bool); b[idx[off[i]:off[i + 1]]] = True; bits[i] = np.packbits(b)
        if merge > 1:                                              # coarser slice = OR of consecutive slices
            bits = np.bitwise_or.reduce(bits.reshape(n // merge, merge, -1), axis=1)
        self.bits, self.mon = bits, mon
        self.c0 = c0[:n:merge]
        self.c1 = c1[merge - 1:n:merge]
        self._sl = {}
        self.kv_pages = int(d["kv_pages"])
        self.cycle_ms = float(np.median(np.diff(self.c0)) * 1000) if len(self.c0) > 1 else float("nan")
        S = len(self.bits)
        # request windows -> slice ranges [a, b)
        self.rng = {}
        k = np.zeros(S + 1)
        for r in reqs:
            a = int(np.searchsorted(self.c1, r["t_send"], side="right"))   # first slice ending after arrival
            b = int(np.searchsorted(self.c0, r["t_first"], side="left"))   # slices starting before first token
            self.rng[r["rid"]] = (a, max(a, b))
            k[a] += 1; k[max(a, b)] -= 1
        self.k = np.maximum(np.cumsum(k)[:S], 1)
        self.cache = {}
        # background subtraction (variant): pages already being touched in the `background` slices just before a
        # request arrived belong to requests already in flight (decode reads every running sequence's KV each
        # step), so they are removed from that request's pages.
        self.background = background
        self.bg = {}
        if background:
            for r in reqs:
                a = self.rng[r["rid"]][0]
                lo = max(0, a - background)
                self.bg[r["rid"]] = np.unique(np.concatenate([self.sl(i) for i in range(lo, a)])) if a > lo else None

    def sl(self, i):
        """monitored KV pages accessed in slice i"""
        v = self._sl.get(i)
        if v is None:
            v = self._sl[i] = self.mon[np.nonzero(np.unpackbits(self.bits[i])[:self.mon.size])[0]]
        return v

    def weights(self, ranges):
        parts = [(self.sl(i), 1.0 / self.k[i]) for a, b in ranges for i in range(a, b) if self.sl(i).size]
        if not parts:
            return np.zeros(0, np.int64), np.zeros(0)
        ix = np.concatenate([p for p, _ in parts]); w = np.concatenate([np.full(p.size, v) for p, v in parts])
        u, inv = np.unique(ix, return_inverse=True)
        return u, np.bincount(inv, weights=w)

    def _sub(self, rid, uw):
        u, w = uw
        b = self.bg.get(rid)
        if b is None or not u.size:
            return u, w
        keep = ~np.isin(u, b, assume_unique=True)
        return u[keep], w[keep]

    def w(self, rid):
        if rid not in self.cache:
            self.cache[rid] = self._sub(rid, self.weights([self.rng[rid]]))
        return self.cache[rid]

    def n_slices(self, rid):
        a, b = self.rng[rid]; return b - a

    @staticmethod
    def _minus(r, o):
        (a, b), (c, d) = r, o
        out = []
        if a < min(b, c): out.append((a, min(b, c)))
        if max(a, d) < b: out.append((max(a, d), b))
        return out

    def score(self, i, j, dense=None):
        """weighted overlap of requests i, j; slices both overlap are excluded. `dense` = densified w(i) (speed)."""
        ri, rj = self.rng[i], self.rng[j]
        if max(ri[0], rj[0]) < min(ri[1], rj[1]):                  # windows share slices
            ui, wi = self._sub(i, self.weights(self._minus(ri, rj))); uj, wj = self._sub(j, self.weights(self._minus(rj, ri)))
            if not wi.size or not wj.size:
                return 0.0
            _, a, b = np.intersect1d(ui, uj, assume_unique=True, return_indices=True)
            return float(np.minimum(wi[a], wj[b]).sum() / min(wi.sum(), wj.sum()))
        ui, wi = self.w(i); uj, wj = self.w(j)
        if not wi.size or not wj.size:
            return 0.0
        if dense is not None:
            return float(np.minimum(dense[uj], wj).sum() / min(wi.sum(), wj.sum()))
        _, a, b = np.intersect1d(ui, uj, assume_unique=True, return_indices=True)
        return float(np.minimum(wi[a], wj[b]).sum() / min(wi.sum(), wj.sum()))

    def dense(self, i):
        u, w = self.w(i); v = np.zeros(self.kv_pages + 1); v[u] = w; return v


def ttft_ratio(r):
    t = np.asarray(r["ev_t"]); ttft = r["t_first"] - r["t_send"]
    g = np.diff(t[1:]) if t.size > 3 else np.array([np.nan])
    m = np.nanmean(g) if g.size else np.nan
    return ttft / m if m and m > 0 else np.nan


OVERLAPS = ("page overlap", "page overlap, bg-subtracted")


def host_attacks(R, trs, rs):
    """trs: {"page overlap": HostTrace (converged attribution), "page overlap, bg-subtracted": HostTrace variant}"""
    tr0 = trs[OVERLAPS[0]]
    W = [r for r in R if r["kind"] == "workload"]
    n = len(W)
    user = np.array([r["user"] for r in W]); app = np.array([r["app"] for r in W])
    L = np.array([r["req_bytes"] for r in W], float)
    TR = np.array([ttft_ratio(r) for r in W], float); TR = np.where(np.isnan(TR), np.nanmedian(TR), TR)
    ts = np.array([r["t_send"] for r in W]); te = np.array([r["t_end"] for r in W])
    out = {"n_requests": n, "slice_cycle_ms": tr0.cycle_ms,
           "mean_slices_per_window": float(np.mean([tr0.n_slices(r["rid"]) for r in W])),
           "mean_pages_per_window": {k: float(np.mean([t.w(r["rid"])[0].size for r in W])) for k, t in trs.items()},
           "mean_k": float(np.mean(tr0.k))}

    def pairs_for(pos_fn, neg_fn, cap_pos):
        P, N = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if pos_fn(i, j): P.append((i, j))
                elif neg_fn(i, j): N.append((i, j))
        P = [P[k] for k in rs.permutation(len(P))[:cap_pos]]
        N = [N[k] for k in rs.permutation(len(N))[:min(MAX_NEG, NEG_PER_POS * max(1, len(P)))]]
        return P + N, np.r_[np.ones(len(P)), np.zeros(len(N))].astype(int)

    def overlap_scores(tr, pairs):
        ov = np.empty(len(pairs)); cur, dense = -1, None
        for k, (i, j) in enumerate(pairs):
            if i != cur:
                cur, dense = i, tr.dense(W[i]["rid"])
            ov[k] = tr.score(W[i]["rid"], W[j]["rid"], dense)
        return ov

    tasks = {"same-user": pairs_for(lambda i, j: user[i] == user[j], lambda i, j: True, MAX_POS),
             "same-app": pairs_for(lambda i, j: user[i] != user[j] and app[i] == app[j],
                                   lambda i, j: app[i] != app[j], MAX_POS)}
    for tname, (pairs, y) in tasks.items():
        pairs = np.array(pairs, int)
        order = np.argsort(pairs[:, 0], kind="stable"); pairs, y = pairs[order], y[order]
        ovs = {k: overlap_scores(t, pairs) for k, t in trs.items()}
        i, j = pairs[:, 0], pairs[:, 1]
        early, late = np.minimum(i, j), np.maximum(i, j)
        dl, dt = np.abs(L[i] - L[j]), np.abs(TR[i] - TR[j])
        gap = np.abs(ts[late] - te[early])
        t = {"positives": int(y.sum()), "negatives": int((1 - y).sum())}
        for sname, sc in list(ovs.items()) + [("length", -dl), ("timing", -dt), ("timing proximity", -gap)]:
            a, ci = auc_ci(y, sc); t[sname] = {"auc": a, "ci95": ci}
        Xb = np.column_stack([dl, dt, L[i] + L[j], TR[i] + TR[j], np.log1p(gap)])
        t["baselines (LR)"] = {"auc": cv_auc_lr(Xb, y)}
        for k, ov in ovs.items():
            t[f"baselines + {k} (LR)"] = {"auc": cv_auc_lr(np.column_stack([Xb, ov]), y)}
            t[f"{k}: mean pos/neg"] = [float(ov[y == 1].mean()), float(ov[y == 0].mean())]
        out[tname] = t
    # next-turn: candidates = workload requests that finished within the horizon before this one arrived
    acc = {k: [] for k in list(trs) + ["timing proximity"]}
    chance, covered, nc = [], 0, 0
    for j in range(n):
        if W[j]["turn"] == 0:
            continue
        nc += 1
        cand = np.nonzero((te <= ts[j]) & (ts[j] - te <= NEXT_TURN_HORIZON_S))[0]
        truth = [i for i in cand if user[i] == user[j] and W[i]["turn"] == W[j]["turn"] - 1]
        if not truth or cand.size < 2:
            continue
        covered += 1
        for k, tr in trs.items():
            dense = tr.dense(W[j]["rid"])
            sc = np.array([tr.score(W[j]["rid"], W[i]["rid"], dense) for i in cand])
            top = cand[sc == sc.max()]
            acc[k].append(np.mean([user[i] == user[j] for i in top]))
        px = ts[j] - te[cand]; top = cand[px == px.min()]
        acc["timing proximity"].append(np.mean([user[i] == user[j] for i in top]))
        chance.append(1.0 / cand.size)
    out["next-turn"] = {f"{k} acc": (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
    out["next-turn"].update({"chance": float(np.mean(chance)) if chance else float("nan"),
                             "mean candidates": float(np.mean([1 / c for c in chance])) if chance else float("nan"),
                             "evaluated": covered, "later turns": nc})
    return out


# ---------------------------------------------------------------- other-user attacker
def probe_attacks(R):
    out = {}
    for atype in ("same_org", "cross_org"):
        P = [r for r in R if r["kind"] == "probe_attack" and r["attacker_type"] == atype]
        if not P:
            continue
        y = np.array([r["truth_hit"] for r in P], int)
        raw = -np.array([r["t_first"] - r["t_send"] for r in P])
        nr = np.array([ttft_ratio(r) for r in P]); nr = -np.where(np.isnan(nr), np.nanmedian(nr), nr)
        a1, c1 = auc_ci(y, raw); a2, c2 = auc_ci(y, nr)
        out[atype] = {"raw TTFT": {"auc": a1, "ci95": c1}, "normalized TTFT": {"auc": a2, "ci95": c2},
                      "n": len(P), "cached tokens hit/miss": [float(np.mean([r["cached_tokens"] for r in P if r["truth_hit"]])),
                                                              float(np.mean([r["cached_tokens"] for r in P if not r["truth_hit"]]))]}
    return out


# ---------------------------------------------------------------- network attacker
def stream_features(r, pad, rs):
    t = np.asarray(r["ev_t"], float); b = np.asarray(r["ev_bytes"], float)
    if pad:
        b = b.copy(); b[:-1] += rs.integers(0, PAD_D + 1, b.size - 1)    # every JSON event; not the final [DONE]
    g = np.diff(t) if t.size > 1 else np.zeros(1)
    q = lambda x, p: float(np.percentile(x, p))
    return [t.size, b.sum(), t[-1] if t.size else 0, t[0] if t.size else 0,
            g.mean(), g.std(), q(g, 50), q(g, 90), b.mean(), b.std(), b.min(), b.max(), q(b, 10), q(b, 50), q(b, 90)]


def network_attacks(R, rs):
    W = [r for r in R if r["kind"] == "workload"]
    groups = np.array([r["user"] for r in W])
    apps = sorted({r["app"] for r in W}); ya = np.array([apps.index(r["app"]) for r in W])
    ot = np.array([r["out_tokens"] for r in W]); yl = (ot > np.median(ot)).astype(int)
    out = {}
    for pad in (False, True):
        X = np.array([stream_features(r, pad, rs) for r in W])
        # folds keep each user's requests together and balance classes across folds (plain grouped folds shift the
        # class mix between train and test, which pushes AUROC below 0.5 on small samples)
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        Pa = cross_val_predict(clf, X, ya, cv=cv, groups=groups, method="predict_proba")
        auc_app = float(np.mean([roc_auc_score(ya == c, Pa[:, c]) for c in range(len(apps))]))
        Pl = cross_val_predict(clf, X, yl, cv=cv, groups=groups, method="predict_proba")[:, 1]
        a, ci = auc_ci(yl, Pl)
        key = "padded" if pad else "unpadded"
        out[key] = {"app (macro OvR)": auc_app, "long vs short": {"auc": a, "ci95": ci},
                    "mean bytes/event": float(X[:, 8].mean())}
    return out


# ---------------------------------------------------------------- cost
def cost(R):
    W = [r for r in R if r["kind"] == "workload"]
    pt = np.array([r["prompt_tokens"] for r in W], float); ct = np.array([r["cached_tokens"] for r in W], float)
    ttft = np.array([r["t_first"] - r["t_send"] for r in W])
    itl = np.concatenate([np.diff(r["ev_t"][1:]) for r in W if len(r["ev_t"]) > 3])
    t0 = min(r["t_send"] for r in R); t1 = max(r["t_end"] for r in R)
    return {"uncached prompt tokens": float((pt - ct).sum()), "prompt tokens": float(pt.sum()),
            "cache hit fraction": float(ct.sum() / pt.sum()), "TTFT mean s": float(ttft.mean()),
            "TTFT p50 s": float(np.median(ttft)), "TTFT p90 s": float(np.percentile(ttft, 90)),
            "ITL mean ms": float(itl.mean() * 1000), "ITL p90 ms": float(np.percentile(itl, 90) * 1000),
            "achieved req/s": len(R) / (t1 - t0), "duration s": t1 - t0}


# ---------------------------------------------------------------- main
def main():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RUN, "replay-*-L*.json"))):
        m = re.search(r"replay-(\w+)-L([\d.]+)\.json$", f)
        runs[(m.group(2), m.group(1))] = f
    loads = sorted({l for l, _ in runs}, key=float)
    res = {}
    for (load, setting), f in sorted(runs.items()):
        d = json.load(open(f)); R = d["requests"]
        for k, r in enumerate(R):
            r["rid"] = k
        npz = f.replace("replay-", "trace-").replace(".json", ".npz")
        e = {"setting": setting, "load": float(load), "cost": cost(R), "probe": probe_attacks(R),
             "network": network_attacks(R, np.random.default_rng(1))}
        if os.path.exists(npz):
            for m in MERGES:
                trs = {OVERLAPS[0]: HostTrace(npz, R, m), OVERLAPS[1]: HostTrace(npz, R, m, background=BG_SLICES)}
                e[f"host_merge{m}"] = host_attacks(R, trs, np.random.default_rng(2))
        res[f"{setting}-L{load}"] = e
        print(f"analyzed {setting} load {load}", flush=True)
    json.dump(res, open(os.path.join(RUN, "summary.json"), "w"), indent=1, default=float)
    md = report(res, loads)
    open(os.path.join(RUN, "summary.md"), "w").write(md)
    print(md)


def f2(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def fci(d):
    return f"{f2(d['auc'])} [{f2(d['ci95'][0])}, {f2(d['ci95'][1])}]"


def report(res, loads):
    L = ["# GapSound v2 results", "", "AUROC (0.5 = chance) with 95% bootstrap CI; next-turn = top-1 accuracy.", ""]
    for load in loads:
        cols = [s for s in SETTINGS if f"{s}-L{load}" in res]
        E = [res[f"{s}-L{load}"] for s in cols]
        hdr = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        L += [f"## Load {float(load):.0%} of capacity", ""]
        for m in MERGES:
            key = f"host_merge{m}"
            if not all(key in e for e in E):
                continue
            L += [f"### Host attacker (slice = {m} tracer cycle{'s' if m > 1 else ''}, "
                  f"~{E[0][key]['slice_cycle_ms']:.0f} ms)", ""] + hdr
            for task in ("same-user", "same-app"):
                for s in OVERLAPS + ("length", "timing", "timing proximity"):
                    L.append(f"| {task}: {s} | " + " | ".join(fci(e[key][task][s]) for e in E) + " |")
                for s in ("baselines (LR)",) + tuple(f"baselines + {k} (LR)" for k in OVERLAPS):
                    L.append(f"| {task}: {s} | " + " | ".join(f2(e[key][task][s]["auc"]) for e in E) + " |")
            for s in tuple(f"{k} acc" for k in OVERLAPS) + ("timing proximity acc", "chance"):
                L.append(f"| next-turn: {s} | " + " | ".join(f2(e[key]["next-turn"][s]) for e in E) + " |")
            for k in OVERLAPS:
                L.append(f"| mean pages per window ({k}) | " + " | ".join(f"{e[key]['mean_pages_per_window'][k]:.0f}" for e in E) + " |")
            L.append("")
        L += ["### Other-user attacker (cache-hit probe)", ""] + hdr
        for at in ("same_org", "cross_org"):
            for s in ("raw TTFT", "normalized TTFT"):
                L.append(f"| {at}: {s} | " + " | ".join(fci(e["probe"][at][s]) if at in e["probe"] else "n/a" for e in E) + " |")
        L += ["", "### Network attacker (response stream), unpadded / padded", ""] + hdr
        L.append("| app (macro OvR) | " + " | ".join(f"{f2(e['network']['unpadded']['app (macro OvR)'])} / "
                                                     f"{f2(e['network']['padded']['app (macro OvR)'])}" for e in E) + " |")
        L.append("| long vs short | " + " | ".join(f"{f2(e['network']['unpadded']['long vs short']['auc'])} / "
                                                   f"{f2(e['network']['padded']['long vs short']['auc'])}" for e in E) + " |")
        L += ["", "### Cost", ""] + hdr
        for k in ("uncached prompt tokens", "cache hit fraction", "TTFT p50 s", "TTFT p90 s", "ITL mean ms",
                  "achieved req/s"):
            L.append(f"| {k} | " + " | ".join(f"{e['cost'][k]:.0f}" if e['cost'][k] > 100 else f2(e['cost'][k]) for e in E) + " |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
