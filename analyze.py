"""Score the emulated host's linking attacks for each setting.

Tasks (over all pairs of requests within a setting):
  same-user : are the two requests from the same user?
  same-app  : among pairs from DIFFERENT users, are they from the same app (same system prompt)?
Host scores:
  overlap   : fraction of shared KV pages, |Pi & Pj| / min(|Pi|, |Pj|)   (page identity)
  length    : -|kv_pages_i - kv_pages_j|                                (what length alone gives)
  timing    : -|ratio_i - ratio_j|, ratio = time-to-first-token / mean inter-token gap
              (prefill cost in decode steps, as a host sees it at the VM's NIC; cancels machine-speed drift)
Also: logistic regression on length+timing features, with and without overlap (5-fold CV).
Usage: python analyze.py results/exp-<ts>-shared.json results/exp-<ts>-salted.json
"""
import json, sys, os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)


def auc_ci(y, s, B=500):
    y, s = np.asarray(y), np.asarray(s)
    if y.min() == y.max():
        return float("nan"), (float("nan"), float("nan"))
    a = roc_auc_score(y, s); bs = []
    for _ in range(B):
        k = rng.integers(0, len(y), len(y))
        if y[k].min() != y[k].max():
            bs.append(roc_auc_score(y[k], s[k]))
    return a, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def cv_auc(X, y):
    if y.min() == y.max():
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def analyze(path):
    d = json.load(open(path)); R = d["requests"]; n = len(R)
    P = [set(r["kv_idx"]) for r in R]
    kv = np.array([r["kv_pages"] for r in R], float)
    tkey = "ttft_ratio" if "ttft_ratio" in R[0] else "cpu_s"
    cpu = np.array([r[tkey] for r in R], float)          # host timing feature
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            ov = len(P[i] & P[j]) / max(1, min(len(P[i]), len(P[j])))
            rows.append((ov, abs(kv[i] - kv[j]), abs(cpu[i] - cpu[j]), kv[i] + kv[j], cpu[i] + cpu[j],
                         R[i]["user"] == R[j]["user"], R[i]["app"] == R[j]["app"], j == i + 1))
    A = np.array(rows, float)
    ov, dkv, dcpu, skv, scpu, su, sa, cons = A.T
    res = {"setting": d["setting"], "n_requests": n, "n_pairs": len(rows)}
    tasks = {"same-user": (np.ones(len(rows), bool), su),
             "same-app (different users)": (su == 0, sa)}
    for tname, (mask, y) in tasks.items():
        y = y[mask].astype(int); t = {}
        for sname, s in (("overlap", ov), ("length", -dkv), ("timing", -dcpu)):
            a, ci = auc_ci(y, s[mask]); t[sname] = {"auc": a, "ci95": ci}
        Xb = np.column_stack([dkv, dcpu, skv, scpu])[mask]
        t["length+timing (LR)"] = {"auc": cv_auc(Xb, y)}
        t["length+timing+overlap (LR)"] = {"auc": cv_auc(np.column_stack([Xb, ov[mask]]), y)}
        t["positives"] = int(y.sum())
        res[tname] = t
    # link each later message to an earlier request: is the max-overlap earlier request the same user?
    # ties share credit: if k earlier requests tie for the max overlap, score = fraction of them from the same user
    by_turn = {}
    for j in range(n):
        t = R[j]["turn"]
        if t == 0:
            continue
        ovs = [len(P[i] & P[j]) for i in range(j)]
        m = max(ovs); tied = [i for i in range(j) if ovs[i] == m]
        by_turn.setdefault(t, []).append(sum(R[i]["user"] == R[j]["user"] for i in tied) / len(tied))
    allv = [v for vs in by_turn.values() for v in vs]
    res["next-turn linking accuracy"] = float(np.mean(allv)) if allv else None
    for t in sorted(by_turn):
        res[f"next-turn linking accuracy, turn {t}"] = float(np.mean(by_turn[t]))
    diff_app = sa == 0
    res["noise: overlap, unrelated pairs (consecutive)"] = float(ov[diff_app & (cons == 1)].mean())
    res["noise: overlap, unrelated pairs (non-consecutive)"] = float(ov[diff_app & (cons == 0)].mean())
    res["overlap, same app diff user (mean)"] = float(ov[(su == 0) & (sa == 1)].mean())
    res["overlap, same user (mean)"] = float(ov[su == 1].mean())
    res["timing_feature"] = tkey
    # other-user attacker: does a fast time-to-first-token reveal that someone else's prefix is cached?
    for atype in ("same_org", "cross_org"):
        P_ = [p for p in d.get("probes", []) if p["attacker"] == atype]
        if P_:
            y = np.array([p["truth_hit"] for p in P_], int)
            a1, ci1 = auc_ci(y, -np.array([p["ttft_s"] for p in P_]))
            a2, ci2 = auc_ci(y, -np.array([p["ttft_ratio"] for p in P_]))
            res[f"other-user probe AUC, {atype} (raw TTFT)"] = a1
            res[f"other-user probe AUC, {atype} (normalized)"] = a2
    res["cost"] = {"uncached_tokens_computed": float(sum(r["prompt_tokens"] - r["hit_tokens"] for r in R)),
                   "mean_ttft_ratio": float(np.mean([r.get("ttft_ratio", np.nan) for r in R])),
                   "mean_latency_s": float(np.mean([r["latency_s"] for r in R])),
                   "mean_cpu_s": float(np.mean([r["cpu_s"] for r in R])),
                   "total_hit_tokens": float(sum(r["hit_tokens"] for r in R)),
                   "total_prompt_tokens": float(sum(r["prompt_tokens"] for r in R)),
                   }
    return res


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


results = [analyze(p) for p in sys.argv[1:]]
import re
base = re.sub(r"-(shared|orgsalt|salted|nocache)\.json$", "", os.path.basename(sys.argv[1]))
outdir = os.path.dirname(sys.argv[1])
L = [f"# GapSound results: {base}", ""]
for task in ("same-user", "same-app (different users)"):
    L += [f"## Task: {task}", "", "| Host attack | " + " | ".join(r["setting"] for r in results) + " |",
          "|---|" + "---|" * len(results)]
    for s in ("overlap", "length", "timing"):
        L.append(f"| {s} AUC [95% CI] | " + " | ".join(
            f"{fmt(r[task][s]['auc'])} [{fmt(r[task][s]['ci95'][0])}, {fmt(r[task][s]['ci95'][1])}]" for r in results) + " |")
    for s in ("length+timing (LR)", "length+timing+overlap (LR)"):
        L.append(f"| {s} AUC | " + " | ".join(fmt(r[task][s]["auc"]) for r in results) + " |")
    L.append("| positive pairs | " + " | ".join(str(r[task]["positives"]) for r in results) + " |")
    L.append("")
L += ["## Other measures", "", "| Measure | " + " | ".join(r["setting"] for r in results) + " |", "|---|" + "---|" * len(results)]
turn_keys = sorted({k for r in results for k in r if k.startswith("next-turn linking accuracy, turn")})
probe_keys = sorted({k for r in results for k in r if k.startswith("other-user probe AUC")})
if probe_keys:
    L += ["## Other-user attacker (cache-hit probe, AUROC)", "", "| Probe | " + " | ".join(r["setting"] for r in results) + " |",
          "|---|" + "---|" * len(results)]
    for k in probe_keys:
        L.append(f"| {k.replace('other-user probe AUC, ', '')} | " + " | ".join(fmt(r.get(k)) for r in results) + " |")
    L.append("")
for k in ["next-turn linking accuracy"] + turn_keys + ["overlap, same user (mean)", "overlap, same app diff user (mean)",
          "noise: overlap, unrelated pairs (consecutive)", "noise: overlap, unrelated pairs (non-consecutive)"]:
    L.append(f"| {k} | " + " | ".join(fmt(r.get(k)) for r in results) + " |")
for k in ("uncached_tokens_computed", "mean_ttft_ratio", "total_hit_tokens", "total_prompt_tokens", "mean_latency_s", "mean_cpu_s"):
    L.append(f"| cost: {k} | " + " | ".join(fmt(r["cost"][k]) for r in results) + " |")
md = "\n".join(L)
open(os.path.join(outdir, f"{base}-summary.md"), "w").write(md)
json.dump(results, open(os.path.join(outdir, f"{base}-summary.json"), "w"), indent=1, default=float)
print(md)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, task in zip(axes, ("same-user", "same-app (different users)")):
        attacks = ("overlap", "length", "timing"); w = 0.8 / len(results)
        for k, r in enumerate(results):
            a = [r[task][s]["auc"] for s in attacks]
            lo = [r[task][s]["auc"] - r[task][s]["ci95"][0] for s in attacks]
            hi = [r[task][s]["ci95"][1] - r[task][s]["auc"] for s in attacks]
            ax.bar(np.arange(3) + k * w, a, w, yerr=[lo, hi], capsize=3, label=r["setting"])
        ax.axhline(0.5, color="gray", lw=1, ls="--")
        ax.set_xticks(np.arange(3) + w * (len(results) - 1) / 2); ax.set_xticklabels(attacks)
        ax.set_title(f"Host links: {task}"); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("AUC (0.5 = chance)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{base}-auc.png"), dpi=150)
    print("plot saved")
except Exception as e:
    print("plot skipped:", e)
