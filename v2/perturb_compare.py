"""Tracer slowdown under real replay traffic: compare replay runs with and without the tracer (same sessions/seed).
Usage: python perturb_compare.py <baseline.json> <traced.json> [more traced.json ...]"""
import json, sys, numpy as np


def stats(path):
    R = [r for r in json.load(open(path))["requests"] if r["kind"] == "workload"]
    ttft = np.array([r["t_first"] - r["t_send"] for r in R])
    itl = np.concatenate([np.diff(r["ev_t"][1:]) for r in R if len(r["ev_t"]) > 3])
    return {"n": len(R), "TTFT p50": np.median(ttft), "TTFT p90": np.percentile(ttft, 90),
            "ITL p50": np.median(itl), "ITL p90": np.percentile(itl, 90), "ITL mean": itl.mean()}


b = stats(sys.argv[1])
print("baseline", {k: round(float(v), 4) for k, v in b.items()})
for p in sys.argv[2:]:
    s = stats(p)
    print(p.split("/")[-1], " ".join(f"{k} {100*(s[k]/b[k]-1):+.1f}%" for k in b if k != "n"), f"(n={s['n']})")
