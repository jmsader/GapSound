"""Tracer check v2: TLB flush + which KV pages each request touches (overlap between requests)."""
import json, sys, time, re
import numpy as np, requests
sys.path.insert(0, "/mnt/gapsound")
import hostview as hv

server_pid = int(sys.argv[1]); out_path = sys.argv[2]
URL = "http://localhost:8000"; MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

wpid, rss_table = hv.worker_pid(server_pid)
snap0 = hv.snapshot(wpid)
KV = hv.find_kv_vma(snap0)
print(f"worker pid {wpid}; RSS MB {rss_table}; KV region {hex(KV) if KV else None} "
      f"({(snap0[KV][0]-KV)/2**20:.0f} MB)" if KV else "KV region NOT FOUND")
HEAP = [s for s, v in snap0.items() if v[1] == "[heap]"]

def metrics():
    txt = requests.get(URL + "/metrics", timeout=10).text
    g = lambda k: float(re.search(rf"^{k}{{[^}}]*}} ([0-9.e+]+)", txt, re.M).group(1))
    return g("vllm:prefix_cache_hits_total")

def window(name, prompt=None, salt=None, max_tokens=8, idle_s=2.0):
    before = hv.snapshot(wpid)
    h0 = metrics()
    hv.mark_idle(); hv.flush_tlb(wpid)
    t0 = time.time(); ptoks = 0
    if prompt is None:
        time.sleep(idle_s)
    else:
        body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
        if salt: body["cache_salt"] = salt
        ptoks = requests.post(URL + "/v1/completions", json=body, timeout=600).json()["usage"]["prompt_tokens"]
    lat = time.time() - t0
    bm = hv.read_idle()
    after = hv.snapshot(wpid)
    hit = metrics() - h0
    rows = {int(r["start"], 16): r for r in hv.diff(before, after, bm)}
    kv_idx = rows[KV]["accessed_idx"] if KV in rows else np.array([], int)
    heap_acc = sum(rows[h]["accessed"] for h in HEAP if h in rows)
    other = sum(r["accessed"] for s, r in rows.items() if s != KV and s not in HEAP)
    return {"name": name, "salt": salt, "prompt_tokens": ptoks, "latency_s": round(lat, 3),
            "cache_hit_tokens": hit, "kv_pages": int(kv_idx.size), "heap_pages": heap_acc,
            "other_pages": other, "kv_idx": kv_idx.tolist()}

def text(seed, n_sent):
    import random
    rnd = random.Random(seed)
    words = ("patient billing insurance claim policy record visit doctor clinic referral lab result "
             "prescription pharmacy appointment schedule copay deductible coverage privacy consent").split()
    return " ".join(" ".join(rnd.choice(words) for _ in range(12)) + "." for _ in range(n_sent))

SHORT = "Question: What are the office hours?\nAnswer:"
LONG = "Context: " + text(1, 60) + "\nQuestion: Summarize the context.\nAnswer:"
OTHER = "Context: " + text(2, 60) + "\nQuestion: Summarize the context.\nAnswer:"   # different prefix

plan = [("idle-1", None, None), ("idle-2", None, None), ("short", SHORT, None),
        ("A long, cold", LONG + " v1", None),
        ("B long again, no salt (shared hit)", LONG + " v2", None),
        ("C different long prompt, cold", OTHER + " v1", None),
        ("D long, salt=userX (cold)", LONG + " v3", "userX"),
        ("E long, salt=userX again (hit)", LONG + " v4", "userX"),
        ("F long, salt=userY (cold)", LONG + " v5", "userY"),
        ("idle-3", None, None)]
recs = []
print(f"{'window':38s} {'tok':>4s} {'lat':>6s} {'hit':>5s} {'KVpg':>6s} {'heap':>7s} {'other':>6s}  overlap-with-earlier-requests")
seen = {}
for name, prompt, salt in plan:
    r = window(name, prompt, salt)
    s = set(r["kv_idx"])
    ov = []
    for pn, ps in seen.items():
        if s and ps:
            f = len(s & ps) / len(s)
            if f > 0.02: ov.append(f"{pn.split()[0]}:{f:.0%}")
    if prompt is not None: seen[name] = s
    print(f"{name:38s} {r['prompt_tokens']:4d} {r['latency_s']:6.2f} {r['cache_hit_tokens']:5.0f} "
          f"{r['kv_pages']:6d} {r['heap_pages']:7d} {r['other_pages']:6d}  {' '.join(ov) or '-'}")
    recs.append(r)

json.dump({"worker_pid": wpid, "kv_start": hex(KV) if KV else None, "windows": recs},
          open(out_path, "w"))
print("TRACE_CHECK_DONE ->", out_path)
