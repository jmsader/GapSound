"""Full experiment: can an emulated host link requests by user or by app, with a shared vs salted prefix cache?

Workload: N_APPS apps, each with its own system prompt; USERS_PER_APP users per app; each user has a
TURNS-turn conversation (each turn resends the whole history, like a real chat). The same workload and
the same random interleaving are used in every setting, so settings differ only in cache_salt.

Per request the emulated host records: which KV-cache pages were touched (identity), how many (length),
and the model worker's CPU time (host-side timing).

Usage (root): run_exp.py <server_pid> <shared|orgsalt|salted|nocache> <out.json> [seed]
  orgsalt = one cache_salt per app (organization), as OpenAI/Anthropic isolate caches
  nocache = host-side defense: server started with prefix caching disabled (no salt)
"""
import hashlib, json, os, random, re, sys, time
import numpy as np, requests
sys.path.insert(0, "/mnt/gapsound")
import hostview as hv

server_pid = int(sys.argv[1]); SETTING = sys.argv[2]; OUT = sys.argv[3]
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
assert SETTING in ("shared", "orgsalt", "salted", "nocache")
URL = "http://localhost:8000"; MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
USERS_PER_APP, TURNS, SYS_SENTENCES, MAX_TOKENS = 6, 4, 40, 16
CLK = os.sysconf("SC_CLK_TCK")

APPS = {
    "clinic": ("Riverside Clinic patient services",
               "patient appointment referral lab result prescription copay insurance doctor nurse visit record privacy consent pharmacy billing"),
    "tax":    ("Northfield tax preparation office",
               "return deduction refund income filing receipt audit credit dependent withholding schedule quarterly estimate penalty bracket"),
    "legal":  ("Harbor legal aid clinic",
               "lease tenant eviction notice court hearing contract landlord deposit appeal filing deadline attorney evidence settlement"),
    "hr":     ("Summit company HR benefits desk",
               "benefits enrollment leave payroll retirement dental vision policy manager overtime timesheet onboarding bonus vacation handbook"),
}
GENERIC = "please help me understand what I should do about my situation and the next steps for this".split()


def sentences(rnd, vocab, n, k):
    return " ".join((" ".join(rnd.choice(vocab) for _ in range(k))).capitalize() + "." for _ in range(n))


# ---------------- workload (deterministic: identical across settings) ----------------
SYSTEM, CONVO, USERS = {}, {}, []
for app, (org, words) in APPS.items():
    vocab = words.split()
    r = random.Random(f"{SEED}-sys-{app}")
    SYSTEM[app] = f"You are the assistant for {org}. Follow these office policies exactly. " + \
                  sentences(r, vocab, SYS_SENTENCES, 12)
    for j in range(USERS_PER_APP):
        u = f"{app}-u{j}"; USERS.append((u, app))
        ru = random.Random(f"{SEED}-user-{u}")
        # ~100-token user messages and ~160-token replies: each turn adds 2+ KV blocks of history
        CONVO[u] = [("User: " + sentences(ru, vocab + GENERIC, 6, 12),
                     "Assistant: " + sentences(ru, vocab, 10, 12)) for _ in range(TURNS)]


def prompt(u, app, t):
    parts = []
    for i in range(t):
        parts += [CONVO[u][i][0], CONVO[u][i][1]]
    parts.append(CONVO[u][t][0])
    return SYSTEM[app] + "\n\n" + "\n".join(parts) + "\nAssistant:"


rs = random.Random(f"{SEED}-schedule"); done = {u: 0 for u, _ in USERS}; ORDER = []
while any(d < TURNS for d in done.values()):
    u, app = rs.choice([(u, a) for u, a in USERS if done[u] < TURNS])
    ORDER.append((u, app, done[u])); done[u] += 1
def salt_for(u, app):
    if SETTING == "salted":
        return hashlib.sha256(f"gateway-secret-user-{u}".encode()).hexdigest()[:16]
    if SETTING == "orgsalt":
        return hashlib.sha256(f"gateway-secret-org-{app}".encode()).hexdigest()[:16]
    return None
SALT = {u: salt_for(u, a) for u, a in USERS}

def metrics():
    txt = requests.get(URL + "/metrics", timeout=10).text
    def g(pat):
        m = re.search(rf"^{pat}{{[^}}]*}} ([0-9.e+-]+)", txt, re.M)
        return float(m.group(1)) if m else None
    return {"hits": g("vllm:prefix_cache_hits_total") or 0.0, "queries": g("vllm:prefix_cache_queries_total") or 0.0,
            "kv_usage": g(r"vllm:\S*cache_usage\S*")}


def cpu_ticks(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12])          # utime + stime, all threads of the worker


def send(text, salt):
    """Streamed request. Records when each token chunk arrives: what a host sees at the VM's virtual NIC.
    Returns (usage, ttft_s, itl_s): time to first token and mean gap between later tokens."""
    body = {"model": MODEL, "prompt": text, "max_tokens": MAX_TOKENS, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    if salt:
        body["cache_salt"] = salt
    t0 = time.time(); times = []; usage = None
    with requests.post(URL + "/v1/completions", json=body, timeout=600, stream=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"request failed: {r.status_code} {r.text[:300]}")
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("choices"):
                times.append(time.time())
            if obj.get("usage"):
                usage = obj["usage"]
    if usage is None or not times:
        raise RuntimeError("stream ended without tokens or usage")
    ttft = times[0] - t0
    itl = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else float("nan")
    return usage, ttft, itl


# ---------------- emulated host setup ----------------
wpid, rss = hv.worker_pid(server_pid)
snap0 = hv.snapshot(wpid)
KV = hv.find_kv_vma(snap0)
if KV is None:
    sys.exit("KV region not found")
KV_VMA = [(KV, snap0[KV][0], snap0[KV][1])]
print(f"[{SETTING}] worker {wpid}, KV region {(snap0[KV][0]-KV)/2**20:.0f} MB, {len(ORDER)} requests", flush=True)



send("Warm-up request, not recorded. " * 5, None)      # first-request overheads land here

recs = []
for i, (u, app, t) in enumerate(ORDER):
    m0 = metrics()
    hv.mark_idle(); hv.flush_tlb(wpid)
    c0 = cpu_ticks(wpid); t0 = time.time()
    usage, ttft, itl = send(prompt(u, app, t), SALT[u])
    lat = time.time() - t0; cpu_s = (cpu_ticks(wpid) - c0) / CLK
    bm = hv.read_idle()
    pfn = hv.snapshot(wpid, KV_VMA)[KV][2]
    idx = np.nonzero(hv.accessed_mask(pfn, bm))[0]
    m1 = metrics()
    rec = {"i": i, "user": u, "app": app, "turn": t, "prompt_tokens": usage["prompt_tokens"],
           "out_tokens": usage.get("completion_tokens"), "latency_s": round(lat, 3), "cpu_s": round(cpu_s, 3),
           "ttft_s": round(ttft, 4), "itl_s": round(itl, 4), "ttft_ratio": round(ttft / itl, 3),
           "hit_tokens": m1["hits"] - m0["hits"],
           "kv_pages": int(idx.size), "kv_usage_after": m1["kv_usage"], "kv_idx": idx.tolist()}
    recs.append(rec)
    if i == 2 and all(r["kv_pages"] == 0 for r in recs) or i == 2 and all(r["cpu_s"] < 0.05 for r in recs):
        sys.exit(f"[{SETTING}] TRACER_SEES_NOTHING: first 3 requests show no KV pages or no worker CPU; wrong process?")
    print(f"[{SETTING}] {i+1:3d}/{len(ORDER)} {u:10s} turn {t} tok {rec['prompt_tokens']:4d} "
          f"ttft {ttft:5.2f}s itl {itl:5.3f}s ratio {ttft/itl:6.1f} hit {rec['hit_tokens']:5.0f} kv {idx.size:5d}", flush=True)

# ---------------- other-user attacker: cache-hit probe (PromptPeek precondition) ----------------
# A victim sends a request starting with a fresh unique document. An attacker then sends a request starting with
# that same document (truth = hit possible) or with a different, never-sent document (truth = miss), and judges
# from its own time-to-first-token. Attackers: same_org (another user of the victim's app), cross_org (another app).
ALLWORDS = sorted({w for _, (_, ws) in APPS.items() for w in ws.split()} | set(GENERIC))
N_PROBE = 24


def doc(tag):
    r = random.Random(f"{SEED}-doc-{tag}")
    return "Document: " + sentences(r, ALLWORDS, 30, 14)


by_app = {}
for u, a in USERS:
    by_app.setdefault(a, []).append(u)
apps_list = list(by_app)
rp = random.Random(f"{SEED}-probes")
probes = []
for k in range(N_PROBE):
    vapp = apps_list[k % len(apps_list)]
    victim = by_app[vapp][k % len(by_app[vapp])]
    attackers = {"same_org": rp.choice([u for u in by_app[vapp] if u != victim]),
                 "cross_org": rp.choice(by_app[rp.choice([a for a in apps_list if a != vapp])])}
    for atype, att in attackers.items():
        real, ctrl = doc(f"{k}-{atype}-real"), doc(f"{k}-{atype}-ctrl")
        send(real + "\nQuestion: summarize this document.\nAssistant:", SALT[victim])   # victim's request
        order = [(True, real), (False, ctrl)]
        rp.shuffle(order)
        for truth, d_ in order:
            usage, ttft, itl = send(d_ + "\nQuestion: list the key terms.\nAssistant:", SALT[att])
            probes.append({"k": k, "attacker": atype, "victim": victim, "attacker_user": att, "truth_hit": truth,
                           "ttft_s": round(ttft, 4), "itl_s": round(itl, 4), "ttft_ratio": round(ttft / itl, 3),
                           "prompt_tokens": usage["prompt_tokens"]})
    print(f"[{SETTING}] probe {k+1:2d}/{N_PROBE} " + " ".join(
        f"{p['attacker']}:{'hit' if p['truth_hit'] else 'ctl'}={p['ttft_s']:.2f}s" for p in probes[-4:]), flush=True)

json.dump({"setting": SETTING, "seed": SEED, "probes": probes,
           "config": {"apps": len(APPS), "users_per_app": USERS_PER_APP, "turns": TURNS,
                      "system_sentences": SYS_SENTENCES, "max_tokens": MAX_TOKENS, "model": MODEL},
           "worker_rss_mb": rss, "metrics_end": metrics(), "requests": recs}, open(OUT, "w"))
print(f"[{SETTING}] RUN_DONE -> {OUT}", flush=True)
