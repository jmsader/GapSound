"""Tracer perturbation benchmark: fixed streaming load (C concurrent requests, ignore_eos, fixed lengths, taken from
the pre-generated workload) and the inter-token latency (ITL) and TTFT it sees. Run once without the tracer and once
per slice size; the job script starts/stops slicer.py around each run.
Usage: python bench_tracer.py <pregen.json> <label> <out.json> [concurrency=8] [rounds=3]
"""
import asyncio, json, random, statistics as st, sys, time
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
W = json.load(open(sys.argv[1])); LABEL = sys.argv[2]; OUT = sys.argv[3]
C = int(sys.argv[4]) if len(sys.argv) > 4 else 8
ROUNDS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
reqs = []
for s in W["sessions"]:
    msgs = [{"role": "system", "content": W["system"][s["app"]]}]
    for t, um in enumerate(s["user_msgs"]):
        msgs.append({"role": "user", "content": um})
        reqs.append(list(msgs))
        msgs.append({"role": "assistant", "content": s["replies"][t]})
random.Random(1).shuffle(reqs)


async def one(http, i, msgs, gaps, ttfts):
    body = {"model": MODEL, "messages": msgs, "max_tokens": 128, "ignore_eos": True, "stream": True,
            "cache_salt": f"bench-{LABEL}-{i}"}          # no cache hits: same work in every run
    t = time.monotonic(); last = None
    async with http.post(URL, json=body) as r:
        async for raw in r.content:
            if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                continue
            now = time.monotonic()
            if last is None:
                ttfts.append(now - t)
            else:
                gaps.append(now - last)
            last = now


async def main():
    gaps, ttfts = [], []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as http:
        t0 = time.monotonic()
        for r in range(ROUNDS):
            batch = [reqs[(r * C + k) % len(reqs)] for k in range(C)]
            await asyncio.gather(*[one(http, r * C + k, m, gaps, ttfts) for k, m in enumerate(batch)])
        dur = time.monotonic() - t0
    q = lambda x, p: sorted(x)[int(p * (len(x) - 1))]
    res = {"label": LABEL, "C": C, "rounds": ROUNDS, "seconds": dur, "itl_mean_ms": 1000 * st.mean(gaps),
           "itl_p50_ms": 1000 * q(gaps, .5), "itl_p90_ms": 1000 * q(gaps, .9), "itl_p99_ms": 1000 * q(gaps, .99),
           "ttft_mean_ms": 1000 * st.mean(ttfts)}
    json.dump(res, open(OUT, "w"))
    print("BENCH", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


asyncio.run(main())
