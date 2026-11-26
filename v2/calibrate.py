"""Measure the server's saturation throughput for this workload (requests/s), like vLLM's benchmark_serving with
request_rate=inf: every turn of the pre-generated workload becomes one request (recorded history, recorded reply
length, ignore_eos), all are submitted at once in random order, and throughput = requests / makespan.
Run it against the server configuration that defines capacity (see run_all.sh).
Usage: python calibrate.py <pregen.json> <n_requests> <out.json>
"""
import asyncio, json, random, sys, time
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
W = json.load(open(sys.argv[1])); N = int(sys.argv[2]); OUT = sys.argv[3]
reqs = []
for s in W["sessions"]:
    msgs = [{"role": "system", "content": W["system"][s["app"]]}]
    for t, um in enumerate(s["user_msgs"][:len(s["replies"])]):
        msgs.append({"role": "user", "content": um})
        reqs.append((list(msgs), s["reply_tokens"][t]))
        msgs.append({"role": "assistant", "content": s["replies"][t]})
random.Random(0).shuffle(reqs)
reqs = reqs[:N]


async def one(http, i, msgs, mt, done):
    body = {"model": MODEL, "messages": msgs, "max_tokens": mt, "ignore_eos": True,
            "cache_salt": f"calib-{i}"}   # unique salt: no cross-request cache hits (worst case = cost of nocache)
    async with http.post(URL, json=body) as r:
        j = await r.json()
        assert r.status == 200, j
    done.append((time.monotonic(), j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"]))


async def main():
    done = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                     timeout=aiohttp.ClientTimeout(total=7200)) as http:
        # warm-up (not timed)
        await one(http, -1, reqs[0][0], 8, [])
        t0 = time.monotonic()
        await asyncio.gather(*[one(http, i, m, mt, done) for i, (m, mt) in enumerate(reqs)])
    t1 = time.monotonic()
    ts = sorted(d[0] for d in done)
    # throughput_rps = requests / makespan (vLLM benchmark_serving's request throughput) is what run_all.sh uses;
    # steady_rps (middle 80% of completions) is reported for reference only: in a burst, completions bunch up,
    # so it overstates what the server sustains.
    a, b = int(0.1 * len(ts)), int(0.9 * len(ts))
    steady = (b - a) / (ts[b - 1] - ts[a]) if ts[b - 1] > ts[a] else len(ts) / (t1 - t0)
    res = {"n": len(done), "makespan_s": t1 - t0, "throughput_rps": len(done) / (t1 - t0), "steady_rps": steady,
           "prompt_tokens_mean": sum(d[1] for d in done) / len(done),
           "out_tokens_mean": sum(d[2] for d in done) / len(done)}
    json.dump(res, open(OUT, "w"), indent=1)
    print("CALIBRATE", json.dumps(res), flush=True)


asyncio.run(main())
