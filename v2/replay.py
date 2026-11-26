"""Replay the pre-generated workload against a running vLLM server, as live chat traffic.

- Sessions (one per user) start as a Poisson process. Session rate = load * capacity / mean turns, so the request
  rate is `load` (fraction) of the server's measured saturation throughput (calibrate.py).
- Turns are closed-loop, like a person chatting: the next message is sent only after the reply finished streaming,
  plus a think time = time to read the reply (238 wpm, Brysbaert 2019) + time to type the next message
  (52 wpm, Dhakal et al. 2018) + a short pause (lognormal, median 2 s).
- Every request replays the recorded history (system prompt, user messages, recorded replies) and generates exactly
  the recorded reply length (ignore_eos + max_tokens), the standard serving-benchmark practice. Sampling uses the
  model defaults. Responses stream; for every streamed event we record arrival time and size in bytes (the
  application payload a network observer sees inside TLS records).
- Other-user probes run during the live traffic: a victim sends a fresh app-themed document; then an attacker from the
  same organization, or from another one, sends a request starting with that exact prefix (truth: hit possible) and
  one starting with a different, never-sent document (truth: miss), in random order, judging by time to first token.
- Cache settings set the cache_salt the gateway would attach: shared (none), orgsalt (per app), salted (per user),
  nocache (none; server runs without prefix caching).

Usage: python replay.py <pregen.json> <setting> <load> <capacity_rps> <out.json> [seed] [n_probe_per_type] [n_sessions]
"""
import asyncio, hashlib, json, math, random, sys, time
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
pg_path, SETTING, LOAD, CAP, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
N_PROBE = int(sys.argv[7]) if len(sys.argv) > 7 else 24
N_SESS = int(sys.argv[8]) if len(sys.argv) > 8 else 0
assert SETTING in ("shared", "orgsalt", "salted", "nocache")
READ_WPM, TYPE_WPM, PAUSE_MEDIAN_S = 238.0, 52.0, 2.0
import os
THINK_SCALE = float(os.environ.get("THINK_SCALE", "1.0"))   # tests only (shortens runs); the experiment uses 1.0

W = json.load(open(pg_path))
SESS = sorted(W["sessions"], key=lambda s: s["order"])
if N_SESS:
    SESS = SESS[:N_SESS]
rng = random.Random(f"replay-{SEED}")
mean_turns = sum(len(s["replies"]) for s in SESS) / len(SESS)
req_rate = LOAD * CAP
sess_rate = req_rate / mean_turns


def salt(user, app):
    if SETTING == "salted":
        return hashlib.sha256(f"gateway-secret-user-{user}".encode()).hexdigest()[:16]
    if SETTING == "orgsalt":
        return hashlib.sha256(f"gateway-secret-org-{app}".encode()).hexdigest()[:16]
    return None


def think_time(reply_text, next_msg, r):
    read = len(reply_text.split()) / READ_WPM * 60
    typ = len(next_msg.split()) / TYPE_WPM * 60
    pause = r.lognormvariate(math.log(PAUSE_MEDIAN_S), 0.5)
    return THINK_SCALE * (read + typ + pause)


T_START = None
RECS = []
ERRORS = []


async def send(http, messages, max_tokens, salt_value, meta):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    if salt_value:
        body["cache_salt"] = salt_value
    req_bytes = len(json.dumps(body).encode())          # request size an on-path observer sees (TLS ~ +constant)
    t_send = time.monotonic()
    ev_t, ev_b, usage, first = [], [], None, None
    async with http.post(URL, json=body) as r:
        if r.status != 200:
            txt = await r.text()
            raise RuntimeError(f"HTTP {r.status}: {txt[:300]}")
        async for raw in r.content:                      # one SSE line per iteration
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            now = time.monotonic()
            payload = line[6:]
            if payload == b"[DONE]":
                ev_t.append(now); ev_b.append(len(raw)); break
            obj = json.loads(payload)
            ev_t.append(now); ev_b.append(len(raw))
            if obj.get("choices") and obj["choices"][0].get("delta", {}).get("content") and first is None:
                first = now
            if obj.get("usage"):
                usage = obj["usage"]
    t_end = time.monotonic()
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    rec = dict(meta, t_send=t_send, req_bytes=req_bytes, t_first=first if first is not None else t_end, t_end=t_end,
               prompt_tokens=(usage or {}).get("prompt_tokens"), out_tokens=(usage or {}).get("completion_tokens"),
               cached_tokens=cached, ev_t=[round(x - t_send, 5) for x in ev_t], ev_bytes=ev_b, salt=salt_value)
    RECS.append(rec)
    return rec


async def session(http, s, start_at):
    r = random.Random(f"session-{SEED}-{s['sid']}")
    await asyncio.sleep(max(0.0, start_at - (time.monotonic() - T_START)))
    msgs = [{"role": "system", "content": W["system"][s["app"]]}]
    for t, um in enumerate(s["user_msgs"]):
        msgs.append({"role": "user", "content": um})
        try:
            await send(http, list(msgs), s["reply_tokens"][t], salt(s["user"], s["app"]),
                       {"kind": "workload", "user": s["user"], "app": s["app"], "turn": t, "sid": s["sid"],
                        "msg_class": s["msg_classes"][t], "rec_reply_tokens": s["reply_tokens"][t]})
        except Exception as e:                       # keep the run going; the error is recorded and counted
            ERRORS.append({"user": s["user"], "turn": t, "error": repr(e)[:300]})
            print(f"[{SETTING}] REQUEST_ERROR {s['user']} turn {t}: {repr(e)[:200]}", flush=True)
            return
        msgs.append({"role": "assistant", "content": s["replies"][t]})
        if t + 1 < len(s["user_msgs"]):
            await asyncio.sleep(think_time(s["replies"][t], s["user_msgs"][t + 1], r))


sys.path.insert(0, __file__.rsplit("/", 1)[0])
from content import DOC_WORDS


def doc(app, tag):
    r = random.Random(f"doc-{SEED}-{tag}")
    words = DOC_WORDS[app]
    sents = [" ".join(r.choice(words) for _ in range(r.randint(10, 16))).capitalize() + "." for _ in range(30)]
    return "Please review this document I received:\n\n" + " ".join(sents)


async def probe_episode(*a):
    try:
        await _probe_episode(*a)
    except Exception as e:
        ERRORS.append({"probe": a[1], "error": repr(e)[:300]})
        print(f"[{SETTING}] PROBE_ERROR {repr(e)[:200]}", flush=True)


async def _probe_episode(http, k, atype, at, victim, attacker):
    await asyncio.sleep(max(0.0, at - (time.monotonic() - T_START)))
    vapp = victim["app"]
    sysmsg = {"role": "system", "content": W["system"][vapp]}
    real, ctrl = doc(vapp, f"{k}-{atype}-real"), doc(vapp, f"{k}-{atype}-ctrl")
    meta = {"kind": "probe_victim", "probe": k, "attacker_type": atype, "user": victim["user"], "app": vapp}
    await send(http, [sysmsg, {"role": "user", "content": real + "\n\nWhat does this mean for me?"}], 32,
               salt(victim["user"], vapp), meta)
    await asyncio.sleep(1.0)
    order = [(True, real), (False, ctrl)]
    random.Random(f"probe-order-{SEED}-{k}-{atype}").shuffle(order)
    for truth, d in order:
        # the attacker reproduces the victim's prefix (system prompt + document) and asks something else
        await send(http, [sysmsg, {"role": "user", "content": d + "\n\nList the key terms in it."}], 8,
                   salt(attacker["user"], attacker["app"]),
                   {"kind": "probe_attack", "probe": k, "attacker_type": atype, "truth_hit": truth,
                    "user": attacker["user"], "app": attacker["app"], "victim": victim["user"]})


async def main():
    global T_START
    # Poisson session starts
    starts, t = [], 0.0
    for _ in SESS:
        t += rng.expovariate(sess_rate)
        starts.append(t)
    horizon = starts[-1]
    by_app = {}
    for s in SESS:
        by_app.setdefault(s["app"], []).append(s)
    apps = sorted(by_app)
    pr = random.Random(f"probes-{SEED}")
    episodes = []
    for k in range(N_PROBE):
        for atype in ("same_org", "cross_org"):
            vapp = apps[k % len(apps)]
            victim = pr.choice(by_app[vapp])
            if atype == "same_org":
                attacker = pr.choice([s for s in by_app[vapp] if s["user"] != victim["user"]])
            else:
                attacker = pr.choice(by_app[pr.choice([a for a in apps if a != vapp])])
            episodes.append((k, atype, pr.uniform(0.1, 0.9) * horizon, victim, attacker))
    print(f"[{SETTING}] load {LOAD:.2f} x capacity {CAP:.3f} req/s -> {req_rate:.3f} req/s; {len(SESS)} sessions, "
          f"mean turns {mean_turns:.2f}, session starts over {horizon:.0f} s; {len(episodes)} probe episodes",
          flush=True)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=7200)) as http:
        T_START = time.monotonic()
        tasks = [session(http, s, st) for s, st in zip(SESS, starts)]
        tasks += [probe_episode(http, *e) for e in episodes]

        async def progress():
            while True:
                await asyncio.sleep(30)
                n = sum(r["kind"] == "workload" for r in RECS)
                print(f"[{SETTING}] t={time.monotonic()-T_START:5.0f}s  workload requests done {n}", flush=True)
        pt = asyncio.create_task(progress())
        await asyncio.gather(*tasks)
        pt.cancel()
    json.dump({"setting": SETTING, "load": LOAD, "capacity_rps": CAP, "seed": SEED, "t_start": T_START,
               "requests": RECS, "errors": ERRORS, "read_wpm": READ_WPM, "type_wpm": TYPE_WPM, "think_scale": THINK_SCALE}, open(OUT, "w"))
    wl = [r for r in RECS if r["kind"] == "workload"]
    dur = max(r["t_end"] for r in RECS) - T_START
    print(f"[{SETTING}] REPLAY_DONE {len(wl)} workload + {len(RECS)-len(wl)} probe requests in {dur:.0f}s "
          f"(achieved {len(RECS)/dur:.3f} req/s), {len(ERRORS)} errors -> {OUT}", flush=True)


asyncio.run(main())
