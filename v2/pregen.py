"""Pre-generate the bot's replies once, with the served model and its default sampling settings
(vLLM applies the model's generation_config). Each session is played turn by turn; the real reply goes into
the history for the next turn, exactly as a chat app would send it. Sessions run concurrently to save time.
Replies are later replayed (replay.py): histories are fixed, and each live response is generated to the
recorded length, the standard serving-benchmark practice.
Usage: python pregen.py <workload.json> <out.json> [max_tokens=1024] [concurrency=16]
"""
import asyncio, json, sys, time
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
wl_path, out_path = sys.argv[1], sys.argv[2]
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 16
W = json.load(open(wl_path))


async def run_session(sess, http, sem):
    msgs = [{"role": "system", "content": W["system"][sess["app"]]}]
    rec = {"replies": [], "reply_tokens": [], "finish": [], "prompt_tokens": []}
    async with sem:
        for t, um in enumerate(sess["user_msgs"]):
            msgs.append({"role": "user", "content": um})
            body = {"model": MODEL, "messages": msgs, "max_tokens": MAX_TOKENS}
            async with http.post(URL, json=body) as r:
                j = await r.json()
            if r.status != 200:
                # e.g. context too long: the conversation ends here, as an app would cut it off
                rec["stopped_at"] = t
                rec["stop_reason"] = str(j)[:200]
                break
            ch = j["choices"][0]
            reply = ch["message"]["content"]
            rec["replies"].append(reply)
            rec["reply_tokens"].append(j["usage"]["completion_tokens"])
            rec["prompt_tokens"].append(j["usage"]["prompt_tokens"])
            rec["finish"].append(ch["finish_reason"])
            msgs.append({"role": "assistant", "content": reply})
    return rec


async def main():
    sem = asyncio.Semaphore(CONC)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as http:
        recs = await asyncio.gather(*[run_session(s, http, sem) for s in W["sessions"]])
    for s, r in zip(W["sessions"], recs):
        n = len(r["replies"])
        s["user_msgs"] = s["user_msgs"][:n]          # drop turns that could not be served
        s["msg_classes"] = s["msg_classes"][:n]
        s.update({k: v for k, v in r.items()})
    W["sessions"] = [s for s in W["sessions"] if s["replies"]]
    W["pregen"] = {"model": MODEL, "max_tokens": MAX_TOKENS, "sampling": "model default (generation_config)",
                   "seconds": round(time.time() - t0, 1)}
    json.dump(W, open(out_path, "w"), indent=1)
    rt = [x for s in W["sessions"] for x in s["reply_tokens"]]
    pt = [x for s in W["sessions"] for x in s["prompt_tokens"]]
    fin = [x for s in W["sessions"] for x in s["finish"]]
    print(f"PREGEN_DONE {len(W['sessions'])} sessions, {len(rt)} turns in {time.time()-t0:.0f}s | reply tokens "
          f"mean {sum(rt)/len(rt):.0f} max {max(rt)} | prompt tokens mean {sum(pt)/len(pt):.0f} max {max(pt)} | "
          f"hit cap {fin.count('length')}")


asyncio.run(main())
