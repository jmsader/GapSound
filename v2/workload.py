"""Build the GapSound workload (deterministic for a seed): one chat session per simulated user.

Realism sources (cited in the paper):
- Turns per conversation follow WildChat (Zhao et al., ICLR 2024): mean 2.52 turns, ~41% multi-turn.
  Model: P(1 turn) = 0.59; otherwise 2 + Geometric with mean 2.71, so the overall mean is 2.52. Capped at 10.
- User-message lengths: short/medium/long-paste mixture (0.40/0.35/0.25) giving a mean near the ~70 tokens per
  user prompt reported for LMSYS-Chat-1M (Zheng et al., ICLR 2024).
Replies are not here: they come from pre-generation with the served model (pregen.py).
Usage: python workload.py <n_sessions> <out.json> [seed]
"""
import json, random, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from content import SYSTEM, OPENERS, FOLLOWUPS, NAMES, MONTHS

CLASS_P = [("short", 0.40), ("medium", 0.35), ("long", 0.25)]
LEADS = ["", "", "", "Hi, ", "Hello. ", "Quick question: ", "Hi there! ", "Sorry to bother you, "]
TAILS = ["", "", "", " Thanks!", " Thank you.", " Any help is appreciated.", " thanks"]


def turns_wildchat(r):
    if r.random() < 0.59:
        return 1
    extra = 0
    while r.random() > 1 / 3.71:          # geometric on {0,1,...} with mean 2.71
        extra += 1
    return min(2 + extra, 10)


def pick_class(r):
    x, acc = r.random(), 0.0
    for c, p in CLASS_P:
        acc += p
        if x < acc:
            return c
    return "short"


def fill(t, r):
    amts = [f"{r.randint(1, 98) * 50 + r.randint(0, 49):,}" for _ in range(7)]
    keys = dict(n=r.randint(2, 9), name=r.choice(NAMES), month=r.choice(MONTHS), d=r.randint(1, 28),
                amt=amts[0], amt2=amts[1], amt3=amts[2], amt4=amts[3], amt5=amts[4], amt6=amts[5], amt7=amts[6])
    return t.format(**keys)


def message(bank, used, r):
    cls = pick_class(r)
    pool = [i for i, (c, _) in enumerate(bank) if c == cls and i not in used] or \
           [i for i in range(len(bank)) if i not in used] or list(range(len(bank)))
    i = r.choice(pool)
    used.add(i)
    text = fill(bank[i][1], r)
    lead = r.choice(LEADS)
    if lead:
        lower = lead.endswith((", ", ": ")) and not text.startswith(("I ", "I'"))
        text = lead + (text[0].lower() + text[1:] if lower else text)
    return text + r.choice(TAILS), bank[i][0]


def build(n_sessions, seed=0):
    r = random.Random(f"gapsound-workload-{seed}")
    apps = list(SYSTEM)
    sessions = []
    for s in range(n_sessions):
        app = apps[s % len(apps)]
        rs = random.Random(f"gapsound-session-{seed}-{s}")
        n_turns = turns_wildchat(rs)
        used_o, used_f = set(), set()
        msgs, classes = [], []
        m, c = message(OPENERS[app], used_o, rs); msgs.append(m); classes.append(c)
        for _ in range(n_turns - 1):
            m, c = message(FOLLOWUPS[app], used_f, rs); msgs.append(m); classes.append(c)
        sessions.append({"sid": s, "user": f"{app}-u{s // len(apps)}", "app": app, "user_msgs": msgs,
                         "msg_classes": classes})
    r.shuffle(sessions)                    # session start order is random, not round-robin by app
    for k, s in enumerate(sessions):
        s["order"] = k
    return {"seed": seed, "system": SYSTEM, "sessions": sessions}


if __name__ == "__main__":
    n, out = int(sys.argv[1]), sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    w = build(n, seed)
    json.dump(w, open(out, "w"), indent=1)
    t = [len(s["user_msgs"]) for s in w["sessions"]]
    words = [len(m.split()) for s in w["sessions"] for m in s["user_msgs"]]
    print(f"{n} sessions, {sum(t)} user turns, mean turns {sum(t)/n:.2f}, multi-turn {sum(x > 1 for x in t)/n:.0%}, "
          f"mean user words {sum(words)/len(words):.0f} (~{1.33*sum(words)/len(words):.0f} tokens)")
