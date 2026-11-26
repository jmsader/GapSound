# GapSound

Side channels in confidential (TEE-based) LLM inference, and how the defenses people already deploy interact.
CSE 588 project.

A confidential VM keeps the cloud host from reading a model server's memory, but the host can still watch *which*
memory pages are touched and *when*. Other users of the same server can time the shared prefix cache, and anyone on
the network can watch the encrypted response stream. Each of these threats has a deployed fix (per-org or per-user
cache isolation, turning prefix caching off, response padding). This project measures every attacker against every
combination of those fixes, to see where a fix for one attacker helps or hurts another.

## Attackers and questions
- **Host** (emulated hypervisor watching KV-cache page accesses): are two requests from the same user? from the same
  app? which earlier request is this message's previous turn?
- **Other user** (PromptPeek-style probe): is someone else's prompt in the cache? (time to first token)
- **Network** (Whisper Leak-style): which app is this? is the reply long or short? (sizes and timing of the stream)

Defenses: shared cache, per-org salt, per-user salt, no prefix caching; each with and without response padding.
Traffic: simulated chat sessions at 10/50/90% of server capacity, plus a memory-pressure pass. Metric: AUROC.

## Layout
- `v2/` — the current experiment (see `v2/README.md` for details and design decisions)
  - `workload.py`, `content.py` — scripted multi-turn chat sessions for four apps (tax, HR, clinic, legal)
  - `pregen.py` — the model writes every reply once; `replay.py` replays sessions as live, concurrent traffic
  - `calibrate.py` — server capacity; `slicer.py` — time-sliced page-access tracer (runs as root)
  - `analyze2.py` — all attackers × all defenses; `run_all.sh` — the whole pipeline
  - `sim_test.py`, `bench_tracer.py`, `perturb_compare.py`, `check_layout.py` — tests and tracer benchmarks
- `hostview.py` — page-tracking helpers (pagemap, idle-page bitmap) shared by both versions
- `run_exp.*`, `analyze.py`, `smoke.sh`, `trace_check.*` — the earlier, sequential version (checkpoint)
- `setup.sh` — VM setup (vLLM CPU build, Python env); `runner.py` — runs job scripts dropped into `jobs/`
- `results/` — summaries of finished runs

## Running
Inside an Ubuntu VM (tested: UTM on Apple silicon, 16 vCPUs, 16 GB RAM) with this folder at `/mnt/gapsound`:

    bash setup.sh                          # once
    sudo bash v2/run_all.sh                # full run, ~15 h; resumable with RUN=<results dir>

The model is Qwen2.5-0.5B-Instruct served by vLLM on CPU. Results land in `results/v2-<timestamp>/summary.md`.

## Limitations
The host is emulated from inside the guest (Linux idle-page tracking, not real nested page tables); user messages
are scripted and replies are pre-generated then replayed; one small model on CPU; the tracer slightly perturbs
timing.
