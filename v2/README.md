# GapSound v2 — full experiment

## Run it (inside the UTM VM)
    sudo pkill -f runner.py            # optional: stop the job runner so nothing else uses the server port
    sudo bash /mnt/gapsound/v2/run_all.sh
Results: `/mnt/gapsound/results/v2-<timestamp>/summary.md` (+ `summary.json`, `pressure/summary.md`).
It is resumable: if it stops, rerun with `sudo RUN=/mnt/gapsound/results/v2-<timestamp> bash /mnt/gapsound/v2/run_all.sh`
and finished pieces (pregen, capacity, each setting x load) are skipped.
Knobs (env vars): LOADS, SETTINGS, DURATION (s of arrivals per run), MIN_SESS, N_PROBE, SLICE_MS, KV_GB,
PRESSURE, PRESSURE_KV_GB, N_SESS, SEED. Defaults are the converged design.

## What runs
1. `workload.py`  — 400 scripted chat sessions, one per user, 4 apps (tax, HR, clinic, legal), themed system prompts
   and messages; turn counts follow WildChat (mean ~2.5, ~41% multi-turn); short/medium/long-paste messages.
2. `pregen.py`    — the served model writes every reply once (model's default sampling); histories carry real replies.
3. `calibrate.py` — capacity = request throughput with no cross-request cache hits (= most expensive defense), so
   10/50/90% load is stable for every setting and all settings get identical traffic.
4. For each load (10/50/90%) x setting (shared / orgsalt / salted / nocache): fresh vLLM server, tracer
   (`slicer.py`, core 13), live replay (`replay.py`, cores 14-15): Poisson session arrivals, closed-loop turns with
   reading (238 wpm) + typing (52 wpm) + pause think times, streamed responses recorded per event (time, bytes),
   other-user cache probes (same-org / cross-org attacker) interleaved.
5. Memory-pressure pass: load 50%, 1 GB KV cache, all four settings (`pressure/`).
6. `analyze2.py`  — every adversary x every defense combination; padding applied post hoc (both views per run).

## Design decisions made while building (and why)
- Tracer: no TLB flush (job 005: flush raised inter-token latency 73% and found the same pages), and only one page
  per 32 KB KV chunk is monitored (job 009: chunks are accessed as aligned units; cost 35 -> 8 ms per cycle).
  Slice 25 ms (~33 ms cycle); perturbation within run-to-run noise (~5%).
- CPU layout: model threads on cores 0-11 ("guest vCPUs"), tracer core 13 ("host"), clients 14-15.
- Host attribution adds a second row: "bg-subtracted" removes pages already being touched just before a request
  arrived (in-flight decodes read every running sequence's KV each step). The converged 1/k + pair-exclusion row is
  reported unchanged next to it.

## Time
Prints an estimate per load after calibration. Rough: pregen ~35 min, calibrate ~8 min, 10% load ~1.5 h per setting
(MIN_SESS=100 sessions at low rate), 50% and 90% ~40 min per setting, pressure ~40 min per setting: ~15 h total.
