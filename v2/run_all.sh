#!/bin/bash
# GapSound v2: full experiment. Run as root inside the VM:   sudo bash /mnt/gapsound/v2/run_all.sh
# (or drop a job file that calls it). Resumable: re-running with the same RUN dir skips finished pieces.
#
#   1. workload.py  : N_SESS scripted sessions (WildChat-like turn counts, app-themed messages)
#   2. pregen.py    : the model writes every reply once (default sampling; histories carry the real replies)
#   3. calibrate.py : saturation throughput (req/s) with no cross-request cache hits = capacity of the most
#                     expensive defense (no prefix caching), so every setting stays stable at the highest load
#   4. for each LOAD x SETTING: fresh server, tracer (core 13), replay of live traffic with probes (cores 14-15)
#   5. PRESSURE=1 adds the memory-pressure pass (smaller KV cache, load 0.5)
#   6. analyze2.py  : all adversaries x all defense combinations (padding applied post hoc)
set -u
cd /mnt/gapsound/v2
source serve.sh
RUN=${RUN:-/mnt/gapsound/results/v2-$(date +%Y%m%d-%H%M%S)}
N_SESS=${N_SESS:-400}            # sessions pre-generated (the replay uses as many as each load needs)
SEED=${SEED:-0}
LOADS=${LOADS:-"0.1 0.5 0.9"}
SETTINGS=${SETTINGS:-"shared orgsalt salted nocache"}
DURATION=${DURATION:-1800}       # seconds of session arrivals per run
MIN_SESS=${MIN_SESS:-100}        # at least this many sessions per run (low load runs longer)
N_PROBE=${N_PROBE:-24}           # probe episodes per attacker type per run
SLICE_MS=${SLICE_MS:-25}         # tracer window (jobs/009: sampled tracer costs 8 ms/cycle; 10-100 ms slices all within run-to-run ITL noise, ~5%)
KV_GB=${KV_GB:-2}
PRESSURE=${PRESSURE:-1}
PRESSURE_KV_GB=${PRESSURE_KV_GB:-1}
mkdir -p $RUN/logs
echo "RUN=$RUN"; exec > >(tee -a $RUN/run_all.log) 2>&1
[ "${THINK_SCALE:-1}" != 1 ] && echo "WARNING: THINK_SCALE=$THINK_SCALE (test mode, not the experiment)"
echo "config: N_SESS=$N_SESS SEED=$SEED LOADS=$LOADS SETTINGS=$SETTINGS DURATION=$DURATION MIN_SESS=$MIN_SESS N_PROBE=$N_PROBE SLICE_MS=$SLICE_MS KV_GB=$KV_GB PRESSURE=$PRESSURE"
trap 'stop_server; pkill -f slicer.py; exit 130' INT TERM

flag() { [ "$1" = nocache ] && echo "--no-enable-prefix-caching" || echo "--enable-prefix-caching"; }

# 1-2. workload + replies
if [ ! -s $RUN/pregen.json ]; then
  taskset -c $CLIENT_CPUS python workload.py $N_SESS $RUN/workload.json $SEED || exit 1
  export VLLM_CPU_KVCACHE_SPACE=$KV_GB
  start_server $RUN/logs/srv-pregen.log --enable-prefix-caching || exit 1
  taskset -c $CLIENT_CPUS python pregen.py $RUN/workload.json $RUN/pregen.json 1024 16 || { stop_server; exit 1; }
  stop_server
fi
# 3. capacity
if [ ! -s $RUN/capacity.json ]; then
  export VLLM_CPU_KVCACHE_SPACE=$KV_GB
  start_server $RUN/logs/srv-calibrate.log --enable-prefix-caching || exit 1
  taskset -c $CLIENT_CPUS python calibrate.py $RUN/pregen.json 200 $RUN/capacity.json || { stop_server; exit 1; }
  stop_server
fi
CAP=$(python -c "import json; print(json.load(open('$RUN/capacity.json'))['throughput_rps'])")
MT=$(python -c "import json; w=json.load(open('$RUN/pregen.json'))['sessions']; print(sum(len(s['replies']) for s in w)/len(w))")
NAVAIL=$(python -c "import json; print(len(json.load(open('$RUN/pregen.json'))['sessions']))")
echo "capacity $CAP req/s, mean turns $MT, sessions available $NAVAIL"
nsess() { python -c "
import math; n=max($MIN_SESS, math.ceil($1*$CAP*$DURATION/$MT)); print(min(n, $NAVAIL))"; }
for L in $LOADS; do
  python -c "
n=$(nsess $L); r=$L*$CAP; print(f'load $L: {r:.3f} req/s, {n} sessions, arrivals over ~{n*$MT/r/60:.0f} min per setting')"
done

one_run() {  # one_run <setting> <load> <kv_gb> <outdir>
  local S=$1 L=$2 KV=$3 OUT=$4
  [ -s $OUT/replay-$S-L$L.json ] && { echo "skip $S L$L (done)"; return 0; }
  export VLLM_CPU_KVCACHE_SPACE=$KV
  start_server $OUT/logs/srv-$S-L$L.log $(flag $S) || return 1
  rm -f /tmp/gs_stop
  taskset -c $TRACER_CPU python slicer.py $SRV_PID $SLICE_MS $OUT/trace-$S-L$L.npz /tmp/gs_stop > $OUT/logs/slicer-$S-L$L.log 2>&1 &
  local TP=$!
  sleep 3
  kill -0 $TP 2>/dev/null || { echo "TRACER_FAILED"; cat $OUT/logs/slicer-$S-L$L.log; stop_server; return 1; }
  taskset -c $CLIENT_CPUS python replay.py $RUN/pregen.json $S $L $CAP $OUT/replay-$S-L$L.json.tmp $SEED $N_PROBE $(nsess $L)
  local rc=$?
  touch /tmp/gs_stop; wait $TP; cat $OUT/logs/slicer-$S-L$L.log | tail -1
  grep -c '"POST /v1/chat/completions' $OUT/logs/srv-$S-L$L.log | sed 's/^/server POSTs: /'
  stop_server
  [ $rc -eq 0 ] && mv $OUT/replay-$S-L$L.json.tmp $OUT/replay-$S-L$L.json
  return $rc
}

# 4. main grid
for L in $LOADS; do for S in $SETTINGS; do
  echo "=== $(date +%T) setting $S load $L"; one_run $S $L $KV_GB $RUN || echo "RUN_FAILED $S L$L"
done; done
# 5. memory pressure
if [ "$PRESSURE" = 1 ]; then
  mkdir -p $RUN/pressure/logs
  for S in $SETTINGS; do
    echo "=== $(date +%T) pressure: setting $S load 0.5, KV ${PRESSURE_KV_GB} GB"
    one_run $S 0.5 $PRESSURE_KV_GB $RUN/pressure || echo "RUN_FAILED pressure $S"
  done
fi
# 6. analysis
python analyze2.py $RUN && [ "$PRESSURE" = 1 ] && python analyze2.py $RUN/pressure
echo "ALL_DONE $(date +%T) -> $RUN/summary.md"
