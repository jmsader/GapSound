# timeout: 10800
# 1) pre-generation + capacity for the MAIN run (run_all.sh will reuse these and skip both steps)
# 2) tracer slowdown at 90% load with realistic think times: no tracer vs 25 ms slices (same sessions, same seed)
set -u
cd /mnt/gapsound/v2
source serve.sh
RUN=/mnt/gapsound/results/v2-main; mkdir -p $RUN/logs $RUN/perturb
export VLLM_CPU_KVCACHE_SPACE=2
if [ ! -s $RUN/pregen.json ]; then
  taskset -c $CLIENT_CPUS python workload.py 400 $RUN/workload.json 0 || exit 1
  start_server $RUN/logs/srv-pregen.log --enable-prefix-caching || exit 1
  taskset -c $CLIENT_CPUS python pregen.py $RUN/workload.json $RUN/pregen.json 1024 16 || { stop_server; exit 1; }
  stop_server
fi
if [ ! -s $RUN/capacity.json ]; then
  start_server $RUN/logs/srv-calibrate.log --enable-prefix-caching || exit 1
  taskset -c $CLIENT_CPUS python calibrate.py $RUN/pregen.json 200 $RUN/capacity.json || { stop_server; exit 1; }
  stop_server
fi
CAP=$(python -c "import json; print(json.load(open('$RUN/capacity.json'))['throughput_rps'])")
MT=$(python -c "import json; w=json.load(open('$RUN/pregen.json'))['sessions']; print(sum(len(s['replies']) for s in w)/len(w))")
N=$(python -c "import math; print(min(400, math.ceil(0.9*$CAP*600/$MT)))")
echo "capacity $CAP req/s, mean turns $MT -> 90% load, $N sessions over ~10 min of arrivals"
P=$RUN/perturb
for MODE in none 25; do
  [ -s $P/replay-$MODE.json ] && continue
  start_server $P/srv-$MODE.log --enable-prefix-caching || exit 1
  if [ $MODE != none ]; then
    rm -f /tmp/gs_stop
    taskset -c $TRACER_CPU python slicer.py $SRV_PID $MODE $P/trace-$MODE.npz /tmp/gs_stop > $P/slicer-$MODE.log 2>&1 & TP=$!
    sleep 3
  fi
  taskset -c $CLIENT_CPUS python replay.py $RUN/pregen.json shared 0.9 $CAP $P/replay-$MODE.json 0 24 $N
  [ $MODE != none ] && { touch /tmp/gs_stop; wait $TP; tail -1 $P/slicer-$MODE.log; }
  stop_server
done
python perturb_compare.py $P/replay-none.json $P/replay-25.json
