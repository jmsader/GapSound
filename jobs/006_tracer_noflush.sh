# timeout: 1500
cd /mnt/gapsound/v2
source serve.sh
export VLLM_CPU_KVCACHE_SPACE=2
start_server logs/srv006.log --enable-prefix-caching || exit 1
grep -A4 "OpenMP thread binding" logs/srv006.log | cut -c1-160
SP=$SRV_PID
mkdir -p /tmp/b6 && rm -f /tmp/b6/*
B() { taskset -c $CLIENT_CPUS python bench_tracer.py /tmp/p8.json $1 /tmp/b6/$1.json; }
S() {  # S <label> <slice_ms> [env]
  rm -f /tmp/b6/stop; env $3 taskset -c $TRACER_CPU python slicer.py $SP $2 /tmp/b6/$1.npz /tmp/b6/stop & TP=$!
  sleep 3; B $1; touch /tmp/b6/stop; wait $TP
  python -c "
import numpy as np; d=np.load('/tmp/b6/$1.npz'); n=np.diff(d['off']); pa=d['parts']; cyc=np.diff(pa[:,0]).mean()*1000
print('$1: pages/slice mean', n.mean().round(0), 'p90', np.percentile(n,90).round(0), '| cycle ms', cyc.round(1),
      '| window ms', np.median((d['t1']-d['t0'])*1000).round(1), '| distinct pages seen', np.unique(d['idx']).size)"
}
taskset -c $CLIENT_CPUS python bench_tracer.py /tmp/p8.json warm /tmp/b6/warm.json 8 1 >/dev/null
B none_a
S slice10 10
S slice25 25
S slice50 50
S slice100 100
B none_b
stop_server
