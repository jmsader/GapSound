# timeout: 1800
cd /mnt/gapsound/v2
source serve.sh
export VLLM_CPU_KVCACHE_SPACE=2
start_server logs/srv005.log --enable-prefix-caching || exit 1
grep -A4 "OpenMP thread binding" logs/srv005.log | cut -c1-160
SP=$SRV_PID
mkdir -p /tmp/b5 && rm -f /tmp/b5/*
B() { taskset -c $CLIENT_CPUS python bench_tracer.py /tmp/p8.json $1 /tmp/b5/$1.json; }
S() {  # S <label> <slice_ms> [env]
  rm -f /tmp/b5/stop; env $3 taskset -c $TRACER_CPU python slicer.py $SP $2 /tmp/b5/$1.npz /tmp/b5/stop & TP=$!
  sleep 3; B $1; touch /tmp/b5/stop; wait $TP
  python -c "
import numpy as np; d=np.load('/tmp/b5/$1.npz'); n=np.diff(d['off']); pa=d['parts']; cyc=np.diff(pa[:,0]).mean()*1000
print('$1: pages/slice mean', n.mean().round(0), 'p90', np.percentile(n,90).round(0), '| cycle ms', cyc.round(1),
      '| window ms', np.median((d['t1']-d['t0'])*1000).round(1), '| distinct pages seen', np.unique(d['idx']).size)"
}
taskset -c $CLIENT_CPUS python bench_tracer.py /tmp/p8.json warm /tmp/b5/warm.json 8 1 >/dev/null
B none_a
S flush50 50
S noflush50 50 SLICER_NOFLUSH=1
S flush25 25
S flush100 100
S flush200 200
B none_b
stop_server
