# timeout: 1800
cd /mnt/gapsound/v2
echo "--- KV layout check on the full-page trace from job 006"; python check_layout.py /tmp/b6/slice50.npz
source serve.sh
export VLLM_CPU_KVCACHE_SPACE=2
start_server logs/srv009.log --enable-prefix-caching || exit 1
SP=$SRV_PID
mkdir -p /tmp/b9 && rm -f /tmp/b9/*
B() { taskset -c $CLIENT_CPUS python bench_tracer.py /tmp/p8.json $1 /tmp/b9/$1.json; }
P() { python -c "
import numpy as np; d=np.load('/tmp/b9/$1.npz'); n=np.unpackbits(d['bits'],axis=1).sum(1); pa=d['parts']
print('$1: monitored', d['mon'].size, '| pages/slice mean', n.mean().round(0), 'p90', np.percentile(n,90).round(0), '| cycle ms', (np.diff(pa[:,0]).mean()*1000).round(1),
      '| cost ms mark/read', (np.median(pa[:,2])*1000).round(1), (np.median(pa[:,4])*1000).round(1), '| distinct', np.unpackbits(np.bitwise_or.reduce(d['bits'],0)).sum())"; }
S() { rm -f /tmp/b9/stop; env $3 taskset -c $TRACER_CPU python slicer.py $SP $2 /tmp/b9/$1.npz /tmp/b9/stop & TP=$!
      sleep 3; B $1; touch /tmp/b9/stop; wait $TP; P $1; }
taskset -c $CLIENT_CPUS python bench_tracer.py /tmp/p8.json warm /tmp/b9/warm.json 8 1 >/dev/null
B none_a
timeout 4 taskset -c $TRACER_CPU python slicer.py $SP 20 /tmp/b9/idle.npz; P idle
S s10 10
S s25 25
S s50 50
S s100 100
S full50 50 SLICER_SAMPLE=1
B none_b
stop_server
grep BENCH /mnt/gapsound/jobs/009_sampled_tracer.out | sed 's/BENCH //' | python -c "
import sys,json
R=[json.loads(l) for l in sys.stdin]; b=[r for r in R if r['label'].startswith('none')]
bm=sum(r['itl_mean_ms'] for r in b)/2; bp=sum(r['itl_p50_ms'] for r in b)/2; bt=sum(r['ttft_mean_ms'] for r in b)/2
for r in R: print(r['label'].ljust(9), 'ITL mean %+5.1f%%  p50 %+5.1f%%  TTFT %+5.1f%%'%(100*(r['itl_mean_ms']/bm-1),100*(r['itl_p50_ms']/bp-1),100*(r['ttft_mean_ms']/bt-1)))"
