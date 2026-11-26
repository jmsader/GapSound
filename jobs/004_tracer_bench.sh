# timeout: 1500
cd /mnt/gapsound/v2
source serve.sh
export VLLM_CPU_KVCACHE_SPACE=2
start_server logs/srv004.log --enable-prefix-caching || exit 1
SP=$SRV_PID
mkdir -p /tmp/b4 && rm -f /tmp/b4/*
python bench_tracer.py /tmp/p8.json warm /tmp/b4/warm.json 8 1 >/dev/null     # warm-up
python bench_tracer.py /tmp/p8.json none_a /tmp/b4/none_a.json || exit 1
# idle check: with no traffic, a slice should see (almost) no KV pages accessed
timeout 4 python slicer.py $SP 20 /tmp/b4/idle.npz; python -c "
import numpy as np; d=np.load('/tmp/b4/idle.npz'); n=np.diff(d['off'])
print('IDLE slices', n.size, 'KV pages/slice mean', n.mean().round(1), 'max', n.max(), 'of', d['kv_pages'])"
for MS in 2 5 10 20 50; do
  rm -f /tmp/b4/stop; python slicer.py $SP $MS /tmp/b4/s$MS.npz /tmp/b4/stop & TP=$!
  sleep 3
  python bench_tracer.py /tmp/p8.json slice$MS /tmp/b4/slice$MS.json
  touch /tmp/b4/stop; wait $TP
  python -c "
import numpy as np; d=np.load('/tmp/b4/s$MS.npz'); n=np.diff(d['off']); c=d['cost']*1000; dt=(d['t1']-d['t0'])*1000
print('SLICE $MS ms: slices', n.size, '| pages/slice mean', n.mean().round(0), 'p90', np.percentile(n,90).round(0),
      '| tracer cost ms median', np.median(c).round(2), 'p90', np.percentile(c,90).round(2),
      '| observed window ms median', np.median(dt).round(2), '| duty (cost/(cost+window))', (c.sum()/(c.sum()+dt.sum())).round(3))"
done
python bench_tracer.py /tmp/p8.json none_b /tmp/b4/none_b.json
stop_server
