# timeout: 5400
# all four settings + memory-pressure pass, sampled bitmap tracer at the chosen slice; think times x0.1 (test only)
R=/mnt/gapsound/results/v2-test2; rm -rf $R; mkdir -p $R
cp /mnt/gapsound/results/v2-test/{workload,pregen,capacity}.json $R/
THINK_SCALE=0.1 RUN=$R LOADS=0.9 DURATION=60 MIN_SESS=16 N_PROBE=6 PRESSURE=1 bash /mnt/gapsound/v2/run_all.sh
ls -la $R $R/pressure
python -c "
import numpy as np, glob
for f in sorted(glob.glob('$R/**/trace-*.npz', recursive=True)):
    d=np.load(f); n=np.unpackbits(d['bits'],axis=1).sum(1); pa=d['parts']
    print(f.split('/')[-2:], 'slices', len(n), 'cycle ms', round(float(np.diff(pa[:,0]).mean()*1000),1), 'monitored pages/slice mean', round(float(n.mean()),0), 'kv_pages', int(d['kv_pages']))"
