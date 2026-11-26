# timeout: 900
set -x
whoami; nproc; free -g | head -2; python --version; python -c "import vllm, aiohttp, numpy, sklearn; print('vllm', vllm.__version__, 'aiohttp', aiohttp.__version__)"
cat /sys/kernel/mm/transparent_hugepage/enabled
pgrep -af "vllm serve" || echo "no stray server"
cd /mnt/gapsound/v2
python workload.py 8 /tmp/w8.json 0
export VLLM_CPU_KVCACHE_SPACE=2
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --enforce-eager --max-model-len 8192 \
  --enable-prompt-tokens-details --port 8000 > /tmp/srv001.log 2>&1 &
for i in $(seq 1 90); do curl -s localhost:8000/health >/dev/null && break; sleep 2; done
grep -E "KV cache size|generation config|Default sampling|sampling" /tmp/srv001.log | head
python pregen.py /tmp/w8.json /tmp/p8.json 1024 8
python -c "
import json; w=json.load(open('/tmp/p8.json'))
s=w['sessions'][0]; print(s['user'], s['app'], 'turns', len(s['replies']), 'reply_tokens', s['reply_tokens'], 'prompt_tokens', s['prompt_tokens'])
print('REPLY0:', s['replies'][0][:300])"
cp /tmp/p8.json /mnt/gapsound/v2/test_p8.json
