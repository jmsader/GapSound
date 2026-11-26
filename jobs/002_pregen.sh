# timeout: 900
echo "--- previous server log (job 001)"; tail -30 /tmp/srv001.log
cd /mnt/gapsound/v2
source serve.sh
export VLLM_CPU_KVCACHE_SPACE=2
start_server logs/srv002.log --enable-prefix-caching || exit 1
grep -E "KV cache|generation config|sampling|Maximum concurrency" logs/srv002.log | head
python pregen.py /tmp/w8.json /tmp/p8.json 1024 8 || exit 1
python -c "
import json; w=json.load(open('/tmp/p8.json'))
for s in w['sessions'][:3]:
    print(s['user'], s['app'], 'turns', len(s['replies']), 'reply_tokens', s['reply_tokens'], 'prompt_tokens', s['prompt_tokens'], 'finish', s.get('finish'))
print('REPLY0:', w['sessions'][0]['replies'][0][:400])"
cp /tmp/w8.json /tmp/p8.json /mnt/gapsound/v2/
stop_server
