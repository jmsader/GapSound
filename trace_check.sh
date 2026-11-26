#!/usr/bin/env bash
# Run:  sudo chmod -R a+rwX /mnt/gapsound && bash /mnt/gapsound/trace_check.sh
set -u
source "$HOME/gs/bin/activate"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4
export VLLM_CPU_KVCACHE_SPACE=2
TS=$(date +%Y%m%d-%H%M%S); LOG="$HOME/trace_check-$TS.log"; SLOG="$HOME/trace_server-$TS.log"
exec > >(tee "$LOG") 2>&1

echo "== THP off (so tracking is at 4 KB pages, like a host's EPT view)"
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null
cat /sys/kernel/mm/transparent_hugepage/enabled

echo "== starting server"
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --enforce-eager --max-model-len 2048 --port 8000 >"$SLOG" 2>&1 &
SPID=$!
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null && break; kill -0 $SPID || { echo "server died"; tail -30 "$SLOG"; exit 1; }; sleep 5; done
echo "server up"; sleep 5

sudo "$HOME/gs/bin/python" /mnt/gapsound/trace_check.py $SPID /mnt/gapsound/results/trace_check-$TS.json

kill $SPID; sleep 3
cp "$LOG" "$SLOG" /mnt/gapsound/logs/ && echo "logs copied"
