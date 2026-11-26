#!/usr/bin/env bash
# Full experiment: shared vs salted cache, fresh server for each, then analysis.
# Run:  sudo chmod -R a+rwX /mnt/gapsound && bash /mnt/gapsound/run_exp.sh
set -u
source "$HOME/gs/bin/activate"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4
export VLLM_CPU_KVCACHE_SPACE=2
SEED=${1:-0}
TS=$(date +%Y%m%d-%H%M%S); LOG="$HOME/exp-$TS.log"
exec > >(tee "$LOG") 2>&1

sudo -v                                            # ask for the password once...
( while true; do sudo -n true; sleep 50; kill -0 $$ 2>/dev/null || exit; done ) &   # ...and keep it valid
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null

SETTINGS=${SETTINGS:-"shared orgsalt salted nocache"}
for SETTING in $SETTINGS; do
  if [ "$SETTING" = nocache ]; then CACHEFLAG="--no-enable-prefix-caching"; else CACHEFLAG="--enable-prefix-caching"; fi
  # make sure no leftover server (e.g. from an interrupted run) is answering on port 8000
  pkill -f "vllm serve" 2>/dev/null; sleep 3
  if curl -s localhost:8000/health >/dev/null; then echo "PORT_8000_BUSY: another server is still running; aborting"; exit 1; fi
  SLOG="$HOME/exp-$TS-$SETTING-server.log"
  echo "===== $SETTING: starting fresh server ====="
  vllm serve Qwen/Qwen2.5-0.5B-Instruct $CACHEFLAG --enforce-eager --max-model-len 4096 --port 8000 >"$SLOG" 2>&1 &
  SPID=$!
  for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 "$SLOG"; exit 1; }; sleep 5; done
  sleep 5
  sudo "$HOME/gs/bin/python" /mnt/gapsound/run_exp.py $SPID $SETTING /mnt/gapsound/results/exp-$TS-$SETTING.json $SEED \
    || echo "RUN_FAILED: $SETTING"
  kill $SPID; wait $SPID 2>/dev/null
  for i in $(seq 1 30); do curl -s localhost:8000/health >/dev/null || break; sleep 2; done
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  cp "$SLOG" /mnt/gapsound/logs/
done

echo "===== analysis ====="
python /mnt/gapsound/analyze.py $(for S in $SETTINGS; do echo /mnt/gapsound/results/exp-$TS-$S.json; done)
echo "EXP_DONE"
cp "$LOG" /mnt/gapsound/logs/
