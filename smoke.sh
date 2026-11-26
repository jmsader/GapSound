#!/usr/bin/env bash
# Smoke test: does vLLM serve on CPU, does prefix caching hit, and does cache_salt isolate it?
# Run:  sudo chmod -R a+rwX /mnt/gapsound && bash /mnt/gapsound/smoke.sh
set -u
source "$HOME/.bashrc" >/dev/null 2>&1; source "$HOME/gs/bin/activate"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4
export VLLM_CPU_KVCACHE_SPACE=2
TS=$(date +%Y%m%d-%H%M%S); LOG="$HOME/smoke-$TS.log"; SLOG="$HOME/smoke-server-$TS.log"
exec > >(tee "$LOG") 2>&1
MODEL=Qwen/Qwen2.5-0.5B-Instruct

start_server() {  # $1 = dtype
  echo "== starting vllm serve (dtype=$1)"
  vllm serve $MODEL --enable-prefix-caching --enforce-eager --max-model-len 2048 --dtype "$1" --port 8000 >"$SLOG" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do
    curl -s localhost:8000/health >/dev/null 2>&1 && { echo "server up after ${i}x5s"; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "server exited"; return 1; }
    sleep 5
  done
  echo "server timeout"; return 1
}

start_server auto || { kill $SPID 2>/dev/null; sleep 3; start_server float32; } || {
  echo "SMOKE_FAIL: server did not start"; tail -40 "$SLOG"; cp "$LOG" "$SLOG" /mnt/gapsound/logs/; exit 1; }

hits() { curl -s localhost:8000/metrics | grep -E '^vllm:prefix_cache_(hits|queries)' | tr '\n' ' '; echo; }

# ~400-token shared prefix
PREFIX=$(python - <<'PY'
print(("You are a careful assistant for a hospital records office. Follow policy section 4.2 on privacy, "
       "never reveal patient identifiers, and answer briefly. ") * 12)
PY
)
req() {  # $1 = salt ("" for none), $2 = question
  python - "$1" "$2" "$PREFIX" <<'PY'
import json, sys, time, requests
salt, q, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "prompt": prefix + "\nQuestion: " + q + "\nAnswer:",
        "max_tokens": 16, "temperature": 0}
if salt: body["cache_salt"] = salt
t = time.time(); r = requests.post("http://localhost:8000/v1/completions", json=body, timeout=300)
dt = time.time() - t
out = r.json()
txt = out.get("choices", [{}])[0].get("text", out)
print(f"salt={salt or '-':6s} status={r.status_code} latency={dt:.2f}s text={json.dumps(txt)[:70]}")
PY
}

echo "== metrics at start";              hits
echo "== R1 salt=userA (cold)";          req userA "What hours is the office open?"; hits
echo "== R2 salt=userA (same prefix)";   req userA "Who do I call about billing?";   hits
echo "== R3 salt=userB (other user)";    req userB "Who do I call about billing?";   hits
echo "== R4 no salt";                    req ""    "Who do I call about billing?";   hits
echo "== R5 no salt (same as R4 prefix)";req ""    "Where is parking?";              hits

kill $SPID 2>/dev/null; sleep 2
echo "SMOKE_DONE"
cp "$LOG" "$SLOG" /mnt/gapsound/logs/ && echo "logs copied"
