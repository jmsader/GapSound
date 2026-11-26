#!/bin/bash
# usage: source serve.sh <log> <extra vllm flags...>   -> sets SRV_PID; returns 1 if the server died or never came up
start_server() {
  local log=$1; shift
  if curl -s localhost:8000/health >/dev/null; then echo "PORT_8000_BUSY"; return 1; fi
  # vLLM's weight-download lock left in /tmp by a run as another user makes filelock refuse (we run as root)
  rm -f /tmp/*Qwen-Qwen2.5-0.5B-Instruct.lock
  # CPU layout: the model's compute threads get cores 0-11 (the "guest vCPUs"); the API server stays on 0-12;
  # core 13 is the emulated host's tracer core and cores 14-15 run the clients (see TRACER_CPU / CLIENT_CPUS)
  VLLM_CPU_OMP_THREADS_BIND=0-11 taskset -c 0-12 vllm serve Qwen/Qwen2.5-0.5B-Instruct --enforce-eager --max-model-len 8192 --enable-prompt-tokens-details \
    --port 8000 "$@" > "$log" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 300); do
    curl -s localhost:8000/health >/dev/null && { echo "server up after ${i}s (pid $SRV_PID)"; return 0; }
    kill -0 $SRV_PID 2>/dev/null || { echo "SERVER_DIED"; tail -40 "$log"; return 1; }
    sleep 1
  done
  echo "SERVER_TIMEOUT"; tail -40 "$log"; return 1
}
TRACER_CPU=13
CLIENT_CPUS=14,15
stop_server() { pkill -f "vllm serve"; for i in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || return 0; sleep 1; done; pkill -9 -f "vllm serve"; }
