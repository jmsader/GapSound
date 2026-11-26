#!/usr/bin/env bash
# GapSound VM setup. Run as your normal user (NOT with sudo):  bash /mnt/gapsound/setup.sh
# It asks for your password once for apt. Everything is logged; the log is copied to /mnt/gapsound/logs/.
set -u
TS=$(date +%Y%m%d-%H%M%S)
LOCAL_LOG="$HOME/setup-$TS.log"
exec > >(tee "$LOCAL_LOG") 2>&1

step() { echo; echo "===== $* ====="; }

step "1. System info"
uname -a; cat /etc/os-release | head -4; nproc; free -h; df -h / | tail -1

step "2. Page tracking check (needs root)"
grep -E 'IDLE_PAGE_TRACKING|MEM_SOFT_DIRTY' /boot/config-$(uname -r) || echo "config grep: not found"
ls -l /sys/kernel/mm/page_idle/bitmap 2>&1
if sudo head -c 16 /sys/kernel/mm/page_idle/bitmap | xxd; then echo "PAGE_IDLE_OK"; else echo "PAGE_IDLE_FAIL"; fi

step "3. apt packages"
sudo apt-get update -y -q
sudo apt-get install -y -q --no-install-recommends libtcmalloc-minimal4 jq curl git ca-certificates xxd numactl build-essential python3.12-dev && echo "APT_OK" || echo "APT_FAIL"

step "4. uv"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version || echo "UV_FAIL"

step "5. Python venv ~/gs (Python 3.12)"
[ -d "$HOME/gs" ] || uv venv --python 3.12 "$HOME/gs"
source "$HOME/gs/bin/activate"
python --version

step "6. vLLM CPU wheel (aarch64)"
URL=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest \
      | jq -r '.assets[].browser_download_url' | grep -E 'cpu.*aarch64\.whl$' | head -1)
echo "wheel: ${URL:-NONE FOUND}"
if [ -n "${URL:-}" ]; then
  uv pip install "$URL" --torch-backend cpu && echo "VLLM_INSTALL_OK" || echo "VLLM_INSTALL_FAIL"
else
  echo "VLLM_INSTALL_FAIL (no aarch64 cpu wheel in latest release)"
fi

step "7. Analysis libraries"
uv pip install numpy pandas scikit-learn matplotlib requests && echo "LIBS_OK" || echo "LIBS_FAIL"

step "8. Environment variables"
TCM=$(ls /usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4 2>/dev/null)
grep -q GAPSOUND_ENV "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<EOT
# GAPSOUND_ENV
export PATH="\$HOME/.local/bin:\$PATH"
export LD_PRELOAD=$TCM
export VLLM_CPU_KVCACHE_SPACE=4
source \$HOME/gs/bin/activate
EOT
echo "added to ~/.bashrc (LD_PRELOAD=$TCM)"

step "9. vLLM import check"
LD_PRELOAD=$TCM python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)" && echo "VLLM_IMPORT_OK" || echo "VLLM_IMPORT_FAIL"

step "DONE"
cp "$LOCAL_LOG" /mnt/gapsound/logs/ 2>/dev/null && echo "log copied to /mnt/gapsound/logs/" \
  || echo "Could not write to /mnt/gapsound (permissions). Log is at $LOCAL_LOG"
