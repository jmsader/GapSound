"""GapSound job runner. Runs inside the UTM VM:
    sudo ~/gs/bin/python /mnt/gapsound/runner.py
Watches /mnt/gapsound/jobs/ for *.sh files, runs each once (oldest name first), one at a time.
Output streams to <job>.out; <job>.done holds the exit code. Ctrl+C stops the runner.
A job may set a timeout with a first-lines comment:  # timeout: 900   (seconds, default 600).
"""
import glob, os, re, signal, subprocess, sys, time

JOBS = "/mnt/gapsound/jobs"
USER_HOME = "/home/jmsader"
ENV = dict(os.environ,
           HOME=USER_HOME, HF_HOME=f"{USER_HOME}/.cache/huggingface",
           PATH=f"{USER_HOME}/gs/bin:{USER_HOME}/.local/bin:" + os.environ.get("PATH", "/usr/bin:/bin"),
           VIRTUAL_ENV=f"{USER_HOME}/gs",
           LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4",
           VLLM_CPU_KVCACHE_SPACE="2", PYTHONUNBUFFERED="1")

print(f"runner: watching {JOBS} (Ctrl+C to stop)", flush=True)
while True:
    try:
        for job in sorted(glob.glob(f"{JOBS}/*.sh")):
            done = job[:-3] + ".done"
            if os.path.exists(done):
                continue
            head = open(job).read(500)
            m = re.search(r"#\s*timeout:\s*(\d+)", head)
            tmo = int(m.group(1)) if m else 600
            out = job[:-3] + ".out"
            print(f"runner: {os.path.basename(job)} (timeout {tmo}s)", flush=True)
            t0 = time.time()
            with open(out, "w") as fo:
                p = subprocess.Popen(["bash", job], stdout=fo, stderr=subprocess.STDOUT, env=ENV,
                                     cwd="/mnt/gapsound", start_new_session=True)
                try:
                    rc = p.wait(timeout=tmo)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = "TIMEOUT"
            # never leave a server behind
            subprocess.run(["pkill", "-f", "vllm serve"], stderr=subprocess.DEVNULL)
            os.chmod(out, 0o666)
            with open(done, "w") as fd:
                fd.write(f"{rc} {time.time() - t0:.1f}s\n")
            os.chmod(done, 0o666)
            print(f"runner: {os.path.basename(job)} -> {rc} in {time.time() - t0:.0f}s", flush=True)
        time.sleep(2)
    except KeyboardInterrupt:
        print("runner: stopped", flush=True)
        subprocess.run(["pkill", "-f", "vllm serve"], stderr=subprocess.DEVNULL)
        sys.exit(0)
