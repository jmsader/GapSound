"""Time-sliced emulated-host tracer (runs as root, in its own process).

Every slice: set the idle bit of every KV-cache page frame, wait, then read the bits back:
frames whose idle bit was cleared were accessed during the slice. Only the KV region's frames are touched in the
bitmap (writing a 0 bit leaves a frame unaffected), which keeps the per-slice cost low.
The worker and its KV region are found exactly as before (hostview.worker_pid / find_kv_vma).

Output (.npz): t0[i], t1[i] (time.monotonic, observation window), parts[i] = (c0, c1, mark s, flush s, read s) with
[c0, c1] the effective window, and bits[i] = packed bitmap over mon (monitored KV page offsets): 1 = accessed.
Usage (root): slicer.py <server_pid> <slice_ms> <out.npz> [stop_file]
Stops on SIGTERM/SIGINT or when stop_file appears.
"""
import os, signal, sys, time
import numpy as np
sys.path.insert(0, "/mnt/gapsound")
import hostview as hv

server_pid, slice_ms, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
stop_file = sys.argv[4] if len(sys.argv) > 4 else None
BITMAP = "/sys/kernel/mm/page_idle/bitmap"
CHUNK_WORDS = 512                                  # 4096 bytes per sysfs read/write call
# Monitor every SAMPLE-th page of the KV region. vLLM's CPU KV cache stores each (layer, block, K-or-V) as one
# contiguous 32 KB chunk (128 tokens x 2 KV heads x 64 dims x 2 bytes = 8 pages), so every chunk contains exactly
# one monitored page: block identity is fully kept, tracer cost and data volume drop 8x.
SAMPLE = int(os.environ.get("SLICER_SAMPLE", "8"))

wpid, _ = hv.worker_pid(server_pid)
snap = hv.snapshot(wpid)
KV = hv.find_kv_vma(snap)
if KV is None:
    sys.exit("slicer: KV region not found")
KV_VMA = [(KV, snap[KV][0], snap[KV][1])]


def layout():
    """PFN of every KV page -> bitmap word runs + per-word masks of KV bits."""
    pfn = hv.snapshot(wpid, KV_VMA)[KV][2]
    ok = np.nonzero(pfn >= 0)[0]
    ok = ok[ok % SAMPLE == 0]
    p = pfn[ok]
    words = p >> 6
    uw, inv = np.unique(words, return_inverse=True)
    masks = np.zeros(uw.size, dtype=np.uint64)
    np.bitwise_or.at(masks, inv, np.left_shift(np.uint64(1), (p & 63).astype(np.uint64)))
    # contiguous runs of words, split into <= CHUNK_WORDS pieces
    runs, start = [], 0
    for i in range(1, uw.size + 1):
        if i == uw.size or uw[i] != uw[i - 1] + 1 or i - start == CHUNK_WORDS:
            runs.append((start, i)); start = i
    return ok, p, uw, inv, masks, runs


ok, p, uw, inv, masks, runs = layout()
fd_w = os.open(BITMAP, os.O_WRONLY)
fd_r = os.open(BITMAP, os.O_RDONLY)
mask_bytes = [(int(uw[a]) * 8, masks[a:b].tobytes()) for a, b in runs]


def mark():
    for off, buf in mask_bytes:
        os.pwrite(fd_w, buf, off)


def read_accessed():
    words = np.empty(uw.size, dtype=np.uint64)
    for a, b in runs:
        words[a:b] = np.frombuffer(os.pread(fd_r, (b - a) * 8, int(uw[a]) * 8), dtype=np.uint64)
    idle = ((words[inv] >> (p & 63).astype(np.uint64)) & np.uint64(1)) == 1
    return ok[~idle]


T0, T1, BITS, COST, PARTS = [], [], [], [], []
KV_PAGES = int((snap[KV][0] - KV) // 4096)
MON = np.arange(0, KV_PAGES, SAMPLE)              # monitored KV page offsets; slice i's bitmap is over MON
NBYTES = (MON.size + 7) // 8
# TLB flush (clear_refs) after marking: OFF by default. Benchmark (job 005): with the flush the worker's inter-token
# latency rose 73% (TLB shootdowns on all its cores); without it +4%, and detection was unchanged (8.9k vs 8.7k
# accessed KV pages per 50 ms slice, identical set of distinct pages). SLICER_FLUSH=1 turns it back on.
NOFLUSH = os.environ.get("SLICER_FLUSH") != "1"
running = True


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
print(f"slicer: worker {wpid}, KV {(snap[KV][0]-KV)/2**20:.0f} MB, {p.size} frames in {uw.size} words / "
      f"{len(runs)} runs, slice {slice_ms} ms, cpu affinity {sorted(os.sched_getaffinity(0))}"
      + ("" if NOFLUSH else " +TLB flush"), flush=True)
period = slice_ms / 1000.0
last_layout = time.monotonic()
while running and not (stop_file and os.path.exists(stop_file)):
    c0 = time.monotonic()
    mark()
    cm = time.monotonic()
    if not NOFLUSH:
        hv.flush_tlb(wpid)
    t0 = time.monotonic()
    nxt = t0 + period                              # observation window = period, after marking + flush
    while time.monotonic() < nxt:
        time.sleep(max(0.0, min(0.002, nxt - time.monotonic())))
    t1 = time.monotonic()
    acc = read_accessed()
    c1 = time.monotonic()
    # an access is recorded if it happens after its page was marked and before it was read back, so the
    # effective window is [c0, c1]; t0/t1 is the core window. Both are saved.
    b = np.zeros(MON.size, dtype=bool); b[acc // SAMPLE] = True
    T0.append(t0); T1.append(t1); BITS.append(np.packbits(b)); COST.append((t0 - c0) + (c1 - t1))
    PARTS.append((c0, c1, cm - c0, t0 - cm, c1 - t1))
    if c1 - last_layout > 5.0:                     # pages can move (compaction); refresh the PFN layout
        new = layout()
        if new[1].size != p.size or not np.array_equal(new[1], p):
            ok, p, uw, inv, masks, runs = new
            mask_bytes = [(int(uw[a]) * 8, masks[a:b].tobytes()) for a, b in runs]
        last_layout = time.monotonic()

np.savez_compressed(out, t0=np.array(T0), t1=np.array(T1), mon=MON, sample=SAMPLE,
                    bits=np.array(BITS, dtype=np.uint8).reshape(-1, NBYTES), flush=not NOFLUSH,
                    cost=np.array(COST), parts=np.array(PARTS).reshape(-1, 5),
                    slice_ms=slice_ms, kv_pages=KV_PAGES, worker=wpid)
c = np.array(COST) * 1000
if len(c):
    pa = np.array(PARTS) * 1000
    print(f"slicer: {len(T0)} slices, tracer cost per slice median {np.median(c):.1f} ms p90 "
          f"{np.percentile(c, 90):.1f} ms (mark {np.median(pa[:,2]):.1f} / flush {np.median(pa[:,3]):.1f} / "
          f"read {np.median(pa[:,4]):.1f}) -> {out}", flush=True)
else:
    print("slicer: no slices", flush=True)
