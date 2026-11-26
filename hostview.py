"""Emulated host view of a process's memory, at 4 KB page granularity.

A real malicious hypervisor sees which guest-physical pages a VM touches (EPT/NPT accessed bits,
or page faults). Inside our VM we emulate that with Linux idle page tracking:
  1. mark every page idle in /sys/kernel/mm/page_idle/bitmap
  2. let the workload run
  3. read the bitmap back: pages whose idle bit got cleared were accessed.
/proc/<pid>/pagemap maps the process's virtual pages to physical frame numbers (PFNs). Needs root.
"""
import os, re, struct
import numpy as np

PAGE = 4096
BITMAP = "/sys/kernel/mm/page_idle/bitmap"
PFN_MASK = (1 << 55) - 1
PRESENT = 1 << 63


def anon_vmas(pid):
    """Writable anonymous (or heap) mappings: where weights, activations and the KV cache live."""
    out = []
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split()
            start, end = (int(x, 16) for x in parts[0].split("-"))
            perms = parts[1]
            path = parts[5] if len(parts) > 5 else ""
            if "rw" in perms and (path == "" or path.startswith("[heap") or path.startswith("[anon")):
                out.append((start, end, path))
    return out


def snapshot(pid, vmas=None):
    """Return dict vma_start -> (vpages ndarray, pfn ndarray with -1 for not-present)."""
    vmas = vmas if vmas is not None else anon_vmas(pid)
    snap = {}
    with open(f"/proc/{pid}/pagemap", "rb") as pm:
        for start, end, path in vmas:
            n = (end - start) // PAGE
            pm.seek(start // PAGE * 8)
            raw = pm.read(n * 8)
            ent = np.frombuffer(raw, dtype=np.uint64)
            present = (ent & np.uint64(PRESENT)) != 0
            pfn = np.where(present, (ent & np.uint64(PFN_MASK)).astype(np.int64), -1)
            snap[start] = (end, path, pfn)
    return snap


def max_pfn(snap):
    m = 0
    for end, path, pfn in snap.values():
        if pfn.size:
            m = max(m, int(pfn.max()))
    return m


CHUNK = 4096          # sysfs bin files transfer at most one page per read/write call
_BITMAP_BYTES = None


def bitmap_bytes():
    """Size of the idle bitmap (covers every physical frame); found once by reading to the end."""
    global _BITMAP_BYTES
    if _BITMAP_BYTES is None:
        fd = os.open(BITMAP, os.O_RDONLY); off = 0
        try:
            while True:
                try:
                    b = os.pread(fd, CHUNK, off)
                except OSError:
                    break
                if not b:
                    break
                off += len(b)
        finally:
            os.close(fd)
        _BITMAP_BYTES = off
    return _BITMAP_BYTES


def mark_idle(_unused=None):
    """Set the idle bit for every physical frame (kernel skips frames it can't track)."""
    n = bitmap_bytes(); ones = b"\xff" * CHUNK
    fd = os.open(BITMAP, os.O_WRONLY)
    try:
        for off in range(0, n, CHUNK):
            try:
                os.pwrite(fd, ones[:min(CHUNK, n - off)], off)
            except OSError:
                break
    finally:
        os.close(fd)


def read_idle(_unused=None):
    n = bitmap_bytes(); parts = []
    fd = os.open(BITMAP, os.O_RDONLY)
    try:
        for off in range(0, n, CHUNK):
            parts.append(os.pread(fd, min(CHUNK, n - off), off))
    finally:
        os.close(fd)
    return np.frombuffer(b"".join(parts), dtype=np.uint64)


def accessed_mask(pfn, bitmap):
    """True where the page is present and its idle bit was cleared (= accessed)."""
    ok = pfn >= 0
    res = np.zeros(pfn.shape, dtype=bool)
    p = pfn[ok]
    w = p >> 6
    inb = w < bitmap.size
    idle = np.zeros(p.shape, dtype=bool)
    idle[inb] = ((bitmap[w[inb]] >> (p[inb] & 63).astype(np.uint64)) & np.uint64(1)) == 1
    res[ok] = ~idle
    return res


def diff(before, after, bitmap):
    """Per-VMA counts: pages accessed during the window, and pages that became present (newly written)."""
    rows = []
    for start, (end, path, pfn_a) in after.items():
        acc = accessed_mask(pfn_a, bitmap)
        if start in before:
            pfn_b = before[start][2]
            n = min(pfn_b.size, pfn_a.size)
            newly = int(((pfn_b[:n] < 0) & (pfn_a[:n] >= 0)).sum()) + int((pfn_a[n:] >= 0).sum())
        else:
            newly = int((pfn_a >= 0).sum())
        rows.append({"start": hex(start), "size_mb": round((end - start) / 2**20, 1), "path": path,
                     "resident": int((pfn_a >= 0).sum()), "accessed": int(acc.sum()), "newly_present": newly,
                     "accessed_idx": np.nonzero(acc)[0]})
    return rows


def worker_pid(server_pid):
    """vLLM runs the model in a child process; pick the descendant with the largest resident memory."""
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                st = f.read()
            ppid = int(st.rsplit(")", 1)[1].split()[1])
            kids.setdefault(ppid, []).append(int(d))
        except Exception:
            pass
    desc, stack = [], [server_pid]
    while stack:
        p = stack.pop()
        for c in kids.get(p, []):
            desc.append(c); stack.append(c)
    def rss(p):
        try:
            with open(f"/proc/{p}/statm") as f:
                return int(f.read().split()[1])
        except Exception:
            return 0
    cand = sorted(desc + [server_pid], key=rss, reverse=True)
    return cand[0], {p: rss(p) * PAGE // 2**20 for p in cand[:6]}


def flush_tlb(pid):
    """clear_refs=1 clears the process's PTE accessed bits AND flushes its TLB, so hot pages
    must re-set their accessed bit on next use (page_idle alone skips the flush and misses them)."""
    with open(f"/proc/{pid}/clear_refs", "w") as f:
        f.write("1")


def find_kv_vma(snap, min_mb=512):
    """The KV cache is one big region reserved and fully touched at startup."""
    best = None
    for start, (end, path, pfn) in snap.items():
        size = end - start
        res = int((pfn >= 0).sum()) * PAGE
        if size >= min_mb * 2**20 and res >= 0.95 * size and path == "":
            if best is None or size > best[1]:
                best = (start, size)
    return best[0] if best else None
