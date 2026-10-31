"""测单个专家（gate + up + down，共约 18.8 MB）从 SSD 读出来要多久：切成多大的块、用几个线程最快。

O_DIRECT 直读 GGUF 里随机位置的专家（每次换一个专家，避开缓存），每种配置读 N 个专家，报每个专家的读取时间。
只读盘、不用 GPU。用法：python ssd_expert_bench.py GGUF [N]
"""
import mmap, os, random, statistics, sys, threading, time

PATH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
EXPERT = 18_800_640
ALIGN = 4096
fd = os.open(PATH, os.O_RDONLY | os.O_DIRECT)
size = os.fstat(fd).st_size
rnd = random.Random(1)


def read_expert(off, piece, threads, bufs):
    pieces = [(off + a, min(piece, EXPERT - a)) for a in range(0, EXPERT, piece)]
    nxt = [0]
    lock = threading.Lock()

    def work(w):
        while True:
            with lock:
                i = nxt[0]
                nxt[0] += 1
            if i >= len(pieces):
                return
            o, n = pieces[i]
            n = (n + ALIGN - 1) // ALIGN * ALIGN
            os.preadv(fd, [memoryview(bufs[w])[:n]], o)

    ts = [threading.Thread(target=work, args=(w,)) for w in range(1, threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    work(0)
    for t in ts:
        t.join()
    return (time.perf_counter() - t0) * 1000


print(f"{'块大小':>8s} {'线程':>4s} {'中位 ms':>8s} {'p90 ms':>7s} {'GB/s':>6s}")
for piece_mb in (1, 2, 4, 6.3):
    piece = int(piece_mb * 1048576) // ALIGN * ALIGN
    for threads in (1, 3, 4, 8, 12, 16):
        bufs = [mmap.mmap(-1, piece + ALIGN) for _ in range(threads)]
        lat = []
        for _ in range(N):
            off = rnd.randrange(0, (size - EXPERT * 2) // ALIGN) * ALIGN
            lat.append(read_expert(off, piece, threads, bufs))
        lat.sort()
        med = statistics.median(lat)
        print(f"{piece_mb:>6.1f}MB {threads:>4d} {med:8.2f} {lat[int(len(lat) * .9)]:7.2f} {EXPERT / med / 1e6:6.2f}", flush=True)
