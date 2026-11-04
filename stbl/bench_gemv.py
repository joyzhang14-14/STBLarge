"""Bandwidth of the libstbl.so GEMV kernels at the decode shapes (random weights, no model).

Seconds of GPU time and well under 1 GB, so it may run outside an stb task: python bench_gemv.py
"""

import ctypes
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(HERE, "libstbl.so"))
P, I = ctypes.c_void_p, ctypes.c_int
dev = "cuda"
DIM, INTER = 5120, 2304


def timed(fn, nbytes, reps=200):
    for _ in range(10):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    us = a.elapsed_time(b) / reps * 1000
    return us, nbytes / us / 1e3


def report(name, us, gbs):
    print(f"{name:34s} {us:8.1f} us  {gbs:6.1f} GB/s", flush=True)


def bench_mxfp4(rows, k):
    w = torch.randint(0, 256, (rows, k // 32 * 17), device=dev, dtype=torch.uint8)
    w[:, ::17] = 120  # sane E8M0 scales
    x = torch.randn(k, device=dev)
    y = torch.empty(rows, device=dev)
    report(f"gemv_mxfp4 {rows}x{k}", *timed(lambda: lib.stbl_gemv_mxfp4(P(y.data_ptr()), P(w.data_ptr()), P(x.data_ptr()), I(rows), I(k), None), w.numel()))


def bench_fp8(rows, k):
    w = torch.randint(0, 120, (rows, k), device=dev, dtype=torch.uint8)
    s = torch.full((rows // 32 + 1, k // 32), 120, device=dev, dtype=torch.uint8)
    x = torch.randn(k, device=dev)
    y = torch.empty(rows, device=dev)
    report(f"gemv_fp8 {rows}x{k}", *timed(lambda: lib.stbl_gemv_fp8(P(y.data_ptr()), P(w.data_ptr()), P(s.data_ptr()), P(x.data_ptr()), I(rows), I(k), I(1), None), w.numel()))


def bench_bf16(rows, k):
    w = torch.randn(rows, k, device=dev).to(torch.bfloat16)
    x = torch.randn(k, device=dev)
    y = torch.empty(rows, device=dev)
    report(f"gemv_bf16 {rows}x{k}", *timed(lambda: lib.stbl_gemv_bf16(P(y.data_ptr()), P(w.data_ptr()), P(x.data_ptr()), I(rows), I(k), I(1), I(0), None), w.numel() * 2))


def bench_copy():
    a = torch.empty(64 << 20, device=dev, dtype=torch.uint8)
    b = torch.empty_like(a)
    report("device copy 64 MiB (read+write)", *timed(lambda: b.copy_(a), 2 * a.numel(), 50))




def bench_hc_mixes(hc=4, iters=20):
    n, mix = hc * DIM, (2 + hc) * hc
    x = torch.randn(n, device=dev)
    fn = torch.randn(mix, n, device=dev) * 0.01
    scale, base = torch.ones(3, device=dev), torch.zeros(mix, device=dev)
    pre, post, comb = (torch.empty(t, device=dev) for t in (hc, hc, hc * hc))
    Fl = ctypes.c_float
    report(f"hc_mixes hc={hc}", *timed(lambda: lib.stbl_hc_mixes(*(P(t.data_ptr()) for t in (pre, post, comb, x, fn, scale, base)),
                                                              I(hc), I(DIM), I(iters), Fl(1e-6), Fl(1e-6), None), fn.numel() * 4))




def bench_bf16_grouped(rows=8192, k=4096, group_rows=1024):
    w = torch.randn(rows, k, device=dev).to(torch.bfloat16)
    x = torch.randn(rows // group_rows * k, device=dev)
    y = torch.empty(rows, device=dev)
    report(f"gemv_bf16 grouped {rows}x{k}", *timed(lambda: lib.stbl_gemv_bf16(P(y.data_ptr()), P(w.data_ptr()), P(x.data_ptr()), I(rows), I(k), I(1), I(group_rows), None), w.numel() * 2))


def bench_sparse_attn(n_idx, heads=64, d=512, window=128):
    q = torch.randn(heads * d, device=dev)
    kv = torch.randn(window + 4096, d, device=dev)
    idx = torch.arange(n_idx, device=dev, dtype=torch.int32)
    sink = torch.zeros(heads, device=dev)
    out = torch.empty(heads * d, device=dev)
    report(f"sparse_attn n_idx={n_idx}", *timed(lambda: lib.stbl_sparse_attn(P(out.data_ptr()), P(q.data_ptr()), P(kv.data_ptr()), I(window), P(kv[window:].data_ptr()), P(idx.data_ptr()), I(n_idx), P(sink.data_ptr()), I(heads), I(d), ctypes.c_float(0.044), None), n_idx * d * 4))


if __name__ == "__main__":
    bench_copy()
    bench_mxfp4(INTER, DIM)
    bench_mxfp4(DIM, INTER)
    bench_fp8(INTER, DIM)
    bench_fp8(DIM, INTER)
    bench_bf16(4096, DIM)
    bench_hc_mixes()
    print("-- attention shapes")
    bench_fp8(32768, 1280)
    bench_fp8(5120, 8192)
    bench_fp8(1280, 5120)
    bench_fp8(512, 5120)
    bench_bf16_grouped()
    for n in (128, 256, 640):
        bench_sparse_attn(n)
