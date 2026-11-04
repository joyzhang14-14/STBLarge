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


bench_copy()
bench_mxfp4(INTER, DIM)
bench_mxfp4(DIM, INTER)
bench_fp8(INTER, DIM)
bench_fp8(DIM, INTER)
bench_bf16(4096, DIM)
