"""Bitwise comparison of kernels in two builds of libstbl.so (MXFP4 GEMV, hc_mixes, sparse_attn), random inputs.

usage: cmp_kernels.py NEW.so OLD.so   (seconds, under 1 GB: may run outside an stb task)
"""
import ctypes, sys
import torch
new = ctypes.CDLL(sys.argv[1]); old = ctypes.CDLL(sys.argv[2])
P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
dev = "cuda"
p = lambda t: P(t.data_ptr())
bad = 0
for seed in range(20):
    torch.manual_seed(seed)
    for rows, k in ((2304, 5120), (5120, 2304)):
        w = torch.randint(0, 256, (rows, k // 32 * 17), device=dev, dtype=torch.uint8)
        w[:, ::17] = torch.randint(110, 130, (rows, k // 32), device=dev, dtype=torch.uint8)
        x = torch.randn(k, device=dev).bfloat16().float() * (10 ** torch.empty(k, device=dev).uniform_(-2, 1))
        new.stbl_act_quant_fp8(p(x), I(k), None)
        a, b = torch.empty(rows, device=dev), torch.empty(rows, device=dev)
        new.stbl_gemv_mxfp4(p(a), p(w), p(x), I(rows), I(k), None)
        old.stbl_gemv_mxfp4(p(b), p(w), p(x), I(rows), I(k), None)
        torch.cuda.synchronize()
        d = (a != b).sum().item(); bad += d
        if d: print(f"mxfp4 seed {seed} {rows}x{k}: {d} rows differ")
    hc, dim = 4, 5120
    n, mix = hc * dim, (2 + hc) * hc
    x = torch.randn(n, device=dev) * 3
    fn = torch.randn(mix, n, device=dev) * 0.02
    scale = torch.rand(3, device=dev); base = torch.randn(mix, device=dev)
    outs = []
    for lib in (new, old):
        pre, post, comb = (torch.empty(t, device=dev) for t in (hc, hc, hc * hc))
        lib.stbl_hc_mixes(*(p(t) for t in (pre, post, comb, x, fn, scale, base)), I(hc), I(dim), I(20), F(1e-6), F(1e-6), None)
        torch.cuda.synchronize(); outs.append(torch.cat([pre, post, comb]))
    d = (outs[0] != outs[1]).sum().item(); bad += d
    if d: print(f"hc_mixes seed {seed}: {d} values differ")
    heads, d, window = 64, 512, 128
    for n_idx in (13, 128, 300, 640):
        q = torch.randn(heads * d, device=dev).bfloat16().float()
        kv = torch.randn(window + 2048, d, device=dev).bfloat16().float()
        idx = torch.randint(0, window + 2048, (n_idx,), device=dev, dtype=torch.int32)
        idx[torch.rand(n_idx, device=dev) < 0.05] = -1
        sink = torch.randn(heads, device=dev)
        outs = []
        for lib in (new, old):
            o = torch.empty(heads * d, device=dev)
            lib.stbl_sparse_attn(p(o), p(q), p(kv), I(window), P(kv[window:].data_ptr()), p(idx), I(n_idx), p(sink), I(heads), I(d), F(0.044), None)
            torch.cuda.synchronize(); outs.append(o)
        dd = (outs[0] != outs[1]).sum().item(); bad += dd
        if dd: print(f"sparse_attn seed {seed} n_idx {n_idx}: {dd} values differ")
print("bitwise identical" if bad == 0 else f"{bad} differences")
