"""GPU checks of libstbl.so kernels against the official model.py modules and ref/kernel.py.

Each test feeds the same random bf16-valued inputs to the official code (with the reference kernels)
and to the STBLarge kernel, and requires agreement within float summation-order noise.
Run as an stb task (uses the GPU): python test_kernels.py
"""

import ctypes
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("STBL_OFFICIAL", "/home/joyzhang14/models/DeepSeek-V4.1-Flash-1017")
sys.path[:0] = [os.path.join(HERE, "..", "ref"), os.path.join(CKPT, "inference")]
import kernel as K  # noqa: E402  (reference kernels)
import model as M  # noqa: E402  (official model.py)

torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
lib = ctypes.CDLL(os.path.join(HERE, "libstbl.so"))
P = ctypes.c_void_p
I = ctypes.c_int
Fl = ctypes.c_float


def ptr(t):
    return P(t.data_ptr())


def f32(t):
    return t.float().contiguous()


def bf(shape, scale=1.0):
    return (torch.randn(*shape, device=dev, dtype=torch.float32) * scale).to(torch.bfloat16)


def check(name, got, want, tol):
    err = (got.float() - want.float()).abs().max().item()
    ref = want.float().abs().max().item()
    ok = err <= tol * max(ref, 1e-6)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max err {err:.3e} (rel {err / max(ref, 1e-30):.2e})")
    return ok


def sync():
    torch.cuda.synchronize()


results = []


def test_act_quant_fp8():
    x = bf((4, 256), 3.0)
    want = x.clone()
    K.act_quant(want, 32, "ue8m0", torch.float8_e8m0fnu, True)
    got = f32(x)
    lib.stbl_act_quant_fp8(ptr(got), I(got.numel()), None); sync()
    results.append(check("act_quant_fp8", got, want, 0))


def test_act_quant_fp4():
    for block, e4 in ((32, 0), (16, 1)):
        x = bf((4, 512), 2.0)
        want = x.clone()
        K.fp4_act_quant(want, block, True, torch.float8_e4m3fn if e4 else torch.float8_e8m0fnu)
        got = f32(x)
        lib.stbl_act_quant_fp4(ptr(got), I(got.numel()), I(block), I(e4), None); sync()
        results.append(check(f"act_quant_fp4 block {block}", got, want, 0))


def fp8_linear(n, k):
    lin = M.Linear(k, n, dtype=torch.float8_e4m3fn).to(dev)
    lin.weight.data = (torch.randn(n, k, device=dev) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    lin.scale.data = (2.0 ** torch.randint(-14, -6, ((n + 31) // 32, k // 32), device=dev).float()).to(torch.float8_e8m0fnu)
    return lin


def test_gemv_fp8():
    n, k = 1000, 1280
    lin = fp8_linear(n, k)
    x = bf((1, k))
    want = M.linear(x, lin.weight)
    xq = f32(x)
    lib.stbl_act_quant_fp8(ptr(xq), I(k), None)
    got = torch.empty(n, device=dev, dtype=torch.float32)
    lib.stbl_gemv_fp8(ptr(got), ptr(lin.weight), ptr(lin.scale), ptr(xq), I(n), I(k), I(1), None); sync()
    results.append(check("gemv_fp8 (official linear)", got, want[0], 1e-2))


def test_gemv_mxfp4():
    n, k = 512, 2560
    codes = torch.randint(0, 256, (n, k // 2), device=dev, dtype=torch.uint8)
    scales = torch.randint(118, 126, (n, k // 32), device=dev, dtype=torch.uint8)
    w = codes.view(torch.float4_e2m1fn_x2)
    w.scale = scales.view(torch.float8_e8m0fnu)
    x = bf((1, k))
    want = M.linear(x, w)
    # GGUF MXFP4 blocks: scale byte, then values 0..15 low nibbles / 16..31 high nibbles
    c = codes.view(n, k // 32, 16)
    vals = torch.stack([c & 15, c >> 4], -1).view(n, k // 32, 32)
    packed = torch.cat([scales.view(n, k // 32, 1), (vals[..., :16] | (vals[..., 16:] << 4))], -1).contiguous()
    xq = f32(x)
    lib.stbl_act_quant_fp8(ptr(xq), I(k), None)
    got = torch.empty(n, device=dev, dtype=torch.float32)
    lib.stbl_gemv_mxfp4(ptr(got), ptr(packed), ptr(xq), I(n), I(k), None); sync()
    results.append(check("gemv_mxfp4 (official fp4 linear)", got, want[0], 1e-2))


def test_gemv_bf16_f32():
    n, k = 777, 4096
    w = bf((n, k), 0.02)
    x = bf((1, k))
    got = torch.empty(n, device=dev, dtype=torch.float32)
    lib.stbl_gemv_bf16(ptr(got), ptr(w), ptr(f32(x)), I(n), I(k), I(1), I(0), None); sync()
    results.append(check("gemv_bf16 (F.linear bf16)", got, F.linear(x, w)[0], 1e-2))
    wf, xf = f32(w), f32(x)
    got = torch.empty(n, device=dev, dtype=torch.float32)
    lib.stbl_gemv_f32(ptr(got), ptr(wf), ptr(xf), I(n), I(k), I(0), None); sync()
    results.append(check("gemv_f32 (F.linear fp32)", got, F.linear(xf, wf)[0], 1e-5))


def test_rms_norm():
    dim = 5120
    norm = M.RMSNorm(dim, 1e-20).to(dev)
    norm.weight.data = bf((dim,), 0.5)
    x = bf((1, dim), 2.0)
    got = torch.empty(dim, device=dev, dtype=torch.float32)
    lib.stbl_rms_norm(ptr(got), ptr(f32(x)), ptr(f32(norm.weight)), I(1), I(dim), Fl(1e-20), None); sync()
    results.append(check("rms_norm", got, norm(x)[0], 1e-2))


def test_rope():
    cs = M.precompute_freqs_cis(64, 64, 65536, 160000, 16, 32, 1).to(dev)
    heads, width, pos = 8, 512, 37
    x = bf((1, 1, heads, width))
    want = x.clone()
    M.apply_rotary_emb(want[..., -64:], cs[pos:pos + 1])
    table = torch.stack([cs[pos].real, cs[pos].imag], -1).float().contiguous()
    got = f32(x)
    lib.stbl_rope(ptr(got), ptr(table), I(heads), I(width), I(64), I(0), None); sync()
    results.append(check("rope", got, want, 1e-2))
    inv = want.clone()
    M.apply_rotary_emb(inv[..., -64:], cs[pos:pos + 1], True)
    lib.stbl_rope(ptr(got), ptr(table), I(heads), I(width), I(64), I(1), None); sync()
    results.append(check("rope inverse", got, inv, 1e-2))


def test_hc():
    args = M.ModelArgs(dim=256, hc_mult=4, n_layers=1)
    blk = M.Block.__new__(M.Block)
    torch.nn.Module.__init__(blk)
    blk.norm_eps, blk.hc_mult, blk.hc_sinkhorn_iters, blk.hc_eps = 1e-20, 4, 20, 1e-6
    x = bf((1, 1, 4, args.dim))
    fn = torch.randn(24, 4 * args.dim, device=dev, dtype=torch.float32) * 0.05
    scale = torch.rand(3, device=dev, dtype=torch.float32)
    base = torch.randn(24, device=dev, dtype=torch.float32) * 0.1
    pre, post, comb = blk.hc_mixes(x, fn, scale, base)
    g_pre, g_post, g_comb = (torch.empty(n, device=dev, dtype=torch.float32) for n in (4, 4, 16))
    lib.stbl_hc_mixes(ptr(g_pre), ptr(g_post), ptr(g_comb), ptr(f32(x)), ptr(fn), ptr(scale), ptr(base), I(4),
                      I(args.dim), I(20), Fl(1e-20), Fl(1e-6), None); sync()
    results.append(check("hc mixes pre", g_pre, pre.flatten(), 1e-4))
    results.append(check("hc mixes post", g_post, post.flatten(), 1e-4))
    results.append(check("hc mixes comb", g_comb, comb.flatten(), 1e-4))
    y = blk.hc_pre(x, pre)
    got = torch.empty(args.dim, device=dev, dtype=torch.float32)
    lib.stbl_hc_pre(ptr(got), ptr(f32(x)), ptr(f32(pre)), I(4), I(args.dim), None); sync()
    results.append(check("hc_pre", got, y.flatten(), 1e-2))
    sub = bf((1, 1, args.dim))
    want = blk.hc_post(sub, x, post, comb)
    got = torch.empty(4 * args.dim, device=dev, dtype=torch.float32)
    lib.stbl_hc_post(ptr(got), ptr(f32(sub)), ptr(f32(x)), ptr(f32(post)), ptr(f32(comb)), I(4), I(args.dim), None)
    sync()
    results.append(check("hc_post", got, want.flatten(), 1e-2))


def test_sparse_attn():
    heads, d, n1, n2 = 64, 512, 128, 300
    q = bf((1, 1, heads, d), 0.1)
    kv1, kv2 = bf((n1, d)), bf((n2, d))
    sink = torch.randn(heads, device=dev, dtype=torch.float32)
    idx = torch.cat([torch.arange(60, device=dev), torch.tensor([-1, -1], device=dev), n1 + torch.arange(0, n2, 3, device=dev)]).int()
    want = K.sparse_attn(q, torch.cat([kv1, kv2])[None], sink, idx.view(1, 1, -1), d ** -0.5)
    got = torch.empty(heads * d, device=dev, dtype=torch.float32)
    lib.stbl_sparse_attn(ptr(got), ptr(f32(q)), ptr(f32(kv1)), I(n1), ptr(f32(kv2)), ptr(idx), I(idx.numel()), ptr(sink),
                         I(heads), I(d), Fl(d ** -0.5), None); sync()
    results.append(check("sparse_attn", got, want.flatten(), 1e-2))


def test_swiglu_router():
    g, u = bf((2304,), 4.0), bf((2304,), 4.0)
    w = 0.37
    gf, uf = g.float().clamp(max=10.0), u.float().clamp(-10.0, 10.0)
    want = (w * (F.silu(gf) * uf)).to(torch.bfloat16)
    got = torch.empty(2304, device=dev, dtype=torch.float32)
    lib.stbl_swiglu(ptr(got), ptr(f32(g)), ptr(f32(u)), Fl(w), Fl(10.0), I(2304), None); sync()
    results.append(check("swiglu", got, want, 1e-2))
    s = torch.randn(384, device=dev, dtype=torch.float32) * 8
    got = s.clone()
    lib.stbl_sqrt_softplus(ptr(got), I(384), None); sync()
    results.append(check("sqrt_softplus", got, F.softplus(s).sqrt(), 1e-6))


def test_index_score():
    heads, d, n = 32, 128, 200
    q, k = bf((1, 1, heads, d), 0.5), bf((1, n, d), 0.5)
    w = bf((1, 1, heads))
    want = (torch.einsum("bshd,btd->bsht", q, k).relu_() * w.unsqueeze(-1)).sum(dim=2)
    got = torch.empty(n, device=dev, dtype=torch.float32)
    lib.stbl_index_score(ptr(got), ptr(f32(q)), ptr(f32(k)), ptr(f32(w)), I(n), I(heads), I(d), None); sync()
    results.append(check("index_score", got, want.flatten(), 2e-2))


def test_compress_pool():
    ratio, width = 2, 512
    kv = torch.randn(ratio, width, device=dev, dtype=torch.float32)
    sc = torch.randn(ratio, width, device=dev, dtype=torch.float32)
    want = (kv * sc.softmax(dim=0)).sum(dim=0).to(torch.bfloat16)
    got = torch.empty(width, device=dev, dtype=torch.float32)
    lib.stbl_compress_pool(ptr(got), ptr(kv), ptr(sc), I(ratio), I(width), None); sync()
    results.append(check("compress_pool", got, want, 1e-2))


def test_engram_gate():
    dim, hc = 512, 4
    eng = M.Engram.__new__(M.Engram)
    torch.nn.Module.__init__(eng)
    eng.dim, eng.hc_mult, eng.eps, eng.clamp_value = dim, hc, 1e-20, 1e-6
    eng.q_weight = torch.nn.Parameter(bf((hc, dim)))
    eng.k_weight = torch.nn.Parameter(bf((hc, dim)))
    h = bf((1, 1, hc, dim))
    kv = bf(((hc + 1) * dim,))
    key = kv[: hc * dim].float().view(hc, dim)
    weight = eng.q_weight.float() * eng.k_weight.float()
    hf = h.float()
    rstd = torch.rsqrt(hf.square().mean(-1) + 1e-20) * torch.rsqrt(key.square().mean(-1) + 1e-20)
    dot = (hf * weight * key).sum(-1) * rstd * dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    want = (hf + gate.unsqueeze(-1) * kv[hc * dim:].float()).to(torch.bfloat16)
    got = f32(h)
    lib.stbl_engram_gate(ptr(got), ptr(f32(kv)), ptr(f32(eng.q_weight)), ptr(f32(eng.k_weight)), I(dim), I(hc),
                         Fl(1e-20), None); sync()
    results.append(check("engram_gate", got, want.flatten(), 1e-2))


for name, fn in list(globals().items()):
    if name.startswith("test_"):
        fn()
print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
