"""CPU checks of the PyTorch kernel stand-ins against brute-force definitions. Run: python test_kernel.py"""

import math

import torch

import kernel as K

torch.manual_seed(0)


def test_pow2_ceil():
    t = torch.tensor([1e-6, 0.25, 0.3, 1.0, 1.0000001, 3.0, 448.0, 1000.0])
    want = torch.tensor([2.0 ** math.ceil(math.log2(v)) for v in t.tolist()])
    assert torch.equal(K._pow2_ceil(t), want)


def test_fp4_round_ties_even():
    grid = K.FP4_TABLE[:8].tolist()
    for v in torch.linspace(-6, 6, 4801).tolist() + [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]:
        best = min(grid, key=lambda g: (abs(abs(v) - g), grid.index(g) % 2))
        got = K._round_fp4(torch.tensor([v]))[0].item()
        assert got == math.copysign(best, v), (v, got, best)


def test_act_quant():
    x = (torch.randn(3, 128) * torch.logspace(-3, 3, 128)).to(torch.bfloat16)
    y, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    xf = x.float().view(3, 4, 32)
    s_want = torch.tensor([[2.0 ** math.ceil(math.log2(max(b.abs().max().item(), 1e-4) / 448)) for b in r] for r in xf])
    assert torch.equal(s.float(), s_want)
    assert torch.equal(y.float(), (xf / s_want[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float().view(3, 128))
    z = x.clone()
    K.act_quant(z, 32, "ue8m0", torch.float8_e8m0fnu, True)
    assert torch.equal(z, (y.float().view(3, 4, 32) * s_want[..., None]).view(3, 128).to(torch.bfloat16))


def test_fp4_act_quant_roundtrip():
    x = torch.randn(2, 64).to(torch.bfloat16)
    packed, s = K.fp4_act_quant(x, 32, False)
    z = x.clone()
    K.fp4_act_quant(z, 32, True)
    raw = packed.view(torch.uint8)
    vals = torch.stack([K.FP4_TABLE[(raw & 15).long()], K.FP4_TABLE[(raw >> 4).long()]], -1).view(2, 64)
    assert torch.equal((vals.view(2, 2, 32) * s.float()[..., None]).view(2, 64).to(torch.bfloat16), z)


def test_fp8_gemm():
    a = torch.randn(5, 64)
    b = torch.randn(40, 64)
    aq, as_ = K.act_quant(a.to(torch.bfloat16), 32, "ue8m0", torch.float8_e8m0fnu)
    bs = torch.full((2, 2), 2.0 ** -3).to(torch.float8_e8m0fnu)
    bq = (b * 8).clamp(-448, 448).to(torch.float8_e4m3fn)
    c = K.fp8_gemm(aq, as_, bq, bs, torch.float8_e8m0fnu, 32).float()
    A = aq.float().view(5, 2, 32) * as_.float()[..., None]
    want = A.view(5, 64) @ (bq.float() * 2.0 ** -3).t()
    assert torch.allclose(c, want, rtol=1e-5, atol=1e-5)


def test_sparse_attn():
    b, m, h, d, n = 1, 2, 3, 16, 7
    q, kv = torch.randn(b, m, h, d).to(torch.bfloat16), torch.randn(b, n, d).to(torch.bfloat16)
    sink = torch.randn(h)
    idx = torch.tensor([[[0, 3, -1, 6], [-1, -1, -1, -1]]], dtype=torch.int32)
    o = K.sparse_attn(q, kv, sink, idx, 0.25).float()
    sel = kv[0, [0, 3, 6]].float()
    s = (q[0, 0].float() @ sel.t()) * 0.25
    mx = s.amax(-1, keepdim=True)
    p = torch.exp(s - mx)
    want = (p.to(torch.bfloat16).float() @ sel) / (p.sum(-1, keepdim=True) + torch.exp(sink[:, None] - mx))
    assert torch.allclose(o[0, 0], want.to(torch.bfloat16).float())
    assert torch.equal(o[0, 1], torch.zeros(h, d))


for name, fn in list(globals().items()):
    if name.startswith("test_"):
        fn()
        print("ok", name)
