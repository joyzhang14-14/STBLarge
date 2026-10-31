"""Pure-PyTorch stand-ins for the tilelang kernels of the official DeepSeek-V4.1 inference code.

The official model.py imports `act_quant, fp4_act_quant, fp4_gemm, fp8_gemm, hc_split_sinkhorn,
sparse_attn` from kernel.py. Putting this directory first on sys.path makes it use these instead.
Each function follows the tilelang kernel's arithmetic: fp32 compute, the same scale rounding,
round-to-nearest-even casts. GEMMs dequantize the exact fp8/fp4 values and accumulate in fp32, so
they differ from the tensor-core kernels only in summation order.
"""

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _pow2_ceil(t: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(t)) for positive normal fp32 t, by the same exponent/mantissa test as fast_log2_ceil."""
    bits = t.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    return ((e + 127) << 23).view(torch.float32)


def _blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    n = x.size(-1)
    assert n % block_size == 0
    return x.contiguous().float().view(-1, n // block_size, block_size)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 quantization. inplace=True writes quant+dequant back to x in its own dtype."""
    xf = _blocks(x, block_size)
    amax = xf.abs().amax(-1).clamp_min(1e-4)
    t = amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    s = _pow2_ceil(t) if scale_fmt is not None else t
    y = (xf / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s.unsqueeze(-1)).view(x.shape).to(x.dtype))
        return x
    return y.view(x.shape), s.to(scale_dtype).view(*x.shape[:-1], -1)


def _round_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 values already clamped to [-6, 6] onto the e2m1 grid, ties to even."""
    a = v.abs()
    step = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0))
    return torch.copysign(torch.round(a / step) * step, v)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """FP4 with E8M0 (power-of-two) scales, or E4M3 scales for the compressed KV."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    xf = _blocks(x, block_size)
    amax = xf.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * 2**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(FP4_MAX * 2**-126)
        s = _pow2_ceil(amax * torch.tensor(1.0 / FP4_MAX, dtype=torch.float32))
    q = _round_fp4((xf / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((q * s.unsqueeze(-1)).view(x.shape).to(x.dtype))
        return x
    codes = _fp4_codes(q).view(*x.shape[:-1], -1, 2)
    packed = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    return packed.view(torch.float4_e2m1fn_x2), s.to(scale_dtype).view(*x.shape[:-1], -1)


def _fp4_codes(q: torch.Tensor) -> torch.Tensor:
    mag = torch.searchsorted(FP4_TABLE[:8].to(q.device), q.abs().contiguous())
    return (mag | (torch.signbit(q).int() << 3)).to(torch.int32)


def _deq_act(a, a_s, block):
    m = a.numel() // a.size(-1)
    return (a.float().view(m, -1, block) * a_s.float().view(m, -1, 1)).view(m, -1)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[M,N] = A[M,K] @ B[N,K]^T, A per (row, K block) scales, B per (N block, K block) scales."""
    K, N = a.size(-1), b.size(0)
    A = _deq_act(a, a_s, block_size)
    sb = b_s.float().repeat_interleave(block_size, 0)[:N].repeat_interleave(block_size, 1)[:, :K]
    B = b.float() * sb
    return (A @ B.t()).to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T; B packs two values per byte, low nibble first."""
    K, N = a.size(-1), b.size(0)
    A = _deq_act(a, a_s, act_block_size)
    raw = b.view(torch.uint8)
    table = FP4_TABLE.to(raw.device)
    vals = torch.stack([table[(raw & 15).long()], table[(raw >> 4).long()]], dim=-1).view(N, K)
    B = (vals.view(N, K // 32, 32) * b_s.float().view(N, K // 32, 1)).view(N, K)
    return (A @ B.t()).to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Attention of each query over the kv rows its topk_idxs name (-1 = none), with a per-head
    sink logit in the denominator. Probabilities are cast to bf16 before multiplying V, as the
    kernel does; a row with no valid index yields zeros."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    gathered = torch.gather(kv.unsqueeze(1).expand(b, m, -1, d), 2,
                            idx.clamp_min(0).unsqueeze(-1).expand(b, m, idx.size(-1), d))
    gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros((), dtype=kv.dtype, device=kv.device))
    s = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered.float()) * softmax_scale
    s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = s.amax(-1).clamp_min(-1e30)
    p = torch.exp(s - mx.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - mx)
    o = torch.einsum("bmht,bmtd->bmhd", p.to(torch.bfloat16).float(), gathered.float()) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    sc, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * sc[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
