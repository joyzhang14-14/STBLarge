"""Make the reference compute the way the ds4 GGUF engine does, to tell ds4's quantization apart from
reference bugs: if the reference reproduces ds4's tokens with this on, the reference itself is sound.

What ds4 (ee7e840, MXFP4-experts GGUF) changes against the official numerics:
- FP8 linears (attention q_a/q_b/kv/wo_b, shared experts) and the output head are
  re-quantized to Q8_0 (int8, one fp16 scale per 32 inputs), and their activations are quantized
  to int8 per 32 (scale amax/127) instead of FP8;
- attention wo_a is Q8_0 too;
- routed MXFP4 experts take int8 (q8_1) activations instead of FP8;
- hc_*_fn is stored as F16; embedding, compressor, indexer wk / weights_proj and Engram q/k weights
  are stored as F16 (only magnitudes below the F16 range change); Engram wkv and indexer q_b are
  F16 and their inputs are not quantized.
"""

import torch
import torch.nn.functional as F

import model as M

_linear = M.linear


def q8_codes(x: torch.Tensor, fp16_scale: bool):
    """int8 codes and scales per 32 along the last dim, scale amax/127 (fp16 for weights)."""
    b = x.float().reshape(-1, x.shape[-1] // 32, 32)
    d = b.abs().amax(-1, keepdim=True) / 127.0
    if fp16_scale:
        d = d.half().float()
    q = torch.round(b / torch.where(d > 0, d, torch.ones_like(d))).clamp(-128, 127)
    return q.to(torch.int8), d


def q8_values(x: torch.Tensor, fp16_scale: bool) -> torch.Tensor:
    q, d = q8_codes(x, fp16_scale)
    return (q.float() * d).reshape(x.shape)


def q8_weight(linear: torch.nn.Module):
    """Q8_0 of an FP8 linear's dequantized weight, a block of rows at a time to bound memory."""
    n, k = linear.weight.shape
    codes = torch.empty(n, k, dtype=torch.int8, device=linear.weight.device)
    scales = torch.empty(n, k // 32, dtype=torch.float16, device=linear.weight.device)
    for r in range(0, n, 2048):
        rows = slice(r, min(r + 2048, n))
        w = linear.weight[rows].float() * linear.scale[r // 32:(rows.stop + 31) // 32].float().repeat_interleave(32, 0)[:rows.stop - r].repeat_interleave(32, 1)[:, :k]
        q, d = q8_codes(w, True)
        codes[rows], scales[rows] = q.view(-1, k), d.view(-1, k // 32).half()
    return codes, scales


def fp8_weight(linear: torch.nn.Module) -> torch.Tensor:
    w, s = linear.weight.float(), linear.scale.float()
    n, k = w.shape
    return w * s.repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)[:, :k]


def f16_roundtrip_(p: torch.Tensor):
    p.data.copy_(p.data.half().to(p.dtype))


def linear(x, weight, bias=None):
    mode = getattr(weight, "ds4_mode", None)
    if mode == "q8":
        n, k = weight.ds4_q.shape
        w = (weight.ds4_q.float().view(n, -1, 32) * weight.ds4_d.float().unsqueeze(-1)).view(n, k)
        return (q8_values(x, False) @ w.t()).to(torch.get_default_dtype())
    if mode == "plain":
        return (x.float() @ weight.ds4_w.float().t()).to(torch.get_default_dtype())
    if weight.dtype == torch.float4_e2m1fn_x2:
        import kernel as K
        n, half_k = weight.shape
        raw = weight.view(torch.uint8)
        table = K.FP4_TABLE.to(raw.device)
        vals = torch.stack([table[(raw & 15).long()], table[(raw >> 4).long()]], -1).view(n, half_k * 2)
        w = (vals.view(n, -1, 32) * weight.scale.float().view(n, -1, 1)).view(n, -1)
        return (q8_values(x, False) @ w.t()).to(torch.get_default_dtype())
    return _linear(x, weight, bias)


def head_forward(self, x, full_logits=False):
    if not full_logits:
        x = x[:, -1]
    return F.linear(q8_values(x.float(), False), self.weight)


def apply(model: torch.nn.Module):
    M.linear = linear
    M.ParallelHead.forward = head_forward
    for name, mod in model.named_modules():
        if isinstance(mod, M.Linear) and mod.weight.dtype == torch.float8_e4m3fn:
            if name.endswith(("engram.wkv", "indexer.wq_b")):
                mod.weight.ds4_mode, mod.weight.ds4_w = "plain", fp8_weight(mod).half()
            else:
                mod.weight.ds4_mode = "q8"
                mod.weight.ds4_q, mod.weight.ds4_d = q8_weight(mod)
        elif name.endswith("attn.wo_a"):
            mod.weight.data.copy_(q8_values(mod.weight.float(), True).to(mod.weight.dtype))
    head = model.head.weight.data
    for r in range(0, head.size(0), 8192):
        head[r:r + 8192] = q8_values(head[r:r + 8192], True)
    f16_roundtrip_(model.embed.weight)
    for name, p in model.named_parameters():
        if (name.endswith(("hc_attn_fn", "hc_ffn_fn")) or ".compressor.wkv." in name or ".compressor.wgate." in name
                or name.endswith(("indexer.wk.weight", "indexer.weights_proj.weight", "engram.q_weight", "engram.k_weight"))):
            f16_roundtrip_(p)
