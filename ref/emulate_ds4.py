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


def q8_values(x: torch.Tensor, fp16_scale: bool) -> torch.Tensor:
    """Round to int8 per 32 along the last dim with scale amax/127 (scale kept fp16 for weights)."""
    shape = x.shape
    b = x.float().reshape(-1, shape[-1] // 32, 32)
    d = b.abs().amax(-1, keepdim=True) / 127.0
    if fp16_scale:
        d = d.half().float()
    q = torch.where(d > 0, torch.round(b / torch.where(d > 0, d, torch.ones_like(d))), torch.zeros_like(b))
    return (q.clamp(-128, 127) * d).reshape(shape)


def fp8_weight(linear: torch.nn.Module) -> torch.Tensor:
    w, s = linear.weight.float(), linear.scale.float()
    n, k = w.shape
    return w * s.repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)[:, :k]


def f16_roundtrip_(p: torch.Tensor):
    p.data.copy_(p.data.half().to(p.dtype))


def linear(x, weight, bias=None):
    mode = getattr(weight, "ds4_mode", None)
    if mode == "q8":
        y = q8_values(x, False) @ weight.ds4_w.t()
        return y.to(torch.get_default_dtype())
    if mode == "plain":
        return (x.float() @ weight.ds4_w.t()).to(torch.get_default_dtype())
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
            w = fp8_weight(mod)
            if name.endswith(("engram.wkv", "indexer.wq_b")):
                mod.weight.ds4_mode, mod.weight.ds4_w = "plain", w.half().float()
            else:
                mod.weight.ds4_mode, mod.weight.ds4_w = "q8", q8_values(w, True)
        elif name.endswith("attn.wo_a"):
            mod.weight.data.copy_(q8_values(mod.weight.float(), True).to(mod.weight.dtype))
    model.head.weight.data.copy_(q8_values(model.head.weight.data, True))
    f16_roundtrip_(model.embed.weight)
    for name, p in model.named_parameters():
        if (name.endswith(("hc_attn_fn", "hc_ffn_fn")) or ".compressor.wkv." in name or ".compressor.wgate." in name
                or name.endswith(("indexer.wk.weight", "indexer.weights_proj.weight", "engram.q_weight", "engram.k_weight"))):
            f16_roundtrip_(p)
