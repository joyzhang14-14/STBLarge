"""Reference generation with the official DeepSeek-V4.1 model.py on one GB10.

The official inference/model.py runs unmodified; kernel.py in this directory replaces its tilelang
kernels. Weights come straight from the official safetensors: dense ones are loaded once (with the
same conversions as the official convert.py), routed experts are read on first use and kept in an
LRU, and Engram table rows are read from the files on lookup. Text only (vision off), no DSpark.

usage: run_ref.py OUT.pt --prompt TEXT [--system TEXT] [--tokens N] [--ctx N] | --ids 1,2,3
Writes prompt ids, generated ids and per-step fp32 logits (greedy decoding).
STBL_REF_EMULATE=ds4 swaps in the ds4 GGUF engine's numerics (emulate_ds4.py) for comparison.
STBL_REF_DUMP=FILE saves the residual stream after every block, [tokens, layers, hc*dim] bf16.
"""

import argparse
import collections
import json
import os
import struct
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("STBL_OFFICIAL", "/home/joyzhang14/models/DeepSeek-V4.1-Flash-1017")
sys.path[:0] = [HERE, os.path.join(CKPT, "inference"), os.path.join(CKPT, "encoding")]

import model as M  # noqa: E402  (official inference/model.py)


class Checkpoint:
    """Name -> (file, dtype, shape, byte offset) from the safetensors headers; tensors read by mmap."""

    DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
              "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8}

    def __init__(self, root):
        self.info, self.maps = {}, {}
        files = sorted(set(json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"].values()))
        for f in files:
            path = os.path.join(root, f)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            self.maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
            for name, v in header.items():
                if name != "__metadata__":
                    a, b = v["data_offsets"]
                    self.info[name] = (path, v["dtype"], v["shape"], 8 + n + a, b - a)

    def raw(self, name):
        path, _, _, off, size = self.info[name]
        return self.maps[path][off:off + size]

    def get(self, name, device="cuda"):
        _, dtype, shape, _, _ = self.info[name]
        t = torch.from_numpy(np.array(self.raw(name))).view(self.DTYPES[dtype]).view(shape)
        return t.to(device)

    def rows(self, name, idx):
        """Rows of a 2-D tensor (idx: int64 numpy array) without reading the rest of it."""
        path, dtype, shape, off, _ = self.info[name]
        row = shape[1] * (1 if dtype.startswith(("F8", "I8")) else 4)
        table = self.maps[path][off:off + shape[0] * row].reshape(shape[0], row)
        return torch.from_numpy(table[idx]).view(self.DTYPES[dtype])


class LazyExpert(torch.nn.Module):
    """Stands in for one routed Expert; materializes the real official Expert on first call."""

    cache = collections.OrderedDict()
    capacity = int(os.environ.get("STBL_REF_EXPERTS", "1500"))
    ckpt = None
    loads = 0

    def __init__(self, prefix, dim, inter_dim, dtype, swiglu_limit):
        super().__init__()
        self.prefix, self.args = prefix, (dim, inter_dim, dtype, swiglu_limit)

    def forward(self, x, weights=None):
        cache = LazyExpert.cache
        e = cache.get(self.prefix)
        if e is None:
            dim, inter, dtype, limit = self.args
            with torch.device("cuda"):
                e = M.Expert(dim, inter, dtype=dtype, swiglu_limit=limit)
            for w in ("w1", "w2", "w3"):
                lin = getattr(e, w)
                lin.weight.data = self.ckpt.get(f"{self.prefix}.{w}.weight").view(torch.float4_e2m1fn_x2)
                lin.scale.data = self.ckpt.get(f"{self.prefix}.{w}.scale")
            cache[self.prefix] = e
            LazyExpert.loads += 1
            if len(cache) > self.capacity:
                cache.popitem(last=False)
        else:
            cache.move_to_end(self.prefix)
        return e(x, weights)


class LazyEngramEmbedding(torch.nn.Module):
    """Official ParallelEngramEmbedding (world_size 1) with the fp8 table left in the files."""

    ckpt = None

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def forward(self, indices):
        idx = indices.reshape(-1).cpu().numpy().astype(np.int64)
        values = self.ckpt.rows(f"{self.prefix}.weight", idx).to(indices.device)
        scales = self.ckpt.rows(f"{self.prefix}.scale", idx).to(indices.device)
        values = values.float().unflatten(-1, (-1, M.fp8_block_size)) * scales.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).view(*indices.shape, -1)


def build(args, tokenizer, ckpt):
    """Build the official Transformer with routed experts and Engram tables left lazy."""
    real_expert, real_engram = M.Expert, M.ParallelEngramEmbedding
    LazyExpert.ckpt = LazyEngramEmbedding.ckpt = ckpt
    pending = []

    def expert(dim, inter_dim, dtype=None, swiglu_limit=0.0):
        if dtype == torch.float4_e2m1fn_x2:
            lazy = LazyExpert(None, dim, inter_dim, dtype, swiglu_limit)
            pending.append(lazy)
            return lazy
        return real_expert(dim, inter_dim, dtype=dtype, swiglu_limit=swiglu_limit)

    M.Expert = expert
    M.ParallelEngramEmbedding = lambda num_embeddings, dim: LazyEngramEmbedding(None)
    try:
        with torch.device("cuda"):
            model = M.Transformer(args, tokenizer)
    finally:
        M.Expert, M.ParallelEngramEmbedding = real_expert, real_engram
    for name, mod in model.named_modules():
        if isinstance(mod, LazyExpert):
            mod.prefix = name
        elif isinstance(mod, LazyEngramEmbedding):
            mod.prefix = name
    assert all(p.prefix for p in pending)
    return model


def load_dense(model, ckpt):
    """Fill every dense parameter from the checkpoint, applying the official convert.py rules
    that matter at world_size 1 (wo_a dequantized to bf16; other dtypes cast by the parameter)."""
    loaded = 0
    for name, p in model.named_parameters():
        if name.endswith("wo_a.weight"):
            w = ckpt.get(name).float()
            s = ckpt.get(name.replace("weight", "scale")).float()
            bo, bi = w.size(0) // s.size(0), w.size(1) // s.size(1)
            w = (w.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)) * s[:, None, :, None]).flatten(2, 3).flatten(0, 1)
            p.data.copy_(w.to(torch.bfloat16))
        elif name in ckpt.info:
            src = ckpt.get(name)
            assert tuple(src.shape) == tuple(p.shape), (name, src.shape, p.shape)
            p.data.copy_(src)
        else:
            raise KeyError(f"no checkpoint tensor for {name}")
        loaded += 1
    return loaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prompt")
    ap.add_argument("--system", default="")
    ap.add_argument("--ids")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--thinking-mode", default="chat", choices=["chat", "thinking"])
    ap.add_argument("--stepwise", action="store_true", help="prefill one token at a time")
    a = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    with open(os.path.join(CKPT, "inference", "config.json")) as f:
        cfg = json.load(f)
    cfg.update(vision_n_layers=0, dspark_block_size=0, max_batch_size=1, max_seq_len=a.ctx, temperature=0)
    args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in M.ModelArgs.__dataclass_fields__})

    if a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    else:
        from encoding import encode_messages
        messages = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
        ids = tokenizer.encode(encode_messages(messages, thinking_mode=a.thinking_mode))

    t0 = time.time()
    ckpt = Checkpoint(CKPT)
    model = build(args, tokenizer, ckpt)
    n = load_dense(model, ckpt)
    if os.environ.get("STBL_REF_EMULATE") == "ds4":
        import emulate_ds4
        emulate_ds4.apply(model)
        print("ref: emulating the ds4 GGUF numerics", flush=True)
    torch.set_default_device("cuda")
    print(f"ref: {n} dense tensors loaded in {time.time() - t0:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on GPU, prompt {len(ids)} tokens", flush=True)

    dump_path, dumped, routes = os.environ.get("STBL_REF_DUMP"), [], []
    if dump_path:  # residual stream after every block [tokens, layers, hc * dim] bf16, and routed expert ids
        for layer in model.layers:
            layer.register_forward_hook(lambda mod, inp, out: dumped.append(out[0][0].flatten(-2).to(torch.bfloat16).cpu()))
            layer.ffn.gate.register_forward_hook(lambda mod, inp, out: routes.append(out[1].cpu()))

    tokens = torch.tensor([ids], dtype=torch.long)
    out_ids, logits_all = [], []
    pos, cur = 0, tokens
    if a.stepwise:
        for t in ids[:-1]:
            model.forward(torch.tensor([[t]], dtype=torch.long), pos)
            pos += 1
        cur = tokens[:, -1:]
    for step in range(a.tokens):
        t1 = time.time()
        next_id, logits, _ = model.forward(cur, pos)
        pos += cur.size(1)
        tok = int(next_id[0])
        out_ids.append(tok)
        logits_all.append(logits[0].float().cpu())
        cur = torch.tensor([[tok]], dtype=torch.long)
        print(f"ref: step {step} token {tok} {tokenizer.decode([tok])!r} "
              f"{time.time() - t1:.1f}s experts loaded {LazyExpert.loads}", flush=True)
        if tok == tokenizer.eos_token_id:
            break
    torch.save({"prompt_ids": ids, "ids": out_ids, "logits": torch.stack(logits_all),
                "text": tokenizer.decode(out_ids)}, a.out)
    if dump_path:
        n_layers = len(model.layers)
        calls = [torch.stack(dumped[i:i + n_layers], dim=1) for i in range(0, len(dumped), n_layers)]
        torch.save(torch.cat(calls), dump_path)
        calls = [torch.stack(routes[i:i + n_layers], dim=1) for i in range(0, len(routes), n_layers)]
        torch.save(torch.cat(calls), dump_path.removesuffix(".pt") + "-routes.pt")
    print("ref: text", repr(tokenizer.decode(out_ids)))


if __name__ == "__main__":
    main()
