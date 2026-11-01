"""Single-layer isolation test with real weights: the official model (ref/run_ref.py loader) runs a token
sequence one token at a time while hooks record, for chosen layers, each step's input residual stream
(before Engram where the layer has one), incoming pre_mix, output stream and outgoing pre_mix. The same
inputs go through the engine's stbl_block_test for that layer alone; per-step errors should stay at the
bf16 rounding level. Loads the model: run as an stb task.

usage: test_layers.py REF_RESULT.pt [layers, default 0,1,2,14,20] [stages]
REF_RESULT.pt is a run_ref.py output; its prompt_ids + ids form the sequence (teacher forcing).
stages (stbl_block_test): default 0 for every layer plus 1,2 for Engram layers; 3 = attention alone,
4 = MoE alone (also compares the routed expert sets; the margin is the reference's 6th-7th selection gap).
"""
import ctypes
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))
import run_ref as R  # noqa: E402

M = R.M
GGUF = os.environ.get("STBL_GGUF", "/home/joyzhang14/models/stbl/DeepSeek-V4.1-Flash-1017-STBL.gguf")


def main():
    seq_src = torch.load(sys.argv[1])
    ids = seq_src["prompt_ids"] + seq_src["ids"]
    layers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2,14,20").split(",")]
    stages = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    torch.set_default_dtype(torch.bfloat16)
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(R.CKPT)
    cfg = json.load(open(os.path.join(R.CKPT, "inference", "config.json")))
    cfg.update(vision_n_layers=0, dspark_block_size=0, max_batch_size=1, max_seq_len=1024, temperature=0)
    args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in M.ModelArgs.__dataclass_fields__})
    ckpt = R.Checkpoint(R.CKPT)
    model = R.build(args, tok, ckpt)
    R.load_dense(model, ckpt)
    torch.set_default_device("cuda")

    rec = {il: [] for il in layers}
    for il in layers:
        blk = model.layers[il]
        if blk.engram is not None:
            blk.engram.register_forward_pre_hook(lambda mod, a, il=il: rec[il].append({"h": a[0][0, 0].float().cpu()}))
        def pre(mod, a, il=il):
            if mod.engram is None:
                rec[il].append({"h": a[0][0, 0].float().cpu()})
            rec[il][-1]["h_block"] = a[0][0, 0].float().cpu()
            rec[il][-1]["pre"] = a[2][0, 0].float().cpu()
        def post(mod, a, out, il=il):
            rec[il][-1]["out"] = out[0][0, 0].float().cpu()
            rec[il][-1]["pre_out"] = out[1][0, 0].float().cpu()
        blk.register_forward_pre_hook(pre)
        blk.register_forward_hook(post)
        blk.attn.register_forward_pre_hook(lambda mod, a, il=il: rec[il][-1].update(attn_in=a[0][0, 0].float().cpu()))
        blk.attn.register_forward_hook(lambda mod, a, out, il=il: rec[il][-1].update(attn_out=out[0, 0].float().cpu()))
        blk.ffn.register_forward_pre_hook(lambda mod, a, il=il: rec[il][-1].update(ffn_in=a[0][0, 0].float().cpu()))
        blk.ffn.register_forward_hook(lambda mod, a, out, il=il: rec[il][-1].update(ffn_out=out[0, 0].float().cpu()))
        def gate(mod, a, out, il=il):
            s = (F.softplus(a[0].float() @ mod.weight.float().t()).sqrt() + mod.bias.float())[0]
            top = s.topk(mod.topk + 1).values
            rec[il][-1].update(route=sorted(out[1][0].tolist()), margin=float(top[-2] - top[-1]))
        blk.ffn.gate.register_forward_hook(gate)
    for pos, t in enumerate(ids):
        model.forward(torch.tensor([[t]], dtype=torch.long), pos)
    print(f"reference: {len(ids)} tokens recorded", flush=True)

    lib = ctypes.CDLL(os.path.join(HERE, "libstbl.so"))
    lib.stbl_open.restype = ctypes.c_void_p
    lib.stbl_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.stbl_block_test.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 4 + [ctypes.c_int]
    lib.stbl_last_route.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.stbl_error.restype = ctypes.c_char_p
    lib.stbl_error.argtypes = [ctypes.c_void_p]
    err = ctypes.create_string_buffer(512)
    eng = lib.stbl_open(GGUF.encode(), 1024, 400, err, 512)
    if not eng:
        sys.exit(f"stbl: {err.value.decode()}")
    out = torch.empty(4 * args.dim, dtype=torch.float32, device="cpu")  # host buffers for the C API
    pre_out = torch.empty(4, dtype=torch.float32, device="cpu")
    route = (ctypes.c_int * 16)()
    passes = [(il, 0) for il in layers] + [(il, s) for il in layers if model.layers[il].engram is not None for s in (1, 2)]
    if stages:
        passes = [(il, s) for il in layers for s in stages]
    prev = {}  # (layer, stage) -> outputs of the earlier pass: a repeated stage checks determinism
    for il, stage in passes:
        errs, flips, stats, outs = [], [], [], []
        for pos, r in enumerate(rec[il]):
            h = {2: r["h_block"], 3: r["attn_in"], 4: r["ffn_in"]}.get(stage, r["h"]).flatten().contiguous()
            pre = r["pre"].contiguous()
            if lib.stbl_block_test(eng, il, pos, ids[pos], h.data_ptr(), pre.data_ptr(), out.data_ptr(), pre_out.data_ptr(), stage):
                sys.exit(f"stbl: {lib.stbl_error(eng).decode()}")
            want = {1: r["h_block"], 3: r["attn_out"], 4: r["ffn_out"]}.get(stage, r["out"]).flatten()
            if stage == 1 or stage >= 3:
                pre_out.copy_(r["pre_out"])
            got = out[: want.numel()]
            outs.append(got.clone())
            if stage == 4:
                k = lib.stbl_last_route(eng, route)
                if sorted(route[:k]) != r["route"]:
                    flips.append((pos, round(r["margin"], 4), sorted(set(route[:k]) ^ set(r["route"]))))
            e = float((got - want).abs().max() / want.abs().max())
            ep = float((pre_out - r["pre_out"]).abs().max())
            # bf16 ulp of each reference element: errors of at most one ulp are rounding flips
            ulp = torch.ldexp(torch.ones_like(want), torch.frexp(want)[1] - 8).clamp_min(2.0 ** -133)
            big = int(((got - want).abs() > 1.01 * ulp).sum())
            l2 = float((got - want).norm() / want.norm().clamp_min(1e-30))
            errs.append((e, ep, pos))
            stats.append((l2, big))
        es = torch.tensor([x[0] for x in errs])
        worst = sorted(errs, reverse=True)[:6]
        if stage == 4:
            print(f"layer {il} route mismatches {len(flips)}/{len(errs)} (pos, ref margin, differing ids): {flips}", flush=True)
        print(f"layer {il} {['engram+block', 'engram only', 'block only', 'attention only', 'moe only'][stage]}: rel err median {es.median():.1e} max {es.max():.1e}; worst (err, pre_mix err, pos): "
              + "  ".join(f"({e:.1e},{p:.0e},{q})" for e, p, q in worst), flush=True)
        if (il, stage) in prev:
            diff = [pos for pos, (a, b) in enumerate(zip(prev[il, stage], outs)) if not torch.equal(a, b)]
            print(f"layer {il} stage {stage} repeat: {len(diff)} positions differ from the earlier pass {diff[:20]}", flush=True)
        prev[il, stage] = outs
        l2s = torch.tensor([x[0] for x in stats])
        bigs = [x[1] for x in stats]
        print(f"    L2 rel median {l2s.median():.1e} max {l2s.max():.1e}; elements off by >1 ulp: median {sorted(bigs)[len(bigs) // 2]} "
              f"max {max(bigs)} of {want.numel()} (worst pos {max(range(len(bigs)), key=bigs.__getitem__)})", flush=True)


if __name__ == "__main__":
    main()
