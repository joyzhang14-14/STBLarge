#!/usr/bin/env python3
"""Convert the official DeepSeek-V4.1-Flash-1017 checkpoint to STBLarge's GGUF without changing a value.

Every tensor keeps its source numbers exactly:
- routed experts: native MXFP4 codes and E8M0 scales, only re-laid out as GGUF MXFP4 blocks
  (scale byte first, values 0..15 in low nibbles, 16..31 in high nibbles);
- FP8 linears (attention, shared experts, indexer q_b, Engram kv): the E4M3 bytes as STBL_E4M3
  plus their 32x32-block E8M0 scales as a separate `.scale` tensor (STBL_E8M0);
- attn wo_a: dequantized to BF16, as the official convert.py does (exact: e4m3 * 2^k fits BF16);
- BF16 tensors stay BF16, F32 stay F32; norm and router weights widen BF16 -> F32 (exact);
- Engram tables: rows packed as 256 E4M3 bytes + 8 E8M0 scales, unchanged.
Every conversion is checked (repacked codes, round trips) and fails instead of rounding.
Resumable: an interrupted run continues with --resume.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine", "gguf-tools"))

import glm53_quantize as G  # noqa: E402
from deepseek41_metadata import GGUF_ALIGNMENT, metadata  # noqa: E402
from deepseek41_quantize import scale_name, validate_scales, write_engram  # noqa: E402

QTYPE_STBL_E4M3 = 200
QTYPE_STBL_E8M0 = 201
G.QTYPE_NAMES.update({QTYPE_STBL_E4M3: "STBL_E4M3", QTYPE_STBL_E8M0: "STBL_E8M0"})
G.QTYPE_LAYOUT.update({QTYPE_STBL_E4M3: (1, 1), QTYPE_STBL_E8M0: (1, 1)})
F32, BF16, MXFP4, I8 = G.QTYPE_F32, G.QTYPE_BF16, G.QTYPE_MXFP4, G.QTYPE_I8

FP8_LUT = None


def fp8_values(codes):
    """E4M3 byte -> float32, all 256 codes (0x7f/0xff are NaN and rejected by the caller)."""
    global FP8_LUT
    if FP8_LUT is None:
        c = np.arange(256, dtype=np.uint32)
        sign = np.where(c & 0x80, -1.0, 1.0)
        exp, man = (c >> 3) & 15, c & 7
        FP8_LUT = (sign * np.where(exp == 0, man / 8.0 * 2.0 ** -6, (1 + man / 8.0) * 2.0 ** (exp.astype(np.int32) - 7))).astype(np.float32)
    return FP8_LUT[codes]


def build_plan(db, config):
    c = config["text_config"]
    if config["quantization_config"]["weight_block_size"] != [32, 32]:
        raise ValueError("expected native 32x32 FP8 blocks")
    dim, inter = c["hidden_size"], c["moe_intermediate_size"]
    heads, hd = c["num_attention_heads"], c["head_dim"]
    qrank, orank, groups = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    hc, experts, vocab = c["hc_mult"], c["n_routed_experts"], c["vocab_size"]
    ih, idim = c["index_n_heads"], c["index_head_dim"]
    plan, disk, consumed = [], [], set()

    def claim(name, shape, dtype):
        info = db.info(name)
        if info["shape"] != list(shape) or info["dtype"] != dtype:
            raise ValueError(f"{name}: {info['dtype']} {info['shape']}, expected {dtype} {list(shape)}")
        consumed.add(name)

    def plain(name, source, shape, dtype, qtype, role):
        """BF16/F32 source kept as is, or BF16 widened to F32."""
        claim(source, shape, dtype)
        plan.append(G.TensorPlan(name, tuple(reversed(shape)), qtype, role, source=source,
                                 transform=f"{dtype}->{G.QTYPE_NAMES[qtype]}"))

    def fp8(name, source, shape, role):
        """E4M3 weight plus its E8M0 block scales, both byte for byte."""
        claim(source, shape, "F8_E4M3")
        sshape = [(d + 31) // 32 for d in shape]
        claim(scale_name(source), sshape, "F8_E8M0")
        plan.append(G.TensorPlan(name, tuple(reversed(shape)), QTYPE_STBL_E4M3, role, source=source, raw_copy=True))
        plan.append(G.TensorPlan(name.removesuffix(".weight") + ".scale", tuple(reversed(sshape)), QTYPE_STBL_E8M0,
                                 role, source=scale_name(source), raw_copy=True))

    def fp8_to_bf16(name, source, shape, role):
        claim(source, shape, "F8_E4M3")
        claim(scale_name(source), [(d + 31) // 32 for d in shape], "F8_E8M0")
        plan.append(G.TensorPlan(name, tuple(reversed(shape)), BF16, role, source=source, transform="E4M3*E8M0->BF16"))

    plain("token_embd.weight", "embed.weight", (vocab, dim), "BF16", BF16, "embedding")
    plain("output_norm.weight", "norm.weight", (dim,), "BF16", F32, "norm")
    plain("output.weight", "head.weight", (vocab, dim), "BF16", BF16, "output")
    for layer in range(c["num_hidden_layers"]):
        src, dst = f"layers.{layer}", f"blk.{layer}"
        for site in ("attn", "ffn"):
            plain(f"{dst}.hc_{site}_fn.weight", f"{src}.hc_{site}_fn", (hc * (hc + 2), hc * dim), "F32", F32, "mhc")
            plain(f"{dst}.hc_{site}_base.weight", f"{src}.hc_{site}_base", (hc * (hc + 2),), "F32", F32, "mhc")
            plain(f"{dst}.hc_{site}_scale.weight", f"{src}.hc_{site}_scale", (3,), "F32", F32, "mhc")
            plain(f"{dst}.{site}_norm.weight", f"{src}.{site}_norm.weight", (dim,), "BF16", F32, "norm")
        plain(f"{dst}.attn_sinks.weight", f"{src}.attn.attn_sink", (heads,), "F32", F32, "attention")
        fp8(f"{dst}.attn_q_a.weight", f"{src}.attn.wq_a.weight", (qrank, dim), "attention")
        fp8(f"{dst}.attn_q_b.weight", f"{src}.attn.wq_b.weight", (heads * hd, qrank), "attention")
        fp8(f"{dst}.attn_kv.weight", f"{src}.attn.wkv.weight", (hd, dim), "attention")
        fp8_to_bf16(f"{dst}.attn_output_a.weight", f"{src}.attn.wo_a.weight", (groups * orank, heads * hd // groups), "attention")
        fp8(f"{dst}.attn_output_b.weight", f"{src}.attn.wo_b.weight", (dim, groups * orank), "attention")
        plain(f"{dst}.attn_q_a_norm.weight", f"{src}.attn.q_norm.weight", (qrank,), "BF16", F32, "norm")
        plain(f"{dst}.attn_kv_a_norm.weight", f"{src}.attn.kv_norm.weight", (hd,), "BF16", F32, "norm")
        if layer in c["kv_source_layer_ids"]:
            plain(f"{dst}.attn_compressor_kv.weight", f"{src}.attn.compressor.wkv.weight", (hd, dim), "BF16", BF16, "compressor")
            plain(f"{dst}.attn_compressor_norm.weight", f"{src}.attn.compressor.norm.weight", (hd,), "BF16", F32, "norm")
            plain(f"{dst}.indexer.attn_k.weight", f"{src}.attn.indexer.wk.weight", (idim, hd), "BF16", BF16, "compressor")
            plain(f"{dst}.indexer.k_norm.weight", f"{src}.attn.indexer.k_norm.weight", (idim,), "BF16", F32, "norm")
            if c["compress_ratios"][layer] > 1:
                plain(f"{dst}.attn_compressor_gate.weight", f"{src}.attn.compressor.wgate.weight", (hd, dim), "BF16", BF16, "compressor")
        if layer in c["index_source_layer_ids"]:
            fp8(f"{dst}.indexer.attn_q_b.weight", f"{src}.attn.indexer.wq_b.weight", (ih * idim, qrank), "indexer")
            plain(f"{dst}.indexer.proj.weight", f"{src}.attn.indexer.weights_proj.weight", (ih, dim), "BF16", BF16, "indexer")
        plain(f"{dst}.ffn_gate_inp.weight", f"{src}.ffn.gate.weight", (experts, dim), "BF16", F32, "router")
        plain(f"{dst}.exp_probs_b.bias", f"{src}.ffn.gate.bias", (experts,), "F32", F32, "router")
        plain(f"{dst}.exp_probs_b_vl.bias", f"{src}.ffn.gate.bias_vl", (experts,), "F32", F32, "router")
        for part, source, shape in (("gate", "w1", (inter, dim)), ("up", "w3", (inter, dim)), ("down", "w2", (dim, inter))):
            fp8(f"{dst}.ffn_{part}_shexp.weight", f"{src}.ffn.shared_experts.{source}.weight", shape, "shared")
            pattern = f"{src}.ffn.experts.{{expert}}.{source}.weight"
            for expert in range(experts):
                claim(pattern.format(expert=expert), (shape[0], shape[1] // 2), "I8")
                claim(scale_name(pattern.format(expert=expert)), (shape[0], shape[1] // 32), "F8_E8M0")
            plan.append(G.TensorPlan(f"{dst}.ffn_{part}_exps.weight", (*reversed(shape), experts), MXFP4, "experts",
                                     source=pattern, expert_layer=layer, expert_part=part, expert_count=experts))
        if layer in c["engram_layer_ids"]:
            index = c["engram_layer_ids"].index(layer)
            rows = c["engram_num_embeddings"][index]
            engram = f"{src}.engram"
            claim(f"{engram}.embed.weight", (rows, 256), "F8_E4M3")
            claim(f"{engram}.embed.scale", (rows, 8), "F8_E8M0")
            disk.append(G.TensorPlan(f"{dst}.engram_embd.weight", (264, rows), I8, "engram_disk",
                                     source=f"{engram}.embed.weight", raw_copy=True))
            for part in ("q", "k"):
                plain(f"{dst}.engram_{part}_norm.weight", f"{engram}.{part}_weight", (hc, dim), "BF16", F32, "engram")
            fp8(f"{dst}.engram_kv.weight", f"{engram}.wkv.weight", ((hc + 1) * dim, 24 * 256), "engram")
    omitted = {name for name in db.tensors if name.startswith(("mtp.", "vision.", "aligner.", "image_"))}
    if consumed | omitted != set(db.tensors):
        raise ValueError(f"unclaimed source tensors: {sorted(set(db.tensors) - consumed - omitted)[:10]}")
    offset = 0
    for item in plan + disk:
        item.offset = offset
        item.nbytes = G.qtype_nbytes(item.qtype, item.shape)
        offset += G.align(item.nbytes, GGUF_ALIGNMENT)
    return plan + disk


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def encode_plain(db, item):
    raw = db.read(item.source)
    if item.transform in ("BF16->BF16", "F32->F32"):
        return raw
    if item.transform == "BF16->F32":
        return bf16_to_f32(raw).tobytes()
    raise ValueError(f"{item.name}: unknown transform {item.transform}")


def encode_raw_fp8(db, item):
    data = db.read(item.source)
    codes = np.frombuffer(data, dtype=np.uint8)
    if item.qtype == QTYPE_STBL_E4M3 and np.any((codes & 127) == 127):
        raise ValueError(f"{item.name}: NaN E4M3 code")
    if item.qtype == QTYPE_STBL_E8M0 and np.any(codes == 255):
        raise ValueError(f"{item.name}: NaN E8M0 scale")
    return data


def encode_wo_a(db, item):
    info, sinfo = db.info(item.source), db.info(scale_name(item.source))
    n, k = info["shape"]
    codes = np.frombuffer(db.read(item.source), dtype=np.uint8).reshape(n, k)
    scales = np.frombuffer(db.read(scale_name(item.source)), dtype=np.uint8).reshape(sinfo["shape"])
    if np.any((codes & 127) == 127) or np.any(scales == 255):
        raise ValueError(f"{item.name}: NaN code or scale")
    s = np.ldexp(np.float32(1), scales.astype(np.int32) - 127).astype(np.float32)
    values = fp8_values(codes) * np.repeat(np.repeat(s, 32, 0)[:n], 32, 1)[:, :k]
    bits = values.view(np.uint32)
    if np.any(bits & 0xFFFF):
        raise ValueError(f"{item.name}: a dequantized value does not fit BF16 exactly")
    return (bits >> 16).astype(np.uint16).tobytes()


def encode_expert(db, name):
    info, sinfo = db.info(name), db.info(scale_name(name))
    rows, packed_cols = info["shape"]
    codes = np.frombuffer(db.read(name), dtype=np.uint8).reshape(-1, 16)
    scales = np.frombuffer(db.read(scale_name(name)), dtype=np.uint8)
    if sinfo["shape"] != [rows, packed_cols // 16] or np.any(scales == 255):
        raise ValueError(f"{name}: invalid FP4 scales")
    unpacked = np.empty((codes.shape[0], 32), dtype=np.uint8)
    unpacked[:, 0::2], unpacked[:, 1::2] = codes & 15, codes >> 4
    packed = np.empty((codes.shape[0], 17), dtype=np.uint8)
    packed[:, 0] = scales
    packed[:, 1:] = unpacked[:, :16] | (unpacked[:, 16:] << 4)
    if not (np.array_equal(packed[:, 1:] & 15, unpacked[:, :16]) and np.array_equal(packed[:, 1:] >> 4, unpacked[:, 16:])):
        raise ValueError(f"{name}: MXFP4 repacking changed a code")
    return packed.tobytes()


def write(args, plan, records, db):
    data_start, data_bytes = G.print_plan(plan, records, [], GGUF_ALIGNMENT)
    partial, journal = args.out + ".partial", args.out + ".partial.json"
    identity = json.dumps([(name, db.info(name)) for name in sorted(db.tensors)], sort_keys=True)
    signature = hashlib.sha256((G.conversion_signature(plan, records, []) + identity).encode()).hexdigest()
    if os.path.exists(args.out):
        raise ValueError(f"refusing to overwrite {args.out}")
    completed = 0
    if os.path.exists(partial) or os.path.exists(journal):
        if not args.resume or not (os.path.exists(partial) and os.path.exists(journal)):
            raise ValueError("partial file and journal require --resume")
        completed = G.load_resume_state(journal, signature, plan)
    end = data_start + (plan[completed - 1].offset + G.align(plan[completed - 1].nbytes, GGUF_ALIGNMENT) if completed else 0)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free
    if free < data_start + data_bytes - end + (32 << 30):
        raise ValueError("insufficient disk space for remaining output plus 32 GiB reserve")
    header = b"GGUF" + G.struct.pack("<IQQ", 3, len(plan), len(records))
    header += b"".join(records) + b"".join(G.tensor_header(item) for item in plan)
    header += bytes(data_start - len(header))
    if os.path.exists(partial):
        with open(partial, "rb") as fp:
            if fp.read(data_start) != header or os.fstat(fp.fileno()).st_size < end:
                raise ValueError("partial GGUF is truncated or has a different header")
    else:
        with open(partial, "xb") as fp:
            fp.write(header)
            fp.flush()
            os.fsync(fp.fileno())
        G.save_resume_state(journal, signature, 0)
    with open(partial, "r+b") as fp:
        fp.truncate(end)
        fp.seek(end)
        for index in range(completed, len(plan)):
            item = plan[index]
            started = time.monotonic()
            if fp.tell() != data_start + item.offset:
                raise ValueError(f"incorrect offset for {item.name}")
            if item.role == "engram_disk":
                write_engram(fp, item, db, np)
            elif item.is_expert:
                for expert in range(item.expert_count):
                    data = encode_expert(db, item.source.format(expert=expert))
                    if len(data) != item.nbytes // item.expert_count:
                        raise ValueError(f"{item.name}: wrong expert size")
                    fp.write(data)
            elif item.raw_copy:
                fp.write(encode_raw_fp8(db, item))
            elif item.transform == "E4M3*E8M0->BF16":
                fp.write(encode_wo_a(db, item))
            else:
                fp.write(encode_plain(db, item))
            if fp.tell() != data_start + item.offset + item.nbytes:
                raise ValueError(f"incorrect payload size for {item.name}")
            fp.write(bytes(G.align(item.nbytes, GGUF_ALIGNMENT) - item.nbytes))
            fp.flush()
            os.fsync(fp.fileno())
            G.save_resume_state(journal, signature, index + 1)
            print(f"[{index + 1}/{len(plan)}] {item.name}: {item.nbytes / (1 << 30):.3f} GiB, "
                  f"{time.monotonic() - started:.1f}s", flush=True)
    os.rename(partial, args.out)
    os.unlink(journal)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    config, records = metadata(args.hf, args.source_revision)
    records.append(G.kv_string("deepseek41.quantization", "STBLarge lossless: native MXFP4 experts, native E4M3+E8M0 linears"))
    db = G.SourceDB(args.hf, index_validator=lambda _: None, scale_validator=validate_scales)
    try:
        plan = build_plan(db, config)
        if args.dry_run:
            G.print_plan(plan, records, [], GGUF_ALIGNMENT)
            for item in plan[:6] + plan[-3:]:
                print(json.dumps(dataclasses.asdict(item), sort_keys=True))
        else:
            write(args, plan, records, db)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"convert_official: {error}")
