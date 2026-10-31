#!/usr/bin/env python3
"""Check an STBLarge GGUF against the official checkpoint by inverting every conversion.

Reads each GGUF tensor back and turns it into the official encoding (MXFP4 blocks -> packed
nibbles + E8M0 scales, F32 -> BF16 bits, BF16 wo_a -> E4M3 codes under the official scales),
then compares bytes with the safetensors source. Routed experts and Engram rows are sampled
(--experts per tensor, --engram-rows per table); everything else is checked whole.
"""

import argparse
import json
import os
import random
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import stbl_gguf as G  # noqa: E402

LAYOUT = {0: (1, 4), 30: (1, 2), 39: (32, 17), 24: (1, 1), 200: (1, 1), 201: (1, 1)}


def read_gguf(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        align = 32

        def s():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode()

        def skip(t):
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if t == 8:
                s()
            elif t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                for _ in range(n):
                    skip(et)
            else:
                f.read(sizes[t])

        for _ in range(n_kv):
            key = s()
            t = struct.unpack("<I", f.read(4))[0]
            if key == "general.alignment":
                align = struct.unpack("<I", f.read(4))[0]
            else:
                skip(t)
        tensors = []
        for _ in range(n_tensors):
            name = s()
            nd = struct.unpack("<I", f.read(4))[0]
            shape = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            qtype, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, shape, qtype, off))
        data = (f.tell() + align - 1) // align * align
    return tensors, data


def nbytes(shape, qtype):
    block, size = LAYOUT[qtype]
    return shape[0] // block * size * int(np.prod(shape[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--engram-rows", type=int, default=20000)
    a = ap.parse_args()
    random.seed(0)
    db = G.SourceDB(a.hf, index_validator=lambda _: None, scale_validator=lambda _: None)
    tensors, data = read_gguf(a.gguf)
    mm = np.memmap(a.gguf, dtype=np.uint8, mode="r")
    names = {t[0] for t in tensors}
    src_of = {}
    sys.path.insert(0, HERE)
    import convert_official as C
    for item in C.build_plan(db, json.load(open(os.path.join(a.hf, "config.json")))):
        src_of[item.name] = item
    assert names == set(src_of), "GGUF tensor names differ from the conversion plan"
    checked = bad = 0
    for name, shape, qtype, off in tensors:
        item = src_of[name]
        assert tuple(shape) == tuple(item.shape) and qtype == item.qtype, name
        blob = mm[data + off:data + off + nbytes(shape, qtype)]
        if item.role == "engram_disk":
            rows = shape[1]
            for r in random.sample(range(rows), min(a.engram_rows, rows)):
                got = np.asarray(blob[r * 264:(r + 1) * 264])
                want = np.frombuffer(db.iter_read(item.source, r * 256, 256).__next__(), np.uint8)
                want_s = np.frombuffer(db.iter_read(C.scale_name(item.source), r * 8, 8).__next__(), np.uint8)
                if not (np.array_equal(got[:256], want) and np.array_equal(got[256:], want_s)):
                    bad += 1
                    print("MISMATCH", name, "row", r)
                    break
        elif item.is_expert:
            per = nbytes(shape, qtype) // item.expert_count
            for e in random.sample(range(item.expert_count), a.experts):
                blocks = np.asarray(blob[e * per:(e + 1) * per]).reshape(-1, 17)
                lo, hi = blocks[:, 1:] & 15, blocks[:, 1:] >> 4
                vals = np.concatenate([lo, hi], axis=1)
                codes = (vals[:, 0::2] | (vals[:, 1::2] << 4)).astype(np.uint8)
                src = item.source.format(expert=e)
                if not (codes.tobytes() == db.read(src) and blocks[:, 0].tobytes() == db.read(C.scale_name(src))):
                    bad += 1
                    print("MISMATCH", name, "expert", e)
        elif item.raw_copy:
            if np.asarray(blob).tobytes() != db.read(item.source):
                bad += 1
                print("MISMATCH", name)
        elif item.transform == "E4M3*E8M0->BF16":
            vals = (np.asarray(blob).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            n, k = db.info(item.source)["shape"]
            sc = np.frombuffer(db.read(C.scale_name(item.source)), np.uint8).reshape(db.info(C.scale_name(item.source))["shape"])
            s = np.ldexp(np.float32(1), sc.astype(np.int32) - 127)
            q = vals.reshape(n, k) / np.repeat(np.repeat(s, 32, 0)[:n], 32, 1)[:, :k]
            table = C.fp8_values(np.arange(256, dtype=np.uint8))
            order = np.argsort(table, kind="stable")
            pos = np.searchsorted(table[order], q.ravel())
            codes = order[np.clip(pos, 0, 255)].astype(np.uint8)
            exact = table[codes] == q.ravel()
            src = np.frombuffer(db.read(item.source), np.uint8)
            same = np.where(table[src] == 0, table[codes] == 0, codes == src)
            if not (exact.all() and same.all()):
                bad += 1
                print("MISMATCH", name)
        else:
            raw = db.read(item.source)
            if item.transform == "BF16->F32":
                back = (np.asarray(blob).view(np.uint32) >> 16).astype(np.uint16).tobytes()
                low = (np.asarray(blob).view(np.uint32) & 0xFFFF).any()
                ok = back == raw and not low
            else:
                ok = np.asarray(blob).tobytes() == raw
            if not ok:
                bad += 1
                print("MISMATCH", name)
        checked += 1
    print(f"checked {checked} tensors, mismatches {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
