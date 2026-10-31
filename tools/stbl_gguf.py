"""GGUF writing and safetensors reading helpers for STBLarge's converter.

Copied from ds4 (antirez/ds4 ee7e840) gguf-tools: glm53_manifest.py, glm53_quantize.py and
deepseek41_quantize.py, keeping only what tools/convert_official.py uses. MIT License,
Copyright (c) 2026 The ds4.c authors, (c) 2023-2026 The ggml authors, (c) 2023 DeepSeek.
"""

import dataclasses
import hashlib
import json
import os
import struct
import threading

# --- from ds4 gguf-tools/glm53_manifest.py ---

DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

def fail(message):
    raise ValueError(message)


def checked_product(values, label):
    result = 1
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"{label}: invalid shape dimension {value!r}")
        result *= value
        if result > (1 << 63) - 1:
            fail(f"{label}: tensor element count overflows int64")
    return result


def load_index(path):
    with open(path, "rb") as fp:
        document = json.load(fp)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        fail(f"{path}: missing or empty weight_map")
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name:
            fail(f"{path}: invalid tensor name")
        if not isinstance(shard, str) or os.path.basename(shard) != shard:
            fail(f"{path}: invalid shard for {name}")
    return document, weight_map


def load_safetensors_header(path):
    file_size = os.path.getsize(path)
    with open(path, "rb") as fp:
        raw_length = fp.read(8)
        if len(raw_length) != 8:
            fail(f"{path}: short safetensors header length")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > file_size - 8:
            fail(f"{path}: header extends beyond file")
        if header_length > (1 << 30):
            fail(f"{path}: unreasonable header length {header_length}")
        raw_header = fp.read(header_length)
    try:
        document = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{path}: invalid safetensors header: {error}")

    payload_size = file_size - 8 - header_length
    tensors = {}
    spans = []
    for name, entry in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            fail(f"{path}: bad metadata for {name}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            fail(f"{path}: unsupported dtype {dtype!r} for {name}")
        if not isinstance(shape, list):
            fail(f"{path}: invalid shape for {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            fail(f"{path}: invalid data offsets for {name}")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            fail(f"{path}: out-of-range data offsets for {name}")
        nbytes = checked_product(shape, name) * DTYPE_BYTES[dtype]
        if end - start != nbytes:
            fail(f"{path}: byte count mismatch for {name}: {end - start} != {nbytes}")
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "offset": 8 + header_length + start,
            "nbytes": nbytes,
        }
        spans.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            fail(f"{path}: gap or overlap before {name}: {start} != {cursor}")
        cursor = end
    if cursor != payload_size:
        fail(f"{path}: unclaimed payload bytes: {payload_size - cursor}")
    return tensors


# --- from ds4 gguf-tools/glm53_quantize.py ---

GGUF_VERSION = 3

GGUF_ALIGNMENT = 32

GGUF_UINT32 = 4

GGUF_FLOAT32 = 6

GGUF_BOOL = 7

GGUF_STRING = 8

GGUF_ARRAY = 9

GGUF_UINT64 = 10

QTYPE_F32 = 0

QTYPE_F16 = 1

QTYPE_Q8_0 = 8

QTYPE_Q2_K = 10

QTYPE_Q4_K = 12

QTYPE_Q8_K = 15

QTYPE_IQ2_XXS = 16

QTYPE_I8 = 24

QTYPE_BF16 = 30

QTYPE_MXFP4 = 39

QTYPE_NAMES = {
    QTYPE_F32: "F32",
    QTYPE_F16: "F16",
    QTYPE_Q8_0: "Q8_0",
    QTYPE_Q2_K: "Q2_K",
    QTYPE_Q4_K: "Q4_K",
    QTYPE_Q8_K: "Q8_K",
    QTYPE_IQ2_XXS: "IQ2_XXS",
    QTYPE_I8: "I8",
    QTYPE_BF16: "BF16",
    QTYPE_MXFP4: "MXFP4",
}

QTYPE_LAYOUT = {
    QTYPE_F32: (1, 4),
    QTYPE_F16: (1, 2),
    QTYPE_Q8_0: (32, 34),
    QTYPE_Q2_K: (256, 84),
    QTYPE_Q4_K: (256, 144),
    QTYPE_Q8_K: (256, 292),
    QTYPE_IQ2_XXS: (256, 66),
    QTYPE_I8: (1, 1),
    QTYPE_BF16: (1, 2),
    QTYPE_MXFP4: (32, 17),
}

def align(value, alignment=GGUF_ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def qtype_nbytes(qtype, shape):
    block, block_bytes = QTYPE_LAYOUT[qtype]
    if not shape or shape[0] % block:
        fail(f"shape {shape} is incompatible with {QTYPE_NAMES[qtype]}")
    return shape[0] // block * block_bytes * product(shape[1:])


def pack_string(value):
    data = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<Q", len(data)) + data


def kv_string(key, value):
    return pack_string(key) + struct.pack("<I", GGUF_STRING) + pack_string(value)


def kv_u32(key, value):
    return pack_string(key) + struct.pack("<II", GGUF_UINT32, value)


def kv_u64(key, value):
    return pack_string(key) + struct.pack("<IQ", GGUF_UINT64, value)


def kv_f32(key, value):
    return pack_string(key) + struct.pack("<If", GGUF_FLOAT32, value)


def kv_bool(key, value):
    return pack_string(key) + struct.pack("<IB", GGUF_BOOL, bool(value))


def kv_u32_array(key, values):
    return (
        pack_string(key)
        + struct.pack("<IIQ", GGUF_ARRAY, GGUF_UINT32, len(values))
        + struct.pack(f"<{len(values)}I", *values)
    )


@dataclasses.dataclass
class TensorPlan:
    name: str
    shape: tuple
    qtype: int
    role: str
    source: str | None = None
    row_start: int = 0
    row_count: int | None = None
    expert_layer: int | None = None
    expert_part: str | None = None
    expert_count: int | None = None
    transform: str | None = None
    raw_copy: bool = False
    expert_scale: bool = False
    offset: int = 0
    nbytes: int = 0

    @property
    def is_expert(self):
        return self.expert_layer is not None


class SourceDB:
    def __init__(
        self,
        hf_dir,
        index_validator=lambda _: None,
        scale_validator=lambda _: None,
        tensor_prefix=None,
    ):
        self.hf_dir = hf_dir
        index_path = os.path.join(hf_dir, "model.safetensors.index.json")
        document, self.weight_map = load_index(index_path)
        index_validator(self.weight_map)
        if tensor_prefix is not None:
            self.weight_map = {name: shard for name, shard in self.weight_map.items()
                               if name.startswith(tensor_prefix)}
            if not self.weight_map:
                fail(f"source index has no tensors matching {tensor_prefix!r}")
        self.declared_bytes = document.get("metadata", {}).get("total_size")
        self.tensors = {}
        self._fds = {}
        self._fd_lock = threading.Lock()

        for shard in sorted(set(self.weight_map.values())):
            path = os.path.join(hf_dir, shard)
            if not os.path.isfile(path):
                fail(f"missing source shard {path}")
            for name, info in load_safetensors_header(path).items():
                if tensor_prefix is not None and not name.startswith(tensor_prefix):
                    continue
                if self.weight_map.get(name) != shard:
                    fail(f"index assigns {name} to {self.weight_map.get(name)!r}, not {shard}")
                if name in self.tensors:
                    fail(f"duplicate source tensor {name}")
                self.tensors[name] = dict(info, shard=shard)

        if set(self.tensors) != set(self.weight_map):
            missing = sorted(set(self.weight_map) - set(self.tensors))
            fail(f"source headers are incomplete; first missing tensor is {missing[0]}")
        scale_validator(self.tensors)

    def info(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            fail(f"source tensor not found: {name}")

    def _fd(self, shard):
        with self._fd_lock:
            fd = self._fds.get(shard)
            if fd is None:
                fd = os.open(os.path.join(self.hf_dir, shard), os.O_RDONLY)
                self._fds[shard] = fd
            return fd

    def read(self, name):
        info = self.info(name)
        data = os.pread(self._fd(info["shard"]), info["nbytes"], info["offset"])
        if len(data) != info["nbytes"]:
            fail(f"short payload read for {name}")
        return data

    def iter_read(self, name, byte_start=0, byte_count=None, chunk_size=16 << 20):
        info = self.info(name)
        if byte_count is None:
            byte_count = info["nbytes"] - byte_start
        if byte_start < 0 or byte_count < 0 or byte_start + byte_count > info["nbytes"]:
            fail(f"invalid payload range for {name}")
        fd = self._fd(info["shard"])
        offset = info["offset"] + byte_start
        remaining = byte_count
        while remaining:
            length = min(remaining, chunk_size)
            data = os.pread(fd, length, offset)
            if len(data) != length:
                fail(f"short payload read for {name}")
            yield data
            offset += length
            remaining -= length

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def tensor_header(item):
    return (
        pack_string(item.name)
        + struct.pack("<I", len(item.shape))
        + struct.pack(f"<{len(item.shape)}Q", *item.shape)
        + struct.pack("<IQ", item.qtype, item.offset)
    )


def print_plan(plan, kv_records, tokenizer_records, alignment=GGUF_ALIGNMENT):
    by_type = {}
    by_role = {}
    for item in plan:
        by_type[item.qtype] = by_type.get(item.qtype, 0) + item.nbytes
        by_role[item.role] = by_role.get(item.role, 0) + item.nbytes
    kv_bytes = sum(map(len, kv_records)) + sum(map(len, tokenizer_records))
    tensor_info_bytes = sum(len(tensor_header(item)) for item in plan)
    data_offset = align(4 + 4 + 8 + 8 + kv_bytes + tensor_info_bytes, alignment)
    data_bytes = sum(align(item.nbytes, alignment) for item in plan)
    print(f"tensors: {len(plan)}")
    print(f"metadata_records: {len(kv_records) + len(tokenizer_records)}")
    print(f"metadata_bytes: {data_offset}")
    print(f"tensor_bytes: {sum(item.nbytes for item in plan)}")
    print(f"file_bytes: {data_offset + data_bytes}")
    for qtype, nbytes in sorted(by_type.items()):
        print(f"type_bytes: {QTYPE_NAMES[qtype]} {nbytes}")
    for role, nbytes in sorted(by_role.items()):
        print(f"role_bytes: {role} {nbytes}")
    return data_offset, data_bytes


def file_sha256(path):
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(8 << 20):
            digest.update(chunk)
    return digest.digest()


def conversion_signature(plan, kv_records, tokenizer_records, imatrix_path=None):
    digest = hashlib.sha256()
    for record in kv_records + tokenizer_records:
        digest.update(record)
    for item in plan:
        digest.update(tensor_header(item))
        digest.update((item.source or "").encode())
        digest.update((item.transform or "").encode())
        digest.update(struct.pack("<I", item.expert_count or 0))
        digest.update(bytes((item.raw_copy, item.expert_scale)))
    imatrix_digest = file_sha256(imatrix_path)
    if imatrix_digest is not None:
        digest.update(b"imatrix\0")
        digest.update(imatrix_digest)
    return digest.hexdigest()


def save_resume_state(path, signature, completed):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as fp:
        json.dump({"version": 1, "signature": signature, "completed": completed}, fp)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, path)


def load_resume_state(path, signature, plan):
    with open(path, "r", encoding="utf-8") as fp:
        state = json.load(fp)
    if state.get("version") != 1 or state.get("signature") != signature:
        fail(f"resume state does not match this conversion: {path}")
    completed = state.get("completed")
    if not isinstance(completed, int) or completed < 0 or completed > len(plan):
        fail(f"invalid completed tensor count in {path}")
    return completed


# --- from ds4 gguf-tools/deepseek41_quantize.py ---

def scale_name(name):
    return name.removesuffix(".weight") + ".scale"


def validate_scales(tensors):
    for name, info in tensors.items():
        if not name.startswith(("layers.", "embed.", "head.", "norm.")):
            continue
        dtype, shape = info["dtype"], info["shape"]
        if dtype not in ("F8_E4M3", "I8") or not name.endswith(".weight"):
            continue
        if len(shape) != 2:
            raise ValueError(f"{name}: expected a matrix")
        if dtype == "I8":
            expected = [shape[0], shape[1] // 16]
        elif ".engram.embed." in name:
            expected = [shape[0], shape[1] // 32]
        else:
            expected = [(dim + 31) // 32 for dim in shape]
        scale = tensors.get(scale_name(name))
        if not scale or scale["dtype"] != "F8_E8M0" or scale["shape"] != expected:
            raise ValueError(f"{name}: expected E8M0 scales {expected}")


def write_engram(fp, item, db, np):
    # Keep weights and scales together without changing either byte. A chunk
    # holds 16K rows (4 MiB of weights), independent of the 94 GiB table size.
    rows = item.shape[1]
    for start in range(0, rows, 16384):
        count = min(16384, rows - start)
        weights = b"".join(db.iter_read(item.source, start * 256, count * 256))
        scales = b"".join(db.iter_read(scale_name(item.source), start * 8, count * 8))
        w = np.frombuffer(weights, dtype=np.uint8).reshape(count, 256)
        s = np.frombuffer(scales, dtype=np.uint8).reshape(count, 8)
        if np.any((w & 127) == 127) or np.any(s == 255):
            raise ValueError(f"{item.name}: nonfinite Engram row near {start}")
        packed = np.empty((count, 264), dtype=np.uint8)
        packed[:, :256], packed[:, 256:] = w, s
        fp.write(packed.tobytes())


