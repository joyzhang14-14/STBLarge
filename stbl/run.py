"""Greedy generation with the STBLarge engine (libstbl.so), saved like ref/run_ref.py for comparison.

usage: run.py OUT.pt --prompt TEXT [--system TEXT] [--tokens N] [--ctx N] [--experts N]
The official tokenizer and chat encoding come from the checkpoint directory (STBL_OFFICIAL).
"""

import argparse
import ctypes
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("STBL_OFFICIAL", "/home/joyzhang14/models/DeepSeek-V4.1-Flash-1017")
GGUF = os.environ.get("STBL_GGUF", "/home/joyzhang14/models/stbl/DeepSeek-V4.1-Flash-1017-STBL.gguf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default="")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=0)  # 0: sized from free memory
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(CKPT, "encoding"))
    from encoding import encode_messages
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    messages = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
    ids = tok.encode(encode_messages(messages, thinking_mode="chat"))

    lib = ctypes.CDLL(os.path.join(HERE, "libstbl.so"))
    lib.stbl_open.restype = ctypes.c_void_p
    lib.stbl_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.stbl_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.stbl_vocab.argtypes = [ctypes.c_void_p]
    lib.stbl_error.restype = ctypes.c_char_p
    lib.stbl_error.argtypes = [ctypes.c_void_p]
    lib.stbl_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    lib.stbl_timing.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.stbl_lookahead.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.stbl_prefetch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    def snapshot():
        sec = (ctypes.c_double * 16)()
        n = lib.stbl_timing(m, sec, 16)
        hits, misses = ctypes.c_uint64(), ctypes.c_uint64()
        lib.stbl_stats(m, ctypes.byref(hits), ctypes.byref(misses))
        return list(sec[:n]), hits.value, misses.value
    err = ctypes.create_string_buffer(512)
    t0 = time.time()
    m = lib.stbl_open(GGUF.encode(), a.ctx, a.experts, err, 512)
    if not m:
        sys.exit(f"stbl: {err.value.decode()}")
    vocab = lib.stbl_vocab(m)
    print(f"stbl: loaded in {time.time() - t0:.0f}s, prompt {len(ids)} tokens", flush=True)

    logits = torch.empty(vocab, dtype=torch.float32)
    t0 = time.time()
    nxt = -1
    for t in ids:
        nxt = lib.stbl_step(m, t, ctypes.c_void_p(logits.data_ptr()))
        if nxt < 0:
            sys.exit(f"stbl: {lib.stbl_error(m).decode()}")
    print(f"stbl: prefill {len(ids) / (time.time() - t0):.2f} tok/s", flush=True)
    before = snapshot()
    out_ids, all_logits = [], []
    t0 = time.time()
    for step in range(a.tokens):
        out_ids.append(nxt)
        all_logits.append(logits.clone())
        print(f"stbl: step {step} token {nxt} {tok.decode([nxt])!r}", flush=True)
        if nxt == tok.eos_token_id or step == a.tokens - 1:
            break
        nxt = lib.stbl_step(m, nxt, ctypes.c_void_p(logits.data_ptr()))
        if nxt < 0:
            sys.exit(f"stbl: {lib.stbl_error(m).decode()}")
    dt = time.time() - t0
    after = snapshot()
    steps = max(len(out_ids) - 1, 1)
    print(f"stbl: generation {steps / dt:.2f} tok/s, experts hit {after[1]} miss {after[2]}"
          f" (generation: {(after[2] - before[2]) / steps:.1f} misses/token)", flush=True)
    if os.environ.get("STBL_TIMING", "0") not in ("", "0"):
        names = ["hc/norm", "engram", "attention", "router", "expert read", "expert compute", "shared expert", "head"]
        per = [(a_ - b_) / steps * 1000 for a_, b_ in zip(after[0], before[0])]
        print("stbl: timing ms/token " + "  ".join(f"{n} {v:.1f}" for n, v in zip(names, per))
              + f"  total {sum(per):.1f}", flush=True)
    if os.environ.get("STBL_LOOKAHEAD", "0") not in ("", "0"):
        la = (ctypes.c_uint64 * 4)()
        lib.stbl_lookahead(m, la)
        print(f"stbl: lookahead top {os.environ['STBL_LOOKAHEAD']}: uncached guesses {la[0]}, routed {la[1]}"
              f" ({la[1] / max(la[0], 1):.0%}); misses {la[2]}, guessed {la[3]} ({la[3] / max(la[2], 1):.0%})", flush=True)
    if os.environ.get("STBL_PREFETCH", "8") not in ("", "0"):
        pf = (ctypes.c_uint64 * 4)()
        lib.stbl_prefetch(m, pf)
        print(f"stbl: prefetch top {os.environ.get("STBL_PREFETCH", "8")}: issued {pf[0]}, routed and landed {pf[1]},"
              f" routed while reading {pf[2]}, cancelled {pf[3]}", flush=True)
    torch.save({"prompt_ids": ids, "ids": out_ids, "logits": torch.stack(all_logits), "text": tok.decode(out_ids)}, a.out)
    print("stbl: text", repr(tok.decode(out_ids)))


if __name__ == "__main__":
    main()
