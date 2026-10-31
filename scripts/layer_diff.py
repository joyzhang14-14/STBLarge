"""Layer-by-layer comparison of residual-stream dumps: reference (STBL_REF_DUMP, [T, L, hc*dim] bf16)
against the engine (STBL_DUMP, raw fp32 in the same order).

usage: layer_diff.py REF_LAYERS.pt ENGINE_LAYERS.f32
For each token prints the relative max error after every layer (|e-r|max / |r|max) and the first
layer where it exceeds 1e-2.
"""
import sys

import numpy as np
import torch


def main():
    ref = torch.load(sys.argv[1]).float()
    T, L, W = ref.shape
    eng = torch.from_numpy(np.fromfile(sys.argv[2], dtype=np.float32))
    eng = eng[: eng.numel() // (L * W) * L * W].view(-1, L, W)
    T = min(T, eng.shape[0])
    print(f"tokens {T}, layers {L}, width {W}")
    for t in range(T):
        err = ((eng[t] - ref[t]).abs().amax(-1) / ref[t].abs().amax(-1).clamp_min(1e-30)).tolist()
        first = next((l for l, e in enumerate(err) if e > 1e-2), None)
        row = " ".join(f"{e:.0e}" for e in err)
        print(f"token {t:3d} first>1e-2: {first}  | {row}")


if __name__ == "__main__":
    main()
