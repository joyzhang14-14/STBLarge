"""For each token: first layer whose routed experts differ between two runs, next to the first layer
where the residual stream differs by more than 1e-2 (relative max error).

usage: route_diff.py REF_LAYERS.pt OTHER_LAYERS(.pt|.f32)
Routes are read from the matching files: <name>-routes.pt (reference) or stbl-routes.i32 (engine).
"""
import os
import sys

import numpy as np
import torch


def load(path, L=None, W=None):
    if path.endswith(".pt"):
        layers = torch.load(path).float()
        routes = torch.load(path.removesuffix(".pt") + "-routes.pt")
        return layers, routes
    layers = torch.from_numpy(np.fromfile(path, dtype=np.float32))
    layers = layers[: layers.numel() // (L * W) * L * W].view(-1, L, W)
    routes = torch.from_numpy(np.fromfile(os.path.join(os.path.dirname(path), "stbl-routes.i32"), dtype=np.int32))
    return layers, routes.view(-1, L, routes.numel() // layers.shape[0] // L)


def main():
    ref, rr = load(sys.argv[1])
    T, L, W = ref.shape
    oth, ro = load(sys.argv[2], L, W)
    T = min(T, oth.shape[0], ro.shape[0])
    both = same_layer = 0
    for t in range(T):
        err = ((oth[t] - ref[t]).abs().amax(-1) / ref[t].abs().amax(-1).clamp_min(1e-30)).tolist()
        first_err = next((l for l, e in enumerate(err) if e > 1e-2), None)
        first_route = next((l for l in range(L) if set(rr[t, l].tolist()) != set(ro[t, l].tolist())), None)
        if first_err is not None and first_route is not None:
            both += 1
            same_layer += first_route == first_err
        print(f"token {t:3d}: routes differ from layer {first_route}, residual >1e-2 from layer {first_err}")
    print(f"tokens with both: {both}, of which the route flip is at the same layer: {same_layer}")


if __name__ == "__main__":
    main()
