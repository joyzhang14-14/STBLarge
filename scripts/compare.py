"""Compare two generation results saved by ref/run_ref.py or stbl/run.py (greedy, same prompt).

usage: compare.py REF.pt OTHER.pt
Prints the prompt check, the first differing token, and per-step logit agreement over the shared
prefix: max |dlogit|, KL(ref || other) of the softmax, and the reference top-1 margin.
"""
import sys

import torch


def main():
    ref, oth = torch.load(sys.argv[1]), torch.load(sys.argv[2])
    if ref["prompt_ids"] != oth["prompt_ids"]:
        sys.exit("prompts differ")
    a, b = ref["ids"], oth["ids"]
    n = min(len(a), len(b))
    div = next((i for i in range(n) if a[i] != b[i]), None)
    upto = n if div is None else div + 1
    worst_kl, worst_d = 0.0, 0.0
    for i in range(upto):
        la, lb = ref["logits"][i].double(), oth["logits"][i].double()
        pa = torch.log_softmax(la, -1)
        kl = float((pa.exp() * (pa - torch.log_softmax(lb, -1))).sum())
        d = float((la - lb).abs().max())
        worst_kl, worst_d = max(worst_kl, kl), max(worst_d, d)
        if i < 3 or i == upto - 1:
            top = la.topk(2)
            print(f"  step {i}: max|dlogit| {d:.4f}  KL {kl:.2e}  ref margin {float(top.values[0] - top.values[1]):.3f}")
    if div is None:
        print(f"same {n} tokens; worst max|dlogit| {worst_d:.4f}, worst KL {worst_kl:.2e}")
    else:
        top = ref["logits"][div].topk(2)
        print(f"first difference at token {div} (ref margin {float(top.values[0] - top.values[1]):.3f}); "
              f"before it worst max|dlogit| {worst_d:.4f}, worst KL {worst_kl:.2e}")


if __name__ == "__main__":
    main()
