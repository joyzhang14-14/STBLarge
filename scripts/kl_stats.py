"""KL statistics of several runs against one reference run, over the shared greedy prefix.

usage: kl_stats.py REF_DIR RUN_DIR [RUN_DIR ...]   (each holds <prompt>.pt from run_ref.py / stbl/run.py)
Per run: first differing token per prompt (with the reference top-1 margin there) and the median /
90th percentile / max KL(ref || run) over all steps before the first difference.
"""
import glob
import os
import sys

import torch


def main():
    ref_dir, runs = sys.argv[1], sys.argv[2:]
    names = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(ref_dir, "*.pt")))
    for run in runs:
        kls, notes = [], []
        for n in names:
            p = os.path.join(run, n + ".pt")
            if not os.path.exists(p):
                continue
            a, b = torch.load(os.path.join(ref_dir, n + ".pt")), torch.load(p)
            m = min(len(a["ids"]), len(b["ids"]))
            div = next((i for i in range(m) if a["ids"][i] != b["ids"][i]), None)
            for i in range(m if div is None else div + 1):
                pa = torch.log_softmax(a["logits"][i].double(), -1)
                kls.append(float((pa.exp() * (pa - torch.log_softmax(b["logits"][i].double(), -1))).sum()))
            if div is None:
                notes.append(f"{n}:同{m}")
            else:
                top = a["logits"][div].topk(2)
                notes.append(f"{n}:第{div}个分岔(领先{float(top.values[0] - top.values[1]):.2f})")
        k = torch.tensor(kls)
        print(f"{run}\n  KL 中位 {k.median():.2e}  P90 {k.quantile(0.9):.2e}  最大 {k.max():.2e}  共 {len(kls)} 步")
        print("  " + "  ".join(notes))


if __name__ == "__main__":
    main()
