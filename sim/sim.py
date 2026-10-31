"""STBLarge 缓存 / 预取模拟器：回放 ds4 的路由轨迹（DS4_ROUTE_TRACE），比较缓存策略和专家预测器。

预测器只决定「提前把哪些专家读进缓存」，不改路由，所以模型输出和不预测时逐字一致。

轨迹每行：单调时钟 ns、层号、这次路由的槽位数（token 数 × 6）、每个不同的专家 ID + h（命中）/ m（缺失）。
用法：python sim.py --test TRACE [--train TRACE ...] [--cache 4256] [--json OUT]
"""
import argparse, collections, json

N_LAYER = 40


def load(path):
    recs = []
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        recs.append({"t": int(f[0]), "layer": int(f[1]), "slots": int(f[2]),
                     "ids": [int(x[:-1]) for x in f[3:]], "hit": [x[-1] == "h" for x in f[3:]]})
    return recs


def split_steps(recs):
    """层号不再增加就是新的一步。每步标 prefill（一次多个 token）/ decode（6 个槽位）/ verify（DSpark 验证）。"""
    steps, cur = [], []
    for r in recs:
        if cur and r["layer"] <= cur[-1]["layer"]:
            steps.append(cur)
            cur = []
        cur.append(r)
    if cur:
        steps.append(cur)
    out = []
    seen_decode = False
    for s in steps:
        slots = max(r["slots"] for r in s)
        if slots == 6:
            kind, seen_decode = "decode", True
        elif seen_decode:
            kind = "verify"
        else:
            kind = "prefill"
        out.append({"kind": kind, "recs": s, "t0": s[0]["t"]})
    return out


def calibrate(steps):
    """用实测时间拟合每步耗时 = 计算 a + 每次缺失 b × 缺失数（只看 decode / verify 步）。"""
    xs, ys = [], []
    for s, nxt in zip(steps, steps[1:]):
        if s["kind"] == "prefill" or nxt["kind"] == "prefill":
            continue
        xs.append(sum(not h for r in s["recs"] for h in r["hit"]))
        ys.append((nxt["t0"] - s["t0"]) / 1e6)
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx else 0.0
    return {"compute_ms": my - b * mx, "miss_ms": b, "steps": n, "mean_step_ms": my, "mean_misses": mx}


class LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, collections.OrderedDict()

    def touch(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return True
        self.d[k] = 1
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return False

    def prefetch(self, k):
        """读进缓存但不算一次使用；已在缓存里就不用读盘。返回是否真的读了盘。"""
        if k in self.d:
            return False
        if len(self.d) >= self.cap:
            self.d.popitem(last=False)
        self.d[k] = 1
        return True


class LFU(LRU):
    """按出现次数淘汰，次数一样淘汰最久没用的。"""
    def __init__(self, cap):
        super().__init__(cap)
        self.freq = collections.Counter()

    def touch(self, k):
        self.freq[k] += 1
        if k in self.d:
            self.d.move_to_end(k)
            return True
        self.d[k] = 1
        if len(self.d) > self.cap:
            victim = min(self.d, key=lambda x: self.freq[x])
            del self.d[victim]
        return False


def belady(accesses, cap):
    """理论最优（知道未来）：缺失时淘汰下次用得最晚的。只算缺失数，作上限参考。"""
    nxt, last = [0] * len(accesses), {}
    for i in range(len(accesses) - 1, -1, -1):
        nxt[i] = last.get(accesses[i], 1 << 60)
        last[accesses[i]] = i
    cache, misses = {}, []
    for i, k in enumerate(accesses):
        if k in cache:
            cache[k] = nxt[i]
            misses.append(False)
            continue
        misses.append(True)
        if len(cache) >= cap:
            victim = max(cache, key=cache.get)
            del cache[victim]
        cache[k] = nxt[i]
    return misses


def train_transitions(train_steps):
    """统计「第 L 层选了 e → 第 L+1 层选 e'」的同步共现次数。"""
    tr = collections.defaultdict(collections.Counter)
    for s in train_steps:
        if s["kind"] == "prefill":
            continue
        by_layer = {r["layer"]: r["ids"] for r in s["recs"]}
        for L in range(N_LAYER - 1):
            if L in by_layer and L + 1 in by_layer:
                for e in by_layer[L]:
                    for e2 in by_layer[L + 1]:
                        tr[(L, e)][e2] += 1
    return tr


def run(steps, cap, policy="lru", predictor=None, k=6, trans=None, pin_prefill=0):
    """回放：返回 decode / verify 阶段的需求缺失和预取读盘次数，以及每步的需求缺失列表（算时间用）。"""
    cache = (LFU if policy == "lfu" else LRU)(cap)
    pinned = set()
    if pin_prefill:
        freq = collections.Counter((r["layer"], e) for s in steps if s["kind"] == "prefill"
                                   for r in s["recs"] for e in r["ids"])
        pinned = {key for key, _ in freq.most_common(pin_prefill)}
        cache.cap = cap - len(pinned)
    demand = prefetch_reads = accesses = 0
    per_step = []
    prev = None
    for s in steps:
        step_miss = step_pf = 0
        for r in s["recs"]:
            L = r["layer"]
            for e in r["ids"]:
                key = (L, e)
                if s["kind"] != "prefill":
                    accesses += 1
                if key in pinned:
                    continue
                hit = cache.touch(key)
                if not hit and s["kind"] != "prefill":
                    demand += 1
                    step_miss += 1
            if s["kind"] == "prefill" or L + 1 >= N_LAYER:
                continue
            guess = []
            if predictor == "oracle":
                nxt = [x for x in s["recs"] if x["layer"] == L + 1]
                guess = nxt[0]["ids"][:k] if nxt else []
            elif predictor == "transition" and trans is not None:
                score = collections.Counter()
                for e in r["ids"]:
                    score.update(trans.get((L, e), {}))
                guess = [e2 for e2, _ in score.most_common(k)]
            elif predictor == "prev-token" and prev is not None:
                nxt = [x for x in prev["recs"] if x["layer"] == L + 1]
                guess = nxt[0]["ids"][:k] if nxt else []
            for e2 in guess:
                if (L + 1, e2) not in pinned and cache.prefetch((L + 1, e2)):
                    prefetch_reads += 1
                    step_pf += 1
        if s["kind"] != "prefill":
            per_step.append((step_miss, step_pf))
            prev = s
    return {"accesses": accesses, "demand_misses": demand, "prefetch_reads": prefetch_reads,
            "hit_rate": 1 - demand / accesses if accesses else None, "per_step": per_step}


def est_ms(per_step, cal, overlap):
    """估每步时间。不重叠：a + b ×（缺失 + 预取）；重叠：预取读盘藏在计算后面，超出计算时间的部分才计入。"""
    a, b = cal["compute_ms"], cal["miss_ms"]
    tot = 0.0
    for miss, pf in per_step:
        tot += a + b * miss + (max(0.0, b * pf - a) if overlap else b * pf)
    return tot / len(per_step) if per_step else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--cache", type=int, default=4256)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--json")
    a = ap.parse_args()

    steps = split_steps(load(a.test))
    cal = calibrate(steps)
    actual = [sum(not h for r in s["recs"] for h in r["hit"]) for s in steps if s["kind"] != "prefill"]
    acc = sum(len(r["ids"]) for s in steps if s["kind"] != "prefill" for r in s["recs"])
    trans = train_transitions([s for t in a.train for s in split_steps(load(t))]) if a.train else None

    rows = {"ds4 实测": {"hit_rate": 1 - sum(actual) / acc if acc else None,
                        "demand_misses": sum(actual), "prefetch_reads": 0, "accesses": acc}}
    cases = [("LRU", dict(policy="lru")), ("LFU", dict(policy="lfu")),
             ("LRU + 预填充热专家固定 1/4", dict(policy="lru", pin_prefill=a.cache // 4)),
             ("LRU + 上一个 token 预取", dict(policy="lru", predictor="prev-token", k=a.k)),
             ("LRU + 预知下一层（上限）", dict(policy="lru", predictor="oracle", k=a.k))]
    if trans:
        cases.append(("LRU + 层间转移预测", dict(policy="lru", predictor="transition", k=a.k, trans=trans)))
    for name, kw in cases:
        rows[name] = run(steps, a.cache, **kw)
    demand_seq = [(r["layer"], e) for s in steps for r in s["recs"] for e in r["ids"]]
    first = sum(len(r["ids"]) for s in steps if s["kind"] == "prefill" for r in s["recs"])
    bel = belady(demand_seq, a.cache)[first:]
    rows["理论最优淘汰（Belady）"] = {"hit_rate": 1 - sum(bel) / len(bel) if bel else None,
                                 "demand_misses": sum(bel), "prefetch_reads": 0, "accesses": len(bel)}

    n_tok = len(actual)
    for name, r in rows.items():
        if cal and "per_step" in r:
            r["ms_serial"] = est_ms(r["per_step"], cal, False)
            r["ms_overlap"] = est_ms(r["per_step"], cal, True)
        r.pop("per_step", None)
    print(f"步数 {n_tok}（decode / verify），校准：{cal}")
    print(f"{'情况':28s} {'命中率':>7s} {'需求缺失':>8s} {'预取读盘':>8s} {'串行 ms/步':>10s} {'重叠 ms/步':>10s}")
    for name, r in rows.items():
        hr = f"{r['hit_rate'] * 100:6.1f}%" if r.get("hit_rate") is not None else "    -"
        f = lambda v: f"{v:10.1f}" if v is not None else "         -"  # noqa: E731
        print(f"{name:28s} {hr:>7s} {r['demand_misses']:8d} {r['prefetch_reads']:8d} "
              f"{f(r.get('ms_serial'))} {f(r.get('ms_overlap'))}")
    if a.json:
        json.dump({"calibration": cal, "steps": n_tok, "cache": a.cache, "rows": rows},
                  open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
