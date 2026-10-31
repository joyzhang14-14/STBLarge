"""STBLarge 缓存 / 预取模拟器：回放 ds4 的路由轨迹（DS4_ROUTE_TRACE），比较缓存策略和专家预测器。

预测器只决定「提前把哪些专家读进缓存」，不改路由，所以模型输出和不预测时逐字一致。

轨迹每行：单调时钟 ns、层号、这次路由的槽位数（token 数 × 6）、每个不同的专家 ID + h（命中）/ m（缺失）。
"""
import collections

N_LAYER = 40


def load(path):
    """预测行「p 层号 专家…」挂到同一层接下来的那条记录上（r["pred"]）。"""
    recs, pending = [], {}
    for line in open(path):
        f = line.split()
        if f and f[0] == "p":
            pending[int(f[1])] = [int(x) for x in f[2:]]
            continue
        if len(f) < 4:
            continue
        recs.append({"t": int(f[0]), "layer": int(f[1]), "slots": int(f[2]),
                     "ids": [int(x[:-1]) for x in f[3:]], "hit": [x[-1] == "h" for x in f[3:]],
                     "pred": pending.pop(int(f[1]), None)})
    return recs


def pred_recall(steps):
    """影子预测每层猜中的比例：|预测 ∩ 实际| / 实际个数。返回 {层: 比例} 和总体比例。"""
    hit, tot = collections.Counter(), collections.Counter()
    for s in steps:
        if s["kind"] == "prefill":
            continue
        for r in s["recs"]:
            if r.get("pred") is not None:
                hit[r["layer"]] += len(set(r["pred"]) & set(r["ids"]))
                tot[r["layer"]] += len(r["ids"])
    per = {L: hit[L] / tot[L] for L in sorted(tot)}
    return per, (sum(hit.values()) / sum(tot.values()) if tot else None)


def split_steps(recs, prefill_tokens=0):
    """层号不再增加就是新的一步（一次前向）。前 prefill_tokens 步、或一次多个 token 的步记为 prefill。"""
    steps, cur = [], []
    for r in recs:
        if cur and r["layer"] <= cur[-1]["layer"]:
            steps.append(cur)
            cur = []
        cur.append(r)
    if cur:
        steps.append(cur)
    out = []
    for i, s in enumerate(steps):
        many = max(r["slots"] for r in s) > 6
        out.append({"kind": "prefill" if i < prefill_tokens or (many and i < prefill_tokens + 1) else
                    ("verify" if many else "decode"), "recs": s, "t0": s[0]["t"]})
    return out


def calibrate(steps):
    """按层拟合：相邻两层记录的时间差 = 计算 c + 读一个专家 b × 这一层的缺失数（只看生成阶段）。"""
    xs, ys = [], []
    for s in steps:
        if s["kind"] == "prefill":
            continue
        for r, nxt in zip(s["recs"], s["recs"][1:]):
            xs.append(sum(not h for h in r["hit"]))
            ys.append((nxt["t"] - r["t"]) / 1e6)
    n = len(xs)
    if n < 10:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx else 0.0
    c = my - b * mx
    gen = [s for s in steps if s["kind"] != "prefill"]
    step_ms = [(n2["t0"] - s1["t0"]) / 1e6 for s1, n2 in zip(gen, gen[1:])]
    return {"layer_compute_ms": c, "miss_ms": b, "samples": n,
            "step_ms": sum(step_ms) / len(step_ms) if step_ms else None,
            "misses_per_step": sum(xs) / len(gen) if gen else None}


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


def belady_misses(steps, cap):
    """理论最优淘汰（知道未来）：缺失时淘汰下次用得最晚的。返回生成阶段的缺失数。"""
    seq = [((r["layer"], e), s["kind"] != "prefill") for s in steps for r in s["recs"] for e in r["ids"]]
    nxt, last = [0] * len(seq), {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i][0], 1 << 60)
        last[seq[i][0]] = i
    import heapq
    cache, heap, misses = {}, [], 0
    for i, (k, count) in enumerate(seq):
        if k not in cache:
            misses += count
            while len(cache) >= cap:
                neg, key = heapq.heappop(heap)
                if cache.get(key) == -neg:
                    del cache[key]
        cache[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return misses


def train_transitions(train_steps):
    """统计「第 L 层选了 e → 第 L+1 层选 e'」的共现次数。"""
    tr = collections.defaultdict(collections.Counter)
    for s in train_steps:
        by_layer = {r["layer"]: r["ids"] for r in s["recs"]}
        for L in range(N_LAYER - 1):
            if L in by_layer and L + 1 in by_layer:
                for e in by_layer[L]:
                    for e2 in by_layer[L + 1]:
                        tr[(L, e)][e2] += 1
    return tr


def global_hot(train_steps, n):
    freq = collections.Counter((r["layer"], e) for s in train_steps for r in s["recs"] for e in r["ids"])
    return [k for k, _ in freq.most_common(n)]


def run(steps, cap, policy="lru", predictor=None, k=6, trans=None, pin=(), lookahead=0):
    """回放。返回生成阶段的访问数、需求缺失、预取读盘，以及每步 (需求缺失, 预取读盘)。

    predictor：
      transition  用训练轨迹的层间共现，在第 L 层路由后预取第 L+1 层最可能的 k 个专家
      shadow      引擎记录的影子预测（下一层路由器套在本层路由器输入上）
      oracle      预知下一层（上限）
      token-oracle 预知 lookahead 个 token 之后整步要用的专家（上限）
    pin：常驻不淘汰的 (层, 专家) 集合，占用缓存容量。
    """
    cache = (LFU if policy == "lfu" else LRU)(cap - len(pin))
    pinned = set(pin)
    accesses = demand = prefetch_reads = 0
    per_step = []
    for si, s in enumerate(steps):
        gen = s["kind"] != "prefill"
        step_miss = step_pf = 0
        if predictor == "token-oracle" and gen and si + lookahead < len(steps):
            for r in steps[si + lookahead]["recs"]:
                for e in r["ids"]:
                    if (r["layer"], e) not in pinned and cache.prefetch((r["layer"], e)):
                        prefetch_reads += 1
                        step_pf += 1
        for r in s["recs"]:
            L = r["layer"]
            for e in r["ids"]:
                key = (L, e)
                accesses += gen
                if key in pinned:
                    continue
                if not cache.touch(key) and gen:
                    demand += 1
                    step_miss += 1
            if L + 1 >= N_LAYER or predictor not in ("transition", "oracle", "shadow"):
                continue
            if predictor == "shadow":
                nxt = [x for x in s["recs"] if x["layer"] == L + 1]
                guess = (nxt[0].get("pred") or [])[:k] if nxt else []
            elif predictor == "oracle":
                nxt = [x for x in s["recs"] if x["layer"] == L + 1]
                guess = nxt[0]["ids"][:k] if nxt else []
            else:
                score = collections.Counter()
                for e in r["ids"]:
                    score.update(trans.get((L, e), {}))
                guess = [e2 for e2, _ in score.most_common(k)]
            for e2 in guess:
                if (L + 1, e2) not in pinned and cache.prefetch((L + 1, e2)):
                    if gen:
                        prefetch_reads += 1
                        step_pf += 1
        if gen:
            per_step.append((step_miss, step_pf))
    return {"accesses": accesses, "demand_misses": demand, "prefetch_reads": prefetch_reads,
            "hit_rate": 1 - demand / accesses if accesses else None, "per_step": per_step}


def est_ms(per_step, cal, overlap):
    """估每步时间。不重叠：40 × c + b ×（缺失 + 预取）；重叠：预取读盘在后台和计算并行，只有超出计算时间的部分计入。"""
    C, b = N_LAYER * cal["layer_compute_ms"], cal["miss_ms"]
    tot = 0.0
    for miss, pf in per_step:
        tot += C + b * miss + (max(0.0, b * pf - C) if overlap else b * pf)
    return tot / len(per_step) if per_step else None
