"""STBLarge 缓存 / 预取模拟器：回放 ds4 的路由轨迹（DS4_ROUTE_TRACE），比较缓存策略和专家预测器。

预测器只决定「提前把哪些专家读进缓存」，不改路由，所以模型输出和不预测时逐字一致。

轨迹每行：单调时钟 ns、层号、这次路由的槽位数（token 数 × 6）、每个不同的专家 ID + h（命中）/ m（缺失）。
"""
import collections

N_LAYER = 40


def load(path):
    """预测行「p 层号 专家…」（提前一层）和「p2 层号 专家…」（提前两层）挂到同一层接下来的那条记录上（r["pred"]、r["pred2"]）。"""
    recs, pending = [], {}
    for line in open(path):
        f = line.split()
        if f and f[0] in ("p", "p2"):
            pending[(f[0], int(f[1]))] = [int(x) for x in f[2:]]
            continue
        if len(f) < 4:
            continue
        recs.append({"t": int(f[0]), "layer": int(f[1]), "slots": int(f[2]),
                     "ids": [int(x[:-1]) for x in f[3:]], "hit": [x[-1] == "h" for x in f[3:]],
                     "pred": pending.pop(("p", int(f[1])), None),
                     "pred2": pending.pop(("p2", int(f[1])), None)})
    return recs


def pred_recall(steps, k=6, key="pred"):
    """预测前 k 名每层猜中的比例：|预测前 k ∩ 实际| / 实际个数。返回 {层: 比例} 和总体比例。"""
    hit, tot = collections.Counter(), collections.Counter()
    for s in steps:
        if s["kind"] == "prefill":
            continue
        for r in s["recs"]:
            if r.get(key) is not None:
                hit[r["layer"]] += len(set(r[key][:k]) & set(r["ids"]))
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
    per_step, per_layer = [], []
    issued = {}  # 层 -> 为它预取、真的读了盘的专家（按预测名次）
    for si, s in enumerate(steps):
        gen = s["kind"] != "prefill"
        step_miss = step_pf = 0
        layers = []
        if predictor == "token-oracle" and gen and si + lookahead < len(steps):
            for r in steps[si + lookahead]["recs"]:
                for e in r["ids"]:
                    if (r["layer"], e) not in pinned and cache.prefetch((r["layer"], e)):
                        prefetch_reads += 1
                        step_pf += 1
        for r in s["recs"]:
            L = r["layer"]
            layer_miss = 0
            for e in r["ids"]:
                key = (L, e)
                accesses += gen
                if key in pinned:
                    continue
                if not cache.touch(key) and gen:
                    demand += 1
                    step_miss += 1
                    layer_miss += 1
            got = issued.pop(L, [])
            need = [i for i, e in enumerate(got) if e in r["ids"]]
            layers.append((len(got), max(need) if need else -1, layer_miss))
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
                    issued.setdefault(L + 1, []).append(e2)
                    if gen:
                        prefetch_reads += 1
                        step_pf += 1
        if gen:
            per_step.append((step_miss, step_pf))
            per_layer.append(layers)
    return {"accesses": accesses, "demand_misses": demand, "prefetch_reads": prefetch_reads,
            "hit_rate": 1 - demand / accesses if accesses else None, "per_step": per_step,
            "per_layer": per_layer}


def est_layer_ms(per_layer, cal, cancel):
    """按层估每步时间。第 L-1 层把第 L 层的预测交给后台读线程，读盘只能藏在约一层的计算时间里。

    cancel=False：到第 L 层时等全部预取读完（ds4 现在的做法）；
    cancel=True：只等第 L 层真正要用的那几个，猜错的取消。
    之后再同步读预取没覆盖到的缺失。
    """
    c, b = cal["layer_compute_ms"], cal["miss_ms"]
    tot = 0.0
    for layers in per_layer:
        for n_issued, last_needed, miss in layers:
            reads = (last_needed + 1) if cancel else n_issued
            tot += c + max(0.0, b * reads - c) + b * miss
    return tot / len(per_layer) if per_layer else None


def est_ms(per_step, cal, overlap):
    """按整步粗估，偏乐观：把预取窗口当成整个 token 的计算时间。按层的估计见 est_layer_ms。"""
    C, b = N_LAYER * cal["layer_compute_ms"], cal["miss_ms"]
    tot = 0.0
    for miss, pf in per_step:
        tot += C + b * miss + (max(0.0, b * pf - C) if overlap else b * pf)
    return tot / len(per_step) if per_step else None


def run_timed(steps, cap, cal, depth=1, k=6, cancel=True, oracle=False):
    """事件模拟：后台一条读队列（先进先出），按预测预取；返回 (生成阶段每步平均 ms, 每 token 按需缺失, 每 token 预取读盘)。

    第 L 层的路由一出来，就把对第 L+1 层（depth>=1，用 pred）和第 L+2 层（depth>=2，用 pred2）的前 k 名里
    不在缓存的专家排进读队列。到了目标层：cancel=True 时只等真正要用的那几个读完，
    还没开始读的猜错项直接丢掉；cancel=False 时等所有为这一层排的都读完。
    按需缺失要等读队列里正在读的那一个读完，再一个个同步读。oracle=True 时用真实路由代替预测（上限）。
    """
    c, b = cal["layer_compute_ms"], cal["miss_ms"]
    cache = LRU(cap)
    t = 0.0
    busy_until = 0.0
    queue = []  # [目标层, 键, 开始时间, 结束时间]
    gen_ms, misses, reads, n_gen = 0.0, 0, 0, 0
    for s in steps:
        gen = s["kind"] != "prefill"
        t0 = t
        by_layer = {r["layer"]: r for r in s["recs"]}
        for r in s["recs"]:
            L = r["layer"]
            needed = {(L, e) for e in r["ids"]}
            mine = [j for j in queue if j[0] == L]
            if cancel:
                started = [j for j in mine if j[2] <= t]
                waiting = [j[3] for j in mine if j[1] in needed and j[2] <= t]
                queued_needed = [j for j in mine if j[1] in needed and j[2] > t]
                # 还没开始的：要用的照样读，不要的丢掉，后面的往前挪
                keep = [j for j in queue if j[0] != L or j[2] <= t or j[1] in needed]
                if len(keep) != len(queue):
                    queue, cur = [], None
                    run_end = max([j[3] for j in keep if j[2] <= t] + [t])
                    for j in keep:
                        if j[2] > t:
                            j[2] = run_end
                            j[3] = run_end + b
                            run_end = j[3]
                        queue.append(j)
                    busy_until = run_end
                    mine = [j for j in queue if j[0] == L]
                wait_to = max([j[3] for j in mine if j[1] in needed] + [t])
            else:
                wait_to = max([j[3] for j in mine] + [t])
            for j in mine:
                if j[1] in needed or j[3] <= wait_to:
                    cache.prefetch(j[1])
            queue = [j for j in queue if j[0] != L]
            t = wait_to
            for e in r["ids"]:
                if not cache.touch((L, e)):
                    t = max(t, min([j[3] for j in queue if j[2] <= t] + [t])) + b
                    misses += gen
            busy_until = max(busy_until, t)
            # 预测下一层、下两层
            for d, key in ((1, "pred"), (2, "pred2")):
                if d > depth or L + d >= N_LAYER or L + d not in by_layer:
                    continue
                guess = by_layer[L + d]["ids"] if oracle else (by_layer[L + d].get(key) or [])
                for e in guess[:k]:
                    kk = (L + d, e)
                    if kk in cache.d or any(j[1] == kk for j in queue):
                        continue
                    start = max(busy_until, t)
                    busy_until = start + b
                    queue.append([L + d, kk, start, busy_until])
                    reads += gen
            t += c
        if gen:
            gen_ms += t - t0
            n_gen += 1
    return (gen_ms / n_gen if n_gen else None, misses / n_gen if n_gen else None, reads / n_gen if n_gen else None)
