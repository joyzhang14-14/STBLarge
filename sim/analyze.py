"""对一个轨迹目录做留一交叉验证：每个提示词当一次测试，其余提示词的轨迹用来训练预测器。

用法：python sim/analyze.py TRACE_DIR [--cache 4256] [--json OUT]
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import sim

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--cache", type=int, default=4256)
ap.add_argument("--json")
a = ap.parse_args()

data = {}
for tr in sorted(glob.glob(f"{a.dir}/*.trace")):
    name = os.path.basename(tr)[:-6]
    err = open(f"{a.dir}/{name}.stderr").read() if os.path.exists(f"{a.dir}/{name}.stderr") else ""
    m = re.findall(r"processing (\d+) input tokens", err)
    gen = re.search(r"generation: ([\d.]+) t/s", err)
    data[name] = {"steps": sim.split_steps(sim.load(tr), int(m[-1]) if m else 0),
                  "tps": float(gen.group(1)) if gen else None}

CASES = [
    ("LRU（复现 ds4）", lambda tr, cap: dict(policy="lru")),
    ("LFU", lambda tr, cap: dict(policy="lfu")),
    ("LRU + 全局热门专家常驻 1/4", lambda tr, cap: dict(policy="lru", pin=sim.global_hot(tr, cap // 4))),
    ("LRU + 层间共现预测 k=6", lambda tr, cap: dict(policy="lru", predictor="transition", k=6,
                                                   trans=sim.train_transitions(tr))),
    ("LRU + 层间共现预测 k=12", lambda tr, cap: dict(policy="lru", predictor="transition", k=12,
                                                    trans=sim.train_transitions(tr))),
    ("LRU + 下一层路由器预测 前 6 名", lambda tr, cap: dict(policy="lru", predictor="shadow", k=6)),
    ("LRU + 下一层路由器预测 前 8 名", lambda tr, cap: dict(policy="lru", predictor="shadow", k=8)),
    ("LRU + 下一层路由器预测 前 12 名", lambda tr, cap: dict(policy="lru", predictor="shadow", k=12)),
    ("LRU + 预知下一层（上限）", lambda tr, cap: dict(policy="lru", predictor="oracle")),
    ("LRU + 预知下 1 个 token（上限）", lambda tr, cap: dict(policy="lru", predictor="token-oracle", lookahead=1)),
    ("LRU + 预知下 5 个 token（上限）", lambda tr, cap: dict(policy="lru", predictor="token-oracle", lookahead=5)),
]

out = {"cache": a.cache, "prompts": {}}
agg = {}
for name, d in data.items():
    steps = d["steps"]
    train = [s for n2, d2 in data.items() if n2 != name for s in d2["steps"]]
    cal = sim.calibrate(steps)
    gen = [s for s in steps if s["kind"] != "prefill"]
    acc = sum(len(r["ids"]) for s in gen for r in s["recs"])
    actual = sum(not h for s in gen for r in s["recs"] for h in r["hit"])
    rows = {"ds4 实测": {"hit_rate": 1 - actual / acc, "demand_misses": actual, "prefetch_reads": 0,
                         "ms_serial": cal["step_ms"], "ms_overlap": cal["step_ms"]}}
    for label, kw in CASES:
        r = sim.run(steps, a.cache, **kw(train, a.cache))
        r["ms_serial"] = sim.est_ms(r["per_step"], cal, False)
        r["ms_overlap"] = sim.est_layer_ms(r["per_layer"], cal, False)
        r["ms_cancel"] = sim.est_layer_ms(r["per_layer"], cal, True)
        r.pop("per_step"); r.pop("per_layer")
        rows[label] = r
    bel = sim.belady_misses(steps, a.cache)
    rows["理论最优淘汰（Belady）"] = {"hit_rate": 1 - bel / acc, "demand_misses": bel, "prefetch_reads": 0,
                                 "ms_serial": sim.est_ms([(bel / len(gen), 0)] * len(gen), cal, False),
                                 "ms_overlap": None}
    per_layer, recall = sim.pred_recall(steps)
    out["prompts"][name] = {"pred_recall": recall, "pred_recall_per_layer": per_layer, "calibration": cal, "measured_tps": d["tps"], "gen_steps": len(gen), "rows": rows}
    for label, r in rows.items():
        g = agg.setdefault(label, {"hit": [], "miss_tok": [], "pf_tok": [], "ser": [], "ovl": [], "cnl": []})
        g["hit"].append(r["hit_rate"]); g["miss_tok"].append(r["demand_misses"] / len(gen))
        g["pf_tok"].append(r["prefetch_reads"] / len(gen))
        if r.get("ms_serial"): g["ser"].append(r["ms_serial"])
        if r.get("ms_overlap"): g["ovl"].append(r["ms_overlap"])
        if r.get("ms_cancel"): g["cnl"].append(r["ms_cancel"])

mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
print(f"{len(data)} 个提示词，缓存 {a.cache} 个专家")
for name, p in out["prompts"].items():
    c = p["calibration"]
    print(f"  {name:14s} 实测 {p['measured_tps']} tok/s，每步 {c['step_ms']:.1f} ms，每层计算 {c['layer_compute_ms']:.2f} ms，"
          f"读一个专家 {c['miss_ms']:.2f} ms，每步缺失 {c['misses_per_step']:.1f}"
          + (f"，前 6 名猜中 {p['pred_recall'] * 100:.1f}%" if p["pred_recall"] is not None else ""))
print(f"{'情况':30s} {'命中率':>7s} {'缺失/token':>9s} {'预取/token':>9s} {'串行 tok/s':>10s} {'等全部 tok/s':>10s} {'取消多余 tok/s':>10s}")
summary = {}
for label, g in agg.items():
    ser, ovl, cnl = mean(g["ser"]), mean(g["ovl"]), mean(g["cnl"])
    summary[label] = {"hit_rate": mean(g["hit"]), "miss_per_token": mean(g["miss_tok"]),
                      "prefetch_per_token": mean(g["pf_tok"]),
                      "tps_serial": 1000 / ser if ser else None, "tps_overlap": 1000 / ovl if ovl else None,
                      "tps_cancel": 1000 / cnl if cnl else None}
    s = summary[label]
    f = lambda v: f"{v:10.2f}" if v else "         -"  # noqa: E731
    print(f"{label:30s} {s['hit_rate'] * 100:6.1f}% {s['miss_per_token']:9.1f} {s['prefetch_per_token']:9.1f} "
          f"{f(s['tps_serial'])} {f(s['tps_overlap'])} {f(s['tps_cancel'])}")
out["summary"] = summary
if a.json:
    json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)
