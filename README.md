# STBLarge

在内存装不下的大 MoE 模型（DeepSeek-V4.1-Flash、GLM-5.3 这一类）上做「专家预测 + SSD 串流」的推理底座。目标机器是 DGX Spark（GB10，128 GB 统一内存，PCIe 5 NVMe）。

## 原则

- **输出不变**：预测只决定提前把哪些专家从 SSD 读进内存，从不改变模型实际选哪个专家。预测错了只是多读一次盘，生成的 token 和不预测时逐字一致。
- **先模拟后实现**：先用真实路由轨迹在 CPU 上回放，量出每种预测方法能省多少读盘，再决定把哪种做进运行时。

## 目录

| 路径 | 内容 |
| --- | --- |
| `sim/sim.py` | 缓存 / 预取模拟器：回放路由轨迹，比较 LRU、LFU、理论最优淘汰，以及几种预测器的命中率、读盘量和估计速度 |
| `scripts/collect_traces.py` | 用带轨迹补丁的 ds4 跑一组提示词，每个提示词存一份路由轨迹 |

运行时目前基于 [antirez/ds4](https://github.com/antirez/ds4)。轨迹补丁在 ds4 的 `stblarge-trace` 分支（Sparky 上的工作目录 `~/apps/ds4-stblarge`）：设了 `DS4_ROUTE_TRACE=文件` 时，每次某一层加载专家就记一行。

## 轨迹格式

每行一次专家加载：

```
<单调时钟 ns> <层号> <槽位数> <专家ID><h|m> ...
```

槽位数 = 这次一起算的 token 数 × 6（每个 token 选 6 个专家）；h = 已在缓存，m = 要从 SSD 读。

## 用法

```sh
python scripts/collect_traces.py OUT_DIR          # 要 GPU：在 Sparky 上登记成 stb 任务跑
python sim/sim.py --test OUT_DIR/c-hashtable.trace --train OUT_DIR/zh-essay.trace ... --cache 4256
```
