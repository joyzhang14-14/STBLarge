"""采集专家路由轨迹：用带 DS4_ROUTE_TRACE 补丁的 ds4 跑一组不同类型的提示词，每个提示词一份轨迹。

不开 DSpark、温度 0、SSD 串流、专家缓存自动大小。轨迹格式见 sim/sim.py 开头。
用法：python collect_traces.py OUT_DIR [--predict] [--prefetch K]
（--predict 记录下一层路由预测，轨迹里多出 p 行；--prefetch K 按预测的前 K 名在后台预取）
"""
import json, os, subprocess, sys

DS4 = os.environ.get("STBL_DS4", "/home/joyzhang14/apps/STBLarge/engine/ds4")
MODEL = os.environ.get("STBL_MODEL", "/home/joyzhang14/models/ds4-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf")
PROMPTS = {
    "c-hashtable": "Write a complete C hash table implementation with string keys, insert, find, delete, and a test main. Output only C code.",
    "py-lru": "用 Python 实现一个线程安全的 LRU 缓存，带过期时间，并写单元测试。",
    "zh-essay": "写一篇八百字的散文，题目是《凌晨三点的实验室》。",
    "en-story": "Write a short story about a lighthouse keeper who finds a message in a bottle from the future.",
    "math": "证明：对任意正整数 n，n^5 - n 能被 30 整除。给出完整推理。",
    "translate": "把下面这段话翻译成英文并解释其中的术语：混合专家模型在推理时只激活一小部分参数，因此可以把不常用的专家放在固态硬盘上按需读取。",
    "json-extract": "从下面的文字中提取人物、地点、时间，输出 JSON：2025 年 3 月 2 日上午，林晓在杭州的研究所里调试一台新到的服务器，下午和同事去西湖边开会。",
    "chat": "我最近总是熬夜，第二天上课很困，有什么实际可行的办法帮我调整作息？",
}


def main(out, predict=False, prefetch=0, only=None, tokens=256, extra_env=None, extra_args=(), trace=True):
    os.makedirs(out, exist_ok=True)
    lines = []
    for name, prompt in PROMPTS.items():
        if only and name not in only:
            continue
        env = dict(os.environ, DS4_ROUTE_TRACE=f"{out}/{name}.trace") if trace else dict(os.environ)
        if predict:
            env["DS4_ROUTE_PREDICT"] = "1"
        if prefetch:
            env["DS4_ROUTE_PREFETCH"] = str(prefetch)
        env.update(extra_env or {})
        cmd = [DS4, "--cuda", "-m", MODEL, "--ssd-streaming", "--ctx", "4096", "--nothink",
               "--temp", "0", "--seed", "12345", "-n", str(tokens), *extra_args, "-p", prompt]
        with open(f"{out}/{name}.stdout", "w") as so, open(f"{out}/{name}.stderr", "w") as se:
            rc = subprocess.run(cmd, stdout=so, stderr=se, env=env).returncode
        size = os.path.getsize(f"{out}/{name}.trace") if os.path.exists(f"{out}/{name}.trace") else 0
        lines.append(f"{name}: 退出码 {rc}，轨迹 {size / 1e6:.1f} MB")
    json.dump(PROMPTS, open(f"{out}/prompts.json", "w"), ensure_ascii=False, indent=1)
    open(f"{out}/summary.txt", "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    k = int(sys.argv[sys.argv.index("--prefetch") + 1]) if "--prefetch" in sys.argv else 0
    print("\n".join(main(sys.argv[1], "--predict" in sys.argv, k)))
