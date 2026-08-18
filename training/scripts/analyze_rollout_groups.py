#!/usr/bin/env python3
"""
分析 VERL rollout dump，诊断 GRPO 训练停滞根因。

用法（训练进行中安全运行，纯只读）:
    # 分析最新 5 个 step
    python analyze_rollout_groups.py --rollout-dir /path/to/outputs/.../rollout --last-n 5

    # 分析指定 step
    python analyze_rollout_groups.py --rollout-dir <dir> --step 30

    # 全部 step + 导出 CSV
    python analyze_rollout_groups.py --rollout-dir <dir> --all --csv summary.csv

Dump 格式（ray_trainer._dump_generations）: 每 step 一个 {step}.jsonl，
每行 {"input": prompt文本, "output": 回答, "score": float, "step": int, ...}。
按 input 文本分组 == GRPO 的 uid 分组（同一 prompt 的 n 条 rollout）。

score 语义（reward_score/deep_research.py）:
    -1   : 格式错误 (bad format)
     0   : 格式对但答案错 (wrong answer)
    >0   : 答案正确 (f1, 最高 1.1)
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="分析 rollout dump 的分组 reward 分布")
    p.add_argument("--rollout-dir", required=True, help="ROLLOUT_SAVE_PATH，含 {step}.jsonl 的目录")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--step", type=int, default=None, help="只分析指定 step")
    g.add_argument("--last-n", type=int, default=None, help="只分析最新 N 个 step")
    g.add_argument("--all", action="store_true", help="分析全部 step")
    p.add_argument("--std-eps", type=float, default=1e-6, help="判定'无梯度信号'的组内 std 阈值")
    p.add_argument("--csv", default=None, help="将逐 step 汇总写到 CSV")
    return p.parse_args()


def list_step_files(rollout_dir, step, last_n, do_all):
    files = glob.glob(os.path.join(rollout_dir, "*.jsonl"))
    # 文件名形如 {step}.jsonl，按 step 数值排序
    def step_of(f):
        try:
            return int(os.path.splitext(os.path.basename(f))[0])
        except ValueError:
            return -1
    files = sorted([f for f in files if step_of(f) >= 0], key=step_of)
    if not files:
        return []
    if step is not None:
        return [f for f in files if step_of(f) == step]
    if last_n is not None:
        return files[-last_n:]
    if do_all:
        return files
    # 默认: 最新 1 个
    return files[-1:]


def load_jsonl(path):
    """逐行解析，容忍训练正在写入导致的最后一行不完整。"""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # 可能是当前 step 尚未写完的残行，跳过
                continue
    return rows


def bucket(score):
    """把 score 归到三档: bad_format(-1) / wrong(0) / correct(>0)。"""
    if score <= -0.5:      # 容忍浮点误差，-1 档
        return "bad_format"
    if score <= 1e-9:      # 0 档
        return "wrong"
    return "correct"


def analyze_step(rows, std_eps):
    groups = defaultdict(list)
    for r in rows:
        if "input" not in r or "score" not in r:
            continue
        groups[r["input"]].append(float(r["score"]))

    n_samples = sum(len(v) for v in groups.values())
    n_groups = len(groups)
    if n_groups == 0:
        return None

    # 全样本 score 三档占比
    all_scores = np.array([s for v in groups.values() for s in v], dtype=float)
    comp = {k: 0 for k in ("bad_format", "wrong", "correct")}
    for s in all_scores:
        comp[bucket(s)] += 1

    group_sizes = np.array([len(v) for v in groups.values()])
    group_stds = np.array([float(np.std(v)) for v in groups.values()])

    # 无信号组: 组内 std≈0 → advantage 全 0 → 无梯度
    no_signal_mask = group_stds <= std_eps
    n_no_signal = int(no_signal_mask.sum())

    # 无信号组再细分: 全 bad_format / 全 wrong / 全同正
    n_all_bad = n_all_wrong = n_all_pos = 0
    for v, ns in zip(groups.values(), no_signal_mask):
        if not ns:
            continue
        b = bucket(v[0])
        if b == "bad_format":
            n_all_bad += 1
        elif b == "wrong":
            n_all_wrong += 1
        else:
            n_all_pos += 1

    return {
        "step": rows[0].get("step", "?"),
        "n_groups": n_groups,
        "n_samples": n_samples,
        "avg_group_size": float(group_sizes.mean()),
        "min_group_size": int(group_sizes.min()),
        "max_group_size": int(group_sizes.max()),
        "mean_reward": float(all_scores.mean()),
        "pct_bad_format": 100.0 * comp["bad_format"] / n_samples,
        "pct_wrong": 100.0 * comp["wrong"] / n_samples,
        "pct_correct": 100.0 * comp["correct"] / n_samples,
        "n_no_signal": n_no_signal,
        "pct_no_signal": 100.0 * n_no_signal / n_groups,
        "n_all_bad": n_all_bad,
        "n_all_wrong": n_all_wrong,
        "n_all_pos": n_all_pos,
        "std_p50": float(np.percentile(group_stds, 50)),
        "std_p75": float(np.percentile(group_stds, 75)),
        "std_p90": float(np.percentile(group_stds, 90)),
        "std_max": float(group_stds.max()),
    }


def print_step(a):
    print(f"\n===== Step {a['step']} =====")
    print(f"  组数={a['n_groups']}  样本={a['n_samples']}  "
          f"组大小 avg={a['avg_group_size']:.1f} (min={a['min_group_size']}, max={a['max_group_size']})")
    print(f"  平均 reward: {a['mean_reward']:+.3f}")
    print(f"  score 三档占比:  格式错(-1)={a['pct_bad_format']:5.1f}%   "
          f"答案错(0)={a['pct_wrong']:5.1f}%   答案对(>0)={a['pct_correct']:5.1f}%")
    print(f"  ⚠ 无梯度信号组 (std≈0): {a['n_no_signal']}/{a['n_groups']} = {a['pct_no_signal']:.1f}%")
    print(f"      ├─ 全格式错(-1): {a['n_all_bad']}")
    print(f"      ├─ 全答案错(0) : {a['n_all_wrong']}")
    print(f"      └─ 全同正分     : {a['n_all_pos']}")
    print(f"  组内 std 分位: p50={a['std_p50']:.3f}  p75={a['std_p75']:.3f}  "
          f"p90={a['std_p90']:.3f}  max={a['std_max']:.3f}")


def print_verdict(analyses):
    """基于聚合结果给出一句话结论。"""
    if not analyses:
        return
    m_nosig = np.mean([a["pct_no_signal"] for a in analyses])
    m_bad = np.mean([a["pct_bad_format"] for a in analyses])
    m_correct = np.mean([a["pct_correct"] for a in analyses])
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)
    print(f"  平均无信号组占比: {m_nosig:.1f}%   "
          f"平均格式错占比: {m_bad:.1f}%   平均答案对占比: {m_correct:.1f}%")
    if m_nosig >= 60:
        print("  ▶ 无信号组占比高 → GRPO 大部分组 advantage=0，几乎拿不到梯度，")
        print("    这正是 reward 不涨、ppo_kl 塌陷的直接原因。")
    if m_bad >= 50:
        print("  ▶ 格式错(-1)占主导 → 病因是'格式不过关'而非'题太难'。")
        print("    优先查: think/answer 标签配对、\\boxed{} 缺失、search/result 顺序、")
        print("    以及 response 被 max_response_length 截断导致标签不闭合。")
    elif m_correct < 5:
        print("  ▶ 格式基本过关但正确率极低 → 病因偏'任务太难/SFT 起点弱'，")
        print("    考虑更简单数据子集或课程式训练。")
    else:
        print("  ▶ 已有一定正确率，组内存在差异 → 有学习信号，可继续观察。")


def main():
    args = parse_args()
    files = list_step_files(args.rollout_dir, args.step, args.last_n, args.all)
    if not files:
        print(f"未在 {args.rollout_dir} 找到 {{step}}.jsonl 文件")
        return

    analyses = []
    for f in files:
        rows = load_jsonl(f)
        if not rows:
            continue
        a = analyze_step(rows, args.std_eps)
        if a is not None:
            analyses.append(a)
            print_step(a)

    print_verdict(analyses)

    if args.csv and analyses:
        import csv
        keys = list(analyses[0].keys())
        with open(args.csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=keys)
            w.writeheader()
            w.writerows(analyses)
        print(f"\n已写出 CSV: {args.csv}")


if __name__ == "__main__":
    main()
