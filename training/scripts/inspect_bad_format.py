#!/usr/bin/env python3
"""
下钻分析: 对 rollout dump 里所有 score==-1 (格式错) 的样本，
复用 reward 中真正的 validate_format() 统计精确失败原因，并检测截断。

用法（训练进行中安全运行，纯只读）:
    python inspect_bad_format.py \
        --rollout-dir /path/to/.../rollout \
        --last-n 3 --show 3

依赖: 需能 import 到 verl.utils.reward_score.deep_research。
脚本会自动把 verl_cbpo 加入 sys.path（相对本文件定位）。
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

# 定位 verl_cbpo 并加入 path，以便 import 真正的 validate_format
_THIS = os.path.dirname(os.path.abspath(__file__))
_VERL_ROOT = os.path.abspath(os.path.join(_THIS, "..", "verl_cbpo"))
if _VERL_ROOT not in sys.path:
    sys.path.insert(0, _VERL_ROOT)

try:
    from verl.utils.reward_score.deep_research import validate_format
    _HAVE_VALIDATE = True
except Exception as e:  # noqa
    print(f"[warn] 无法 import validate_format ({e})，将退化为标签启发式判断")
    _HAVE_VALIDATE = False


def local_reason(text):
    """import 失败时的兜底: 粗判失败原因（与 validate_format 的规则保持一致大类）。"""
    if text.count("<think>") != text.count("</think>"):
        return "<think> </think> not paired"
    if text.count("<think>") == 0 or text.count("</think>") == 0:
        return "<think> or </think> not found"
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return "<answer> or </answer> not found"
    if "\\boxed{" not in text:
        return "answer is missing \\boxed{} format"
    return "other"


def get_reason(text):
    if _HAVE_VALIDATE:
        ok, reason = validate_format(text)
        return reason if not ok else "format is correct"
    return local_reason(text)


def parse_args():
    p = argparse.ArgumentParser(description="下钻 rollout dump 里格式错样本的具体原因")
    p.add_argument("--rollout-dir", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--step", type=int, default=None)
    g.add_argument("--last-n", type=int, default=None)
    g.add_argument("--all", action="store_true")
    p.add_argument("--show", type=int, default=2, help="每个原因打印几条样例输出")
    p.add_argument("--tail-chars", type=int, default=300, help="样例只打印输出末尾多少字符")
    return p.parse_args()


def list_step_files(rollout_dir, step, last_n, do_all):
    def step_of(f):
        try:
            return int(os.path.splitext(os.path.basename(f))[0])
        except ValueError:
            return -1
    files = sorted(
        [f for f in glob.glob(os.path.join(rollout_dir, "*.jsonl")) if step_of(f) >= 0],
        key=step_of,
    )
    if not files:
        return []
    if step is not None:
        return [f for f in files if step_of(f) == step]
    if last_n is not None:
        return files[-last_n:]
    if do_all:
        return files
    return files[-1:]


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    args = parse_args()
    files = list_step_files(args.rollout_dir, args.step, args.last_n, args.all)
    if not files:
        print(f"未在 {args.rollout_dir} 找到 {{step}}.jsonl")
        return

    reason_counter = Counter()
    examples = {}          # reason -> [output样例]
    n_bad = 0
    n_total = 0
    n_no_close_answer = 0  # 疑似截断: 没有 </answer>

    for path in files:
        for r in load_jsonl(path):
            if "score" not in r or "output" not in r:
                continue
            n_total += 1
            if float(r["score"]) > -0.5:   # 只看 -1 档
                continue
            n_bad += 1
            out = r["output"]
            reason = get_reason(out)
            reason_counter[reason] += 1
            if "</answer>" not in out:
                n_no_close_answer += 1
            if len(examples.setdefault(reason, [])) < args.show:
                examples[reason].append(out)

    print("=" * 60)
    print(f"总样本={n_total}  格式错(-1)={n_bad} ({100.0*n_bad/max(n_total,1):.1f}%)")
    print(f"疑似截断(输出无 </answer>): {n_no_close_answer} "
          f"({100.0*n_no_close_answer/max(n_bad,1):.1f}% of 格式错)")
    print("=" * 60)
    print("格式错原因分布:")
    for reason, cnt in reason_counter.most_common():
        print(f"  {cnt:5d}  ({100.0*cnt/max(n_bad,1):5.1f}%)  {reason}")

    print("\n" + "=" * 60)
    print("各原因样例（输出末尾片段）:")
    for reason, _ in reason_counter.most_common():
        print(f"\n----- 原因: {reason} -----")
        for i, out in enumerate(examples.get(reason, [])):
            tail = out[-args.tail_chars:]
            print(f"  [例{i+1}] ...{tail!r}")


if __name__ == "__main__":
    main()
