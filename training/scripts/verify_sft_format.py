#!/usr/bin/env python3
"""
验证 SFT 模型（RL 前）在"内联工具结果"场景下的收尾行为。

目的：复刻 RL rollout 的单轮内联流程，确认模型是否在拿到工具结果后
直接吐 EOS、不继续写 <answer>，从而判断 SFT 格式（多轮 observation 拆分）
与 RL rollout（单轮内联）是否不匹配。

用法（在有 GPU 的环境，如 cbpo_rl）:
    # 推荐：用 RL parquet 里的真实 prompt（最忠实）
    python verify_sft_format.py \
        --model /path/to/workspace/models/Qwen2.5-7B-CBPO-SFT \
        --parquet ../rl_datasets/train_10k_math_only_fixed.parquet \
        --num-prompts 10 --random-prompts --n-samples 8 \
        --temperature 1.0 --real-python --sft-style-result --score-reward

    # 兜底：自带 system prompt + 自定义问题
    python verify_sft_format.py --model <path> \
        --question "Find the remainder when 7^100 is divided by 13."
"""
import argparse
import os
import random
import sys

# 强制走 vLLM V0 引擎，绕开 V1 采样器里无条件 import 的 flashinfer
# （cbpo_rl 环境的 flashinfer 原生库与 torch2.6 ABI 不兼容会报 undefined symbol）。
# 同时对齐 RL 训练的注意力后端。使用 setdefault 便于需要时手动覆盖。
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

# 与 reward_score/deep_research.py __main__ 中一致的 CBPO 系统提示
SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve the given question step by step "
    "with the help of the wikipedia search tool and python interpreter tool. "
    "Given a question, you need to first think about the reasoning process in the "
    "mind and then provide the answer. During thinking, you can invoke the wikipedia "
    "search tool to search and python interpreter tool to calculate the math problem "
    "for fact information about specific topics if needed. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> tags "
    "respectively, and the search query and result are enclosed within <search> "
    "</search> and <result> </result> tags respectively. For example, <think> This "
    "is the reasoning process. </think> <search> search query here </search> <result> "
    "search result here </result> <think> This is the reasoning process. </think> "
    "<python> python code here </python> <result> python interpreter result here "
    "</result> <think> This is the reasoning process. </think> <answer> The final "
    "answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, "
    "the final exact answer is enclosed within \\boxed{} with latex format."
)

# 工具结果正文（不含外层包裹），供两种投喂格式复用
CLEAN_BODY = "42"
ERROR_BODY = 'File "<string>", line 1\n    x = = 5\n        ^\nSyntaxError: invalid syntax'
STOP = ["</python>", "</search>"]


def wrap_result(sft_style: bool, body: str) -> str:
    """把工具结果正文包成注入串。

    inline（默认，复刻当前 RL rollout）：同一 assistant 轮内联裸 <result>。
    sft_style（复刻 SFT 训练格式）：关闭回合 → 新 user 轮 + <tool_response> 包裹
        → 重开 assistant，与 LLaMA-Factory qwen/qwen3 模板 format_observation 一致。
    """
    if sft_style:
        return (
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
            f"<result>\n{body}\n</result>\n"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )
    return f" <result>\n{body}\n</result>"


def build_result_injection(sft_style: bool, inject_error: bool) -> str:
    """假结果（无真实执行时）的注入串。"""
    return wrap_result(sft_style, ERROR_BODY if inject_error else CLEAN_BODY)


def extract_last_tool_content(text: str, tag: str) -> str:
    """提取文本中最后一个 <tag>...</tag> 之间的内容（用于真实执行工具调用）。"""
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    end = text.rfind(close_t)
    if end == -1:
        return ""
    start = text.rfind(open_t, 0, end)
    if start == -1:
        return ""
    return text[start + len(open_t):end].strip()


def parse_args():
    p = argparse.ArgumentParser(description="验证 SFT 模型内联工具结果后的收尾行为")
    p.add_argument("--model", required=True, help="SFT 模型路径（RL 前）")
    p.add_argument("--parquet", default=None, help="RL parquet，用其真实 prompt")
    p.add_argument("--row", type=int, default=0, help="parquet 行号")
    p.add_argument("--num-prompts", type=int, default=1, help="从 parquet 评测多少个 prompt")
    p.add_argument("--random-prompts", action="store_true", help="从 --row 之后随机抽取 prompt")
    p.add_argument("--seed", type=int, default=42, help="随机抽题种子")
    p.add_argument("--prompt-key", default="prompt", help="parquet 中 prompt 字段名")
    p.add_argument("--question", default=None, help="无 parquet 时的自定义问题")
    p.add_argument("--ground-truth", default=None, help="无 parquet 时用于严格 reward 的标准答案")
    p.add_argument("--temperature", type=float, default=0.0, help="0=贪心，行为最确定")
    p.add_argument("--chunk-tokens", type=int, default=1024, help="每轮最多生成 token")
    p.add_argument("--max-rounds", type=int, default=6, help="最多模拟几轮")
    p.add_argument("--max-response-tokens", type=int, default=4096,
                   help="单条轨迹总 response token 预算（含工具结果），应与训练一致")
    p.add_argument("--tool-result-length", type=int, default=1500,
                   help="工具结果最大字符数，应与训练的 python result_length 一致")
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--gpu-mem", type=float, default=0.6, help="gpu_memory_utilization")
    p.add_argument("--n-samples", type=int, default=1, help="跑几条独立轨迹统计成功率（temp>0时才有意义）")
    p.add_argument("--inject-error", action="store_true", help="注入python报错结果而非干净结果，测鲁棒性")
    p.add_argument("--sft-style-result", action="store_true",
                   help="按SFT训练格式投喂结果(关回合+<tool_response>+重开assistant)，对照内联投喂")
    p.add_argument("--real-python", action="store_true",
                   help="真实执行模型生成的python代码并注入真值(复刻rollout的PythonTool)，取代假结果'42'")
    p.add_argument("--python-bin", default=None,
                   help="真实执行时用的python解释器，默认当前解释器(sys.executable)")
    p.add_argument("--real-search", action="store_true",
                   help="真实执行search工具(复刻rollout的BingSearchTool，cache优先+异常兜底)")
    p.add_argument("--search-api-key", default="", help="BrightData api_key(不传则仅靠cache/兜底)")
    p.add_argument("--search-zone", default="", help="BrightData zone")
    p.add_argument("--search-cache", default=None,
                   help="search缓存路径，默认项目 search_cache/search_cache.json")
    p.add_argument("--score-reward", action="store_true",
                   help="使用 deep_research_math 对每条轨迹严格评分并汇总组内信号")
    p.add_argument("--show-prompts", type=int, default=1,
                   help="打印前几个 prompt 的第一条完整轨迹；0 表示不打印")
    return p.parse_args()


def _as_mapping(value):
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def build_prompt(tokenizer, args, row_data=None, row_index=None):
    """返回套好 chat 模板的 prompt、标准答案和 data source。"""
    if row_data is not None:
        raw = row_data[args.prompt_key]
        # VERL prompt 通常是 [{"role","content"}, ...]
        try:
            messages = [dict(m) for m in list(raw)]
        except Exception:
            # 已是纯字符串
            if isinstance(raw, str):
                prompt = raw
                messages = None
            else:
                raise
        if messages is not None:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        reward_model = _as_mapping(row_data.get("reward_model"))
        ground_truth = reward_model.get("ground_truth")
        data_source = row_data.get("data_source", "math")
        print(f"[info] 使用 parquet 第 {row_index} 行的真实 prompt"
              f"，共 {len(messages) if messages is not None else 1} 条 message")
        return prompt, ground_truth, data_source
    else:
        q = args.question or "Find the remainder when 7^100 is divided by 13."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        print(f"[info] 使用自带 system prompt + 问题: {q}")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt, args.ground_truth, "math"


def select_prompt_rows(args):
    if not args.parquet:
        if args.num_prompts != 1:
            raise ValueError("不使用 --parquet 时，--num-prompts 必须为 1")
        return [(None, None)]

    import pandas as pd

    df = pd.read_parquet(args.parquet)
    if args.row < 0 or args.row >= len(df):
        raise IndexError(f"--row={args.row} 超出 parquet 范围 [0, {len(df)})")
    available = list(range(args.row, len(df)))
    if args.num_prompts < 1 or args.num_prompts > len(available):
        raise ValueError(f"--num-prompts 必须在 [1, {len(available)}] 内")
    if args.random_prompts:
        indices = random.Random(args.seed).sample(available, args.num_prompts)
    else:
        indices = available[: args.num_prompts]
    return [(index, df.iloc[index]) for index in indices]


def reward_bucket(score):
    if score < -0.5:
        return "bad_format"
    if score <= 1e-9:
        return "wrong"
    return "correct"


def summarize_prompt(row_index, ground_truth, results):
    scores = [float(result["reward"]["score"]) for result in results]
    buckets = [reward_bucket(score) for score in scores]
    counts = {name: buckets.count(name) for name in ("bad_format", "wrong", "correct")}
    reward_active = max(scores) - min(scores) > 1e-9
    useful_mixed = counts["correct"] > 0 and counts["correct"] < len(scores)
    summary = {
        "row": row_index,
        "ground_truth": ground_truth,
        "scores": scores,
        **counts,
        "reward_active": reward_active,
        "useful_mixed": useful_mixed,
        "pass_at_n": counts["correct"] > 0,
        "no_signal": not reward_active,
        "all_bad_format": counts["bad_format"] == len(scores),
        "all_wrong": counts["wrong"] == len(scores),
        "all_correct": counts["correct"] == len(scores),
    }
    label = "custom" if row_index is None else str(row_index)
    print(
        f"[prompt row={label}] gt={ground_truth!r} scores={scores} | "
        f"格式错={counts['bad_format']} 答错={counts['wrong']} 答对={counts['correct']} | "
        f"reward有差异={reward_active} 有效正确性混合={useful_mixed}"
    )
    return summary


def print_reward_summary(prompt_summaries, n_samples):
    n_prompts = len(prompt_summaries)
    n_trajectories = n_prompts * n_samples
    bad = sum(item["bad_format"] for item in prompt_summaries)
    wrong = sum(item["wrong"] for item in prompt_summaries)
    correct = sum(item["correct"] for item in prompt_summaries)
    active = sum(item["reward_active"] for item in prompt_summaries)
    useful = sum(item["useful_mixed"] for item in prompt_summaries)
    pass_n = sum(item["pass_at_n"] for item in prompt_summaries)
    no_signal = sum(item["no_signal"] for item in prompt_summaries)
    all_bad = sum(item["all_bad_format"] for item in prompt_summaries)
    all_wrong = sum(item["all_wrong"] for item in prompt_summaries)
    all_correct = sum(item["all_correct"] for item in prompt_summaries)

    def pct(value, total):
        return 100.0 * value / max(1, total)

    print("\n" + "=" * 70)
    print("严格数学 Reward 汇总")
    print("=" * 70)
    print(f"  prompts={n_prompts}  trajectories={n_trajectories}  n={n_samples}")
    print(f"  格式错误(-1): {bad}/{n_trajectories} = {pct(bad, n_trajectories):.1f}%")
    print(f"  答案错误(0):  {wrong}/{n_trajectories} = {pct(wrong, n_trajectories):.1f}%")
    print(f"  答案正确(1):  {correct}/{n_trajectories} = {pct(correct, n_trajectories):.1f}%")
    print(f"  pass@{n_samples}:      {pass_n}/{n_prompts} = {pct(pass_n, n_prompts):.1f}%")
    print(f"  reward有差异组: {active}/{n_prompts} = {pct(active, n_prompts):.1f}%")
    print(f"  有效正确性混合组: {useful}/{n_prompts} = {pct(useful, n_prompts):.1f}%")
    print(f"  无信号组: {no_signal}/{n_prompts} = {pct(no_signal, n_prompts):.1f}%")
    print(f"    全格式错={all_bad}  全答错={all_wrong}  全答对={all_correct}")
    print("-" * 70)
    if pct(bad, n_trajectories) > 10.0:
        print("  ❌ 格式错误率高于 10%，先修重复/缺失标签，不建议开始 RL。")
    elif pct(correct, n_trajectories) < 5.0 or pct(useful, n_prompts) < 30.0:
        print("  ❌ 正确轨迹或有效混合组过少，先做课程筛选/SFT warm-up，不建议长训练。")
    else:
        print("  ✅ 已达到短程 RL 预检门槛，可以先跑 20 step，继续监控无信号组。")


def truncate_tool_result(text, max_chars):
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n...[truncated {omitted} chars]...\n{text[-tail:]}"


def run_one_trajectory(llm, tokenizer, prompt, args, python_tool=None, search_tool=None):
    """复刻 rollout 单轮内联流程跑一条轨迹，返回统计 dict。"""
    from vllm import SamplingParams

    cur = prompt
    tool_calls = 0
    ended_by = None
    rounds_info = []
    for r in range(1, args.max_rounds + 1):
        full_response = cur[len(prompt):]
        response_tokens = len(tokenizer.encode(full_response, add_special_tokens=False))
        remaining_tokens = args.max_response_tokens - response_tokens
        if remaining_tokens <= 0:
            ended_by = "response_limit"
            break

        sp = SamplingParams(
            temperature=args.temperature,
            max_tokens=min(args.chunk_tokens, remaining_tokens),
            stop=STOP,
            include_stop_str_in_output=True,
        )
        try:
            out = llm.generate([cur], sp, use_tqdm=False)[0].outputs[0]
        except ValueError as error:
            if "longer than the maximum model length" not in str(error):
                raise
            ended_by = "context_limit"
            rounds_info.append((r, "context_limit", None, 0, str(error)[-240:]))
            break
        cur += out.text
        is_tool = out.finish_reason == "stop" and out.stop_reason in STOP
        is_eos = out.finish_reason == "stop" and out.stop_reason not in STOP
        is_len = out.finish_reason == "length"
        rounds_info.append((r, out.finish_reason, out.stop_reason, len(out.token_ids), out.text[-240:]))
        if is_tool:
            tool_calls += 1
            tag = out.stop_reason.strip("</>")  # python / search
            if python_tool is not None and tag == "python":
                code = extract_last_tool_content(cur, "python")
                body = python_tool.execute(code) or "(empty output)"
            elif search_tool is not None and tag == "search":
                query = extract_last_tool_content(cur, "search")
                try:
                    body = search_tool.execute(query) or "No relevant results."
                except Exception as e:  # noqa: BLE001
                    body = f"No relevant results. ({e})"
            elif tag == "search":
                body = "No relevant results."  # 未开启 --real-search 时的占位
            else:
                body = ERROR_BODY if args.inject_error else CLEAN_BODY
            body = truncate_tool_result(body, args.tool_result_length)
            cur += wrap_result(args.sft_style_result, body)
            continue
        if is_eos:
            ended_by = "eos"
            break
        if is_len:
            # 未命中 stop/EOS，说明仍在生成中途：继续下一轮分块续写，
            # 对齐 RL rollout 的"finish_reason==length 且未超预算则继续"，
            # 避免在单轮 1024 token 处误截断、低估成功率。
            continue
    else:
        ended_by = "max_rounds"
    full_response = cur[len(prompt):]
    return {
        "full_response": full_response,
        "tool_calls": tool_calls,
        "ended_by": ended_by,
        "has_answer": "</answer>" in full_response,
        "ends_after_result": full_response.rstrip().endswith("</result>"),
        "rounds_info": rounds_info,
    }


def main():
    args = parse_args()
    if args.n_samples < 1:
        raise ValueError("--n-samples 必须大于 0")
    if args.max_response_tokens < 1:
        raise ValueError("--max-response-tokens 必须大于 0")
    if args.show_prompts < 0:
        raise ValueError("--show-prompts 不能为负数")

    verl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "verl_cbpo"))
    if verl_root in sys.path:
        sys.path.remove(verl_root)
    sys.path.insert(0, verl_root)

    from vllm import LLM
    from transformers import AutoTokenizer

    prompt_rows = select_prompt_rows(args)
    reward_fn = None
    if args.score_reward:
        from verl.utils.reward_score.deep_research_math import compute_score

        reward_fn = compute_score

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
    )

    python_tool = None
    search_tool = None
    if args.real_python:
        from verl.workers.rollout.tools.python_tool import PythonTool
        python_bin = args.python_bin or sys.executable
        python_tool = PythonTool(python_path=python_bin, result_length=args.tool_result_length)
        print(f"[info] 真实执行 python 工具，解释器: {python_bin}")
    if args.real_search:
        from verl.workers.rollout.tools.search_tool import BingSearchTool
        cache = args.search_cache or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "search_cache", "search_cache.json"))
        search_tool = BingSearchTool(
            api_key=args.search_api_key, zone=args.search_zone,
            cache_file=cache, wikipedia_fallback=True,
        )
        print(f"[info] 真实执行 search 工具，cache: {cache}")

    n = args.n_samples
    all_results = []
    prompt_summaries = []
    for prompt_number, (row_index, row_data) in enumerate(prompt_rows):
        prompt, ground_truth, data_source = build_prompt(
            tokenizer,
            args,
            row_data=row_data,
            row_index=row_index,
        )
        if reward_fn is not None and ground_truth is None:
            raise ValueError("--score-reward 需要 parquet reward_model.ground_truth 或 --ground-truth")

        if prompt_number < args.show_prompts:
            print("\n" + "=" * 70)
            print(f"PROMPT 末尾（row={row_index}）:")
            print("..." + prompt[-400:])
            print("=" * 70)

        results = [
            run_one_trajectory(llm, tokenizer, prompt, args, python_tool, search_tool)
            for _ in range(n)
        ]
        if reward_fn is not None:
            for result in results:
                result["reward"] = reward_fn(data_source, result["full_response"], ground_truth)
            prompt_summaries.append(summarize_prompt(row_index, ground_truth, results))
        all_results.extend(results)

        if prompt_number < args.show_prompts:
            first = results[0]
            print(f"\n[样例轨迹 row={row_index} sample=1] 逐轮:")
            for (r, fr, sr, ntok, tail) in first["rounds_info"]:
                print(f"  Round {r}: finish={fr} stop_reason={sr!r} ntok={ntok}")
                print(f"    末尾: ...{tail!r}")
            print("\n完整生成:")
            print(first["full_response"])
            if reward_fn is not None:
                print(f"\n严格 reward: {first['reward']}")

    total = len(all_results)
    n_answer = sum(1 for result in all_results if result["has_answer"])
    n_eos_no_answer = sum(
        1 for result in all_results if result["ended_by"] == "eos" and not result["has_answer"]
    )
    n_ends_result = sum(1 for result in all_results if result["ends_after_result"])
    n_response_limit = sum(
        1 for result in all_results if result["ended_by"] == "response_limit"
    )
    n_context_limit = sum(
        1 for result in all_results if result["ended_by"] == "context_limit"
    )

    print("\n" + "=" * 70)
    print(f"格式/收尾汇总（prompts={len(prompt_rows)}, n_samples={n}, temperature={args.temperature}, "
          f"sft_style_result={args.sft_style_result}, real_python={args.real_python}）")
    print("=" * 70)
    print(f"  写出 </answer> 的比例:  {n_answer}/{total} = {100.0 * n_answer / total:.1f}%")
    print(f"  EOS 但无 </answer>:     {n_eos_no_answer}/{total} = {100.0 * n_eos_no_answer / total:.1f}%")
    print(f"  停在 </result> 之后:    {n_ends_result}/{total}")
    print(f"  达到 response token 上限: {n_response_limit}/{total}")
    print(f"  达到模型 context 上限:    {n_context_limit}/{total}")
    print("-" * 70)
    if total == 1:
        print("  ⚠ 单条采样不反映概率行为，建议 --temperature 1.0 --n-samples 16 再测")
    elif args.sft_style_result:
        if n_eos_no_answer <= max(1, total // 10):
            print(f"  ✅ 工具结果后提前 EOS 基本消失（{n_eos_no_answer}/{total}）。")
        else:
            print(f"  ⚠ 提前 EOS 仍较多（{n_eos_no_answer}/{total}），需继续检查投喂格式。")
    elif n_answer / total < 0.5:
        print("  ❌ 内联投喂+采样温度下大量写不出 </answer> → 复现RL失败模式")
        print("     下一步用 --sft-style-result 对照，确认是否格式不匹配所致")
    else:
        print("  ✅ 采样温度下多数能写出 </answer>，失败率不高，问题另有原因")

    if prompt_summaries:
        print_reward_summary(prompt_summaries, n)


if __name__ == "__main__":
    main()
