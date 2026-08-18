#!/usr/bin/env python3
"""Sanitize CBPO prompts and build math-only RL parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CBPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_INPUT = CBPO_ROOT / "rl_datasets" / "train_10k_math_only.parquet"
DEFAULT_TRAIN_OUTPUT = CBPO_ROOT / "rl_datasets" / "train_10k_math_only_fixed.parquet"
DEFAULT_VALID_INPUT = CBPO_ROOT / "rl_datasets" / "valid.parquet"
DEFAULT_VALID_OUTPUT = CBPO_ROOT / "rl_datasets" / "valid_math.parquet"

BROKEN_BOXED = "\x08oxed"
FIXED_BOXED = "\\boxed"

# Must stay byte-identical to the SFT system prompt (NEW_SYSTEM_PROMPT in
# transform_dataset_python_only.py / filter_python_only_samples.py) so the
# RL rollout distribution matches what the SFT checkpoint was trained on.
MATH_SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve the given question step by step "
    "with the help of the python interpreter tool. Given a question, you need to "
    "first think about the reasoning process in the mind and then provide the answer. "
    "During thinking, you can invoke the python interpreter tool to calculate or "
    "execute code for fact information if needed. The reasoning process and answer "
    "are enclosed within <think> </think> and <answer> </answer> tags respectively, "
    "and the python code and result are enclosed within <python> </python> and "
    "<result> </result> tags respectively. For example, <think> This is the "
    "reasoning process. </think> <python> python code here </python> <result> "
    "python interpreter result here </result> <think> This is the reasoning "
    "process. </think> <answer> The final answer is \\boxed{answer here} </answer>. "
    "In the last part of the answer, the final exact answer is enclosed within "
    "\\boxed{} with latex format."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--valid-input", type=Path, default=DEFAULT_VALID_INPUT)
    parser.add_argument("--valid-output", type=Path, default=DEFAULT_VALID_OUTPUT)
    return parser.parse_args()


def sanitize(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        count = value.count(BROKEN_BOXED)
        return value.replace(BROKEN_BOXED, FIXED_BOXED), count
    if isinstance(value, list):
        output = []
        replacements = 0
        for item in value:
            cleaned, count = sanitize(item)
            output.append(cleaned)
            replacements += count
        return output, replacements
    if isinstance(value, dict):
        output = {}
        replacements = 0
        for key, item in value.items():
            cleaned, count = sanitize(item)
            output[key] = cleaned
            replacements += count
        return output, replacements
    return value, 0


def use_python_only_math_prompt(row: dict[str, Any]) -> dict[str, Any]:
    """Replace the math system prompt so rollout exposes only the Python tool."""
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        return row

    replaced = False
    updated_prompt = []
    for message in prompt:
        if isinstance(message, dict) and message.get("role") == "system":
            message = {**message, "content": MATH_SYSTEM_PROMPT}
            replaced = True
        updated_prompt.append(message)

    if not replaced:
        updated_prompt.insert(0, {"role": "system", "content": MATH_SYSTEM_PROMPT})
    return {**row, "prompt": updated_prompt}


def prepare(input_path: Path, output_path: Path, math_only: bool) -> tuple[int, int]:
    table = pq.read_table(input_path)
    rows = []
    replacements = 0

    for row in table.to_pylist():
        if math_only and str(row.get("ability", "")).lower() != "math":
            continue
        cleaned, count = sanitize(row)
        if math_only:
            cleaned = use_python_only_math_prompt(cleaned)
        rows.append(cleaned)
        replacements += count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_table = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(cleaned_table, output_path, compression="snappy")

    remaining = sum(str(row).count(BROKEN_BOXED) for row in rows)
    if remaining:
        raise RuntimeError(f"{output_path} still contains {remaining} broken boxed markers")
    return len(rows), replacements


def main() -> None:
    args = parse_args()
    train_rows, train_replacements = prepare(
        args.train_input.expanduser().resolve(),
        args.train_output.expanduser().resolve(),
        math_only=True,
    )
    valid_rows, valid_replacements = prepare(
        args.valid_input.expanduser().resolve(),
        args.valid_output.expanduser().resolve(),
        math_only=True,
    )
    print(
        f"train: rows={train_rows}, boxed_fixes={train_replacements}, output={args.train_output}\n"
        f"valid: rows={valid_rows}, boxed_fixes={valid_replacements}, output={args.valid_output}"
    )


if __name__ == "__main__":
    main()
