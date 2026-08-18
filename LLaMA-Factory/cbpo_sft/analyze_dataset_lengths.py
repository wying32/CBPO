#!/usr/bin/env python3
"""Analyze token lengths for ShareGPT-style SFT data.

This script is intended to run in the same environment used for training, so it
uses the target model tokenizer and chat template when available.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer


DEFAULT_CUTOFFS = [4096, 8192, 10000, 12000, 15000, 16000, 18000, 20000]


def percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    rank = math.ceil((pct / 100.0) * len(sorted_values)) - 1
    rank = min(max(rank, 0), len(sorted_values) - 1)
    return sorted_values[rank]


def convert_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = record.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})

    for message in record.get("conversations", []):
        role = message.get("from")
        content = message.get("value", "")
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        elif role == "observation":
            role = "tool"
        elif role == "function_call":
            role = "assistant"
        else:
            role = "user"
        messages.append({"role": role, "content": content})

    return messages


def encode_length(tokenizer: Any, record: dict[str, Any]) -> int:
    messages = convert_messages(record)
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return len(ids)
    except Exception:
        text = ""
        if record.get("system"):
            text += str(record["system"]) + "\n"
        for message in record.get("conversations", []):
            text += f"{message.get('from', '')}: {message.get('value', '')}\n"
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="JSONL dataset path.")
    parser.add_argument("--model", required=True, help="Tokenizer/model path.")
    parser.add_argument("--top-k", type=int, default=20, help="Print the top-k longest examples.")
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="*",
        default=DEFAULT_CUTOFFS,
        help="Cutoff lengths to report exceedance counts for.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lengths: list[int] = []
    turn_counts: list[int] = []
    role_counts: Counter[str] = Counter()
    longest: list[tuple[int, int, int]] = []

    with args.data.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            length = encode_length(tokenizer, record)
            conversations = record.get("conversations", [])
            lengths.append(length)
            turn_counts.append(len(conversations))
            role_counts.update(message.get("from", "") for message in conversations)
            longest.append((length, line_no, len(conversations)))

    sorted_lengths = sorted(lengths)
    longest.sort(reverse=True)
    total = len(lengths)

    print(f"records: {total}")
    print(f"mean_tokens: {mean(lengths):.2f}")
    print(f"min_tokens: {sorted_lengths[0] if sorted_lengths else 0}")
    for pct in [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9]:
        print(f"p{pct}_tokens: {percentile(sorted_lengths, pct)}")
    print(f"max_tokens: {sorted_lengths[-1] if sorted_lengths else 0}")
    print(f"mean_turns: {mean(turn_counts):.2f}")
    print(f"max_turns: {max(turn_counts) if turn_counts else 0}")
    print(f"role_counts: {dict(role_counts)}")

    print("\ncutoff_exceedance:")
    for cutoff in sorted(args.cutoffs):
        count = sum(1 for length in lengths if length > cutoff)
        pct = (count / total * 100.0) if total else 0.0
        print(f"  >{cutoff}: {count} ({pct:.2f}%)")

    print(f"\ntop_{args.top_k}_longest:")
    for length, line_no, turns in longest[: args.top_k]:
        print(f"  line={line_no} tokens={length} turns={turns}")


if __name__ == "__main__":
    main()
