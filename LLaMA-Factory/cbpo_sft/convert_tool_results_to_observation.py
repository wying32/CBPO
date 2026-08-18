#!/usr/bin/env python3
"""Convert inline <result> blocks in assistant messages to observation turns.

Input can be a ShareGPT JSON array or JSONL; output is ShareGPT JSONL:
  {"conversations": [{"from": "human", ...}, {"from": "gpt", ...}], "system": "..."}

The converter turns:
  gpt: "<think>...</think><search>...</search><result>...</result><think>...</think>"

into:
  gpt: "<think>...</think><search>...</search>"
  observation: "<result>...</result>"
  gpt: "<think>...</think>"
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


RESULT_TAG_RE = re.compile(r"</?result>")
ODD_ROLES = {"human", "observation"}
EVEN_ROLES = {"gpt", "function_call"}


class MalformedResultTags(ValueError):
    """Raised when result tags are unbalanced or nested."""


def valid_sharegpt_turn_order(conversations: list[dict[str, Any]]) -> bool:
    for idx, message in enumerate(conversations):
        role = message.get("from")
        if idx % 2 == 0 and role not in ODD_ROLES:
            return False
        if idx % 2 == 1 and role not in EVEN_ROLES:
            return False
    return len(conversations) % 2 == 0


def result_spans(value: str) -> list[tuple[int, int]]:
    """Return flat, balanced ``<result>...</result>`` spans.

    Nested result tags are treated as corrupt data rather than silently merging
    multiple tool calls into one observation turn.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None

    for match in RESULT_TAG_RE.finditer(value):
        if match.group(0) == "<result>":
            if start is not None:
                raise MalformedResultTags("nested <result> tag")
            start = match.start()
        else:
            if start is None:
                raise MalformedResultTags("orphan </result> tag")
            spans.append((start, match.end()))
            start = None

    if start is not None:
        raise MalformedResultTags("unclosed <result> tag")
    return spans


def split_gpt_message(message: dict[str, Any]) -> list[dict[str, str]]:
    value = message.get("value", "")
    pieces: list[dict[str, str]] = []
    cursor = 0

    for start, end in result_spans(value):
        before = value[cursor:start].strip()
        result = value[start:end].strip()
        if before:
            pieces.append({"from": "gpt", "value": before})

        pieces.append({"from": "observation", "value": result})
        cursor = end

    after = value[cursor:].strip()
    if after:
        pieces.append({"from": "gpt", "value": after})

    return pieces or [message]


def convert_record(record: dict[str, Any], stats: Counter[str]) -> dict[str, Any] | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        stats["missing_conversations"] += 1
        return None

    converted: list[dict[str, Any]] = []
    changed = False
    result_blocks = 0

    for message in conversations:
        role = message.get("from")
        value = message.get("value", "")
        if role == "gpt" and isinstance(value, str) and ("<result>" in value or "</result>" in value):
            try:
                split_messages = split_gpt_message(message)
            except MalformedResultTags:
                stats["dropped_malformed_result_tags"] += 1
                return None
            converted.extend(split_messages)
            result_blocks += sum(1 for item in split_messages if item.get("from") == "observation")
            changed = True
        else:
            converted.append(message)

    if changed and not valid_sharegpt_turn_order(converted):
        stats["dropped_invalid_after_split"] += 1
        return None

    if changed:
        stats["converted_records"] += 1
        stats["observation_turns"] += result_blocks
        out = dict(record)
        out["conversations"] = converted
        return out

    stats["unchanged_records"] += 1
    return record


def iter_records(input_path: Path):
    """Yield ``(record_number, record)`` from a JSON array or JSONL file."""
    with input_path.open("r", encoding="utf-8") as fin:
        first = ""
        while True:
            char = fin.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
                break

        fin.seek(0)
        if first == "[":
            records = json.load(fin)
            if not isinstance(records, list):
                raise ValueError("JSON input must contain a top-level array.")
            for record_number, record in enumerate(records, start=1):
                yield record_number, record
            return

        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at line {line_no}: {err}") from err


def convert_file(input_path: Path, output_path: Path) -> Counter[str]:
    stats: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_path = output_path.with_suffix(output_path.suffix + ".dropped_records.txt")

    with (
        output_path.open("w", encoding="utf-8") as fout,
        dropped_path.open("w", encoding="utf-8") as dropped,
    ):
        for record_number, record in iter_records(input_path):
            stats["input_records"] += 1
            if not isinstance(record, dict):
                stats["dropped_non_object_records"] += 1
                dropped.write(f"{record_number}\n")
                continue

            converted = convert_record(record, stats)
            if converted is None:
                dropped.write(f"{record_number}\n")
                continue

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            stats["output_records"] += 1

    stats["dropped_records"] = stats["input_records"] - stats["output_records"]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Input ShareGPT JSON or JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Output ShareGPT JSONL file.")
    args = parser.parse_args()

    stats = convert_file(args.input, args.output)
    print("Conversion complete:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
