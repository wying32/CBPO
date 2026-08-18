#!/usr/bin/env python3
"""Convert CBPO parquet datasets to JSONL for inspection.

By default this writes a readable JSONL view of train_10k.parquet:

    python3 training/scripts/parquet_to_jsonl.py

Use --mode raw if you want to preserve the original parquet fields exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REPO_CBPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_CBPO_DIR / "rl_datasets" / "train_10k.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a parquet dataset to JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input parquet file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file. Defaults to input filename with .jsonl suffix.",
    )
    parser.add_argument(
        "--mode",
        choices=("readable", "raw"),
        default="readable",
        help="readable extracts common fields; raw preserves the parquet row.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Rows read from parquet at a time.",
    )
    parser.add_argument(
        "--ensure-ascii",
        action="store_true",
        help="Escape non-ASCII characters in JSON output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def first_message_content(prompt: Any, role: str) -> str | None:
    if not isinstance(prompt, list):
        return None
    for message in prompt:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def readable_row(row: dict[str, Any]) -> dict[str, Any]:
    reward_model = row.get("reward_model") or {}
    extra_info = row.get("extra_info") or {}

    return {
        "index": extra_info.get("index"),
        "split": extra_info.get("split"),
        "data_source": row.get("data_source"),
        "ability": row.get("ability"),
        "question": extra_info.get("question") or first_message_content(row.get("prompt"), "user"),
        "answer": reward_model.get("ground_truth"),
        "reward_style": reward_model.get("style"),
        "need_tools_kwargs": extra_info.get("need_tools_kwargs"),
        "system_prompt": first_message_content(row.get("prompt"), "system"),
        "user_prompt": first_message_content(row.get("prompt"), "user"),
        "metadata": row.get("metadata"),
    }


def convert(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    if args.output is None:
        output_path = input_path.with_suffix(".jsonl")
    else:
        output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists, use --overwrite: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(input_path)
    total_written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for batch in parquet_file.iter_batches(batch_size=args.batch_size):
            for row in batch.to_pylist():
                if args.limit is not None and total_written >= args.limit:
                    print(f"Wrote {total_written} rows to {output_path}")
                    return total_written

                item = readable_row(row) if args.mode == "readable" else row
                json.dump(item, f, ensure_ascii=args.ensure_ascii)
                f.write("\n")
                total_written += 1

    print(f"Wrote {total_written} rows to {output_path}")
    return total_written


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
