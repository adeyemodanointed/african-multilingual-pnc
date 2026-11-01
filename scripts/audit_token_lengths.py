"""Measure wordpiece lengths and truncation risk for a PnC training configuration."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from pnc.training import chunk_record_to_token_limit


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[int], percent: int) -> int:
    if not values:
        return 0
    index = max(0, math.ceil((percent / 100) * len(values)) - 1)
    return sorted(values)[index]


def summarize(lengths: list[int], max_length: int, source_records: int) -> dict[str, int | float]:
    return {
        "source_records": source_records,
        "chunks": len(lengths),
        "additional_chunks": len(lengths) - source_records,
        "over_budget_chunks": sum(length > max_length for length in lengths),
        "over_budget_pct": round(100 * sum(length > max_length for length in lengths) / len(lengths), 2) if lengths else 0.0,
        "p50": percentile(lengths, 50),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "max": max(lengths, default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/afroxlmr_small_baseline.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/supervised"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/token_length_audit.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    max_length = config["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True, token=False)
    by_split: dict[str, dict[str, list[int]]] = {}

    for split in ("train", "dev", "test"):
        records = read_jsonl(args.data_dir / f"{split}.jsonl")
        chunks = [
            chunk
            for record in records
            for chunk in chunk_record_to_token_limit(record, tokenizer, max_length)
        ]
        lengths: dict[str, list[int]] = defaultdict(list)
        source_counts: dict[str, int] = defaultdict(int)
        for record in records:
            source_counts[str(record["language"])] += 1
        for start in range(0, len(chunks), args.batch_size):
            batch = chunks[start : start + args.batch_size]
            encoded = tokenizer(
                [item["input_words"] for item in batch],
                is_split_into_words=True,
                truncation=False,
                padding=False,
            )
            for item, input_ids in zip(batch, encoded["input_ids"], strict=True):
                lengths["all"].append(len(input_ids))
                lengths[str(item["language"])].append(len(input_ids))
        by_split[split] = {
            language: summarize(values, max_length, len(records) if language == "all" else source_counts[language])
            for language, values in sorted(lengths.items())
        }
        print(f"Audited {split}: {len(chunks):,} chunks")

    report = {
        "model_name": config["model_name"],
        "max_length": max_length,
        "word_chunk_size": config["word_chunk_size"],
        "by_split": by_split,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
