"""Audit capitalization targets at token-aware chunk boundaries.

This checks whether training exposes AfroXLMR to chunk starts that are actually
mid-sentence. It does not measure model predictions; use it before evaluating a
streaming ASR inference strategy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from pnc.training import chunk_record_to_token_limit


TERMINAL = {"PERIOD", "QUESTION", "EXCLAMATION"}
CASE_LABELS = ("LOWER", "TITLE", "UPPER")
CATEGORIES = ("record_start", "after_terminal", "mid_sentence")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def details(counts: Counter[str]) -> dict[str, object]:
    total = sum(counts.values())
    return {
        "chunk_starts": total,
        "case_counts": {label: counts[label] for label in CASE_LABELS},
        "case_percentages": {label: (100 * counts[label] / total if total else 0.0) for label in CASE_LABELS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/afroxlmr_small_expanded_v1_8epoch.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/supervised_expanded_v1"))
    parser.add_argument("--output-json", type=Path, default=Path("data/interim/chunk_start_capitalization_expanded_v1.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/interim/chunk_start_capitalization_expanded_v1.csv"))
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), default=("train", "dev", "test"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True, token=False)
    max_length = int(config["max_length"])
    counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    overall: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)

    for split in args.splits:
        for record in read_jsonl(args.data_dir / f"{split}.jsonl"):
            words = record["words"]
            punctuation = record["punctuation_labels"]
            casing = record["case_labels"]
            offset = 0
            for chunk in chunk_record_to_token_limit(record, tokenizer, max_length):
                if offset == 0:
                    category = "record_start"
                elif punctuation[offset - 1] in TERMINAL:
                    category = "after_terminal"
                else:
                    category = "mid_sentence"
                label = casing[offset]
                language = str(record["language"])
                counts[(language, split, category)][label] += 1
                overall[category][label] += 1
                if category == "mid_sentence" and len(examples[language]) < 5:
                    examples[language].append({
                        "record_id": record["id"],
                        "chunk_start_word": words[offset],
                        "expected_case": label,
                        "previous_punctuation": punctuation[offset - 1],
                    })
                offset += len(chunk["words"])

    report = {
        "data_dir": str(args.data_dir),
        "model_name": config["model_name"],
        "max_length": max_length,
        "definition": {
            "record_start": "first word in a source record",
            "after_terminal": "chunk begins after period, question mark, or exclamation mark",
            "mid_sentence": "chunk begins after O, comma, colon, or semicolon",
        },
        "overall": {category: details(overall[category]) for category in CATEGORIES},
        "by_language_and_split": {
            language: {
                split: {
                    category: details(counts[(language, split, category)])
                    for category in CATEGORIES
                }
                for split in args.splits
                if any(counts[(language, split, category)] for category in CATEGORIES)
            }
            for language in sorted({language for language, _, _ in counts})
        },
        "mid_sentence_examples": dict(sorted(examples.items())),
    }
    for path in (args.output_json, args.output_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("language", "split", "boundary_type", "case_label", "count", "percentage"))
        writer.writeheader()
        for (language, split, category), case_counts in sorted(counts.items()):
            result = details(case_counts)
            for case_label in CASE_LABELS:
                writer.writerow({
                    "language": language,
                    "split": split,
                    "boundary_type": category,
                    "case_label": case_label,
                    "count": result["case_counts"][case_label],
                    "percentage": f"{result['case_percentages'][case_label]:.4f}",
                })
    mid_sentence = details(overall["mid_sentence"])
    print(f"Audited {sum(item['chunk_starts'] for item in (details(overall[c]) for c in CATEGORIES)):,} token-aware chunk starts.")
    print(f"Mid-sentence chunk starts labelled TITLE: {mid_sentence['case_percentages']['TITLE']:.2f}%.")
    print(f"Wrote {args.output_json} and {args.output_csv}.")


if __name__ == "__main__":
    main()
