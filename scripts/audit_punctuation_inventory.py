"""Count punctuation labels by language and split in prepared PnC supervision.

The report follows the label-inventory idea used by Mark My Words: it makes
explicit which punctuation marks are present, absent, or extremely rare before
they are interpreted as model errors.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pnc.dataset import PUNCTUATION_LABELS


DISPLAY = {
    "O": "O (none)",
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "EXCLAMATION": "!",
    "COLON": ":",
    "SEMICOLON": ";",
}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summary(counts: Counter[str]) -> dict[str, object]:
    total = sum(counts.values())
    ordered_counts = {label: counts[label] for label in PUNCTUATION_LABELS}
    return {
        "word_labels": total,
        "counts": ordered_counts,
        "percentages": {
            label: (100 * counts[label] / total if total else 0.0)
            for label in PUNCTUATION_LABELS
        },
        "present_punctuation": [label for label in PUNCTUATION_LABELS[1:] if counts[label]],
        "missing_punctuation": [label for label in PUNCTUATION_LABELS[1:] if not counts[label]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/supervised_expanded_v1"))
    parser.add_argument("--output-json", type=Path, default=Path("data/interim/punctuation_inventory_expanded_v1.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/interim/punctuation_inventory_expanded_v1.csv"))
    parser.add_argument("--report-md", type=Path, default=Path("data/interim/punctuation_inventory_expanded_v1.md"))
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), default=("train", "dev", "test"))
    args = parser.parse_args()

    by_language_split: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    record_counts: Counter[tuple[str, str]] = Counter()

    for split in args.splits:
        for record in read_jsonl(args.data_dir / f"{split}.jsonl"):
            language = str(record["language"])
            labels = record["punctuation_labels"]
            if not isinstance(labels, list):
                raise ValueError(f"Record {record['id']} has invalid punctuation labels")
            unknown = set(labels) - set(PUNCTUATION_LABELS)
            if unknown:
                raise ValueError(f"Record {record['id']} has unknown labels: {sorted(unknown)}")
            by_language_split[(language, split)].update(labels)
            by_language[language].update(labels)
            by_split[split].update(labels)
            overall.update(labels)
            record_counts[(language, split)] += 1

    report = {
        "data_dir": str(args.data_dir),
        "labels": list(PUNCTUATION_LABELS),
        "overall": summary(overall),
        "by_split": {split: summary(by_split[split]) for split in args.splits},
        "by_language": {language: summary(by_language[language]) for language in sorted(by_language)},
        "by_language_and_split": {
            language: {
                split: {"records": record_counts[(language, split)], **summary(by_language_split[(language, split)])}
                for split in args.splits
                if (language, split) in by_language_split
            }
            for language in sorted(by_language)
        },
    }
    for path in (args.output_json, args.output_csv, args.report_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("language", "split", "label", "symbol", "count", "percentage"))
        writer.writeheader()
        for (language, split), counts in sorted(by_language_split.items()):
            details = summary(counts)
            for label in PUNCTUATION_LABELS:
                writer.writerow({
                    "language": language,
                    "split": split,
                    "label": label,
                    "symbol": DISPLAY[label],
                    "count": details["counts"][label],
                    "percentage": f"{details['percentages'][label]:.4f}",
                })

    lines = [
        "# Punctuation inventory",
        "",
        "Counts are word-level labels in the prepared supervision. `O` means no punctuation follows the word.",
        "",
        "## All selected splits by language",
        "",
        "| Language | Word labels | O | , | . | ? | ! | : | ; | Missing punctuation labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for language in sorted(by_language):
        details = summary(by_language[language])
        counts = details["counts"]
        missing = ", ".join(DISPLAY[label] for label in details["missing_punctuation"]) or "None"
        lines.append(
            f"| {language} | {details['word_labels']:,} | {counts['O']:,} | {counts['COMMA']:,} | "
            f"{counts['PERIOD']:,} | {counts['QUESTION']:,} | {counts['EXCLAMATION']:,} | "
            f"{counts['COLON']:,} | {counts['SEMICOLON']:,} | {missing} |"
        )
    lines.extend(["", "## Overall distribution", "", "| Label | Symbol | Count | Percentage |", "|---|---|---:|---:|"])
    overall_details = summary(overall)
    for label in PUNCTUATION_LABELS:
        lines.append(
            f"| {label} | {DISPLAY[label]} | {overall_details['counts'][label]:,} | "
            f"{overall_details['percentages'][label]:.4f}% |"
        )
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audited {overall_details['word_labels']:,} word labels across {len(by_language)} languages.")
    print(f"Wrote {args.output_json}, {args.output_csv}, and {args.report_md}.")


if __name__ == "__main__":
    main()
