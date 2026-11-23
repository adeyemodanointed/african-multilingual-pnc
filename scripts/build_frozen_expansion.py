"""Combine selected new PnC data with an existing frozen development/test set."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pnc.dataset import prepare_record
from pnc.quality import normalized_text_key


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a larger PnC training split with frozen evaluation data.")
    parser.add_argument(
        "--selected-candidates",
        type=Path,
        action="append",
        required=True,
        help="Candidate JSONL file. Repeat this option to combine sources.",
    )
    parser.add_argument("--frozen-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    old_train = read_jsonl(args.frozen_data_dir / "train.jsonl")
    dev = read_jsonl(args.frozen_data_dir / "dev.jsonl")
    test = read_jsonl(args.frozen_data_dir / "test.jsonl")
    protected = {normalized_text_key(str(record["target_text"])) for record in dev + test}
    train_keys = {normalized_text_key(str(record["target_text"])) for record in old_train}
    new_train: list[dict[str, object]] = []
    audit: dict[str, Counter[str]] = defaultdict(Counter)

    for candidate_path in args.selected_candidates:
        with candidate_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                language = raw["language"]
                audit[language]["selected_candidates_seen"] += 1
                prepared, reason = prepare_record(raw)
                if reason:
                    audit[language][reason] += 1
                    continue
                key = normalized_text_key(str(prepared["target_text"]))
                if key in protected:
                    audit[language]["excluded_frozen_overlap"] += 1
                    continue
                if key in train_keys:
                    audit[language]["duplicate_with_existing_train"] += 1
                    continue
                prepared["split"] = "train"
                new_train.append(prepared)
                train_keys.add(key)
                audit[language]["added_to_train"] += 1

    combined_train = old_train + new_train
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in (("train", combined_train), ("dev", dev), ("test", test)):
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "frozen_evaluation_source": str(args.frozen_data_dir),
        "candidate_sources": [str(path) for path in args.selected_candidates],
        "records": {"train": len(combined_train), "dev": len(dev), "test": len(test)},
        "new_data": {language: dict(sorted(counts.items())) for language, counts in sorted(audit.items())},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote frozen-evaluation expansion to {args.output_dir}")
    print(f"  train: {len(combined_train):,}; dev: {len(dev):,}; test: {len(test):,}")
    for language, counts in sorted(audit.items()):
        print(f"  {language}: added {counts['added_to_train']:,}; existing-train duplicates {counts['duplicate_with_existing_train']:,}")


if __name__ == "__main__":
    main()
