"""Create split multilingual PnC training records from high-confidence candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pnc.dataset import prepare_records


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multilingual PnC supervised records.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/high_confidence_candidates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/supervised"))
    parser.add_argument("--report", type=Path, default=Path("data/interim/supervised_preparation.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records, report = prepare_records(read_jsonl(args.input), seed=args.seed)
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_split[str(record["split"])].append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in by_split[split]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Prepared {len(records):,} records.")
    for split in ("train", "dev", "test"):
        print(f"  {split}: {len(by_split[split]):,}")


if __name__ == "__main__":
    main()
