"""Create a balanced high-confidence PnC candidate corpus for initial training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from pnc.ingest import load_manifest, records_for_source
from pnc.quality import select_high_confidence_records


def primary_records(manifest_path: Path) -> Iterator[dict[str, str]]:
    root = manifest_path.parents[2]
    for source in load_manifest(manifest_path):
        if source.role == "primary_pnc":
            yield from records_for_source(source, root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select high-confidence PnC candidates.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/corpus_inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/high_confidence_candidates.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/interim/high_confidence_selection.json"))
    parser.add_argument("--max-per-language", type=int, default=10_000)
    parser.add_argument(
        "--limits-json",
        type=Path,
        help="Optional JSON object mapping language codes to selection limits.",
    )
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    limits = json.loads(args.limits_json.read_text(encoding="utf-8")) if args.limits_json else args.max_per_language
    selected, report = select_high_confidence_records(
        primary_records(args.manifest),
        max_per_language=limits,
        min_words=args.min_words,
        max_words=args.max_words,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for language in sorted(selected):
            for record in selected[language]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {sum(len(records) for records in selected.values()):,} records to {args.output}.")
    for language, records in sorted(selected.items()):
        print(f"  {language}: {len(records):,}")


if __name__ == "__main__":
    main()
