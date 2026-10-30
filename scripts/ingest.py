"""Build the unified multilingual PnC corpus from the source inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from pnc.ingest import ingest_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest primary multilingual PnC sources.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/corpus_inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/unified_corpus.jsonl"))
    parser.add_argument("--datasets", nargs="*", help="Optional manifest dataset IDs to ingest.")
    parser.add_argument("--include-smoke", action="store_true", help="Include sources marked smoke_only.")
    args = parser.parse_args()

    counts = ingest_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        dataset_ids=set(args.datasets) if args.datasets else None,
        include_smoke=args.include_smoke,
    )
    total = sum(counts.values())
    print(f"Wrote {total:,} records to {args.output}")
    for dataset_id, count in counts.items():
        print(f"  {dataset_id}: {count:,}")


if __name__ == "__main__":
    main()
