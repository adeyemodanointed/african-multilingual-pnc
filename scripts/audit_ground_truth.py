"""Audit primary PnC sources and write a manual ground-truth review sheet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pnc.audit import audit_sources
from pnc.ingest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit punctuation and capitalization ground truth.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/corpus_inventory.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/interim/ground_truth_audit.json"))
    parser.add_argument("--review-sheet", type=Path, default=Path("data/interim/ground_truth_review_sample.csv"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sample_size < 1:
        parser.error("--sample-size must be at least 1")

    primary_sources = [source for source in load_manifest(args.manifest) if source.role == "primary_pnc"]
    summary, review_rows = audit_sources(
        primary_sources,
        root=args.manifest.parents[2],
        sample_size=args.sample_size,
        seed=args.seed,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.review_sheet.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(review_rows[0]) if review_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"Audited {sum(item['records'] for item in summary.values()):,} records from {len(summary)} sources.")
    print(f"Wrote {len(review_rows):,} review rows to {args.review_sheet}.")


if __name__ == "__main__":
    main()
