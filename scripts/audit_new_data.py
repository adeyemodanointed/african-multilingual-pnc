"""Audit compressed CSV corpora before adding them to the PnC pipeline.

The script reads the files in place.  It does not copy, alter, or ingest raw
data.  It writes one compact JSON report that can be used to decide which
sources are suitable for the next supervised-data expansion.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


PUNCTUATION = ",.?!:;"


def top_counts(counter: Counter[str], limit: int = 20) -> list[dict[str, object]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def audit_file(path: Path) -> dict[str, object]:
    counters = {
        "language": Counter(),
        "domain": Counter(),
        "source": Counter(),
        "license": Counter(),
        "punctuation_rows": Counter(),
    }
    total = empty_text = duplicate_hashes = missing_hashes = 0
    seen_hashes: set[str] = set()
    word_total = char_total = 0

    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Text", "Language_Code", "Source", "Text_SHA1"}
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

        for row in reader:
            total += 1
            text = (row.get("Text") or "").strip()
            text_hash = (row.get("Text_SHA1") or "").strip()
            counters["language"][(row.get("Language_Code") or "<missing>").strip()] += 1
            counters["domain"][(row.get("Domain") or "<missing>").strip()] += 1
            counters["source"][(row.get("Source") or "<missing>").strip()] += 1
            counters["license"][(row.get("License") or "<missing>").strip()] += 1

            if not text:
                empty_text += 1
                continue

            word_total += len(text.split())
            char_total += len(text)
            for character in PUNCTUATION:
                if character in text:
                    counters["punctuation_rows"][character] += 1

            if not text_hash:
                missing_hashes += 1
            elif text_hash in seen_hashes:
                duplicate_hashes += 1
            else:
                seen_hashes.add(text_hash)

    non_empty = total - empty_text
    return {
        "file": path.name,
        "compressed_bytes": path.stat().st_size,
        "rows": total,
        "empty_text_rows": empty_text,
        "missing_text_hash_rows": missing_hashes,
        "duplicate_text_hash_rows_within_file": duplicate_hashes,
        "mean_words_per_nonempty_row": round(word_total / non_empty, 2) if non_empty else 0,
        "mean_characters_per_nonempty_row": round(char_total / non_empty, 2) if non_empty else 0,
        "language_counts": top_counts(counters["language"]),
        "domain_counts": top_counts(counters["domain"]),
        "source_counts": top_counts(counters["source"]),
        "license_counts": top_counts(counters["license"]),
        "rows_containing_punctuation": dict(sorted(counters["punctuation_rows"].items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No .csv.gz files found in {args.data_dir}")

    report = {"data_dir": str(args.data_dir), "files": []}
    for path in files:
        print(f"Auditing {path.name} ...", flush=True)
        result = audit_file(path)
        report["files"].append(result)
        print(f"  {result['rows']:,} rows; {result['duplicate_text_hash_rows_within_file']:,} within-file duplicate hashes", flush=True)

    report["total_rows"] = sum(item["rows"] for item in report["files"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    print(f"Total rows: {report['total_rows']:,}")


if __name__ == "__main__":
    main()
