"""Streaming ingestion for the multilingual PnC corpus manifest."""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class CorpusSource:
    dataset_id: str
    language: str
    path: str
    format: str
    text_field: str
    domain_field: str
    source_field: str
    split_field: str
    license_field: str
    role: str
    notes: str


def load_manifest(path: Path) -> list[CorpusSource]:
    """Load the corpus inventory while preserving its explicit source metadata."""
    with path.open(encoding="utf-8", newline="") as handle:
        sources = []
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            if None in row:
                raise ValueError(f"Malformed manifest row {row_number} in {path}")
            sources.append(CorpusSource(**row))
        return sources


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def _rows(source: CorpusSource, path: Path) -> Iterator[dict[str, str]]:
    with _open_text(path) as handle:
        if source.format == "jsonl":
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
                yield {key: "" if item is None else str(item) for key, item in value.items()}
        elif source.format in {"csv", "csv.gz"}:
            yield from csv.DictReader(handle)
        else:
            raise ValueError(f"Unsupported manifest format: {source.format}")


def _value(row: dict[str, str], field: str) -> str:
    return row.get(field, "").strip() if field else ""


def records_for_source(source: CorpusSource, root: Path) -> Iterator[dict[str, str]]:
    """Yield normalized, provenance-preserving records for a single manifest source."""
    source_path = root / source.path
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file is missing: {source_path}")

    for row_number, row in enumerate(_rows(source, source_path), start=1):
        text = _value(row, source.text_field)
        if not text:
            continue
        identifier = _value(row, "id") or _value(row, "ID") or f"{source.dataset_id}_{row_number:08d}"
        yield {
            "id": identifier,
            "language": source.language,
            "raw_text": text,
            "domain": _value(row, source.domain_field),
            "source_dataset": _value(row, source.source_field) or source.dataset_id,
            "source_split": _value(row, source.split_field),
            "license": _value(row, source.license_field),
            "review_status": _value(row, "review_status") or "unknown",
            "dataset_id": source.dataset_id,
        }


def ingest_manifest(
    manifest_path: Path,
    output_path: Path,
    dataset_ids: set[str] | None = None,
    include_smoke: bool = False,
) -> dict[str, int]:
    """Stream selected primary sources into one portable JSONL corpus."""
    root = manifest_path.parents[2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for source in load_manifest(manifest_path):
            if dataset_ids and source.dataset_id not in dataset_ids:
                continue
            if source.role == "smoke_only" and not include_smoke:
                continue
            if source.role not in {"primary_pnc", "smoke_only"}:
                continue
            count = 0
            for record in records_for_source(source, root):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            counts[source.dataset_id] = count
    return counts
