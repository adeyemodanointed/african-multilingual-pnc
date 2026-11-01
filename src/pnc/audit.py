"""Ground-truth quality checks and deterministic review sampling."""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .ingest import CorpusSource, records_for_source


def _percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _rank(record: dict[str, str], seed: int) -> int:
    key = f"{seed}:{record['dataset_id']}:{record['id']}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big")


def _review_flags(text: str) -> tuple[bool, bool]:
    """Return simple formatting flags for a human reviewer to inspect."""
    stripped = text.strip()
    return stripped.endswith((".", "?", "!")), bool(stripped[:1].isupper())


def audit_sources(
    sources: Iterable[CorpusSource],
    root: Path,
    sample_size: int = 100,
    seed: int = 42,
) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    """Audit source formatting and select deterministic review samples by language."""
    summary: dict[str, dict[str, object]] = {}
    sample_heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = {}

    for source in sources:
        counts: Counter[str] = Counter()
        for record in records_for_source(source, root):
            text = record["raw_text"].strip()
            has_terminal_punctuation, starts_with_uppercase = _review_flags(text)
            counts["records"] += 1
            counts["tokens"] += len(text.split())
            counts["has_punctuation"] += int(any(character in text for character in ".,?!"))
            counts["terminal_punctuation"] += int(has_terminal_punctuation)
            counts["has_uppercase"] += int(any(character.isupper() for character in text))
            counts["initial_uppercase"] += int(starts_with_uppercase)
            counts["multiple_spaces"] += int("  " in text)

            language_heap = sample_heaps.setdefault(record["language"], [])
            rank = _rank(record, seed)
            candidate = (-rank, record["id"], record)
            if len(language_heap) < sample_size:
                heapq.heappush(language_heap, candidate)
            elif rank < -language_heap[0][0]:
                heapq.heapreplace(language_heap, candidate)

        total = counts["records"]
        summary[source.dataset_id] = {
            "language": source.language,
            "records": total,
            "tokens": counts["tokens"],
            "has_any_punctuation_pct": _percentage(counts["has_punctuation"], total),
            "terminal_punctuation_pct": _percentage(counts["terminal_punctuation"], total),
            "has_uppercase_pct": _percentage(counts["has_uppercase"], total),
            "initial_uppercase_pct": _percentage(counts["initial_uppercase"], total),
            "multiple_space_records": counts["multiple_spaces"],
        }

    review_rows: list[dict[str, str]] = []
    for language, heap in sorted(sample_heaps.items()):
        for record in sorted((item[2] for item in heap), key=lambda item: item["id"]):
            punctuation_ok, casing_ok = _review_flags(record["raw_text"])
            review_rows.append(
                {
                    "id": record["id"],
                    "language": language,
                    "dataset_id": record["dataset_id"],
                    "source_dataset": record["source_dataset"],
                    "domain": record["domain"],
                    "license": record["license"],
                    "raw_text": record["raw_text"],
                    "has_terminal_punctuation": str(punctuation_ok).lower(),
                    "starts_with_uppercase": str(casing_ok).lower(),
                    "punctuation_review": "",
                    "capitalization_review": "",
                    "text_quality_review": "",
                    "reviewer_notes": "",
                }
            )
    return summary, review_rows
