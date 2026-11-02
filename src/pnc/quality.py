"""Strict candidate selection for the first multilingual PnC training run."""

from __future__ import annotations

import hashlib
import heapq
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Mapping


_WORD_PATTERN = re.compile(r"\S+")
_TERMINAL_PUNCTUATION = (".", "?", "!")


def normalized_text_key(text: str) -> str:
    """Create a stable key for exact duplicate detection without altering the target text."""
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def high_confidence_reason(text: str, min_words: int, max_words: int) -> str | None:
    """Return the exclusion reason, or None when a record passes strict text checks."""
    normalized = unicodedata.normalize("NFC", text).strip()
    words = _WORD_PATTERN.findall(normalized)
    if len(words) < min_words:
        return "too_short"
    if len(words) > max_words:
        return "too_long"
    if not normalized[:1].isupper():
        return "missing_initial_capital"
    if not normalized.endswith(_TERMINAL_PUNCTUATION):
        return "missing_terminal_punctuation"
    if not any(character in normalized for character in ".,?!"):
        return "missing_punctuation"
    if any(unicodedata.category(character).startswith("C") and character not in "\t\n\r" for character in normalized):
        return "contains_control_character"
    return None


def _rank(record: dict[str, str], seed: int) -> int:
    value = f"{seed}:{record['dataset_id']}:{record['id']}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], byteorder="big")


def select_high_confidence_records(
    records: Iterable[dict[str, str]],
    max_per_language: int | Mapping[str, int] = 10_000,
    candidate_multiplier: int = 2,
    min_words: int = 4,
    max_words: int = 160,
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, object]]]:
    """Select balanced deterministic candidates without retaining the full corpus in memory."""
    if candidate_multiplier < 1:
        raise ValueError("Selection limits must be positive.")

    def limit_for(language: str) -> int:
        if isinstance(max_per_language, int):
            limit = max_per_language
        else:
            limit = max_per_language.get(language, max_per_language.get("default", 10_000))
        if limit < 1:
            raise ValueError(f"Selection limit for {language} must be positive.")
        return limit

    heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = defaultdict(list)
    audit: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        language = record["language"]
        audit[language]["seen"] += 1
        reason = high_confidence_reason(record["raw_text"], min_words=min_words, max_words=max_words)
        if reason:
            audit[language][reason] += 1
            continue
        audit[language]["passed_rules"] += 1
        rank = _rank(record, seed)
        candidate = (-rank, record["id"], record)
        heap = heaps[language]
        capacity = limit_for(language) * candidate_multiplier
        if len(heap) < capacity:
            heapq.heappush(heap, candidate)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, candidate)

    selected: dict[str, list[dict[str, str]]] = {}
    report: dict[str, dict[str, object]] = {}
    for language, counts in sorted(audit.items()):
        candidates = sorted((item[2] for item in heaps[language]), key=lambda item: _rank(item, seed))
        seen_text: set[str] = set()
        records_for_language: list[dict[str, str]] = []
        for record in candidates:
            key = normalized_text_key(record["raw_text"])
            if key in seen_text:
                counts["duplicate_in_candidate_pool"] += 1
                continue
            seen_text.add(key)
            records_for_language.append(record)
            if len(records_for_language) == limit_for(language):
                break
        counts["selected"] = len(records_for_language)
        selected[language] = records_for_language
        report[language] = dict(sorted(counts.items()))
    return selected, report
