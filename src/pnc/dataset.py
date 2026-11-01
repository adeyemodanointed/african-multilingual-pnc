"""Supervised PnC record preparation with lossless label validation."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter


PUNCTUATION_LABELS = ("O", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION", "COLON", "SEMICOLON")
CASE_LABELS = ("LOWER", "TITLE", "UPPER")
_PUNCTUATION_TO_LABEL = {
    ",": "COMMA",
    ".": "PERIOD",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    ":": "COLON",
    ";": "SEMICOLON",
}
_TRAILING_PUNCTUATION = re.compile(r"([,.:;?!]+)$")


def normalize_reference_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving language-specific diacritics."""
    text = " ".join(unicodedata.normalize("NFC", text).split())
    text = re.sub(r"\s+([,.:;?!])", r"\1", text)
    output: list[str] = []
    for index, character in enumerate(text):
        output.append(character)
        if character not in ",.:;?!" or index == len(text) - 1:
            continue
        next_character = text[index + 1]
        previous_character = text[index - 1] if index else ""
        if next_character.isspace():
            continue
        if character == "." and previous_character.isdigit() and next_character.isdigit():
            continue
        if next_character in "'\"”’)]}":
            continue
        output.append(" ")
    return "".join(output)


def case_label(word: str) -> str | None:
    letters = "".join(character for character in word if character.isalpha())
    if not letters:
        return None
    if letters.islower():
        return "LOWER"
    if len(letters) > 1 and letters.isupper():
        return "UPPER"
    if letters[:1].isupper() and (len(letters) == 1 or letters[1:].islower()):
        return "TITLE"
    return None


def apply_case(word: str, label: str) -> str:
    if label == "LOWER":
        return word.lower()
    if label == "UPPER":
        return word.upper()
    if label == "TITLE":
        lowered = word.lower()
        for index, character in enumerate(lowered):
            if character.isalpha():
                return lowered[:index] + character.upper() + lowered[index + 1 :]
        return lowered
    raise ValueError(f"Unknown case label: {label}")


def punctuation_label(suffix: str) -> str:
    if not suffix:
        return "O"
    if len(set(suffix)) != 1:
        raise ValueError("combined_punctuation")
    return _PUNCTUATION_TO_LABEL[suffix[-1]]


def render_text(words: list[str], punctuation: list[str], casing: list[str]) -> str:
    label_to_character = {
        "O": "",
        "COMMA": ",",
        "PERIOD": ".",
        "QUESTION": "?",
        "EXCLAMATION": "!",
        "COLON": ":",
        "SEMICOLON": ";",
    }
    return " ".join(
        apply_case(word, case) + label_to_character[label]
        for word, label, case in zip(words, punctuation, casing, strict=True)
    )


def prepare_record(record: dict[str, str], min_words: int = 4, max_words: int = 160) -> tuple[dict[str, object] | None, str | None]:
    """Derive an exactly reconstructible PnC training record or an exclusion reason."""
    target_text = normalize_reference_text(record["raw_text"])
    words: list[str] = []
    punctuation: list[str] = []
    casing: list[str] = []

    for token in target_text.split():
        match = _TRAILING_PUNCTUATION.search(token)
        suffix = match.group(1) if match else ""
        word = token[: -len(suffix)] if suffix else token
        if not word:
            return None, "punctuation_only_token"
        if any(character in word for character in ",.?!;:"):
            return None, "embedded_or_unsupported_punctuation"
        try:
            label = punctuation_label(suffix)
        except ValueError as error:
            return None, str(error)
        case = case_label(word)
        if case is None:
            return None, "unsupported_case_pattern"
        words.append(word)
        punctuation.append(label)
        casing.append(case)

    if len(words) < min_words:
        return None, "too_short_after_tokenization"
    if len(words) > max_words:
        return None, "too_long_after_tokenization"
    if render_text(words, punctuation, casing) != target_text:
        return None, "round_trip_mismatch"

    input_words = [word.lower() for word in words]
    prepared = {
        **record,
        "target_text": target_text,
        "input_text": " ".join(input_words),
        "input_words": input_words,
        "words": words,
        "punctuation_labels": punctuation,
        "case_labels": casing,
    }
    return prepared, None


def split_for_record(record: dict[str, object], seed: int = 42) -> str:
    """Assign a deterministic record-level split when a document ID is unavailable."""
    key = f"{seed}:{record['language']}:{record['dataset_id']}:{record['id']}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big") / 2**64
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "dev"
    return "test"


def prepare_records(records: list[dict[str, str]], seed: int = 42) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    """Prepare records and collect inclusion/exclusion counts by language."""
    accepted: list[dict[str, object]] = []
    report: dict[str, Counter[str]] = {}
    for record in records:
        language = record["language"]
        counts = report.setdefault(language, Counter())
        counts["seen"] += 1
        prepared, reason = prepare_record(record)
        if reason:
            counts[reason] += 1
            continue
        prepared["split"] = split_for_record(prepared, seed)
        counts[f"split_{prepared['split']}"] += 1
        counts["accepted"] += 1
        accepted.append(prepared)
    return accepted, {language: dict(sorted(counts.items())) for language, counts in sorted(report.items())}
