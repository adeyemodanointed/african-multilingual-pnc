"""Dataset and metric helpers for the first multilingual PnC baseline."""

from __future__ import annotations

from collections import Counter
from typing import Iterator


def chunk_record(record: dict[str, object], word_chunk_size: int) -> Iterator[dict[str, object]]:
    """Split long examples at word boundaries while retaining aligned labels."""
    words = record["words"]
    punctuation = record["punctuation_labels"]
    casing = record["case_labels"]
    if not (len(words) == len(punctuation) == len(casing)):
        raise ValueError(f"Misaligned labels for record {record['id']}")
    for start in range(0, len(words), word_chunk_size):
        end = start + word_chunk_size
        yield {
            "id": f"{record['id']}:{start}",
            "language": record["language"],
            "words": words[start:end],
            "punctuation_labels": punctuation[start:end],
            "case_labels": casing[start:end],
        }


def encode_record(tokenizer, record: dict[str, object], punctuation_to_id: dict[str, int], case_to_id: dict[str, int], max_length: int):
    """Align word-level targets to first subword tokens; continuation tokens ignore loss."""
    encoded = tokenizer(
        record["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = encoded.word_ids()
    punctuation_labels: list[int] = []
    case_labels: list[int] = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word_id:
            punctuation_labels.append(-100)
            case_labels.append(-100)
        else:
            punctuation_labels.append(punctuation_to_id[record["punctuation_labels"][word_id]])
            case_labels.append(case_to_id[record["case_labels"][word_id]])
        previous_word_id = word_id
    encoded["punctuation_labels"] = punctuation_labels
    encoded["case_labels"] = case_labels
    encoded["language"] = record["language"]
    return encoded


def label_counts(records: list[dict[str, object]], field: str) -> Counter[str]:
    return Counter(label for record in records for label in record[field])


def inverse_sqrt_weights(counts: Counter[str], labels: list[str]) -> list[float]:
    """Mild class weighting that avoids letting rare labels dominate optimization."""
    total = sum(counts.values())
    raw = [(total / max(1, counts[label])) ** 0.5 for label in labels]
    scale = sum(raw) / len(raw)
    return [weight / scale for weight in raw]


def token_classification_counts(predictions, labels, label_count: int) -> tuple[list[int], list[int], list[int]]:
    """Return per-class true-positive, false-positive, and false-negative counts."""
    true_positive = [0] * label_count
    false_positive = [0] * label_count
    false_negative = [0] * label_count
    for predicted, target in zip(predictions, labels, strict=True):
        if target == -100:
            continue
        if predicted == target:
            true_positive[target] += 1
        else:
            false_positive[predicted] += 1
            false_negative[target] += 1
    return true_positive, false_positive, false_negative


def f1_from_counts(true_positive: list[int], false_positive: list[int], false_negative: list[int]) -> dict[str, list[float] | float]:
    f1: list[float] = []
    support: list[int] = []
    for tp, fp, fn in zip(true_positive, false_positive, false_negative, strict=True):
        denominator = (2 * tp) + fp + fn
        f1.append(0.0 if denominator == 0 else (2 * tp) / denominator)
        support.append(tp + fn)
    included = [score for score, count in zip(f1, support, strict=True) if count]
    return {"per_class_f1": f1, "support": support, "macro_f1_present": sum(included) / len(included) if included else 0.0}
