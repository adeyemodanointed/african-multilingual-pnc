"""Select new PnC training candidates while protecting a frozen evaluation set.

Candidate sources are streamed from the corpus manifest.  A deterministic
hash-based reservoir keeps a balanced sample by language and punctuation
stratum, so very common period-only rows cannot crowd out rare punctuation.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from pnc.dataset import prepare_record
from pnc.quality import high_confidence_reason, normalized_text_key
from pnc.ingest import load_manifest, records_for_source


STRATA = ("rare", "colon", "comma", "other")


def candidate_records(manifest_path: Path) -> Iterator[dict[str, str]]:
    root = manifest_path.parents[2]
    for source in load_manifest(manifest_path):
        if source.role == "candidate_pnc":
            yield from records_for_source(source, root)


def frozen_keys(data_dir: Path) -> set[str]:
    keys: set[str] = set()
    for split in ("dev", "test"):
        path = data_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    keys.add(normalized_text_key(str(record["target_text"])))
    return keys


def stratum_for(labels: list[str]) -> str:
    values = set(labels)
    if values.intersection({"QUESTION", "EXCLAMATION", "SEMICOLON"}):
        return "rare"
    if "COLON" in values:
        return "colon"
    if "COMMA" in values:
        return "comma"
    return "other"


def rank(record: dict[str, str], seed: int) -> int:
    key = f"{seed}:{record['dataset_id']}:{record['id']}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select stratified PnC candidates from candidate_pnc sources.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/corpus_inventory.csv"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    targets: dict[str, dict[str, int]] = config["targets"]
    seed = int(config.get("seed", 42))
    min_words = int(config.get("min_words", 4))
    max_words = int(config.get("max_words", 160))
    protected = frozen_keys(args.frozen_data_dir)

    heaps: dict[str, dict[str, list[tuple[int, str, dict[str, str]]]]] = {
        language: {stratum: [] for stratum in STRATA} for language in targets
    }
    audit: dict[str, Counter[str]] = defaultdict(Counter)

    for record in candidate_records(args.manifest):
        language = record["language"]
        if language not in targets:
            continue
        counts = audit[language]
        counts["seen"] += 1
        text_key = normalized_text_key(record["raw_text"])
        if text_key in protected:
            counts["excluded_frozen_overlap"] += 1
            continue
        reason = high_confidence_reason(record["raw_text"], min_words=min_words, max_words=max_words)
        if reason:
            counts[reason] += 1
            continue
        prepared, reason = prepare_record(record, min_words=min_words, max_words=max_words)
        if reason:
            counts[reason] += 1
            continue
        counts["passed_rules"] += 1
        bucket = stratum_for(list(prepared["punctuation_labels"]))
        counts[f"passed_{bucket}"] += 1
        target = int(targets[language].get(bucket, 0))
        if target == 0:
            continue
        value = rank(record, seed)
        item = (-value, record["id"], record)
        heap = heaps[language][bucket]
        if len(heap) < target:
            heapq.heappush(heap, item)
        elif value < -heap[0][0]:
            heapq.heapreplace(heap, item)

    selected: list[dict[str, str]] = []
    report: dict[str, dict[str, object]] = {}
    for language in sorted(targets):
        counts = audit[language]
        selected_keys: set[str] = set()
        selected_labels: Counter[str] = Counter()
        selected_by_stratum: dict[str, int] = {}
        for bucket in STRATA:
            values = sorted((item[2] for item in heaps[language][bucket]), key=lambda item: rank(item, seed))
            kept = 0
            for record in values:
                text_key = normalized_text_key(record["raw_text"])
                if text_key in selected_keys:
                    counts["duplicate_across_strata"] += 1
                    continue
                selected_keys.add(text_key)
                prepared, reason = prepare_record(record, min_words=min_words, max_words=max_words)
                if reason:
                    raise RuntimeError(f"A selected record no longer parses: {reason}")
                selected_labels.update(prepared["punctuation_labels"])
                selected.append(record)
                kept += 1
            selected_by_stratum[bucket] = kept
        counts["selected"] = sum(selected_by_stratum.values())
        report[language] = {
            "selection": selected_by_stratum,
            "punctuation_label_counts": dict(sorted(selected_labels.items())),
            "audit": dict(sorted(counts.items())),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Protected frozen dev/test texts: {len(protected):,}")
    print(f"Wrote {len(selected):,} stratified candidates to {args.output}")
    for language, values in report.items():
        print(f"  {language}: {values['selection']}")


if __name__ == "__main__":
    main()
