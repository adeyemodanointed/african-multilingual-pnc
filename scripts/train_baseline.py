"""Train the AfroXLMR-small shared multilingual punctuation/case baseline."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pnc.model import build_multitask_model
from pnc.training import chunk_record, encode_record, f1_from_counts, inverse_sqrt_weights, label_counts, token_classification_counts


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collate(batch):
    items = [dict(item) for item in batch]
    punctuation_sequences = [item.pop("punctuation_labels") for item in items]
    case_sequences = [item.pop("case_labels") for item in items]
    languages = [item.pop("language") for item in items]
    padded = tokenizer.pad(items, padding=True, return_tensors="pt")
    sequence_length = padded["input_ids"].shape[1]
    punctuation = torch.tensor([values + [-100] * (sequence_length - len(values)) for values in punctuation_sequences], dtype=torch.long)
    casing = torch.tensor([values + [-100] * (sequence_length - len(values)) for values in case_sequences], dtype=torch.long)
    return padded, punctuation, casing, languages


def focal_cross_entropy(logits, labels, weights, gamma: float):
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    if not valid.any():
        return flat_logits.sum() * 0
    logits = flat_logits[valid]
    labels = flat_labels[valid]
    log_probabilities = functional.log_softmax(logits, dim=-1)
    log_probability = log_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    probability = log_probability.exp()
    return (weights[labels] * (1 - probability).pow(gamma) * -log_probability).mean()


def evaluate(model, dataloader, device, punctuation_count: int, case_count: int):
    model.eval()
    stats = {"punctuation": ([0] * punctuation_count, [0] * punctuation_count, [0] * punctuation_count), "case": ([0] * case_count, [0] * case_count, [0] * case_count)}
    with torch.no_grad():
        for inputs, punctuation, casing, _ in dataloader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            punctuation_logits, case_logits = model(**inputs)
            for key, logits, targets, count in (("punctuation", punctuation_logits, punctuation, punctuation_count), ("case", case_logits, casing, case_count)):
                tp, fp, fn = token_classification_counts(logits.argmax(-1).cpu().flatten().tolist(), targets.flatten().tolist(), count)
                for destination, incoming in zip(stats[key], (tp, fp, fn), strict=True):
                    for index, value in enumerate(incoming):
                        destination[index] += value
    return {key: f1_from_counts(*value) for key, value in stats.items()}


def main() -> None:
    global tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/afroxlmr_small_baseline.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/supervised"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/afroxlmr_small_baseline"))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    punct_labels, case_labels = config["punctuation_labels"], config["case_labels"]
    punct_to_id = {label: index for index, label in enumerate(punct_labels)}
    case_to_id = {label: index for index, label in enumerate(case_labels)}
    raw_splits = {split: read_jsonl(args.data_dir / f"{split}.jsonl") for split in ("train", "dev", "test")}
    chunks = {split: [chunk for record in records for chunk in chunk_record(record, config["word_chunk_size"])] for split, records in raw_splits.items()}
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True, token=False)
    encoded = {split: [encode_record(tokenizer, record, punct_to_id, case_to_id, config["max_length"]) for record in records] for split, records in chunks.items()}
    language_counts = Counter(item["language"] for item in encoded["train"])
    sample_weights = [1 / language_counts[item["language"]] for item in encoded["train"]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    batch_size = training["batch_size"]
    loaders = {
        "train": DataLoader(encoded["train"], batch_size=batch_size, sampler=sampler, collate_fn=collate),
        "dev": DataLoader(encoded["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate),
        "test": DataLoader(encoded["test"], batch_size=batch_size, shuffle=False, collate_fn=collate),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_multitask_model(config["model_name"], len(punct_labels), len(case_labels)).to(device)
    optimizer = AdamW(model.parameters(), lr=training["learning_rate"])
    epochs = args.epochs or training["epochs"]
    accumulation = training["gradient_accumulation_steps"]
    updates = max(1, (len(loaders["train"]) * epochs + accumulation - 1) // accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, updates)
    punct_weights = torch.tensor(inverse_sqrt_weights(label_counts(raw_splits["train"], "punctuation_labels"), punct_labels), device=device)
    case_weights = torch.tensor(inverse_sqrt_weights(label_counts(raw_splits["train"], "case_labels"), case_labels), device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["mixed_precision"] and device.type == "cuda"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (inputs, punctuation, casing, _) in enumerate(loaders["train"], start=1):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            punctuation, casing = punctuation.to(device), casing.to(device)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                punctuation_logits, case_logits = model(**inputs)
                loss = focal_cross_entropy(punctuation_logits, punctuation, punct_weights, training["focal_gamma"])
                loss += functional.cross_entropy(case_logits.reshape(-1, len(case_labels)), casing.reshape(-1), weight=case_weights, ignore_index=-100)
                loss = loss / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            running_loss += loss.item() * accumulation
        dev = evaluate(model, loaders["dev"], device, len(punct_labels), len(case_labels))
        result = {"epoch": epoch, "train_loss": running_loss / len(loaders["train"]), "dev": dev}
        history.append(result)
        score = dev["punctuation"]["macro_f1_present"]
        print(json.dumps(result))
        if score > best_score:
            best_score = score
            torch.save({"model_state": model.state_dict(), "config": config}, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics = {"config": config, "device": str(device), "train_records": len(raw_splits["train"]), "chunk_counts": {split: len(records) for split, records in chunks.items()}, "history": history, "test": evaluate(model, loaders["test"], device, len(punct_labels), len(case_labels))}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test": metrics["test"]}, indent=2))


if __name__ == "__main__":
    main()
