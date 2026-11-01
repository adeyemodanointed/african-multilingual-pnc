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
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pnc.model import build_multitask_model
from pnc.training import chunk_record, f1_from_counts, inverse_sqrt_weights, label_counts, token_classification_counts


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_collate_fn(tokenizer, punctuation_to_id, case_to_id, max_length: int):
    """Tokenize a complete batch at once and align its word-level labels."""
    def collate(batch):
        encoded = tokenizer(
            [item["input_words"] for item in batch],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        punctuation_sequences: list[list[int]] = []
        case_sequences: list[list[int]] = []
        for batch_index, item in enumerate(batch):
            punctuation: list[int] = []
            casing: list[int] = []
            previous_word_id = None
            for word_id in encoded.word_ids(batch_index=batch_index):
                if word_id is None or word_id == previous_word_id:
                    punctuation.append(-100)
                    casing.append(-100)
                else:
                    punctuation.append(punctuation_to_id[item["punctuation_labels"][word_id]])
                    casing.append(case_to_id[item["case_labels"][word_id]])
                previous_word_id = word_id
            punctuation_sequences.append(punctuation)
            case_sequences.append(casing)
        return (
            encoded,
            torch.tensor(punctuation_sequences, dtype=torch.long),
            torch.tensor(case_sequences, dtype=torch.long),
            [item["language"] for item in batch],
        )

    return collate


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


def empty_stats(punctuation_count: int, case_count: int):
    return {
        "punctuation": ([0] * punctuation_count, [0] * punctuation_count, [0] * punctuation_count),
        "case": ([0] * case_count, [0] * case_count, [0] * case_count),
    }


def add_counts(destination, incoming) -> None:
    for existing, new in zip(destination, incoming, strict=True):
        for index, value in enumerate(new):
            existing[index] += value


def summarize_stats(stats):
    return {key: f1_from_counts(*value) for key, value in stats.items()}


def evaluate(model, dataloader, device, punctuation_count: int, case_count: int, description: str):
    model.eval()
    stats = empty_stats(punctuation_count, case_count)
    language_stats = {}
    with torch.no_grad():
        progress = tqdm(dataloader, desc=description, unit="batch", dynamic_ncols=True)
        for inputs, punctuation, casing, languages in progress:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            punctuation_logits, case_logits = model(**inputs)
            tasks = (
                ("punctuation", punctuation_logits.argmax(-1).cpu(), punctuation, punctuation_count),
                ("case", case_logits.argmax(-1).cpu(), casing, case_count),
            )
            for key, predictions, targets, count in tasks:
                add_counts(
                    stats[key],
                    token_classification_counts(predictions.flatten().tolist(), targets.flatten().tolist(), count),
                )
                for batch_index, language in enumerate(languages):
                    per_language = language_stats.setdefault(language, empty_stats(punctuation_count, case_count))
                    add_counts(
                        per_language[key],
                        token_classification_counts(
                            predictions[batch_index].tolist(), targets[batch_index].tolist(), count
                        ),
                    )
    result = summarize_stats(stats)
    result["by_language"] = {
        language: summarize_stats(language_stats[language])
        for language in sorted(language_stats)
    }
    return result


def main() -> None:
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
    chunks = {
        split: [chunk for record in records for chunk in chunk_record(record, config["word_chunk_size"])]
        for split, records in raw_splits.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True, token=False)
    collate = make_collate_fn(tokenizer, punct_to_id, case_to_id, config["max_length"])
    language_counts = Counter(item["language"] for item in chunks["train"])
    sample_weights = [1 / language_counts[item["language"]] for item in chunks["train"]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    batch_size = training["batch_size"]
    loaders = {
        "train": DataLoader(chunks["train"], batch_size=batch_size, sampler=sampler, collate_fn=collate),
        "dev": DataLoader(chunks["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate),
        "test": DataLoader(chunks["test"], batch_size=batch_size, shuffle=False, collate_fn=collate),
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
        progress = tqdm(
            loaders["train"],
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for step, (inputs, punctuation, casing, _) in enumerate(progress, start=1):
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
            progress.set_postfix(
                loss=f"{running_loss / step:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        progress.close()
        print(f"Evaluating development split after epoch {epoch}/{epochs}...", flush=True)
        dev = evaluate(
            model,
            loaders["dev"],
            device,
            len(punct_labels),
            len(case_labels),
            f"Development {epoch}/{epochs}",
        )
        result = {"epoch": epoch, "train_loss": running_loss / len(loaders["train"]), "dev": dev}
        history.append(result)
        score = dev["punctuation"]["macro_f1_present"]
        print(json.dumps(result))
        if score > best_score:
            best_score = score
            torch.save({"model_state": model.state_dict(), "config": config}, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics = {
        "config": config,
        "run": {"effective_epochs": epochs, "epochs_override": args.epochs},
        "device": str(device),
        "train_records": len(raw_splits["train"]),
        "chunk_counts": {split: len(records) for split, records in chunks.items()},
        "history": history,
        "test": evaluate(
            model,
            loaders["test"],
            device,
            len(punct_labels),
            len(case_labels),
            "Test evaluation",
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test": metrics["test"]}, indent=2))


if __name__ == "__main__":
    main()
