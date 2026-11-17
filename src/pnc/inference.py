"""Inference utilities for a saved multitask punctuation/case checkpoint."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from pnc.dataset import render_text
from pnc.model import build_multitask_model


INPUT_PUNCTUATION = re.compile(r"[,.:;?!]")
WHITESPACE = re.compile(r"\s+")


def input_words(text: str) -> list[str]:
    """Produce the lowercase, punctuation-free word input used in training."""
    cleaned = INPUT_PUNCTUATION.sub(" ", text)
    return [word.lower() for word in WHITESPACE.sub(" ", cleaned).strip().split() if word]


def load_checkpoint(checkpoint_path: Path, device: torch.device | None = None):
    """Load the exact architecture and label maps stored with ``best.pt``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    # The demo must work from the already-cached base model without relying on
    # network access during a presentation.
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True, token=False, local_files_only=True)
    model = build_multitask_model(
        config["model_name"],
        len(config["punctuation_labels"]),
        len(config["case_labels"]),
        local_files_only=True,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return tokenizer, model, config, device


def restore_text(text: str, tokenizer, model, config: dict, device: torch.device) -> dict[str, object]:
    """Restore punctuation and casing for one short input (up to 96 words)."""
    words = input_words(text)
    if not words:
        raise ValueError("Enter at least one word.")
    max_words = int(config.get("word_chunk_size", 96))
    if len(words) > max_words:
        raise ValueError(f"Demo inputs support up to {max_words} words; received {len(words)}.")
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=config["max_length"])
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        punctuation_logits, case_logits = model(**encoded)
    punctuation_ids = punctuation_logits[0].argmax(-1).cpu().tolist()
    case_ids = case_logits[0].argmax(-1).cpu().tolist()
    punct_labels = config["punctuation_labels"]
    case_labels = config["case_labels"]
    punctuation: list[str] = []
    casing: list[str] = []
    previous_word_id = None
    for token_index, word_id in enumerate(tokenizer(words, is_split_into_words=True).word_ids()):
        if word_id is None or word_id == previous_word_id:
            continue
        punctuation.append(punct_labels[punctuation_ids[token_index]])
        casing.append(case_labels[case_ids[token_index]])
        previous_word_id = word_id
    if len(punctuation) != len(words):
        raise RuntimeError("Token-to-word alignment failed; try a shorter input.")
    return {
        "input_words": words,
        "punctuation_labels": punctuation,
        "case_labels": casing,
        "restored_text": render_text(words, punctuation, casing),
    }
