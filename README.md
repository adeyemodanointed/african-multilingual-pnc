# Multilingual African Punctuation and Capitalization Restoration

This project starts a fresh, shared PnC pipeline for Yoruba (`yor`), Nigerian Pidgin (`pcm`), Igbo (`ibo`), Swahili (`swa`), and Kinyarwanda (`kin`).

## First build step

The raw corpus inventory is maintained in `data/manifests/corpus_inventory.csv`. The ingestion pipeline will read each source without altering its original file, normalize it into one common record format, derive labels, and create source-aware splits.

Raw data, models, and generated outputs must not be committed to version control.

## Baseline

The first experiment fine-tunes `Davlan/afro-xlmr-small` with one shared encoder and two token-classification heads: punctuation (`O`, comma, period, question, exclamation, colon, semicolon) and word casing (lower, title, upper). It uses language-balanced sampling and focal loss for punctuation because the no-punctuation label dominates.

Each supervised record keeps the reference text unchanged as `target_text`. The model receives only `input_words`: a separately derived lowercase, punctuation-free word sequence. Raw sources and reference targets are never overwritten.

After preparing the supervised data, install the training dependencies and run:

```powershell
conda activate idlf26
python -m pip install -r requirements-train.txt
$env:PYTHONPATH = "src"
python scripts\train_baseline.py
```

Outputs are written under `outputs/afroxlmr_small_baseline/` and remain local.

The best checkpoint is selected by punctuation macro F1 excluding `O` (no punctuation). Before increasing the context length, inspect tokenization coverage with:

```powershell
$env:PYTHONPATH = "src"
python scripts\audit_token_lengths.py
```

The resulting local report is written to `data/interim/token_length_audit.json`.
