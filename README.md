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

The best checkpoint is selected by punctuation macro F1 excluding `O` (no punctuation). Training uses token-aware, word-boundary chunking so every chunk fits the configured token limit without dropping labels. Before increasing the context length, inspect the resulting segmentation with:

```powershell
$env:PYTHONPATH = "src"
python scripts\audit_token_lengths.py
```

The resulting local report is written to `data/interim/token_length_audit.json`.

## Versioned data expansion

The first baseline uses a 10,000-candidate cap per language. To expand Yoruba without overwriting that baseline, select all available strict Yoruba candidates and write a separate supervised dataset:

```powershell
$env:PYTHONPATH = "src"
python scripts\select_high_confidence.py --limits-json configs\selection_yoruba_expanded.json --output data\processed\high_confidence_candidates_yoruba_expanded.jsonl --report data\interim\high_confidence_selection_yoruba_expanded.json
python scripts\prepare_supervised.py --input data\processed\high_confidence_candidates_yoruba_expanded.jsonl --output-dir data\processed\supervised_yoruba_expanded --report data\interim\supervised_yoruba_expanded.json
```

This preserves the original candidate file and supervised splits. The parser also safely retains closing quotes and brackets while assigning punctuation labels; ambiguous mixed-casing and combined-punctuation cases remain excluded.

For the all-language expansion, use `configs/selection_all_languages_expanded.json`. It increases Igbo, Kinyarwanda, and Swahili to 20,000 candidates each, while selecting all available strict Yoruba and Pidgin candidates:

```powershell
$env:PYTHONPATH = "src"
python scripts\select_high_confidence.py --limits-json configs\selection_all_languages_expanded.json --output data\processed\high_confidence_candidates_expanded_v1.jsonl --report data\interim\high_confidence_selection_expanded_v1.json
python scripts\prepare_supervised.py --input data\processed\high_confidence_candidates_expanded_v1.jsonl --output-dir data\processed\supervised_expanded_v1 --report data\interim\supervised_expanded_v1.json
```
