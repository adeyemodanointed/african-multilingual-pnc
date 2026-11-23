# Canonical Africa PnC corpus expansion

This run uses the additional `Africa_PnC_Corpus` download and Bible export
without mixing duplicate representations. It preserves the existing
`supervised_expanded_v1` dataset and its development/test partitions.

The web-data sources are the five `*_ALL_DOMAINS.csv.gz` exports: one each for
Igbo, Kinyarwanda, Nigerian Pidgin, Swahili, and Yoruba. Each all-domain file
combines general, agriculture, finance, law, health, and telecommunications
data. The run also includes the four Bible JSONL exports for Igbo, Nigerian
Pidgin, Swahili, and Yoruba. Do not ingest the per-domain files, `_shards`,
parallel Bible CSVs, or `.txt.gz` exports; they duplicate rows already present
in the canonical files.

## 1. Extract only the canonical files

Run this from the repository root. It writes compressed files to the ignored
`../new_data/africa_pnc_canonical` directory and is safe to rerun.

```powershell
conda activate idlf26
python scripts\extract_africa_pnc_canonical.py
```

## 2. Select high-confidence additions

This streams the five web-corpus files, filters records using the same
lossless-label rules as the baseline, and protects the existing frozen dev/test
texts from overlap.

```powershell
$env:PYTHONPATH = "src"
python scripts\select_stratified_candidates.py `
  --manifest data\manifests\africa_pnc_canonical_v1.csv `
  --config configs\africa_pnc_canonical_v1.json `
  --frozen-data-dir data\processed\supervised_expanded_v1 `
  --output data\processed\africa_pnc_canonical_candidates_v1.jsonl `
  --report data\interim\africa_pnc_canonical_selection_v1.json
```

Review the selection report before building the expansion. In particular,
confirm that every language has enough accepted records and inspect the
punctuation counts for labels with limited support.

## 3. Select every eligible Bible verse

The high target values intentionally exceed the size of this corpus, so every
Bible verse that passes the lossless-label checks is retained rather than being
sampled down.

```powershell
python scripts\select_stratified_candidates.py `
  --manifest data\manifests\bible_pnc_v1.csv `
  --config configs\bible_pnc_v1.json `
  --frozen-data-dir data\processed\supervised_expanded_v1 `
  --output data\processed\bible_pnc_candidates_v1.jsonl `
  --report data\interim\bible_pnc_selection_v1.json
```

## 4. Build an additive training split

Development and test remain byte-for-byte copies of the existing frozen split.
Only the training partition is enlarged.

```powershell
python scripts\build_frozen_expansion.py `
  --selected-candidates data\processed\africa_pnc_canonical_candidates_v1.jsonl `
  --selected-candidates data\processed\bible_pnc_candidates_v1.jsonl `
  --frozen-data-dir data\processed\supervised_expanded_v1 `
  --output-dir data\processed\supervised_africa_pnc_canonical_v1 `
  --report data\interim\supervised_africa_pnc_canonical_v1.json
```

## 5. Validate before training

```powershell
python scripts\audit_token_lengths.py `
  --config configs\afroxlmr_small_new_data_frozen_v1.json `
  --data-dir data\processed\supervised_africa_pnc_canonical_v1 `
  --output data\interim\token_length_audit_africa_pnc_canonical_v1.json
```

Proceed only when `over_budget_chunks` is zero for train, dev, and test.
