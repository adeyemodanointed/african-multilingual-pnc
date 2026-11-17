# New-data expansion with frozen evaluation

This runbook enlarges the PnC training data without changing the development
or test data used by `supervised_expanded_v1`. It uses the five corpora stored
in `../new_data` through manifest entries marked `candidate_pnc`.

The selected sample intentionally reserves capacity for rare punctuation:
questions, exclamations, semicolons, and colons. It is therefore not a simple
random sample of the much larger new corpus.

## 1. Verify the code

```powershell
conda activate idlf26
$env:PYTHONPATH = "src"
python -m unittest tests.test_ingest
```

## 2. Select stratified candidates

This streams the new compressed files in place. It does not move or modify
them. It may take a while because the Swahili corpus alone contains more than
seven million rows.

```powershell
python scripts\select_stratified_candidates.py `
  --config configs\new_data_stratified_v1.json `
  --frozen-data-dir data\processed\supervised_expanded_v1 `
  --output data\processed\new_data_stratified_candidates_v1.jsonl `
  --report data\interim\new_data_stratified_selection_v1.json
```

Check `data/interim/new_data_stratified_selection_v1.json` before continuing.
It records excluded frozen-test overlaps, strict-label exclusions, selected
counts by stratum, and punctuation-label counts.

The target is 220,000 candidates: 50,000 each for Igbo, Kinyarwanda, Swahili,
and Yoruba, plus 20,000 for Nigerian Pidgin. A lower count for any stratum is
valid when the source does not contain enough high-confidence examples.

## 3. Build the training expansion

This keeps the prior train split, copies the existing development/test files
unchanged, and adds only new records that do not duplicate existing training
texts or frozen evaluation texts.

```powershell
python scripts\build_frozen_expansion.py `
  --selected-candidates data\processed\new_data_stratified_candidates_v1.jsonl `
  --frozen-data-dir data\processed\supervised_expanded_v1 `
  --output-dir data\processed\supervised_new_data_frozen_v1 `
  --report data\interim\supervised_new_data_frozen_v1.json
```

The expected result is roughly 268,000 training records while the development
and test partitions remain exactly those from `supervised_expanded_v1`.

## 4. Audit token lengths

```powershell
python scripts\audit_token_lengths.py `
  --config configs\afroxlmr_small_new_data_frozen_v1.json `
  --data-dir data\processed\supervised_new_data_frozen_v1 `
  --output data\interim\token_length_audit_new_data_frozen_v1.json
```

Continue only when the report shows `over_budget_chunks: 0` for every split.

## 5. Train the comparable baseline

Use Colab for this stage if local training is too slow. The initial comparison
is three epochs, because the training split is several times larger than the
previous 48,055-record run. The trainer selects the best epoch using the
unchanged development set.

```powershell
python scripts\train_baseline.py `
  --config configs\afroxlmr_small_new_data_frozen_v1.json `
  --data-dir data\processed\supervised_new_data_frozen_v1 `
  --output-dir outputs\afroxlmr_small_new_data_frozen_v1_3epoch
```

Compare its test `punctuation.macro_f1_non_o` directly with the earlier
expanded baseline (`0.4768`), because both use the same frozen test partition.
