# PnC visual demo

This app restores punctuation and capitalization using the extracted
`afroxlmr_small_expanded_v1_8epoch` checkpoint.

From the `multilingual-pnc` root:

```powershell
conda activate idlf26
python -m pip install -r requirements-train.txt
$env:PYTHONPATH = "src"
streamlit run demo/app.py
```

For the presentation, choose a language, click **Use example**, then
**Restore text**. State that it is the current experimental 8-epoch checkpoint,
not the final selected model.
