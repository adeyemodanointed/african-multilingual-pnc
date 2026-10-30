# Multilingual African Punctuation and Capitalization Restoration

This project starts a fresh, shared PnC pipeline for Yoruba (`yor`), Nigerian Pidgin (`pcm`), Igbo (`ibo`), Swahili (`swa`), and Kinyarwanda (`kin`).

## First build step

The raw corpus inventory is maintained in `data/manifests/corpus_inventory.csv`. The ingestion pipeline will read each source without altering its original file, normalize it into one common record format, derive labels, and create source-aware splits.

Raw data, models, and generated outputs must not be committed to version control.
