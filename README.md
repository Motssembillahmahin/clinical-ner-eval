# clinical-ner-eval

Evaluation harness that benchmarks shortlisted clinical NER models on a combined
clinical dataset, on Colab free-tier (T4). Built around a dataset-agnostic
adapter interface so new benchmarks plug in without touching training or eval code.

## What it does

Fine-tunes and evaluates three shortlisted models under identical conditions,
then scores them on a shared held-out test set using entity-level seqeval F1
(the correct NER metric — token-level over-counts).

Shortlisted models:
- Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)
- BioBERT (`dmis-lab/biobert-base-cased-v1.1`)
- PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)

## Datasets (two tiers)

Tier 1 — available now, combined for the comparison:
- NCBI Disease — PubMed abstracts, Disease only
- BC5CDR — literature, Disease + Chemical
- MACCROBAT — real clinical case reports, ~24 native entity types

All Tier-1 sets are harmonized into a shared `Disease` / `Chemical` BIO label
space so they can be combined and models compared apples-to-apples.

Tier 2 — gated, plug in when access lands (same harness, new adapter):
- n2c2 2010 (real doctor notes; requires DUA) — adapter stub in place
- MIMIC-derived NER (pending PhysioNet access)

This matches the "doctor notes -> expected extracted entities" framing: the
harness ingests note text and scores predicted entities against gold entities.
MACCROBAT delivers genuine clinical-note text today; n2c2/MIMIC slot in later
without code changes.

## Label modes

- `harmonized` — collapse every dataset into Disease/Chemical for cross-model
  comparison (the default; required for combining datasets).
- `native` — keep a single dataset's full label set. Used for a MACCROBAT-only
  showcase of rich note->entity extraction across all ~24 types.

## Architecture

```
src/
  config.py            models, label spaces, hyperparameters
  datasets/
    base.py            common schema + BIO helper
    ncbi.py            adapter
    bc5cdr.py          adapter (defensive about mirror schema)
    maccrobat.py       adapter (harmonized + native modes)
    n2c2.py            gated adapter stub
    __init__.py        registry + combination logic
  tokenize.py          subword tokenization + label alignment
  train.py             train_model(model_key), runs once per model
  evaluate.py          seqeval harness + comparison table
notebooks/
  colab_runner.ipynb   Drive mount -> train sweep -> eval, resumable on T4
```

Adding a benchmark = one new adapter implementing `load()` and `label_list()`.
Nothing else changes.

## Running

Local:
```
make setup     # uv sync
make all       # train all models, then evaluate
```

Colab (free T4): open `notebooks/colab_runner.ipynb`, set runtime to T4, run
cells top to bottom. Checkpoints persist to Google Drive, so the sweep is
resumable across disconnects — a re-run skips models already trained.

MACCROBAT-only native showcase:
```
make showcase
```

## Free-tier notes

Three sequential fine-tunes on T4, roughly 20–40 min each at 3 epochs / batch 16
/ seq-len 256 — comfortably under the ~12h session cap. Checkpoints land in Drive
between models, so a disconnect costs at most the in-progress model.

## Output

`results/comparison.csv` and `results/full_report.json`: per-model, per-entity
precision/recall/F1 plus micro/macro averages. The runner prints a model × entity
F1 pivot for quick review.
