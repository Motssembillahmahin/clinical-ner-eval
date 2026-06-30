# clinical-ner-eval

Benchmarks three biomedical language models on Named Entity Recognition (NER) — finding **diseases** and **chemicals** in clinical/biomedical text — under identical training conditions so results are directly comparable.

> **TL;DR:** BioBERT wins (micro-avg F1 **0.833**), PubMedBERT is close (0.819), Bio_ClinicalBERT trails (0.787) on literature-derived benchmarks.

---

## Results

Evaluated on combined NCBI Disease + BC5CDR test set (1,363 entities):

| Model | Chemical F1 | Disease F1 | **Micro-avg F1** |
|---|---|---|---|
| **BioBERT** | 0.871 | 0.798 | **0.833** |
| PubMedBERT | 0.866 | 0.775 | 0.819 |
| Bio_ClinicalBERT | 0.852 | 0.730 | 0.787 |

Full breakdown and interpretation: [docs/results.md](docs/results.md)

---

## Models benchmarked

| Key | HuggingFace ID | Pre-trained on |
|---|---|---|
| `bio_clinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC-III clinical notes |
| `biobert` | `dmis-lab/biobert-base-cased-v1.1` | PubMed abstracts + PMC full text |
| `pubmedbert` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | PubMed abstracts |

---

## Datasets

| Dataset | Entities | Source |
|---|---|---|
| NCBI Disease | Disease only | PubMed abstracts |
| BC5CDR | Disease + Chemical | BioCreative V |
| MACCROBAT | 24 native types | Clinical case reports (not on Hub — skipped) |

All datasets are harmonized into a shared `Disease` / `Chemical` BIO label space for cross-model comparison.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/setup.md](docs/setup.md) | Installation, Colab vs local, environment variables |
| [docs/training.md](docs/training.md) | How to fine-tune, hyperparameters, Drive persistence, adding a new model |
| [docs/evaluation.md](docs/evaluation.md) | Metrics, test set details, how to run evaluation |
| [docs/results.md](docs/results.md) | Full results table, analysis, interpretation, limitations |

---

## Quickstart

### Colab (recommended — free T4 GPU)

1. Open `colab_runner.ipynb` in Google Colab
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells — training takes ~25 min per model, checkpoints persist to Drive

Pre-trained checkpoints are already saved to Drive. Re-running the notebook skips models that are already trained.

### Local

```bash
git clone git@github.com:Motssembillahmahin/clinical-ner-eval.git
cd clinical-ner-eval
pip install uv && uv sync

# Train all models (~1.5 h on GPU)
python -m src.train

# Evaluate
python -m src.evaluate
```

To target a specific model:
```bash
python -m src.train --model biobert
```

---

## Architecture

```
src/
  config.py          models, label spaces, hyperparameters
  datasets/
    ncbi.py          NCBI Disease adapter
    bc5cdr.py        BC5CDR adapter
    maccrobat.py     MACCROBAT adapter (native 24-type + harmonized)
    n2c2.py          gated adapter stub (requires data use agreement)
    base.py          BIO conversion helper
    __init__.py      dataset registry + combination logic
  tokenize.py        subword tokenization + label alignment
  train.py           fine-tuning (one call per model)
  evaluate.py        seqeval harness + comparison table
colab_runner.ipynb   end-to-end Colab runner
results/
  comparison.csv     F1 per model × entity (rounded)
  full_report.json   full-precision seqeval report
docs/               detailed documentation
```

**Extending:** Adding a new benchmark = one new adapter implementing `load()` and `label_list()`. Nothing else changes. See [docs/training.md](docs/training.md) for adding a new model.

---

## Free-tier notes

Three sequential fine-tunes on Colab T4: roughly 20–30 min each at 3 epochs / batch 16 / seq-len 256 — comfortably under the ~12h session cap. Checkpoints land on Drive between models so a disconnect costs at most the in-progress model.
