# Clinical NER Evaluation — Documentation

This project benchmarks three biomedical language models on named entity recognition (NER) for clinical text — finding diseases and chemicals in sentences — using identical training conditions so results are directly comparable.

## What problem does this solve?

When building a clinical NLP pipeline, you need to know which pre-trained model to fine-tune. The three most commonly shortlisted models (BioBERT, PubMedBERT, Bio_ClinicalBERT) all claim to be the best choice. This harness settles that by putting all three through the same training and test procedure and measuring real F1 scores.

## Quick answer

| Model | Micro-avg F1 | Best at |
|---|---|---|
| **BioBERT** | **0.833** | Both entities |
| PubMedBERT | 0.819 | Chemical detection |
| Bio_ClinicalBERT | 0.787 | — |

See [results.md](results.md) for the full breakdown and interpretation.

## Documentation map

| File | What it covers |
|---|---|
| [setup.md](setup.md) | Installing dependencies, local vs Colab environments |
| [training.md](training.md) | How to fine-tune the models, Drive persistence, resuming after disconnect |
| [evaluation.md](evaluation.md) | How evaluation works, metrics, label space, test set details |
| [results.md](results.md) | Full results table, analysis, interpretation, limitations |

## Repository layout

```
src/
  config.py          model registry, label spaces, hyperparameters
  datasets/
    ncbi.py          NCBI Disease adapter
    bc5cdr.py        BC5CDR adapter
    maccrobat.py     MACCROBAT adapter (native 24-type + harmonized modes)
    n2c2.py          gated adapter stub (requires data use agreement)
    base.py          shared BIO conversion helper
    __init__.py      dataset registry + combination logic
  tokenize.py        subword tokenization + label alignment
  train.py           fine-tuning (one call per model)
  evaluate.py        seqeval scoring + comparison table
colab_runner.ipynb   end-to-end Colab notebook (Drive-persistent)
results/
  comparison.csv     rounded F1 per model × entity
  full_report.json   full-precision seqeval report
docs/               this folder
```
