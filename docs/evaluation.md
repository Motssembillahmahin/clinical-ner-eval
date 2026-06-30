# Evaluation

## Overview

After training, each model is evaluated on a shared held-out test set. The evaluation script loads each checkpoint, runs inference, and computes entity-level F1 scores using seqeval — the standard metric for NER tasks.

---

## Running evaluation

### Via Colab notebook

Run cell 5 in `colab_runner.ipynb`:

```python
from src.evaluate import main as eval_main
eval_main()
```

Make sure `config.CHECKPOINT_DIR` is set to the Drive path first (cell 3).

### Via terminal

```bash
python -m src.evaluate

# With Drive checkpoint path
export CHECKPOINT_DIR=/content/drive/MyDrive/clinical-ner-eval/checkpoints
python -m src.evaluate
```

### Output

The script prints a model × entity F1 table to stdout and writes two files:

- `results/comparison.csv` — rounded scores (precision, recall, F1, support) per model per entity
- `results/full_report.json` — full-precision scores from seqeval

---

## Test set

The evaluation uses the official test splits from each dataset, combined:

| Dataset | Test sentences | Disease entities | Chemical entities |
|---|---|---|---|
| NCBI Disease | ~100 | 707 | — |
| BC5CDR | ~500 | — | 656 |
| **Combined** | **~600** | **707** | **656** |

**Total: 1,363 gold entities** across both types.

The test split is never seen during training or validation. It is only loaded once per model at evaluation time.

---

## Metric: entity-level F1 (seqeval)

We use **seqeval** with default settings, which computes **entity-level** (span-level) F1, not token-level.

### Why entity-level matters

Consider the gold entity `[B-Disease, I-Disease, I-Disease]` (a 3-token disease):

| Prediction | Token accuracy | Entity F1 |
|---|---|---|
| `[B-Disease, I-Disease, O]` | 66% correct | 0.0 — span is wrong |
| `[B-Disease, I-Disease, I-Disease]` | 100% correct | 1.0 — full match |

Token-level accuracy over-counts partial matches. Entity-level F1 is stricter: the model must get the full span boundary right for a prediction to count as a true positive.

### Micro vs macro average

- **Micro avg** — treats every entity instance equally, regardless of type. This is the primary metric since entity counts differ (707 Disease vs 656 Chemical).
- **Macro avg** — treats Disease and Chemical equally regardless of frequency. Useful for checking type balance.
- **Weighted avg** — weights each type by its support count. Nearly identical to micro avg when support counts are similar.

---

## How the evaluation script works

```
for each model checkpoint:
    1. Load tokenizer and model from checkpoint directory
    2. Tokenize the test set (subword alignment via word_ids())
    3. Run Trainer.predict() — batched GPU inference
    4. Align predictions back to word level (drop subword tokens marked -100)
    5. Feed word-level gold labels + predictions to seqeval
    6. Collect precision, recall, F1, support per entity type + averages
```

### Label alignment detail

BERT models tokenize words into subwords (e.g. "hepatitis" → ["hep", "##atitis"]). We assign the label of the first subword to the word, and mark all subsequent subwords `-100` so they are ignored by both the loss function and the evaluation:

```
Word:      hepatitis   B
Subwords:  hep ##atis  B
Label:     B-Disease   -100
```

This matches the standard approach from the original HuggingFace NER tutorial.

---

## Adding a new dataset to evaluation

The evaluation automatically includes all models that have a checkpoint in `CHECKPOINT_DIR`. To change which datasets are evaluated against, pass `dataset_names` to `evaluate_model()`:

```python
from src.evaluate import evaluate_model

# evaluate on NCBI Disease only
report = evaluate_model("biobert", dataset_names=["ncbi_disease"])

# evaluate on BC5CDR only
report = evaluate_model("biobert", dataset_names=["bc5cdr"])
```

---

## Reproducibility

The same trained checkpoint always produces the same evaluation scores (no randomness in inference). The training seed is fixed at 42. To reproduce from scratch:

```bash
python -m src.train   # ~1.5 hours total on T4
python -m src.evaluate
```
