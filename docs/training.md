# Training

## Overview

Each model is fine-tuned independently for 3 epochs on the combined NCBI Disease + BC5CDR dataset. Training one model takes roughly 20–30 minutes on a T4 GPU. All three models use identical hyperparameters so results are directly comparable.

---

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Epochs | 3 | Standard for BERT fine-tuning; loss converges by epoch 2 |
| Batch size (train) | 16 | Fits in T4 16 GB VRAM |
| Batch size (eval) | 32 | Larger is fine for inference-only |
| Learning rate | 2e-5 | Standard BERT fine-tuning range |
| Weight decay | 0.01 | Light regularization |
| Warmup ratio | 0.1 | 10% of steps for LR warm-up |
| Max sequence length | 256 | Covers most clinical sentences |
| Precision | fp16 | Halves memory, speeds training on T4 |
| Seed | 42 | Reproducibility |

---

## Label space

All datasets are harmonized into five BIO tags:

```
O            — not an entity
B-Disease    — beginning of a disease mention
I-Disease    — continuation of a disease mention
B-Chemical   — beginning of a chemical/drug mention
I-Chemical   — continuation of a chemical/drug mention
```

This harmonization allows NCBI Disease (disease-only) and BC5CDR (disease + chemical) to be combined into one training set.

---

## Running training

### Via Colab notebook (recommended)

Run cells 1–4 in `colab_runner.ipynb`:

- **Cell 1** — mounts Google Drive, creates checkpoint directory
- **Cell 2** — installs dependencies
- **Cell 3** — points `config.CHECKPOINT_DIR` at Drive, confirms GPU
- **Cell 4** — trains all three models, skipping any already on Drive

```python
# Cell 4 logic (simplified):
for key in MODELS:
    if checkpoint_already_exists(key):
        print(f'[skip] {key} already trained')
        continue
    train_model(key)
```

The skip logic means you can safely re-run the cell or restart after a disconnect without retraining completed models.

### Via terminal (Colab or local)

```bash
# All models
python -m src.train

# Single model
python -m src.train --model biobert

# With Drive checkpoint dir
export CHECKPOINT_DIR=/content/drive/MyDrive/clinical-ner-eval/checkpoints
python -m src.train
```

---

## What happens during training

For each model, the training loop:

1. Downloads the model weights from HuggingFace Hub (cached after first run)
2. Downloads and tokenizes the combined NCBI + BC5CDR dataset
3. Runs 3 training epochs, evaluating on the validation split after each
4. Saves the best checkpoint (lowest eval loss) to `CHECKPOINT_DIR/<model_key>/`
5. Saves the tokenizer alongside the weights so the checkpoint is self-contained

### Training loss observed (our run)

| Model | Epoch 1 loss | Epoch 2 loss | Epoch 3 loss |
|---|---|---|---|
| Bio_ClinicalBERT | 0.3591 | 0.2728 | 0.2612 |
| BioBERT | 0.2902 | 0.2341 | 0.2267 |
| PubMedBERT | 0.2915 | 0.2233 | 0.2248 |

BioBERT and PubMedBERT converged faster and to a lower loss than Bio_ClinicalBERT, consistent with their higher final F1.

---

## Checkpoint structure

Each saved checkpoint contains:

```
checkpoints/<model_key>/
  config.json            model architecture + label mappings
  model.safetensors      fine-tuned weights (~400 MB)
  tokenizer.json         fast tokenizer
  tokenizer_config.json
  training_args.bin      hyperparameters used
  trainer_state.json     loss curve, best epoch
```

The checkpoint is a standard HuggingFace model directory and can be loaded with `AutoModelForTokenClassification.from_pretrained(path)`.

---

## Google Drive persistence

Checkpoints are written to Google Drive during training so a Colab session disconnect only costs the currently-training model. On the next session:

1. Mount Drive (cell 1)
2. Set `CHECKPOINT_DIR` to Drive path (cell 3)
3. Run training (cell 4) — completed models are skipped automatically

Our trained checkpoints live at:
```
MyDrive/clinical-ner-eval/checkpoints/
  biobert/
  bio_clinicalbert/
  pubmedbert/
```

---

## Adding a new model

1. Add the HuggingFace model ID to `src/config.py`:

```python
MODELS = {
    "bio_clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "your_model": "org/your-model-name",   # add here
}
```

2. Run `python -m src.train --model your_model`. Everything else (tokenization, label alignment, evaluation) is handled automatically.

If the model lacks a `tokenizer.json` on the Hub (like older BERT variants), `src/tokenize.py` falls back to `BertTokenizerFast` automatically. If its `config.json` lacks a `model_type` field, `src/train.py` falls back to `BertForTokenClassification` explicitly.
