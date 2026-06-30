# Setup

## Requirements

- Python 3.10+
- CUDA GPU (for Colab: T4 free tier is sufficient)
- ~3 GB free disk or Google Drive space for checkpoints
- ~500 MB for downloaded datasets (cached by HuggingFace at runtime)

No dataset downloads or model weights need to be pre-staged — everything is pulled automatically at run time.

---

## Option A: Google Colab (recommended, no local GPU needed)

1. Open `colab_runner.ipynb` in Google Colab.
2. Set the runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run cells top to bottom — they handle everything:
   - Mount your Google Drive for durable checkpoints
   - Clone or locate the project code
   - Install dependencies via `uv`
   - Train all three models
   - Evaluate and print the comparison table

Checkpoints are written to `MyDrive/clinical-ner-eval/checkpoints/` so a session disconnect only costs the in-progress model. Re-running the notebook skips already-trained models automatically.

---

## Option B: Local machine

### 1. Clone the repository

```bash
git clone git@github.com:Motssembillahmahin/clinical-ner-eval.git
cd clinical-ner-eval
```

### 2. Install dependencies

The project uses [`uv`](https://github.com/astral-sh/uv) for fast dependency management:

```bash
pip install uv
uv sync
```

Or with plain pip:

```bash
pip install -e .
```

Key dependencies (see `pyproject.toml` for pinned versions):

| Package | Purpose |
|---|---|
| `transformers>=4.44` | Model loading, Trainer API |
| `torch>=2.2` | GPU training |
| `datasets>=2.20,<3.0` | HuggingFace dataset loading |
| `seqeval>=1.2.2` | Entity-level NER metrics |
| `accelerate>=0.33` | Trainer backend |
| `sentencepiece>=0.1.99` | Tokenizer for older BERT models |

### 3. Verify GPU

```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_device_name(0)) # e.g. "Tesla T4"
```

Training on CPU is possible but will take hours per model instead of ~25 minutes.

---

## Environment variables

Two paths can be overridden at runtime without touching the code:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_DIR` | `checkpoints/` | Where trained model weights are saved |
| `RESULTS_DIR` | `results/` | Where CSV/JSON evaluation output lands |

**Colab example** (set in notebook cell 3, already done for you):
```python
import src.config as config
config.CHECKPOINT_DIR = '/content/drive/MyDrive/clinical-ner-eval/checkpoints'
```

**Terminal example:**
```bash
export CHECKPOINT_DIR=/content/drive/MyDrive/clinical-ner-eval/checkpoints
python -m src.train
```
