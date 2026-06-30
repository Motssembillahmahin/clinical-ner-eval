import os

MODELS = {
    "bio_clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}

HARMONIZED_LABELS = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]


def label_maps(labels):
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    return label2id, id2label


# Override at runtime — Colab sets these to Drive paths.
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

TRAIN_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "seed": 42,
}
