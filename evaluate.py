"""Evaluation harness.

Loads each trained checkpoint, predicts on the shared test set, and computes
entity-level precision/recall/F1 via seqeval (the correct metric for NER —
token-level over-counts). Produces a single comparison table across models and
entity types and writes JSON + CSV to RESULTS_DIR.
"""

import json
import os

import numpy as np
import pandas as pd
from seqeval.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)

from .config import CHECKPOINT_DIR, MODELS, RESULTS_DIR, label_maps
from .datasets import DEFAULT_COMBINATION, build_dataset
from .tokenize import build_tokenize_fn


def _align_predictions(preds, label_ids, id2label):
    """Strip -100 positions and convert ids to BIO label strings per sentence."""
    pred_ids = np.argmax(preds, axis=2)
    true_labels, true_preds = [], []
    for p_row, l_row in zip(pred_ids, label_ids):
        sent_labels, sent_preds = [], []
        for p, l in zip(p_row, l_row):
            if l == -100:
                continue
            sent_labels.append(id2label[l])
            sent_preds.append(id2label[p])
        true_labels.append(sent_labels)
        true_preds.append(sent_preds)
    return true_labels, true_preds


def evaluate_model(model_key, dataset_names=None, label_mode="harmonized"):
    ckpt = os.path.join(CHECKPOINT_DIR, model_key)
    ds, labels = build_dataset(dataset_names or DEFAULT_COMBINATION, label_mode)
    _, id2label = label_maps(labels)

    tokenizer, tok_fn = build_tokenize_fn(ckpt)
    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds["test"].column_names)

    model = AutoModelForTokenClassification.from_pretrained(ckpt)
    collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, data_collator=collator, tokenizer=tokenizer)

    output = trainer.predict(tokenized["test"])
    true_labels, true_preds = _align_predictions(
        output.predictions, output.label_ids, id2label
    )

    # output_dict gives per-entity-type precision/recall/f1 + micro/macro avgs
    report = classification_report(
        true_labels, true_preds, output_dict=True, zero_division=0
    )
    return report


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    full = {}

    for model_key in MODELS:
        ckpt = os.path.join(CHECKPOINT_DIR, model_key)
        if not os.path.isdir(ckpt):
            print(f"[skip] no checkpoint for {model_key}")
            continue
        print(f"=== evaluating {model_key} ===")
        report = evaluate_model(model_key)
        full[model_key] = report

        for entity, scores in report.items():
            if not isinstance(scores, dict):
                continue
            rows.append({
                "model": model_key,
                "entity": entity,
                "precision": round(scores["precision"], 4),
                "recall": round(scores["recall"], 4),
                "f1": round(scores["f1-score"], 4),
                "support": scores["support"],
            })

    with open(os.path.join(RESULTS_DIR, "full_report.json"), "w") as f:
        json.dump(full, f, indent=2)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
    df.to_csv(csv_path, index=False)

    # Print a readable pivot: model x entity -> F1
    if not df.empty:
        pivot = df.pivot(index="model", columns="entity", values="f1")
        print("\n=== F1 comparison (model x entity) ===")
        print(pivot.to_string())
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
