"""Zero-shot baseline evaluation.

Loads each base model from HuggingFace Hub with NO fine-tuning, attaches a
randomly-initialized classification head for the 5-label BIO scheme, and
evaluates on the same test set used by evaluate.py.

This answers: how much NER knowledge do these models carry before any task
training? The difference between these scores and the fine-tuned scores in
results/comparison.csv shows the value of fine-tuning on our label scheme.

Usage:
    python -m src.zero_shot
"""

import json
import os

import numpy as np
from seqeval.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)

from . import config as _config
from .config import HARMONIZED_LABELS, MODELS, label_maps
from .datasets import DEFAULT_COMBINATION, build_dataset
from .tokenize import build_tokenize_fn


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def _align_predictions(preds, label_ids, id2label):
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


def evaluate_zero_shot(model_key, dataset_names=None):
    """Run zero-shot NER evaluation for one model.

    Loads the base pre-trained weights from HuggingFace Hub and attaches a
    randomly initialized linear classification head. No training occurs.
    """
    hf_model_id = MODELS[model_key]
    labels = HARMONIZED_LABELS
    label2id, id2label = label_maps(labels)

    ds, _ = build_dataset(dataset_names or DEFAULT_COMBINATION, "harmonized")

    tokenizer, tok_fn = build_tokenize_fn(hf_model_id)
    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds["test"].column_names)

    # ignore_mismatched_sizes=True lets us swap in a fresh classification head
    # sized to our 5 labels instead of whatever the Hub model had (if any).
    model = AutoModelForTokenClassification.from_pretrained(
        hf_model_id,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, data_collator=collator, processing_class=tokenizer)

    output = trainer.predict(tokenized["test"])
    true_labels, true_preds = _align_predictions(
        output.predictions, output.label_ids, id2label
    )

    report = classification_report(
        true_labels, true_preds, output_dict=True, zero_division=0
    )
    return report


def main():
    os.makedirs(_config.RESULTS_DIR, exist_ok=True)
    full = {}
    summary = []

    for model_key in MODELS:
        hf_id = MODELS[model_key]
        print(f"=== zero-shot baseline: {model_key} ({hf_id}) ===")
        report = evaluate_zero_shot(model_key)
        full[model_key] = report

        micro = report.get("micro avg", {})
        chem = report.get("Chemical", {})
        dis = report.get("Disease", {})
        f1 = micro.get("f1-score", 0.0)
        print(f"  Chemical F1 : {chem.get('f1-score', 0.0):.4f}")
        print(f"  Disease F1  : {dis.get('f1-score', 0.0):.4f}")
        print(f"  Micro-avg F1: {f1:.4f}")
        summary.append({
            "model": model_key,
            "chemical_f1": round(chem.get("f1-score", 0.0), 4),
            "disease_f1": round(dis.get("f1-score", 0.0), 4),
            "micro_f1": round(f1, 4),
        })

    out_path = os.path.join(_config.RESULTS_DIR, "zero_shot_report.json")
    with open(out_path, "w") as f:
        json.dump(full, f, indent=2, cls=_NpEncoder)
    print(f"\nResults written to {out_path}")

    print("\n=== Zero-shot baseline (no fine-tuning) ===")
    print(f"{'Model':<22} {'Chemical F1':>12} {'Disease F1':>11} {'Micro F1':>9}")
    print("-" * 58)
    for row in summary:
        print(
            f"{row['model']:<22} {row['chemical_f1']:>12.4f}"
            f" {row['disease_f1']:>11.4f} {row['micro_f1']:>9.4f}"
        )

    print("\n=== Gain from fine-tuning (if fine-tuned results exist) ===")
    finetuned_path = os.path.join(_config.RESULTS_DIR, "full_report.json")
    if os.path.exists(finetuned_path):
        with open(finetuned_path) as f:
            ft = json.load(f)
        print(f"{'Model':<22} {'Zero-shot F1':>13} {'Fine-tuned F1':>14} {'Gain':>6}")
        print("-" * 58)
        for row in summary:
            mk = row["model"]
            ft_f1 = ft.get(mk, {}).get("micro avg", {}).get("f1-score", None)
            if ft_f1 is not None:
                gain = ft_f1 - row["micro_f1"]
                print(
                    f"{mk:<22} {row['micro_f1']:>13.4f} {ft_f1:>14.4f} {gain:>+6.4f}"
                )
    else:
        print("(run python -m src.evaluate first to see fine-tuned comparison)")


if __name__ == "__main__":
    main()
