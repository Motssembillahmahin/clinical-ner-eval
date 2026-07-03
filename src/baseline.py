"""Zero-shot baseline: base (non-fine-tuned) models evaluated on NCBI Disease
and BC5CDR independently.

Each base checkpoint gets a freshly initialized token-classification head (no
training) so this establishes the pre-fine-tuning floor, evaluated per dataset
rather than on the combined test set used in evaluate.py.
"""

import json
import os

import pandas as pd
from seqeval.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)

from . import config as _config
from .config import MODELS, label_maps
from .datasets import build_dataset
from .evaluate import _align_predictions, _NpEncoder
from .tokenize import build_tokenize_fn

DATASETS = ["ncbi", "bc5cdr"]


def evaluate_baseline(model_key, dataset_name, label_mode="harmonized"):
    model_name = MODELS[model_key]
    ds, labels = build_dataset([dataset_name], label_mode)
    label2id, id2label = label_maps(labels)

    tokenizer, tok_fn = build_tokenize_fn(model_name)
    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds["test"].column_names)

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(labels), id2label=id2label, label2id=label2id,
        )
    except ValueError:
        # Older BERT models (e.g. dmis-lab/biobert-base-cased-v1.1) lack
        # model_type in config.json; load explicitly as BertForTokenClassification.
        model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=len(labels), id2label=id2label, label2id=label2id,
        )

    collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, data_collator=collator, processing_class=tokenizer)

    output = trainer.predict(tokenized["test"])
    true_labels, true_preds = _align_predictions(
        output.predictions, output.label_ids, id2label
    )
    return classification_report(
        true_labels, true_preds, output_dict=True, zero_division=0
    )


def main():
    os.makedirs(_config.RESULTS_DIR, exist_ok=True)
    rows = []
    full = {}

    for model_key in MODELS:
        full[model_key] = {}
        for dataset_name in DATASETS:
            print(f"=== baseline {model_key} on {dataset_name} ===")
            report = evaluate_baseline(model_key, dataset_name)
            full[model_key][dataset_name] = report

            for entity, scores in report.items():
                if not isinstance(scores, dict):
                    continue
                rows.append({
                    "model": model_key,
                    "dataset": dataset_name,
                    "entity": entity,
                    "precision": round(scores["precision"], 4),
                    "recall": round(scores["recall"], 4),
                    "f1": round(scores["f1-score"], 4),
                    "support": scores["support"],
                })

    with open(os.path.join(_config.RESULTS_DIR, "baseline_report.json"), "w") as f:
        json.dump(full, f, indent=2, cls=_NpEncoder)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(_config.RESULTS_DIR, "baseline_comparison.csv")
    df.to_csv(csv_path, index=False)

    if not df.empty:
        pivot = df.pivot_table(index=["model", "dataset"], columns="entity", values="f1")
        print("\n=== Baseline F1 (model x dataset x entity) ===")
        print(pivot.to_string())
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
