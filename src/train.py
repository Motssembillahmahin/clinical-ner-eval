"""Training: one `train_model(model_key)` function, run once per shortlisted model.

Each run checkpoints under CHECKPOINT_DIR/<model_key> so a T4 disconnect costs at
most the current model, not the whole sweep. On Colab, set CHECKPOINT_DIR to a
Drive path.
"""

import argparse
import os

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from . import config as _config
from .config import MODELS, TRAIN_CONFIG, label_maps
from .datasets import DEFAULT_COMBINATION, build_dataset
from .tokenize import build_tokenize_fn


def train_model(model_key: str, dataset_names=None, label_mode="harmonized"):
    model_name = MODELS[model_key]
    ds, labels = build_dataset(dataset_names or DEFAULT_COMBINATION, label_mode)
    label2id, id2label = label_maps(labels)

    tokenizer, tok_fn = build_tokenize_fn(model_name)
    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
    except ValueError:
        # Older BERT models (e.g. dmis-lab/biobert-base-cased-v1.1) lack
        # model_type in config.json; load explicitly as BertForTokenClassification.
        model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

    out_dir = os.path.join(_config.CHECKPOINT_DIR, model_key)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_CONFIG["per_device_eval_batch_size"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=TRAIN_CONFIG["seed"],
        report_to="none",
        fp16=True,
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[done] {model_key} -> {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--label-mode", default="harmonized",
                        choices=["harmonized", "native"])
    a = parser.parse_args()

    names = [a.dataset] if a.dataset else None
    keys = [a.model] if a.model else list(MODELS)
    for k in keys:
        print(f"\n=== training {k} ===")
        train_model(k, dataset_names=names, label_mode=a.label_mode)



if __name__ == "__main__":
    main()
