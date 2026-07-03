"""Run a sentence file through the base (untrained head) or fine-tuned checkpoint
of each model, and write token-by-token predicted BIO tags for eyeballing.

Usage:
    python -m scripts.run_examples --stage base
    python -m scripts.run_examples --stage finetuned --model biobert
    python -m scripts.run_examples --stage both --file scripts/test_sentences.txt --limit 20
"""

import argparse
import os
import re

from transformers import AutoModelForTokenClassification, BertForTokenClassification

from src import config as _config
from src.config import HARMONIZED_LABELS, MODELS, label_maps
from src.demo import load_finetuned, predict_with
from src.tokenize import build_tokenize_fn

_TOKEN_RE = re.compile(r"\S+")


def load_base(model_key):
    model_name = MODELS[model_key]
    label2id, id2label = label_maps(HARMONIZED_LABELS)
    tokenizer, _ = build_tokenize_fn(model_name)
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(HARMONIZED_LABELS),
            id2label=id2label, label2id=label2id,
        )
    except ValueError:
        # Older BERT models (e.g. dmis-lab/biobert-base-cased-v1.1) lack
        # model_type in config.json; load explicitly as BertForTokenClassification.
        model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=len(HARMONIZED_LABELS),
            id2label=id2label, label2id=label2id,
        )
    model.eval()
    return tokenizer, model


def read_sentences(path, limit=None):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:limit] if limit else lines


def run_stage(stage, model_key, sentences, out_dir):
    tokenizer, model = load_base(model_key) if stage == "base" else load_finetuned(model_key)

    out_path = os.path.join(out_dir, f"predictions_{stage}_{model_key}.txt")
    with open(out_path, "w") as f:
        for sentence in sentences:
            tokens = _TOKEN_RE.findall(sentence)
            tags = predict_with(tokenizer, model, tokens)
            f.write(f"SENTENCE: {sentence}\n")
            for tok, tag in zip(tokens, tags):
                f.write(f"  {tok:<25}{tag}\n")
            f.write("\n")
    print(f"[{stage}/{model_key}] wrote {len(sentences)} sentences -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["base", "finetuned", "both"], default="both")
    parser.add_argument("--model", choices=list(MODELS), default=None,
                        help="Default: run all three models.")
    parser.add_argument("--file", default="scripts/test_sentences.txt")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N sentences (default: all).")
    parser.add_argument("--out", default="results")
    a = parser.parse_args()

    os.makedirs(a.out, exist_ok=True)
    sentences = read_sentences(a.file, a.limit)
    stages = ["base", "finetuned"] if a.stage == "both" else [a.stage]
    keys = [a.model] if a.model else list(MODELS)

    for stage in stages:
        for model_key in keys:
            if stage == "finetuned":
                ckpt = os.path.join(_config.CHECKPOINT_DIR, model_key)
                if not os.path.isdir(ckpt):
                    print(f"[skip] {stage}/{model_key}: no checkpoint at {ckpt}")
                    continue
            run_stage(stage, model_key, sentences, a.out)


if __name__ == "__main__":
    main()
