"""Qualitative NER demo: run a checkpoint on raw text or a real test example
and show token-level predictions next to ground truth.
"""

import re

import torch
from transformers import AutoModelForTokenClassification

from . import config as _config
from .config import label_maps
from .datasets import build_dataset
from .tokenize import build_tokenize_fn

_TOKEN_RE = re.compile(r"\S+")


def load_finetuned(model_key):
    ckpt = f"{_config.CHECKPOINT_DIR}/{model_key}"
    tokenizer, _ = build_tokenize_fn(ckpt)
    model = AutoModelForTokenClassification.from_pretrained(ckpt)
    model.eval()
    return tokenizer, model


def predict_with(tokenizer, model, tokens):
    """Run an already-loaded model on pre-split tokens, one predicted BIO tag per token."""
    id2label = model.config.id2label

    enc = tokenizer(
        tokens, is_split_into_words=True, return_tensors="pt",
        truncation=True, max_length=256,
    )
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred_ids = logits.argmax(-1).tolist()

    word_ids = enc.word_ids()
    labels = [None] * len(tokens)
    for pos, wid in enumerate(word_ids):
        if wid is not None and labels[wid] is None:
            labels[wid] = id2label[pred_ids[pos]]
    return labels


def predict_tokens(model_key, tokens):
    """Load a fine-tuned checkpoint and run it on pre-split tokens."""
    tokenizer, model = load_finetuned(model_key)
    return predict_with(tokenizer, model, tokens)


def predict_sentence(model_key, text):
    """Run a checkpoint on a raw free-text sentence."""
    tokens = _TOKEN_RE.findall(text)
    preds = predict_tokens(model_key, tokens)
    return list(zip(tokens, preds))


def show_test_example(model_key, dataset_name, index=0):
    """Print gold vs. predicted tags, token by token, for one real test example."""
    ds, labels = build_dataset([dataset_name], "harmonized")
    _, id2label = label_maps(labels)
    example = ds["test"][index]
    tokens = example["tokens"]
    gold = [id2label[i] for i in example["labels"]]

    preds = predict_tokens(model_key, tokens)

    print(f"{'TOKEN':<20}{'GOLD':<15}{'PRED':<15}")
    for tok, g, p in zip(tokens, gold, preds):
        mark = "" if g == p else "  <-- miss"
        print(f"{tok:<20}{g:<15}{p:<15}{mark}")
