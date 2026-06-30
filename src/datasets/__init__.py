"""Dataset registry + combination logic."""

from datasets import DatasetDict, concatenate_datasets

from ..config import HARMONIZED_LABELS, label_maps
from .bc5cdr import BC5CDRAdapter
from .maccrobat import MaccrobatAdapter
from .n2c2 import N2C2Adapter
from .ncbi import NCBIDiseaseAdapter

REGISTRY = {
    "ncbi": NCBIDiseaseAdapter,
    "bc5cdr": BC5CDRAdapter,
    "maccrobat": MaccrobatAdapter,
    "n2c2": N2C2Adapter,
}

DEFAULT_COMBINATION = ["ncbi", "bc5cdr", "maccrobat"]


def _encode_split(split, label2id):
    def enc(example):
        example["labels"] = [label2id[tag] for tag in example["bio"]]
        return example
    return split.map(enc)


def build_dataset(names=None, label_mode="harmonized"):
    names = names or DEFAULT_COMBINATION

    if label_mode == "native":
        if len(names) != 1:
            raise ValueError("native mode supports exactly one dataset")
        adapter = REGISTRY[names[0]](label_mode="native")
        ds = adapter.load()
        labels = adapter.label_list()
        label2id, _ = label_maps(labels)
        ds = DatasetDict({s: _encode_split(ds[s], label2id) for s in ds})
        return ds, labels

    labels = HARMONIZED_LABELS
    label2id, _ = label_maps(labels)

    loaded = []
    for n in names:
        adapter = REGISTRY[n](label_mode="harmonized")
        try:
            d = adapter.load()
        except NotImplementedError as e:
            print(f"[skip] {n}: {e}")
            continue
        d = DatasetDict({s: _encode_split(d[s], label2id) for s in d})
        loaded.append(d)

    if not loaded:
        raise RuntimeError("no datasets loaded")

    keep = ["tokens", "labels"]
    combined = {}
    for split in ["train", "validation", "test"]:
        parts = [
            d[split].remove_columns([c for c in d[split].column_names if c not in keep])
            for d in loaded
            if split in d
        ]
        if parts:
            combined[split] = concatenate_datasets(parts)
    return DatasetDict(combined), labels
