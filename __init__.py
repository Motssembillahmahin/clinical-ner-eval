"""Dataset registry + combination logic.

`build_dataset()` is the single entry point used by training and evaluation.
It can return one adapter's data or the concatenation of several, all emitted in
the shared BIO schema with a single unified label list.
"""

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

# The default Tier-1 combination requested: NCBI + BC5CDR + MACCROBAT.
DEFAULT_COMBINATION = ["ncbi", "bc5cdr", "maccrobat"]


def _encode_split(split, label2id):
    """Map the string `bio` field onto integer ids under the unified label set."""

    def enc(example):
        example["labels"] = [label2id[tag] for tag in example["bio"]]
        return example

    return split.map(enc)


def build_dataset(names=None, label_mode="harmonized"):
    """Load and combine datasets into one DatasetDict under a shared label set.

    In harmonized mode the unified label list is HARMONIZED_LABELS and all
    adapters are forced into that space. Native mode only supports a single
    dataset (combination across different native schemes is undefined).
    """
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

    # Concatenate split-by-split, keeping only the columns the model needs.
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
