"""MACCROBAT 2018 adapter.

Loads from bigbio/maccrobat2018 (KB format), converts character-offset entity
annotations to whitespace-tokenized BIO tags, and creates an 80/10/10 split
(MACCROBAT has no predefined train/val/test split).
"""

import re

from datasets import Dataset, DatasetDict, load_dataset

from .base import BaseNERAdapter

_DISEASE_TYPES = {"Disease_disorder", "Sign_symptom"}
_CHEMICAL_TYPES = {"Medication", "Drug"}

_NATIVE_LABELS = ["O"] + [
    f"{prefix}-{etype}"
    for etype in sorted([
        "Activity", "Administration", "Age", "Area",
        "Biological_attribute", "Biological_structure", "Clinical_event",
        "Color", "Coreference", "Date", "Detailed_description",
        "Diagnostic_procedure", "Disease_disorder", "Distance", "Dosage",
        "Duration", "Family_history", "Frequency", "Height", "Lab_value",
        "Mass", "Medication", "Nonbiological_location", "Occupation",
        "Outcome", "Personal_background", "Qualitative_concept",
        "Quantitative_concept", "Severity", "Sex", "Shape", "Sign_symptom",
        "Subject", "Therapeutic_procedure", "Time", "Travel_history",
        "Treatment", "Units", "Volume", "Weight",
    ])
    for prefix in ("B", "I")
]
_HARMONIZED_LABELS = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]

_TOKEN_RE = re.compile(r"\S+")


def _doc_to_example(doc, label_mode):
    """Convert a bigbio KB document dict to {tokens, bio}."""
    # Reconstruct document text from passages (sorted by offset).
    passages = sorted(doc["passages"], key=lambda p: p["offsets"][0][0])
    if not passages:
        return None

    # Use the first passage only to stay within 256-token budgets and avoid
    # jumbled offsets when passages are non-contiguous.
    text = passages[0]["text"][0]
    base_offset = passages[0]["offsets"][0][0]

    # Whitespace tokenize, recording char spans.
    tokens, spans = [], []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(m.group())
        spans.append((base_offset + m.start(), base_offset + m.end()))

    if not tokens:
        return None

    bio = ["O"] * len(tokens)

    for entity in doc["entities"]:
        etype = entity["type"]
        if label_mode == "harmonized":
            if etype in _DISEASE_TYPES:
                prefix = "Disease"
            elif etype in _CHEMICAL_TYPES:
                prefix = "Chemical"
            else:
                continue
        else:
            prefix = etype

        for char_start, char_end in entity["offsets"]:
            first = True
            for ti, (ts, te) in enumerate(spans):
                if te > char_start and ts < char_end:
                    if bio[ti] == "O":
                        bio[ti] = f"B-{prefix}" if first else f"I-{prefix}"
                    first = False

    return {"tokens": tokens, "bio": bio}


class MaccrobatAdapter(BaseNERAdapter):
    def load(self) -> DatasetDict:
        raw = load_dataset(
            "bigbio/maccrobat2018",
            name="maccrobat2018_bigbio_kb",
            trust_remote_code=True,
        )

        examples = []
        for split in raw:
            for doc in raw[split]:
                ex = _doc_to_example(doc, self.label_mode)
                if ex:
                    examples.append(ex)

        # No predefined splits — deterministic 80/10/10.
        n = len(examples)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        return DatasetDict({
            "train": Dataset.from_list(examples[:n_train]),
            "validation": Dataset.from_list(examples[n_train:n_train + n_val]),
            "test": Dataset.from_list(examples[n_train + n_val:]),
        })

    def label_list(self):
        return _NATIVE_LABELS if self.label_mode == "native" else _HARMONIZED_LABELS
