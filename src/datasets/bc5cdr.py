from datasets import load_dataset, DatasetDict

from .base import BaseNERAdapter

# tner/bc5cdr uses a 'tags' column (Sequence of ClassLabel).
# ClassLabel names: O, B-Chemical, I-Chemical, B-Disease, I-Disease
# We pass them straight through — they already match the harmonized schema.
_HARMONIZE = {
    "B-Disease": "B-Disease",
    "I-Disease": "I-Disease",
    "B-Chemical": "B-Chemical",
    "I-Chemical": "I-Chemical",
    "O": "O",
}
_LABELS = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]


class BC5CDRAdapter(BaseNERAdapter):
    def load(self) -> DatasetDict:
        raw = load_dataset("tner/bc5cdr")
        # tags is Sequence(ClassLabel); .feature gives the inner ClassLabel
        tag_feature = raw["train"].features["tags"].feature

        def convert(example):
            example["bio"] = [
                _HARMONIZE.get(tag_feature.int2str(t), "O")
                for t in example["tags"]
            ]
            return example

        return DatasetDict({s: raw[s].map(convert) for s in raw})

    def label_list(self):
        return _LABELS
