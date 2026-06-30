from datasets import load_dataset, DatasetDict

from .base import BaseNERAdapter

# tner/bc5cdr tags column is Sequence(ClassLabel).
# ClassLabel names: O, B-Chemical, I-Chemical, B-Disease, I-Disease
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
        # datasets<3.0 required: v3 removed loading-script support entirely.
        raw = load_dataset("tner/bc5cdr", trust_remote_code=True)
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
