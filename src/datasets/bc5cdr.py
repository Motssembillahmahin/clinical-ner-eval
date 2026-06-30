from datasets import load_dataset, DatasetDict

from .base import BaseNERAdapter

# bigbio/bc5cdr NER config: tokens (list[str]) + tags (list[str] already in BIO string format)
# Tag names already match our harmonized schema exactly.
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
        # tner/bc5cdr uses a deprecated loading script; bigbio/bc5cdr ships as Parquet.
        raw = load_dataset("bigbio/bc5cdr", name="bc5cdr_bigbio_ner", trust_remote_code=True)

        def convert(example):
            # tags column is already a list of BIO strings
            example["bio"] = [_HARMONIZE.get(t, "O") for t in example["tags"]]
            return example

        return DatasetDict({s: raw[s].map(convert) for s in raw})

    def label_list(self):
        return _LABELS
