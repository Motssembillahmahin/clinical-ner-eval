from datasets import load_dataset, DatasetDict

from .base import BaseNERAdapter

# ncbi_disease ClassLabel: 0=O, 1=B-Disease, 2=I-Disease
_ID2BIO = {0: "O", 1: "B-Disease", 2: "I-Disease"}
_LABELS = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]


class NCBIDiseaseAdapter(BaseNERAdapter):
    def load(self) -> DatasetDict:
        # datasets<3.0 required: v3 removed loading-script support entirely.
        raw = load_dataset("ncbi_disease", trust_remote_code=True)

        def convert(example):
            example["bio"] = [_ID2BIO[t] for t in example["ner_tags"]]
            return example

        return DatasetDict({s: raw[s].map(convert) for s in raw})

    def label_list(self):
        return _LABELS
