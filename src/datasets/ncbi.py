from datasets import load_dataset, DatasetDict

from .base import BaseNERAdapter

# bigbio NER schema: tags is already a list of BIO strings (O, B-Disease, I-Disease)
_HARMONIZE = {"B-Disease": "B-Disease", "I-Disease": "I-Disease", "O": "O"}
_LABELS = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]


class NCBIDiseaseAdapter(BaseNERAdapter):
    def load(self) -> DatasetDict:
        # ncbi_disease uses a deprecated loading script; bigbio/ncbi_disease ships as Parquet.
        raw = load_dataset(
            "bigbio/ncbi_disease", name="ncbi_disease_bigbio_ner", trust_remote_code=True
        )

        def convert(example):
            example["bio"] = [_HARMONIZE.get(t, "O") for t in example["tags"]]
            return example

        return DatasetDict({s: raw[s].map(convert) for s in raw})

    def label_list(self):
        return _LABELS
