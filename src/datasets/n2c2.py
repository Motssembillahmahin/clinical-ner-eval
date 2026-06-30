from .base import BaseNERAdapter


class N2C2Adapter(BaseNERAdapter):
    """Gated adapter stub — requires n2c2 2010 DUA acceptance."""

    def load(self):
        raise NotImplementedError(
            "n2c2 2010 data is gated. Accept the DUA at "
            "https://portal.dbmi.hms.harvard.edu/ and place the raw data under data/n2c2/."
        )

    def label_list(self):
        return ["O", "B-problem", "I-problem", "B-test", "I-test",
                "B-treatment", "I-treatment"]
