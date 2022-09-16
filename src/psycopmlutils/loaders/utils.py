"""Example of how to get tfidf vocab"""

from typing import List

from psycopmlutils.utils import FEATURIZERS_PATH


def get_tfidf_vocab(n_features: int) -> List[str]:
    with open(  # pylint: disable=unspecified-encoding
        FEATURIZERS_PATH / f"tfidf_{str(n_features)}.txt", "r"
    ) as f:
        return f.read().splitlines()


TFIDF_VOCAB = {n: get_tfidf_vocab(n) for n in [100, 500, 1000]}
