from typing import List

import dill as pkl

from psycopmlutils.utils import FEATURIZERS_PATH


def get_tfidf_vocab(n_features: int) -> List[str]:
    with open(FEATURIZERS_PATH / f"tfidf_{str(n_features)}.txt", "r") as f:
        return f.read().splitlines()


TFIDF_100_VOCAB = get_tfidf_vocab(100)
TFIDF_500_VOCAB = get_tfidf_vocab(500)
TFIDF_1000_VOCAB = get_tfidf_vocab(1000)


if __name__ == "__main__":

    for n_features in [100, 500, 1000]:
        with open(FEATURIZERS_PATH / f"tfidf_{n_features}.pkl", "rb") as f:
            tfidf = pkl.load(f)
        vocab = tfidf.get_feature_names()
        vocab = ["tfidf-" + word for word in vocab]
        with open(FEATURIZERS_PATH / f"tfidf_{n_features}_vocab.txt", "w") as f:
            f.write("\n".join(vocab))
