"""Train a TF-IDF featurizer on train set of all clinical notes."""
import pickle as pkl
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from psycopmlutils.loaders.raw.load_text import LoadText
from psycopmlutils.utils import FEATURIZERS_PATH


def create_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_df: float = 0.95,
    min_df: float = 0.01,
    max_features: int = 100,
) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        lowercase=True,
        max_df=max_df,  # remove very common words
        min_df=min_df,  # remove very rare words
        max_features=max_features,
    )
    return vectorizer


if __name__ == "__main__":
    if not FEATURIZERS_PATH.exists():
        FEATURIZERS_PATH.mkdir()

    text = LoadText.load_all_notes(featurizer=None)["text"].tolist()

    for n_features in [100, 500, 1000]:
        vectorizer = create_tfidf_vectorizer(max_features=n_features)
        vectorizer.fit(text)

    with open(FEATURIZERS_PATH / f"tfidf_{n_features}.pkl", "wb") as f:
        pkl.dump(vectorizer, f)
