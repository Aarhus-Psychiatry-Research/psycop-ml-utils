"""Train a TF-IDF featurizer on train set of all clinical notes."""
from pathlib import Path
from typing import List, Tuple

import dill as pkl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wasabi import Printer

from psycopmlutils.loaders.raw import LoadIDs
from psycopmlutils.loaders.raw.load_text import LoadText
from psycopmlutils.utils import FEATURIZERS_PATH


def create_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_df: float = 0.95,
    min_df: float = 0.01,
    max_features: int = 100,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        tokenizer=lambda x: x.split(" "),
        lowercase=True,
        max_df=max_df,  # remove very common words
        min_df=min_df,  # remove very rare words
        max_features=max_features,
    )


def whitespace_tokenizer(text: str) -> List[str]:
    return text.split(" ")


if __name__ == "__main__":

    SYNTHETIC = True
    OVERTACI = False

    msg = Printer(timestamp=True)

    # Train TF-IDF model on all notes
    if OVERTACI:
        if not FEATURIZERS_PATH.exists():
            FEATURIZERS_PATH.mkdir()

        text = LoadText.load_all_notes(featurizer=None, n=None)
        # Subset only train set
        train_ids = LoadIDs.load(split="train")
        train_ids = train_ids["dw_ek_borger"].unique()
        text = text[text["dw_ek_borger"].isin(train_ids)]
        text = text["text"].tolist()

        for n_features in [100, 500, 1000]:
            msg.info(f"Fitting tf-idf with {n_features} features..")
            vectorizer = create_tfidf_vectorizer(max_features=n_features)
            vectorizer.fit(text)

            with open(FEATURIZERS_PATH / f"tfidf_{n_features}.pkl", "wb") as f:
                pkl.dump(vectorizer, f)

    # train TF-IDF on synthetic data
    if SYNTHETIC:
        test_path = Path("tests") / "test_data"
        save_dir = test_path / "test_tfidf"
        if not save_dir.exists():
            save_dir.mkdir()

        text = pd.read_csv(test_path / "synth_txt_data.csv")
        text = text["text"].dropna().tolist()

        msg.info("Fitting tf-idf with 10 features..")
        vectorizer = create_tfidf_vectorizer(max_features=10)
        vectorizer.fit(text)

        with open(save_dir / "tfidf_10.pkl", "wb") as f:
            pkl.dump(vectorizer, f)
