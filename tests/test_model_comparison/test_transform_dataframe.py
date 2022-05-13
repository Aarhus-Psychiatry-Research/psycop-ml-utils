from psycopmlutils.model_comparison import ModelComparison
import pytest

import pandas as pd

from pathlib import Path


def test_transform_folder():
    folder = Path("tests") / "test_model_comparison" / "test_data"
    metadata_cols = ["model_name", "split", "type", "binary"]

    dfs = []
    for diagnosis in ["DEPR", "ASD", "SCHZ"]:
        score_mapping = {0: diagnosis, 1: "TD"}
        model_comparer = ModelComparison(
            id_col="id", score_mapping=score_mapping, metadata_cols=metadata_cols
        )
        df = model_comparer.transform_data_from_folder(folder, f"*{diagnosis}*.jsonl")
        dfs.append(df)

    dfs = pd.concat(dfs)
    dfs.to_json("binary_baselines.jsonl", orient="records", lines=True)
