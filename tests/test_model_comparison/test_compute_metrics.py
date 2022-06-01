from psycopmlutils.model_comparison import ModelComparison
import pytest

import pandas as pd


@pytest.fixture(scope="function")
def multiclass_df():
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "scores": [
                # id 1
                [0.8, 0.1, 0.05, 0.05],
                [0.4, 0.7, 0.1, 0.1],
                # id 2
                [0.1, 0.05, 0.8, 0.05],
                [0.1, 0.7, 0.1, 0.1],
                # id 3
                [0.1, 0.1, 0.7, 0.1],
                [0.2, 0.5, 0.2, 0.1],
                # id 4
                [0.1, 0.1, 0.2, 0.6],
                [0.1, 0.2, 0.1, 0.6],
            ],
            "label": ["ASD", "ASD", "DEPR", "DEPR", "TD", "TD", "SCHZ", "SCHZ"],
            "model_name": ["test"] * 8,
        }
    )


@pytest.fixture(scope="function")
def binary_df():
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "scores": [[0.8, 0.2], [0.5, 0.5], [0.6, 0.4], [0.9, 0.1]],
            "label": ["TD", "TD", "DEPR", "DEPR"],
            "optional_grouping1": ["grouping1"] * 4,
            "optional_grouping2": ["grouping2"] * 4,
        }
    )


@pytest.fixture(scope="function")
def binary_float_df():
    return pd.DataFrame({"scores": [0.6, 0.2, 0.8], "label": [1, 0, 0]})


@pytest.fixture(scope="function")
def multiclass_score_mapping():
    return {0: "ASD", 1: "DEPR", 2: "TD", 3: "SCHZ"}


@pytest.fixture(scope="function")
def binary_score_mapping():
    return {0: "TD", 1: "DEPR"}


def test_multiclass_transform_from_dataframe(multiclass_df, multiclass_score_mapping):
    model_comparer = ModelComparison(
        id2label=multiclass_score_mapping, id_col="id", metadata_cols="model_name"
    )

    res = model_comparer.transform_data_from_dataframe(multiclass_df)

    assert len(res["model_name"].unique()) == 1
    assert len(res["level"].unique()) == 2
    assert res.shape[0] == 40  # (3 metrics per class (4) + 7 overall) * 2


def test_binary_transform_from_dataframe(binary_df, binary_score_mapping):
    model_comparer = ModelComparison(
        id2label=binary_score_mapping,
        id_col="id",
        metadata_cols=["optional_grouping1", "optional_grouping2"],
    )

    res = model_comparer.transform_data_from_dataframe(binary_df)


def test_binary_transform_from_dataframe_with_float(binary_float_df):
    model_comparer = ModelComparison()

    res = model_comparer.transform_data_from_dataframe(binary_float_df)
    assert res[res["score_type"] == "acc"]["value"].values[0] == pytest.approx(0.666667)
