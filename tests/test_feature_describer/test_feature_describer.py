import numpy as np
import pandas as pd
import pytest

from psycopmlutils.data_checks.flattened.feature_describer import (
    generate_feature_description_df,
    generate_feature_description_row,
)
from psycopmlutils.utils import generate_feature_colname


@pytest.fixture()
def predictor_dicts():
    PREDICTOR_DICTS = [
        {
            "predictor_df": "hba1c",
            "lookbehind_days": 100,
            "resolve_multiple": "max",
            "fallback": np.nan,
        },
        {
            "predictor_df": "hdl",
            "lookbehind_days": 100,
            "resolve_multiple": "max",
            "fallback": np.nan,
        },
    ]

    return PREDICTOR_DICTS


@pytest.fixture()
def df():
    """Load the synthetic data set."""
    return pd.read_csv("tests/test_data/synth_prediction_data.csv")


def test_load_dataset(df):
    """Check loading of synthetic dataset."""
    assert df.shape[0] == 10_000


def test_generate_feature_description_row(df, predictor_dicts):
    d = predictor_dicts[0]

    column_name = generate_feature_colname(
        prefix="pred",
        out_col_name=d["predictor_df"],
        interval_days=d["lookbehind_days"],
        resolve_multiple=d["resolve_multiple"],
        fallback=d["fallback"],
    )

    generate_feature_description_row(series=df[column_name], predictor_dict=d)

    generate_feature_description_df(df, predictor_dicts)
