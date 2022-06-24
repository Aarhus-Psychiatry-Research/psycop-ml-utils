import numpy as np
import pandas as pd
from utils_for_testing import (
    assert_flattened_outcome_as_expected,
    assert_flattened_predictor_as_expected,
    str_to_df,
)

from psycopmlutils.timeseriesflattener import (
    FlattenedDataset,
    create_feature_combinations,
)


def test_min_date():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:01
                            1,2023-12-31 00:00:01
                            """

    expected_df_str = """dw_ek_borger,timestamp,
                            1,2023-12-31 00:00:01
                            """

    prediction_times_df = str_to_df(prediction_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        min_date=pd.Timestamp(2022, 12, 31),
        n_workers=4,
    )

    outcome_df = flattened_dataset.df

    for col in [
        "timestamp",
    ]:
        pd.testing.assert_series_equal(
            outcome_df[col].reset_index(drop=True),
            expected_df[col].reset_index(drop=True),
        )
