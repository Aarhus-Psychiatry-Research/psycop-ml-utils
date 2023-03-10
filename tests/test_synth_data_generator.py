"""Generate synth data with outcome."""

import numpy as np
from psycop_ml_utils.synth_data_generator.synth_prediction_times_generator import (
    generate_synth_data,
)
from psycop_ml_utils.utils import PROJECT_ROOT


def test_synth_data_generator():
    """Test synth data generator."""
    WRITE_NEW_TO_DISK = False

    column_specifications = [
        {"citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_001}},
        {"timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 5 * 365}},
        {
            "timestamp_outcome": {
                "column_type": "datetime_uniform",
                "min": 1 * 365,
                "max": 6 * 365,
            },
        },
        {
            "pred_hba1c_within_100_days_max_fallback_np.nan": {
                "column_type": "normal",
                "mean": 48,
                "sd": 5,
                "fallback": np.nan,
            },
        },
        {
            "pred_hdl_within_100_days_max_fallback_np.nan": {
                "column_type": "normal",
                "mean": 1,
                "sd": 0.5,
                "min": 0,
                "fallback": np.nan,
            },
        },
    ]

    n_samples = 10_000

    synth_df = generate_synth_data(
        predictors=column_specifications,
        outcome_column_name="outc_dichotomous_t2d_within_30_days_max_fallback_0",
        n_samples=n_samples,
        logistic_outcome_model="1*pred_hba1c_within_100_days_max_fallback_nan+1*pred_hdl_within_100_days_max_fallback_nan",
        prob_outcome=0.08,
        na_prob=0.1,
    )

    synth_df.describe()

    assert synth_df.shape == (n_samples, len(column_specifications) + 1)

    # Get proportion of all values in dataframe that are NA
    na_prop = synth_df.isna().sum().sum() / synth_df.size

    assert na_prop > 0.02 and na_prop < 0.2

    if WRITE_NEW_TO_DISK:
        save_path = PROJECT_ROOT
        synth_df.to_csv(save_path / "tests" / "test_data" / "synth_prediction_data.csv")
