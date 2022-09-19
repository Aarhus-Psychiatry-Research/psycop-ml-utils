"""Generator for synth prediction data."""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from psycopmlutils.synth_data_generator.synth_col_generators import (
    create_outcome_values,
    generate_data_columns,
)
from psycopmlutils.synth_data_generator.utils import replace_vals_with_na


def generate_synth_data(
    predictors: dict,
    outcome_column_name: str,
    n_samples: int,
    logistic_outcome_model: str,
    intercept: Optional[float] = 0,
    na_prob: Optional[float] = 0.1,
    na_ignore_cols: Optional[list[str]] = None,
    prob_outcome: Optional[float] = 0.08,
    noise_mean_sd: Optional[tuple[float, float]] = (0, 1),
) -> pd.DataFrame:
    """Takes a dict and generates synth data from it.

    Args:
        predictors (dict): A dict representing each column. Key is col_name (str), values are column_type (str), output_type (float|int), min (int), max(int).
        outcome_column_name (str): Name of the outcome column.
        n_samples (int): Number of samples (rows) to generate.
        logistic_outcome_model (str): The statistical model used to generate outcome values, e.g. specified as'1*col_name+1*col_name2'.
        intercept (float, optional): The intercept of the logistic outcome model. Defaults to 0.
        na_prob (float, optional): Probability of changing a value in a predictor column
            to NA.
        na_ignore_cols (list[str], optional): Columns to ignore when creating NAs
        prob_outcome (float): Probability of a given row receiving "1" for the outcome.
        noise_mean_sd (tuple[float, float], optional): mean and sd of the noise.
            Increase SD to obtain more uncertain models.

    Returns:
        pd.DataFrame: The synthetic dataset
    """

    # Initialise dataframe
    df = pd.DataFrame(columns=list(predictors.keys()))

    # Generate data
    df = generate_data_columns(predictors=predictors, n_samples=n_samples, df=df)

    # Sigmoid it to get probabilities with mean = 0.5
    df[outcome_column_name] = create_outcome_values(
        n_samples=n_samples,
        logistic_outcome_model=logistic_outcome_model,
        intercept=intercept,
        noise_mean_sd=noise_mean_sd,
        df=df,
    )

    df[outcome_column_name] = np.where(df[outcome_column_name] < prob_outcome, 1, 0)

    # randomly replace predictors with NAs
    if na_prob:
        df = replace_vals_with_na(na_prob=na_prob, na_ignore_cols=na_ignore_cols, df=df)

    return df


if __name__ == "__main__":
    column_specifications = {
        "citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_001},
        "timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 5 * 365},
        "timestamp_outcome": {
            "column_type": "datetime_uniform",
            "min": 1 * 365,
            "max": 6 * 365,
        },
        "pred_hba1c_within_100_days_max_fallback_np.nan": {
            "column_type": "normal",
            "mean": 48,
            "sd": 5,
            "fallback": np.nan,
        },
        "pred_hdl_within_100_days_max_fallback_np.nan": {
            "column_type": "normal",
            "mean": 1,
            "sd": 0.5,
            "min": 0,
            "fallback": np.nan,
        },
    }

    synth_df = generate_synth_data(
        predictors=column_specifications,
        outcome_column_name="outc_dichotomous_t2d_within_30_days_max_fallback_0",
        n_samples=10_000,
        logistic_outcome_model="1*pred_hba1c_within_100_days_max_fallback_nan+1*pred_hdl_within_100_days_max_fallback_nan",
        prob_outcome=0.08,
    )

    synth_df.describe()

    save_path = Path(__file__).parent.parent.parent.parent
    synth_df.to_csv(save_path / "tests" / "test_data" / "synth_prediction_data.csv")


__all__ = [
    "column_specifications",
    "generate_synth_data",
    "save_path",
    "synth_df",
]
