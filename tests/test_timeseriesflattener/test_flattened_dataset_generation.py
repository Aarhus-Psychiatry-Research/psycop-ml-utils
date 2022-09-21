"""Integration test for the flattened dataset generation."""

# pylint: disable=unused-import


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psycopmlutils.loaders.synth.raw.load_synth_data import (
    load_synth_outcome,
    load_synth_prediction_times,
    synth_predictor,
)
from psycopmlutils.timeseriesflattener.flattened_dataset import FlattenedDataset
from psycopmlutils.utils import load_most_recent_df_matching_pattern


def check_dfs_have_same_contents_by_column(df1, df2):
    """Check that two dataframes have the same contents by column.

    Makes debugging much easier, as it generates a diff df which is easy to read.

    Args:
        df1 (pd.DataFrame): First dataframe.
        df2 (pd.DataFrame): Second dataframe.

    Raises:
        AssertionError: If the dataframes don't have the same contents by column.
    """

    cols_to_test = [c for c in df1.columns if "prediction_time_uuid" not in c]

    for col in cols_to_test:
        # Saving to csv rounds floats, so we need to round here too
        # to avoid false negatives. Otherwise, it thinks the .csv
        # file has different values from the generated_df, simply because
        # generated_df has more decimal places.
        for df in (df1, df2):
            if df[col].dtype not in (np.dtype("O"), np.dtype("<M8[ns]")):
                df[col] = df[col].round(4)

        merged_df = df1.merge(
            df2,
            indicator=True,
            how="outer",
            on=[col, "prediction_time_uuid"],
            suffixes=("_first", "_cache"),
        )

        # Get diff rows
        diff_rows = merged_df[merged_df["_merge"] != "both"]

        # Sort rows and columns for easier comparison
        diff_rows = diff_rows.sort_index(axis=1)
        diff_rows = diff_rows.sort_values(by=[col, "prediction_time_uuid"])

        # Set display options
        pd.options.display.width = 0

        assert len(diff_rows) == 0


def create_flattened_df(cache_dir, predictor_combinations, prediction_times_df):
    """Create a dataset df for testing."""
    first_df = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        n_workers=4,
        feature_cache_dir=cache_dir,
    )
    first_df.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictor_combinations,
    )

    return first_df.df


def delete_dir_with_contents(temp_dir: Path):
    """Delete a directory and all its contents."""
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            file.unlink()

        temp_dir.rmdir()


def test_cache_hitting():
    """Test that the cache is hit when the same data is requested twice."""

    for _ in range(10):
        # Get project root dir
        project_root = Path(__file__).resolve().parents[2]

        temp_dir = project_root / "tests" / "test_data" / "temp"

        # Delete temp dir
        delete_dir_with_contents(temp_dir)

        predictor_combinations = [
            {
                "predictor_df": "synth_predictor",
                "lookbehind_days": 365,
                "resolve_multiple": "max",
                "fallback": np.NaN,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "synth_predictor",
                "lookbehind_days": 730,
                "resolve_multiple": "max",
                "fallback": np.NaN,
                "allowed_nan_value_prop": 0.0,
            },
        ]

        # Create temp dir for testing
        temp_dir.mkdir(exist_ok=True)

        prediction_times_df = load_synth_prediction_times()

        # Create the cache
        first_df = create_flattened_df(
            cache_dir=temp_dir,
            predictor_combinations=predictor_combinations,
            prediction_times_df=prediction_times_df,
        )

        # Load the cache
        cache_df = create_flattened_df(
            cache_dir=temp_dir,
            predictor_combinations=predictor_combinations,
            prediction_times_df=prediction_times_df,
        )

        # If cache_df doesn't hit the cache, it creates its own files
        # Thus, number of files is an indicator of whether the cache was hit
        assert len(list(temp_dir.glob("*"))) == len(predictor_combinations)

        # Assert that each column has the same contents
        check_dfs_have_same_contents_by_column(first_df, cache_df)

        # Delete temp dir
        delete_dir_with_contents(temp_dir)


def load_cache(temp_dir, predictor_combinations, prediction_times_df):
    """Create a new dataset which should hit the cache."""
    cache_df = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        n_workers=4,
        feature_cache_dir=temp_dir,
    )

    cache_df.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictor_combinations,
    )

    return cache_df
