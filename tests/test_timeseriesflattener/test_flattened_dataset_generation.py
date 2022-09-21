"""Integration test for the flattened dataset generation."""

# pylint: disable=unused-import, redefined-outer-name


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
from src.application.t2d.generate_features_and_write_to_disk import (
    log_to_wandb,
    save_feature_set_description_to_disk,
    split_and_save_to_disk,
)


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


def recursively_delete_dir_with_contents(dir_path: Path):
    """Delete a directory and all its contents, even if those contents are
    directories themselves."""
    for item in dir_path.iterdir():
        if item.is_dir():
            recursively_delete_dir_with_contents(item)
        else:
            item.unlink()

    dir_path.rmdir()


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


@pytest.fixture(scope="function")
def synth_prediction_times():
    """Load the prediction times."""
    return load_synth_prediction_times()


@pytest.fixture(scope="function")
def base_predictor_combinations():
    """Basic predictor combinations."""

    return [
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


@pytest.fixture(scope="function")
def temp_dir_path():
    """Get temp dir path."""
    # Get project root dir
    project_root = Path(__file__).resolve().parents[2]

    return project_root / "tests" / "test_data" / "temp"


def init_temp_dir(temp_dir_path):
    """Create a temp dir for testing."""
    # Delete temp dir
    if temp_dir_path.exists():
        recursively_delete_dir_with_contents(temp_dir_path)

    return temp_dir_path.mkdir(exist_ok=True)


def test_cache_hitting(
    temp_dir_path,
    synth_prediction_times,
    base_predictor_combinations,
):
    """Test that the cache is hit when the same data is requested twice."""

    # Create temp dir for testing
    init_temp_dir(temp_dir_path)

    # Create the cache
    first_df = create_flattened_df(
        cache_dir=temp_dir_path,
        predictor_combinations=base_predictor_combinations,
        prediction_times_df=synth_prediction_times,
    )

    # Load the cache
    cache_df = create_flattened_df(
        cache_dir=temp_dir_path,
        predictor_combinations=base_predictor_combinations,
        prediction_times_df=synth_prediction_times,
    )

    # If cache_df doesn't hit the cache, it creates its own files
    # Thus, number of files is an indicator of whether the cache was hit
    assert len(list(temp_dir_path.glob("*"))) == len(base_predictor_combinations)

    # Assert that each column has the same contents
    check_dfs_have_same_contents_by_column(first_df, cache_df)

    # Delete temp dir
    recursively_delete_dir_with_contents(temp_dir_path)


def test_all_non_online_elements_in_pipeline(
    temp_dir_path,
    synth_prediction_times,
    base_predictor_combinations,
):
    """Test that the splitting and saving to disk works as expected."""

    init_temp_dir(temp_dir_path=temp_dir_path)

    flattened_df = create_flattened_df(
        cache_dir=None,
        predictor_combinations=base_predictor_combinations,
        prediction_times_df=synth_prediction_times,
    )

    split_ids = {}

    start_idx = 0

    # Get the first 20% of the IDs
    splits = ["train", "test", "val"]

    for split in splits:
        prop_per_split = 0.2
        end_idx = int(start_idx + len(flattened_df) * prop_per_split)

        # Get 20% of the dataframe
        ids = flattened_df.iloc[start_idx:end_idx]

        split_ids[split] = ids

        start_idx = end_idx

    split_and_save_to_disk(
        flattened_df=flattened_df,
        out_dir=temp_dir_path,
        file_prefix="integration",
        split_ids_dict=split_ids,
        splits=splits,
    )

    save_feature_set_description_to_disk(
        predictor_combinations=base_predictor_combinations,
        flattened_csv_dir=temp_dir_path,
        out_dir=temp_dir_path,
    )

    recursively_delete_dir_with_contents(dir_path=temp_dir_path)
