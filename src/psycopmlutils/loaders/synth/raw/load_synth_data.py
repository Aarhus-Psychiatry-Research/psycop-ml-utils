"""Loaders for synth data."""

from pathlib import Path
from typing import Optional

import pandas as pd

from psycopmlutils.utils import data_loaders


def load_raw_test_csv(filename: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load raw csv.

    Args:
        filename (str): Name of the file to load.
        n_rows (int, optional): Number of rows to load. Defaults to None.
    """
    # Get project root dir
    project_root = Path(__file__).resolve().parents[5]

    df = pd.read_csv(
        project_root / "tests" / "test_data" / "raw" / filename,
        nrows=n_rows,
    )

    # Convert timestamp col to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


@data_loaders.register("synth_predictor")
def synth_predictor(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_raw_float_1.csv", n_rows=n_rows)


@data_loaders.register("synth_outcome")
def load_synth_outcome(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_raw_float_2.csv", n_rows=n_rows)


@data_loaders.register("synth_prediction_times")
def load_synth_prediction_times(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_prediction_times.csv", n_rows=n_rows)