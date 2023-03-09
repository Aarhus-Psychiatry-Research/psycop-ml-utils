"""Utilites for testing."""

from io import StringIO

import numpy as np
import pandas as pd
from pandas import DataFrame


def convert_cols_with_matching_colnames_to_datetime(
    df: DataFrame,
    colname_substr: str,
) -> DataFrame:
    """Convert columns that contain colname_substr in their name to datetimes
    Args:
        df (DataFrame): The df to convert. # noqa: DAR101
        colname_substr (str): Substring to match on. # noqa: DAR101

    Returns:
        DataFrame: The converted df
    """
    df.loc[:, df.columns.str.contains(colname_substr)] = df.loc[
        :,
        df.columns.str.contains(colname_substr),
    ].apply(pd.to_datetime)

    return df


def str_to_df(
    string: str,
    convert_timestamp_to_datetime: bool = True,
    convert_np_nan_to_nan: bool = True,
    convert_str_to_float: bool = False,
) -> DataFrame:
    """Convert a string representation of a dataframe to a dataframe.

    Args:
        string (str): A string representation of a dataframe.
        convert_timestamp_to_datetime (bool): Whether to convert the timestamp column to datetime. Defaults to True.
        convert_np_nan_to_nan (bool): Whether to convert np.nan to np.nan. Defaults to True.
        convert_str_to_float (bool): Whether to convert strings to floats. Defaults to False.

    Returns:
        DataFrame: A dataframe.
    """

    df = pd.read_table(StringIO(string), sep=",", index_col=False)

    if convert_timestamp_to_datetime:
        df = convert_cols_with_matching_colnames_to_datetime(df, "timestamp")

    if convert_np_nan_to_nan:
        # Convert "np.nan" str to the actual np.nan
        df = df.replace("np.nan", np.nan)

    if convert_str_to_float:
        # Convert all str to float
        df = df.apply(pd.to_numeric, axis=0, errors="coerce")

    # Drop "Unnamed" cols
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]
