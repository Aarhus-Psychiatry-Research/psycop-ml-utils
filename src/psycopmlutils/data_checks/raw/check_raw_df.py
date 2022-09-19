"""Check that any raw df conforms to the required format."""

from typing import List, Optional

import pandas as pd


def check_raw_df(  # pylint: disable=too-many-branches
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    subset_duplicates_columns: Optional[List[str]] = None,
    allowed_nan_value_prop: float = 0.0,
    expected_val_dtypes: Optional[List[str]] = None,
    raise_error: bool = True,
) -> List[str]:
    """Check that the raw df conforms to the required format and doesn't
    contain duplicates or missing values.

    Args:
        df (pd.DataFrame): Dataframe to check.
        required_columns (List[str]): List of required columns. Defaults to ["dw_ek_borger", "timestamp", "value"].
        subset_duplicates_columns (List[str]): List of columns to subset on when checking for duplicates. Defaults to ["dw_ek_borger", "timestamp"].
        allowed_nan_value_prop (float): Allowed proportion of missing values. Defaults to 0.0.
        expected_val_dtypes (List[str]): Expected dtype of value column. Defaults to "float64".
        raise_error (bool): Whether to raise an error if the df fails the checks. Defaults to True.

    Returns:
        List[str]: List of errors.

    Raises:
        ValueError: If the df fails the checks and raise_error is True.
    """
    source_failures = []

    if required_columns is None:
        required_columns = ["dw_ek_borger", "timestamp", "value"]

    if subset_duplicates_columns is None:
        subset_duplicates_columns = ["dw_ek_borger", "timestamp", "value"]

    if expected_val_dtypes is None:
        expected_val_dtypes = ["float64", "int64"]

    # Check that the dataframe has a meaningful length
    if df.shape[0] == 0:
        source_failures.append("No rows returned")

    # Check that the dataframe has the required columns
    for col in required_columns:
        if col not in df.columns:
            source_failures.append(f"{col}: not in columns")
            continue

        # Check that columns are correctly formatted
        if "timestamp" in col:
            # Check that column has a valid datetime format
            if df[col].dtype != "datetime64[ns]":
                source_failures.append(f"{col}: invalid datetime format")
        elif "value" in col:
            # Check that column has a valid numeric format
            if df[col].dtype not in expected_val_dtypes:
                source_failures.append(
                    f"{col}: dtype {df[col].dtype}, expected {expected_val_dtypes}",
                )

        # Check for NaN in cols
        na_prop = df[col].isna().sum() / df.shape[0]

        if na_prop > 0:
            if col != "value":
                source_failures.append(f"{col}: {na_prop} NaN")
            else:
                if na_prop > allowed_nan_value_prop:
                    source_failures.append(
                        f"{col}: {na_prop} NaN (allowed {allowed_nan_value_prop})",
                    )

    # Check for duplicates
    duplicates_idx = df.duplicated(subset=subset_duplicates_columns, keep=False)

    if duplicates_idx.any():
        source_failures.append(f"Duplicates found on {subset_duplicates_columns}")
        duplicates = df[duplicates_idx]
    else:
        duplicates = None

    # Raise error if any failures
    if raise_error and len(source_failures) > 0:
        raise ValueError(source_failures)

    return source_failures, duplicates
