from typing import Dict, List, Union

import pandas as pd
from wasabi import Printer

from psycopmlutils.utils import data_loaders


def check_feature_combinations_return_correct_dfs(
    predictor_dict_list: List[Dict[str, Union[str, float, int]]],
    n: int = 1_000,
    required_columns: List[str] = ["dw_ek_borger", "timestamp", "value"],
    subset_duplicates_columns: List[str] = ["dw_ek_borger", "timestamp", "value"],
):
    """Test that all predictor_dfs in predictor_list return a valid df.

    Args:
        predictor_dict_list (Dict[str, Union[str, float, int]]): List of dictionaries where the key predictor_df maps to an SQL database.
        n (int): Number of rows to test. Defaults to 1_000.
        required_columns (List[str]): List of required columns. Defaults to ["dw_ek_borger", "timestamp", "value"].
        subset_duplicates_columns (List[str]): List of columns to subset on when checking for duplicates. Defaults to ["dw_ek_borger", "timestamp"].
    """
    msg = Printer(timestamp=True)

    msg.info("Checking that feature combinations conform to correct formatting")

    # Find all dicts that are unique on keys predictor_df and allowed_nan_value_prop
    unique_subset_dicts = []

    dicts_with_subset_keys = [
        {k: bigdict[k] for k in ("predictor_df", "allowed_nan_value_prop")}
        for bigdict in predictor_dict_list
    ]

    for predictor_dict in dicts_with_subset_keys:
        if predictor_dict not in unique_subset_dicts:
            unique_subset_dicts.append(predictor_dict)

    msg.info(f"Loading {n} rows from each predictor_df")

    loader_fns_dict = data_loaders.get_all()

    failure_dicts = []

    for i, d in enumerate(unique_subset_dicts):
        # Check that it returns a dataframe

        try:
            df = loader_fns_dict[d["predictor_df"]](n=n)
        except KeyError:
            msg.warn(
                f"{d['predictor_df']} does not appear to be a loader function in catalogue, assuming a dataframe. Continuing.",
            )
            continue

        prefix = f"{i+1}/{len(unique_subset_dicts)} {d['predictor_df']}:"

        source_failures, duplicates = check_raw_df(
            df=df,
            required_columns=required_columns,
            subset_duplicates_columns=subset_duplicates_columns,
            allowed_nan_value_prop=d["allowed_nan_value_prop"],
        )

        # Return errors
        if len(source_failures) != 0:
            failure_dicts.append({d["predictor_df"]: source_failures})
            msg.fail(f"{prefix} errors: {source_failures}")
        else:
            msg.good(f"{prefix} Conforms to criteria")

    if not failure_dicts:
        msg.good(
            f"Checked {len(unique_subset_dicts)} predictor_dfs, all returned appropriate dfs",
        )
    else:
        raise ValueError(f"{failure_dicts}")


def check_raw_df(
    df: pd.DataFrame,
    required_columns: List[str] = ["dw_ek_borger", "timestamp", "value"],
    subset_duplicates_columns: List[str] = ["dw_ek_borger", "timestamp", "value"],
    allowed_nan_value_prop: float = 0.0,
) -> List[str]:
    """Check that the raw df conforms to the required format and doesn't
    contain duplicates or missing values.

    Args:
        df (pd.DataFrame): Dataframe to check.
        required_columns (List[str]): List of required columns. Defaults to ["dw_ek_borger", "timestamp", "value"].
        subset_duplicates_columns (List[str]): List of columns to subset on when checking for duplicates. Defaults to ["dw_ek_borger", "timestamp"].
        allowed_nan_value_prop (float): Allowed proportion of missing values. Defaults to 0.0.

    Returns:
        List[str]: List of errors.
    """
    source_failures = []

    # Check that the dataframe has a meaningful length
    if df.shape[0] == 0:
        source_failures.append("No rows returned")

        # Check that the dataframe has the required columns
    for col in required_columns:
        if col not in df.columns:
            source_failures.append(f"{col}: not in columns")

            # Check that columns are correctly formatted
            if "timestamp" in col:
                # Check that column has a valid datetime format
                if col.dtype not in ("datetime64[ns]"):
                    source_failures.append(f"{col}: invalid datetime format")

            # Check for NaN in cols
        na_prop = round(df[col].isna().sum() / df.shape[0], 2)

        if na_prop > 0:
            if col != "value":
                source_failures.append(f"{col}: {na_prop}% NaN")
            else:
                if na_prop > allowed_nan_value_prop:
                    source_failures.append(
                        f"{col}: {na_prop}% NaN (allowed {allowed_nan_value_prop}%)",
                    )

    # Check for duplicates
    duplicates_idx = df.duplicated(subset=subset_duplicates_columns, keep=False)

    if duplicates_idx.any():
        source_failures.append(f"Duplicates found on {subset_duplicates_columns}")
        duplicates = df[duplicates_idx]
    else:
        duplicates = None

    return source_failures, duplicates
