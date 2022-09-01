from typing import Dict, List, Optional, Union

from wasabi import Printer

from psycopmlutils.utils import data_loaders


def check_feature_combinations_return_correct_formatting(
    predictor_dict_list: List[Dict[str, Union[str, float, int]]],
    n: Optional[int] = 1_000,
    required_columns: Optional[List[str]] = ["dw_ek_borger", "timestamp", "value"],
    subset_duplicates_columns: Optional[List[str]] = ["dw_ek_borger", "timestamp"],
):
    """Test that all predictor_dfs in predictor_list return a valid df.

    Args:
        predictor_dict_list (Dict[str, Union[str, float, int]]): List of dictionaries where the key predictor_df maps to an SQL database.
        n (int, optional): Number of rows to test. Defaults to 1_000.
        required_columns (List[str], optional): List of required columns. Defaults to ["dw_ek_borger", "timestamp", "value"].
        subset_duplicates_columns (List[str], optional): List of columns to subset on when checking for duplicates. Defaults to ["dw_ek_borger", "timestamp"].
    """
    msg = Printer(timestamp=True)

    msg.info("Checking that feature combinations comform to correct formatting")

    # Get all unique dataframes sources in predictor_df_vals
    predictor_df_vals = [d["predictor_df"] for d in predictor_dict_list]
    predictor_df_vals = list(set(predictor_df_vals))

    msg.info(f"Loading {n} rows from each predictor_df")

    loader_fns_dict = data_loaders.get_all()

    failure_dicts = []

    for i, predictor_df_str in enumerate(predictor_df_vals):
        # Check that it returns a dataframe
        try:
            df = loader_fns_dict[predictor_df_str](n=n)
        except KeyError:
            msg.warn(
                f"{predictor_df_str} does not appear to be a loader function in catalogue, assuming a dataframe. Continuing.",
            )
            continue

        source_failures = []

        prefix = f"{i}/{len(predictor_df_vals)} {predictor_df_str}:"

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
            n_na = df[col].isna().sum()

            if n_na > 0:
                source_failures.append(f"{col}: {n_na} NaN")

        # Check for duplicates
        if df.duplicated(subset=subset_duplicates_columns).any():
            source_failures.append(f"Duplicates found on {subset_duplicates_columns}")

        # Return errors
        if len(source_failures) != 0:
            failure_dicts.append({predictor_df_str["predictor_df"]: source_failures})
            msg.fail(f"{prefix} failed, errors: {source_failures}")
        else:
            msg.good(f"{prefix} Conforms to criteria")

    if not failure_dicts:
        msg.good(
            f"Checked {len(predictor_df_vals)} predictor_dfs, all returned appropriate dfs",
        )
    else:
        raise ValueError(f"{failure_dicts}")
