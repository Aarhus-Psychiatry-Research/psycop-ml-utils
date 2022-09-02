from typing import Dict, List, Optional, Union

from wasabi import Printer

from psycopmlutils.utils import data_loaders


def check_feature_combinations_return_correct_formatting(
    predictor_dict_list: List[Dict[str, Union[str, float, int]]],
    n: Optional[int] = 100,
    required_columns: Optional[List[str]] = ["dw_ek_borger", "timestamp", "value"],
):
    """Test that all predictor_dfs in predictor_list return a valid df.

    Args:
        predictor_dict_list (Dict[str, Union[str, float, int]]): List of dictionaries where the key predictor_df maps to an SQL database.
        n (int, optional): Number of rows to test. Defaults to 100.
        required_columns (List[str], optional): List of required columns. Defaults to ["dw_ek_borger", "timestamp", "value"].
    """
    msg = Printer(timestamp=True)

    msg.info("Checking that feature combinations conform to correct formatting")

    # Get all unique dataframes sources in predictor_df_vals
    predictor_df_vals = [d["predictor_df"] for d in predictor_dict_list]
    predictor_df_vals = list(set(predictor_df_vals))

    msg.info(f"Loading {n} rows from each predictor_df")

    loader_fns_dict = data_loaders.get_all()

    failed = []

    for i, predictor_df_str in enumerate(predictor_df_vals):
        try:
            df = loader_fns_dict[predictor_df_str](n=n)
        except KeyError:
            msg.warn(
                f"{predictor_df_str} does not appear to be a loader function in catalogue, assuming a dataframe. Continuing.",
            )
            continue

        source_failures = []

        prefix = f"{i+1}/{len(predictor_df_vals)} {predictor_df_str}:"

        if df.shape[0] == 0:
            source_failures.append("No rows returned")

        for col in required_columns:
            if col not in df.columns:
                source_failures.append(f"{col} not in columns")

            if "timestamp" in col:
                # Check that column has a valid datetime format
                if df[col].dtype != "datetime64[ns]":
                    source_failures.append(f"{col} has invalid datetime format")

        if len(source_failures) != 0:
            msg.warn(f"{prefix} {source_failures}, agregating")
            failed.append({predictor_df_str: source_failures})
        else:
            msg.good(f"{prefix} Conforms to criteria")

    if not failed:
        msg.good(
            f"Checked {len(predictor_df_vals)} predictor_dfs, all returned appropriate dfs",
        )
    else:
        raise ValueError(failed)
