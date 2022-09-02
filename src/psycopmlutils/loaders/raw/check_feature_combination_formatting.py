from typing import Dict, List, Union

from wasabi import Printer

from psycopmlutils.loaders.raw.check_raw_df import check_raw_df
from psycopmlutils.utils import data_loaders


def check_feature_combinations_return_correct_dfs(
    predictor_dict_list: List[Dict[str, Union[str, float, int]]],
    n: int = 1_000,
    required_columns: List[str] = ["dw_ek_borger", "timestamp", "value"],
    subset_duplicates_columns: List[str] = ["dw_ek_borger", "timestamp", "value"],
    allowed_nan_value_prop: float = 0.01,
):
    """Test that all predictor_dfs in predictor_list return a valid df.

    Args:
        predictor_dict_list (Dict[str, Union[str, float, int]]): List of dictionaries where the key predictor_df maps to an SQL database.
        n (int): Number of rows to test. Defaults to 1_000.
        required_columns (List[str]): List of required columns. Defaults to ["dw_ek_borger", "timestamp", "value"].
        subset_duplicates_columns (List[str]): List of columns to subset on when checking for duplicates. Defaults to ["dw_ek_borger", "timestamp"].
        allowed_nan_value_prop (float): Allowed proportion of missing values. Defaults to 0.0.
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

        allowed_nan_value_prop = (
            d["allowed_nan_value_prop"]
            if d["allowed_nan_value_prop"]
            else allowed_nan_value_prop
        )

        source_failures, duplicates = check_raw_df(
            df=df,
            required_columns=required_columns,
            subset_duplicates_columns=subset_duplicates_columns,
            allowed_nan_value_prop=allowed_nan_value_prop,
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
