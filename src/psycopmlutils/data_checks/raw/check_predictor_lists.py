"""Check that all feature_dicts conform to correct formatting.

Also check that they return meaningful dictionaries.
"""
from typing import Dict, List, Optional, Union

from wasabi import Printer

from psycopmlutils.data_checks.raw.check_raw_df import check_raw_df
from psycopmlutils.utils import data_loaders


def check_feature_combinations_return_correct_dfs(  # pylint: disable=too-many-branches
    predictor_dict_list: Optional[List[Dict[str, Union[str, float, int]]]],
    n_rows: int = 1_000,
    required_columns: Optional[List[str]] = None,
    subset_duplicates_columns: Optional[List[str]] = None,
    allowed_nan_value_prop: float = 0.01,
    expected_val_dtypes: Optional[List[str]] = None,
):
    """Test that all predictor_dfs in predictor_list return a valid df.

    Args:
        predictor_dict_list (List[Dict[str, Union[str, float, int]]]): List of dictionaries
            where the key predictor_df maps to a catalogue registered data loader
            or is a valid dataframe.
        n_rows (int): Number of rows to test. Defaults to 1_000.
        required_columns (List[str]): List of required columns. Defaults to ["dw_ek_borger", "timestamp", "value"].
        subset_duplicates_columns (List[str]): List of columns to subset on when
            checking for duplicates. Defaults to ["dw_ek_borger", "timestamp"].
        allowed_nan_value_prop (float): Allowed proportion of missing values. Defaults to 0.0.
        expected_val_dtypes (List[str]): Expected value dtype. Defaults to ["float64", "int64"].
    """

    if required_columns is None:
        required_columns = ["dw_ek_borger", "timestamp", "value"]

    if subset_duplicates_columns is None:
        subset_duplicates_columns = ["dw_ek_borger", "timestamp", "value"]

    if expected_val_dtypes is None:
        expected_val_dtypes = ["float64", "int64"]

    msg = Printer(timestamp=True)

    msg.info("Checking that feature combinations conform to correct formatting")

    # Find all dicts that are unique on keys predictor_df and allowed_nan_value_prop
    unique_subset_dicts = []

    required_keys = ["predictor_df", "allowed_nan_value_prop"]

    dicts_with_subset_keys = []

    for d in predictor_dict_list:
        new_d = {k: d[k] for k in required_keys}

        if "loader_kwargs" in d:
            new_d["loader_kwargs"] = d["loader_kwargs"]

        dicts_with_subset_keys.append(new_d)

    for predictor_dict in dicts_with_subset_keys:
        if predictor_dict not in unique_subset_dicts:
            unique_subset_dicts.append(predictor_dict)

    msg.info(f"Loading {n_rows} rows from each predictor_df")

    loader_fns_dict = data_loaders.get_all()

    failure_dicts = []

    for i, d in enumerate(unique_subset_dicts):  # pylint: disable=invalid-name
        # Check that it returns a dataframe

        try:
            if "loader_kwargs" in d:
                df = loader_fns_dict[d["predictor_df"]](
                    n_rows=n_rows, **d["loader_kwargs"]
                )
            else:
                df = loader_fns_dict[d["predictor_df"]](n_rows=n_rows)
        except KeyError:
            msg.warn(
                f"{d['predictor_df']} does not appear to be a loader function in catalogue, assuming a well-formatted dataframe. Continuing.",
            )
            continue

        prefix = f"{i+1}/{len(unique_subset_dicts)} {d['predictor_df']}:"

        allowed_nan_value_prop = (
            d["allowed_nan_value_prop"]
            if d["allowed_nan_value_prop"]
            else allowed_nan_value_prop
        )

        source_failures, _ = check_raw_df(
            df=df,
            required_columns=required_columns,
            subset_duplicates_columns=subset_duplicates_columns,
            allowed_nan_value_prop=allowed_nan_value_prop,
            expected_val_dtypes=expected_val_dtypes,
        )

        # Return errors
        if len(source_failures) != 0:
            failure_dicts.append({d["predictor_df"]: source_failures})
            msg.fail(f"{prefix} errors: {source_failures}")
        else:
            msg.good(
                f"{prefix} passed data validation criteria.",
            )

    if not failure_dicts:
        msg.good(
            f"Checked {len(unique_subset_dicts)} predictor_dfs, all returned appropriate dfs",
        )
    else:
        raise ValueError(f"{failure_dicts}")
