"""Pre-load dataframes to avoid duplicate loading."""

from multiprocessing import Pool
from typing import Union

import pandas as pd
from wasabi import Printer

from psycopmlutils.data_checks.raw.check_raw_df import check_raw_df
from psycopmlutils.utils import data_loaders


def load_df(predictor_df: str, values_to_load: str = None) -> pd.DataFrame:
    """Load a dataframe from a SQL database.

    Args:
        predictor_df (str): The name of the loader function which calls the SQL database.
        values_to_load (dict): Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". Defaults to None.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    msg = Printer(timestamp=True)

    msg.info(f"Loading {predictor_df}")

    loader_fns = data_loaders.get_all()

    if predictor_df not in loader_fns:
        msg.fail(f"Could not find loader for {predictor_df}.")
    else:
        # We need this control_flow since some loader_fns don't take values_to_load
        if values_to_load is not None:
            df = loader_fns[predictor_df](values_to_load=values_to_load)
        else:
            df = loader_fns[predictor_df]()

    msg.info(f"Loaded {predictor_df} with {len(df)} rows")
    return {predictor_df: df}


def load_df_wrapper(predictor_dict: dict[str, Union[str, float, int]]) -> pd.DataFrame:
    """Wrapper to load a dataframe from a dictionary.

    Args:
        predictor_dict (dict[str, Union[str, float, int]]): dictionary where the key predictor_df maps to an SQL database.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    return load_df(
        predictor_df=predictor_dict["predictor_df"],
        values_to_load=predictor_dict.get("values_to_load"),
    )


def error_check_dfs(pre_loaded_dfs: list[dict[str, pd.DataFrame]]) -> None:
    """Error check the pre-loaded dataframes.

    Args:
        pre_loaded_dfs (list): list of pre-loaded dataframes.
    """
    # Error check the laoded dfs
    failures = []

    msg = Printer(timestamp=True)

    for d in pre_loaded_dfs:
        for k in d.keys():
            source_failures, _ = check_raw_df(df=d[k], raise_error=False)

            if len(source_failures) > 0:
                failures.append({k: source_failures})

    if len(failures) > 0:
        raise ValueError(
            f"Pre-loaded dataframes failed source checks. {source_failures}",
        )

    msg.info(f"Pre-loaded {len(pre_loaded_dfs)} dataframes, all conformed to criteria")


def pre_load_unique_dfs(
    unique_predictor_dict_list: list[dict[str, Union[str, float, int]]],
) -> dict[str, pd.DataFrame]:
    """Pre-load unique dataframes to avoid duplicate loading.

    Args:
        unique_predictor_dict_list (list[dict[str, Union[str, float, int]]]): list of dictionaries where the key predictor_df maps to an SQL database.

    Returns:
        dict[str, pd.DataFrame]: A dictionary with keys predictor_df and values the loaded dataframe.
    """

    # Get unique predictor_df values from predictor_dict_list
    unique_dfs = {x["predictor_df"] for x in unique_predictor_dict_list}

    msg = Printer(timestamp=True)

    msg.info(f"Pre-loading {len(unique_dfs)} dataframes")
    n_workers = min(len(unique_dfs), 16)

    with Pool(n_workers) as p:
        pre_loaded_dfs = p.map(load_df_wrapper, unique_predictor_dict_list)

        error_check_dfs(pre_loaded_dfs=pre_loaded_dfs)

        # Combined pre_loaded dfs into one dictionary
        pre_loaded_dfs = {k: v for d in pre_loaded_dfs for k, v in d.items()}

    return pre_loaded_dfs
