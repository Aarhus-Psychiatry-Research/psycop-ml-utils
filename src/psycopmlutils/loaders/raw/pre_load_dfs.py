"""Pre-load dataframes to avoid duplicate loading."""

from multiprocessing import Pool
from typing import Dict, List, Union

import pandas as pd
from wasabi import Printer

from psycopmlutils.data_checks.raw.check_raw_df import check_raw_df
from psycopmlutils.utils import data_loaders


def pre_load_unique_dfs(
    unique_predictor_dict_list: List[Dict[str, Union[str, float, int]]],
) -> Dict[str, pd.DataFrame]:
    """Pre-load unique dataframes to avoid duplicate loading.

    Args:
        predictor_dict_list (List[Dict[str, Union[str, float, int]]]): List of dictionaries where the key predictor_df maps to an SQL database.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary with keys predictor_df and values the loaded dataframe.
    """

    # Get unique predictor_df values from predictor_dict_list
    unique_dfs = {x["predictor_df"] for x in unique_predictor_dict_list}

    msg = Printer(timestamp=True)

    msg.info(f"Pre-loading {len(unique_dfs)} dataframes")
    n_workers = min(len(unique_dfs), 16)
    p = Pool(n_workers)

    pre_loaded_dfs = p.map(load_df_wrapper, unique_predictor_dict_list)

    # Error check the laoded dfs
    failures = []

    for k, df in pre_loaded_dfs:
        source_failures, duplicates = check_raw_df(df=df, raise_error=False)

        failures.append({k: source_failures})

    if len(failures) > 0:
        raise ValueError(
            f"Pre-loaded dataframes failed source checks. {source_failures}",
        )

    # Combined pre_loaded dfs into one dictionary
    pre_loaded_dfs = {k: v for d in pre_loaded_dfs for k, v in d.items()}
    return pre_loaded_dfs


def load_df_wrapper(predictor_dict: Dict[str, Union[str, float, int]]) -> pd.DataFrame:
    """Wrapper to load a dataframe from a dictionary.

    Args:
        predictor_dict (Dict[str, Union[str, float, int]]): Dictionary where the key predictor_df maps to an SQL database.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if "values_to_load" in predictor_dict:
        return load_df(
            predictor_df=predictor_dict["predictor_df"],
            values_to_load=predictor_dict["values_to_load"],
        )
    else:
        return load_df(predictor_df=predictor_dict["predictor_df"])


def load_df(predictor_df: str, values_to_load: str = None) -> pd.DataFrame:
    """Load a dataframe from a SQL database.

    Args:
        predictor_df (str): The name of the SQL database.
        predictor_dict (Dict): A dict describing the predictor.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    msg = Printer(timestamp=True)

    msg.info(f"Loading {predictor_df}")

    loader_fns = data_loaders.get_all()

    if predictor_df not in loader_fns:
        msg.fail(f"Could not find loader for {predictor_df}.")
    else:
        if values_to_load:
            df = loader_fns[predictor_df](values_to_load=values_to_load)
        else:
            df = loader_fns[predictor_df]()

    msg.info(f"Loaded {predictor_df} with {len(df)} rows")
    return {predictor_df: df}
