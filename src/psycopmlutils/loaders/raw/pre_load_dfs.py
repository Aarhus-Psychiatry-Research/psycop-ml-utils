"""Pre-load dataframes to avoid duplicate loading."""

from multiprocessing import Pool
from typing import Dict, List, Union

import pandas as pd
import tqdm
from wasabi import Printer

from psycopmlutils.utils import data_loaders


def pre_load_unique_dfs(
    predictor_dict_list: List[Dict[str, Union[str, float, int]]],
) -> Dict[str, pd.DataFrame]:
    """Pre-load unique dataframes to avoid duplicate loading.

    Args:
        predictor_dict_list (List[Dict[str, Union[str, float, int]]]): List of dictionaries where the key predictor_df maps to an SQL database.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary with keys predictor_df and values the loaded dataframe.
    """

    # Get unique predictor dfs
    unique_predictor_dfs = {
        predictor_dict["predictor_df"] for predictor_dict in predictor_dict_list
    }

    msg = Printer(timestamp=True)

    msg.info(f"Pre-loading {len(unique_predictor_dfs)} dataframes")

    n_workers = min(len(unique_predictor_dfs), 16)

    p = Pool(n_workers)

    pre_loaded_dfs = p.map(load_df, unique_predictor_dfs)

    # Combined pre_loaded dfs into one dictionary
    pre_loaded_dfs = {k: v for d in pre_loaded_dfs for k, v in d.items()}
    return pre_loaded_dfs


def load_df(predictor_df: str) -> pd.DataFrame:
    """Load a dataframe from a SQL database.

    Args:
        predictor_df (str): The name of the SQL database.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    msg = Printer(timestamp=True)

    msg.info(f"Loading {predictor_df}")

    loader_fns = data_loaders.get_all()

    raise KeyError("Please step into subprocess, thx")

    if predictor_df not in loader_fns:
        msg.fail(f"Could not find loader for {predictor_df}.")
    else:
        df = loader_fns[predictor_df]()

    msg.info(f"Loaded {predictor_df} with {len(df)} rows")
    return {predictor_df: df}
