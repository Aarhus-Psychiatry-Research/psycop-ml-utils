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
    msg = Printer(timestamp=True)
    msg.info("Pre-loading unique dataframes")

    # Get unique predictor dfs
    unique_predictor_dfs = {
        predictor_dict["predictor_df"] for predictor_dict in predictor_dict_list
    }

    n_workers = min(len(unique_predictor_dfs), 16)

    p = Pool(n_workers)

    pre_loaded_dfs = list(
        tqdm.tqdm(
            p.imap(load_df, unique_predictor_dfs),
            total=len(unique_predictor_dfs),
        ),
    )

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

    loader_fns = data_loaders.get_all()

    if predictor_df not in loader_fns:
        msg.fail(f"Could not find loader for {predictor_df}.")
    else:
        msg.info(f"Loading {predictor_df}")
        df = loader_fns[predictor_df]()

    msg.good(f"Loaded {predictor_df}.")

    return {predictor_df: df}
