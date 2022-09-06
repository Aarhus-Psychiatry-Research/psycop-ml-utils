"""Pre-load dataframes to avoid duplicate loading."""

from typing import Dict, List, Union

import pandas as pd
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

    loader_fns = data_loaders.get_all()

    dfs = {}

    for pred_d in predictor_dict_list:
        if pred_d["predictor_df"] not in dfs:
            msg.info(f"Loading {pred_d['predictor_df']}")

            if "values_to_load" in pred_d:
                dfs[pred_d["predictor_df"]] = loader_fns[pred_d["predictor_df"]](
                    values_to_load=pred_d["values_to_load"],
                )
            else:
                dfs[pred_d["predictor_df"]] = loader_fns[pred_d["predictor_df"]]()

            msg.info(f"Loaded {pred_d['predictor_df']}")

    return dfs
