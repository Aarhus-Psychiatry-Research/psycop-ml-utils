from pathlib import Path
from typing import List

import catalogue
import pandas as pd

data_loaders = catalogue.create("timeseriesflattener", "data_loaders")

SHARED_RESOURCES_PATH = Path(r"E:\shared_resources")
FEATURE_SETS_PATH = SHARED_RESOURCES_PATH / "feature_sets"
OUTCOME_DATA_PATH = SHARED_RESOURCES_PATH / "outcome_data"
RAW_DATA_VALIDATION_PATH = SHARED_RESOURCES_PATH / "raw_data_validation"


def generate_feature_colname(
    prefix: str,
    out_col_name: str,
    interval_days: int,
    resolve_multiple: str,
    fallback: str,
    values_to_load: str = None,
) -> str:
    """Generates standardized column name from feature collapse information.

    Args:
        prefix (str): Prefix (typically either "pred" or "outc")
        out_col_name (str): Name after the prefix.
        interval_days (int): Fills out "_within_{interval_days}" in the col name.
        resolve_multiple (str): Name of the resolve_multiple strategy.
        fallback (str): Values used for fallback.
        values_to_load (str): Values to load from lab results.

    Returns:
        str: A full column name
    """
    col_name = f"{prefix}_{out_col_name}_within_{interval_days}_days_{resolve_multiple}_fallback_{fallback}"

    if values_to_load:
        col_name += f"_{values_to_load}"

    return col_name


def df_contains_duplicates(df=pd.DataFrame, col_subset=List[str]):
    """Check if a dataframe contains duplicates.

    Args:
        df (pd.DataFrame): Dataframe to check.
        col_subset (List[str]): Columns to check for duplicates.

    Returns:
        bool: True if duplicates are found.
    """
    return df.duplicated(subset=col_subset).any()
