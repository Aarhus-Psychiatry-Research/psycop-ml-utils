from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def load_split_predictors_and_outcomes(
    path: Path,
    split: str,
    include_id: bool,
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads a given data split from a directory and returns predictors and
    outcomes separately.

    Args:
        path (Path): Path to directory containing data files
        split (str): Which split to load
        include_id (bool): Whether to include 'dw_ek_borger' in predictor df
        nrows (Optional[int]): Number of rows to load from each file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple where first element is the
        predictors and second element the outcomes
    """
    split = load_split(path, split, nrows=nrows)
    predictors, outcomes = separate_predictors_and_outcome(split, include_id=include_id)
    return predictors, outcomes


def separate_predictors_and_outcome(
    df: pd.DataFrame,
    include_id: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split predictors and outcomes into two dataframes. Assumes predictors to
    be prefixed with 'pred', and outcomes to be prefixed with 'outc'. Timestamp
    is also returned for predictors, and optionally also dw_ek_borger.

    Args:
        df (pd.DataFrame): Dataframe containing generates features
        include_id (bool): Whether to include 'dw_ek_borger' in predictor df

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple where first element is the
        predictors and second element the outcomes
    """
    pred_regex = (
        "^pred|^timestamp" if not include_id else "^pred|^timestamp|dw_ek_borger"
    )
    predictors = df.filter(regex=pred_regex)
    outcomes = df.filter(regex="^outc")
    return predictors, outcomes


def load_split(path: Path, split: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Loads a given data split as a dataframe from a directory.

    Args:
        path (Path): Path to directory containing data files
        split (str): Which string to look for (e.g. 'train', 'val', 'test')
        nrows (Optional[int]): Whether to only load a subset of the data

    Returns:
        pd.DataFrame: The loaded dataframe
    """
    return pd.read_csv(list(path.glob(f"*{split}*"))[0], nrows=nrows)
