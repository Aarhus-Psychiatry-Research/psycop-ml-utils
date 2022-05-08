import ast
from multiprocessing.sharedctypes import Value

from typing import List, Optional
import pandas as pd

import numpy as np


def aggregate_predictions(df: pd.DataFrame, id_col: str):
    """Calculates the mean prediction by a grouping col (id_col).
    Assumes that df has the columns 'scores': List[float] and
    'label' : str

    Args:
        df (pd.DataFrame): Dataframe with 'scores', 'label' and id_col columns
        id_col (str): Column to group by
    """

    def mean_scores(x: pd.Series):
        gathered = np.stack(x)
        return gathered.mean(axis=0)

    def get_first_entry(x: pd.Series):
        return x.unique()[0]

    return df.groupby(id_col).agg({"scores": mean_scores, "label": get_first_entry})


def idx_to_class(idx: List[int], mapping: dict):
    return [mapping[id] for id in idx]


def get_metadata_cols(df: pd.DataFrame, cols: Optional[List[str]]) -> pd.DataFrame:
    """Extracts model metadata and returns as a single row dataframe.

    Args:
        df (pd.DataFrame): Dataframe with predictions and metadata.
        cols (Optional[List[str]]): Which columns contain metadata. 
            The columns should only contain a single value.

    Raises:
        ValueError: If a metadata col contains more than a single unique value.

    Returns:
        pd.DataFrame: 1 row dataframe with metadata
    """
    if not cols:
        return df
    metadata = {}
    all_columns = df.columns
    for col in cols:
        if col in all_columns:
            val = df[col].unique()
            if len(val) > 1:
                raise ValueError(f"The column '{col}' contains more than one unique value.")
            metadata[col] = val
    return pd.DataFrame.from_records([metadata])


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "scores": [
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.7, 0.1, 0.1],
            ],
            "label": ["ASD", "ASD", "TD", "TD"],
        }
    )

    aggregate_predictions(df, "id")
