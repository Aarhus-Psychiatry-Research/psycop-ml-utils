import ast

from typing import Callable, List
import pandas as pd

import numpy as np


def parse_column_as_list(df: pd.DataFrame, col: str) -> pd.Series:
    """Correctly parse a column as list
    Args:
        df (pd.DataFrame):
        col (str): Column to parse

    Returns:
        pd.Series: the column parsed as a list
    """
    return df[col].apply(ast.literal_eval)


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
