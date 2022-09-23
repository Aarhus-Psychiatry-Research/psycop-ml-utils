"""Utils for model performance."""

import numpy as np
import pandas as pd
from pandas import Series


def scores_to_probs(scores: Series) -> Series:
    """Converts a series of lists of probabilities for each class or a list of
    floats for binary classification a list of floats of maximum length 2.

    Args:
        scores (Union[Series[list[float]], Series[float]]): Series containing probabilities for each class or a list of floats for binary classification.

    Returns:
        Series: Probability of class 1
    """

    if scores.dtype == "float":
        return scores

    return scores.apply(lambda x: x[1])


def labels_to_int(
    labels: Series,
    label2id: dict[str, int],
) -> Series:
    """Converts label to int mapping. Only makes sense for binary models. If
    already int will return as is.

    Args:
        labels (Union[Series[str], Series[int]]): Series containing labels.
        Either as string (e.g. "ASD") or as int.
        label2id (dict[str, int]): dictionary mapping the labels to 0 and 1

    Returns:
        Series: _description_
    """
    if labels.dtype == "int":
        return labels

    return labels.apply(lambda x: label2id[x])


def aggregate_predictions(
    df: pd.DataFrame,
    id_col: str,
    predictions_col: str,
    label_col: str,
) -> pd.DataFrame:
    """Calculates the mean prediction by a grouping col (id_col).

    Args:
        df (pd.DataFrame): Dataframe with 'predictions_col, 'label_col' and `id_col`
        id_col (str): Column to group by
        predictions_col (str): column containing predictions
        label_col (str): column containing labels

    Returns:
        pd.DataFrame: Dataframe with aggregated predictions
    """

    def mean_scores(scores: pd.Series):
        gathered = np.stack(scores)
        return gathered.mean(axis=0)

    def get_first_entry(scores: pd.Series):
        return scores.unique()[0]

    return df.groupby(id_col).agg(
        {predictions_col: mean_scores, label_col: get_first_entry},
    )


def idx_to_class(idx: list[int], mapping: dict) -> list[str]:
    """Returns the label from an id2label mapping.

    Args:
        idx (list[int]): index
        mapping (dict): dictionary mapping index to label

    Returns:
        list[str]: Labels matching the indices
    """
    return [mapping[id] for id in idx]


def select_metadata_cols(
    df: pd.DataFrame,
    metadata_cols: list[str],
    skip_cols: list[str],
) -> pd.DataFrame:
    """Selects columns with model metadata to a 1 row dataframe.

    Args:
        df (pd.DataFrame): Dataframe with predictions and metadata.
        metadata_cols (list[str]): Which columns contain metadata.
            The columns should only contain a single value. If "all", will add all columns.
        skip_cols (list[str]): Columns to definitely not include.

    Raises:
        ValueError: If a metadata col contains more than a single unique value.

    Returns:
        pd.DataFrame: 1 row dataframe with metadata
    """

    metadata = {}

    if isinstance(metadata_cols, str):
        metadata_cols = [metadata_cols]

    for col in df.columns:
        if col in skip_cols:
            continue

        # If asked to add all metadata cols,
        # Save all columns with only 1 unique value
        get_all = metadata_cols[0] == "all"
        is_metadata_col = col in metadata_cols

        if get_all or is_metadata_col:
            if df[col].nunique() == 1:
                metadata[col] = df[col].unique()[0]
            else:
                if is_metadata_col:
                    raise ValueError(
                        f"Meta-data {col} contains more than 1 unique value.",
                    )
                continue

    return pd.DataFrame.from_records([metadata])


def add_metadata_cols(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Adds 1 row dataframe with metadata to the long format performance
    dataframe.

    Args:
        df (pd.DataFrame): Dataframe to add metadata to.
        metadata (pd.Dataframe): 1 row dataframe with metadata

    Returns:
        pd.DataFrame: Dataframe with added metadata
    """
    nrows = df.shape[0]

    meta_dict = {}
    for col in metadata.columns:
        meta_dict[col] = [metadata[col][0]] * nrows
    meta_df = pd.DataFrame.from_records(meta_dict)

    return df.reset_index(drop=True).join(meta_df)
