from pathlib import Path

import pandas as pd

from psycopmlutils.loaders.flattened import load_split_predictors


def get_duplicates_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get duplicates from a dataframe.

    Args:
        df (pd.DataFrame): Dataframe to get duplicates from.

    Returns:
        pd.DataFrame: Dataframe with duplicates.
    """
    return df[df.duplicated(keep=False)]


if __name__ == "__main__":
    df = load_split_predictors(
        path=Path(
            "E:/shared_resources/feature_sets/t2d/adminmanber_4_features_2022_08_30_15_49",
        ),
        split="train",
        include_id=True,
    )

    inspect = get_duplicates_from_df(df=df)[["dw_ek_borger", "timestamp"]].sort_values(
        by=["dw_ek_borger", "timestamp"],
    )

    pass
