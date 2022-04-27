from pandas import DataFrame
from wasabi import msg


def create_cols_for_unique_vals_at_depth(
    df: DataFrame, source_col_name: str, depth: int
) -> DataFrame:
    """Create a column for each unique value at string depth.
    E.g. for [E10, E11, E12] at depth 2, create [E1], at depth 3, create [E10, E11, E12].

    Args:
        df (DataFrame): Input dataframe.
        source_col_name (str): Column name with the strings in the input dataframe.
        depth (int): How far down to go in the string, see above documentation.

    Returns:
        DataFrame
    """
    df["substr"] = df[source_col_name].str[:depth]

    categories = df["substr"].unique().tolist()
    categories.sort()

    msg.info(f"Generating new columns for {categories} from {source_col_name}")

    for category in categories:
        df[category] = df["substr"].map({category: 1})
        df[category].fillna(0, inplace=True)

    return df.drop("substr", axis=1)
