from pandas import DataFrame
from wasabi import msg


def create_cols_for_unique_vals_at_depth(
    df: DataFrame, source_col_name: str, depth: int
) -> DataFrame:
    df["substr"] = df[source_col_name].str[:depth]

    categories = df["substr"].unique().tolist()
    categories.sort()

    msg.info(f"Generating new columns for {categories} from {source_col_name}")

    for category in categories:
        df[category] = df["substr"].map({category: 1})
        df[category].fillna(0, inplace=True)

    return df.drop("substr", axis=1)
