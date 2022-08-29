from pathlib import Path

import catalogue

data_loaders = catalogue.create("timeseriesflattener", "data_loaders")

SHARED_RESOURCES_PATH = Path(r"C:\shared_resources")
FEATURE_SETS_PATH = SHARED_RESOURCES_PATH / "feature_sets"
OUTCOME_DATA_PATH = SHARED_RESOURCES_PATH / "outcome_data"


def generate_feature_colname(
    prefix: str,
    out_col_name: str,
    interval_days: int,
    resolve_multiple: str,
    fallback: str,
) -> str:
    """Generates standardized column name from feature collapse information.

    Args:
        prefix (str): Prefix (typically either "pred" or "outc")
        out_col_name (str): Name after the prefix.
        interval_days (int): Fills out "_within_{interval_days}" in the col name.
        resolve_multiple (str): Name of the resolve_multiple strategy.
        fallback (str): Values used for fallback.

    Returns:
        str: _description_
    """
    return f"{prefix}_{out_col_name}_within_{interval_days}_days_{resolve_multiple}_fallback_{fallback}"
