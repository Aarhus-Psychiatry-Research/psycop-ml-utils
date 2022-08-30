import time

import pandas as pd

from psycopmlutils.utils import RAW_DATA_VALIDATION_PATH


def validate_raw_data(
    df: pd.DataFrame,
    feature_set_name: str,
    timestamp_col_name: str = "timestamp",
    id_col_name: str = "dw_ek_borger",
) -> None:
    """Validates raw data from SQL database. Runs data integrity checks from
    deepchecks, and calculates quantiles.

    Args:
        df (pd.DataFrame): Dataframe to validate
        feature_set_name (str): Name of the feature set
        timestamp_col_name (str): Name of timestamp column
        id_col_name (str): Name of id column
    """
    savepath = (
        RAW_DATA_VALIDATION_PATH / {feature_set_name} / time.strftime("%Y_%m_%d_%H_%M")
    )
    if not savepath.exists():
        savepath.mkdir(parents=True)

    # Deepchecks

    # Deepchecks

    # Quantiles
