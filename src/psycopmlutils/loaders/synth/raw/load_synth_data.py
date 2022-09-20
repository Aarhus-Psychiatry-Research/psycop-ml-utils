"""Loaders for synth data."""

from pathlib import Path
from typing import Optional

import pandas as pd

from psycopmlutils.utils import data_loaders


@data_loaders.register("synth_predictor")
def coercion_duration(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    # Get project base path
    proj_path = Path(__file__).resolve().parents[3]

    df = pd.read_csv(
        proj_path / "tests" / "test_data" / "synth_predictor_data.csv",
        nrows=n_rows,
    )

    return df
