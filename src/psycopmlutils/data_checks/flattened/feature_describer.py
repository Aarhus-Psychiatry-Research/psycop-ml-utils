from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from wasabi import Printer

from psycopmlutils.data_checks.utils import save_df_to_pretty_html
from psycopmlutils.loaders.flattened.local_feature_loaders import load_split_predictors
from psycopmlutils.utils import generate_feature_colname

UNICODE_HIST = {
    0: " ",
    1 / 8: "▁",
    1 / 4: "▂",
    3 / 8: "▃",
    1 / 2: "▄",
    5 / 8: "▅",
    3 / 4: "▆",
    7 / 8: "▇",
    1: "█",
}

HIST_BINS = 8


def create_feature_description_from_dir(
    path: Path,
    predictor_dicts: List[Dict[str, str]],
    splits: Optional[List[str]] = ["train"],
) -> pd.DataFrame:
    """Write a csv with feature descriptions in the directory.

    Args:
        path (Path): Path to directory with data frames.
        predictor_dicts (List[Dict[str, str]]): List of dictionaries with predictor information.
        splits (List[str]): List of splits to include in the description. Defaults to ["train"].
    """
    msg = Printer(timestamp=True)
    save_dir = path / "feature_descriptions"

    if not save_dir.exists():
        save_dir.mkdir()

    for split in splits:
        msg.info(f"{split}: Creating feature description")

        predictors = load_split_predictors(path=path, split=split, include_id=False)

        msg.info(f"{split}: Generating feature description dataframe")
        feature_description_df = generate_feature_description_df(
            df=predictors,
            predictor_dicts=predictor_dicts,
        )

        msg.info(f"{split}: Writing feature description to disk")

        feature_description_df.to_csv(
            save_dir / f"{split}_feature_description.csv",
            index=False,
        )
        # Writing html table as well
        save_df_to_pretty_html(
            path=save_dir / f"{split}_feature_description.html",
            title="Feature description",
            df=feature_description_df,
        )


def generate_feature_description_df(
    df: pd.DataFrame,
    predictor_dicts: List[Dict[str, str]],
) -> pd.DataFrame:
    """Generate a data frame with feature descriptions.

    Args:
        df (pd.DataFrame): Data frame with data to describe.
        predictor_dicts (List[Dict[str, str]]): List of dictionaries with predictor information.

    Returns:
        pd.DataFrame: Data frame with feature descriptions.
    """

    rows = []

    for d in predictor_dicts:
        column_name = generate_feature_colname(
            prefix="pred",
            out_col_name=d["predictor_df"],
            interval_days=d["lookbehind_days"],
            resolve_multiple=d["resolve_multiple"],
            fallback=d["fallback"],
            loader_kwargs=d.get("loader_kwargs", None),
        )

        rows.append(generate_feature_description_row(df[column_name], d))

    # Convert to dataframe
    feature_description_df = pd.DataFrame(rows)

    # Sort feature_description_df by Predictor df to make outputs easier to read
    feature_description_df = feature_description_df.sort_values(by="Predictor df")

    return feature_description_df


def generate_feature_description_row(
    series: pd.Series,
    predictor_dict: Dict[str, str],
) -> Dict:
    """Generate a row with feature description.

    Args:
        series (pd.Series): Series with data to describe.
        predictor_dict (Dict[str, str]): Dictionary with predictor information.

    Returns:
        Dict: Dictionary with feature description.
    """

    d = {
        "Predictor df": predictor_dict["predictor_df"],
        "Lookbehind days": predictor_dict["lookbehind_days"],
        "Resolve multiple": predictor_dict["resolve_multiple"],
        "N unique": series.nunique(),
        "Fallback strategy": predictor_dict["fallback"],
        "Proportion missing": series.isna().mean(),
        "Mean": round(series.mean(), 2),
        "Histogram": create_unicode_hist(series),
        "Proportion using fallback": get_value_proportion(
            series,
            predictor_dict["fallback"],
        ),
    }

    for percentile in [0.01, 0.25, 0.5, 0.75, 0.99]:
        # Get the value representing the percentile
        d[f"{percentile*100}-percentile"] = round(series.quantile(percentile), 1)

    return d


def get_value_proportion(series, value):
    """Get proportion of series that is equal to the value argument."""
    if np.isnan(value):
        return round(series.isna().mean(), 2)
    else:
        return round(series.eq(value).mean(), 2)


def create_unicode_hist(series: pd.Series) -> pd.Series:
    """Return a histogram rendered in block unicode. Given a pandas series of
    numerical values, returns a series with one entry, the original series
    name, and a histogram made up of unicode characters.

    Args:
        series (pd.Series): Numeric column of data frame for analysis

    Returns:
        pd.Series: Index of series name and entry with unicode histogram as
        a string, eg '▃▅█'

    All credit goes to the python package skimpy.
    """
    # Remove any NaNs
    series = series.dropna()

    if series.dtype == "bool":
        series = series.astype("int")

    hist, _ = np.histogram(series, density=True, bins=HIST_BINS)
    hist = hist / hist.max()

    # Now do value counts
    key_vector = np.array(list(UNICODE_HIST.keys()), dtype="float")

    ucode_to_print = "".join(
        [UNICODE_HIST[_find_nearest(key_vector, val)] for val in hist],
    )

    return ucode_to_print


def _find_nearest(array, value):
    """Find the nearest numerical match to value in an array.

    Args:
        array (np.ndarray): An array of numbers to match with.
        value (float): Single value to find an entry in array that is close.

    Returns:
        np.array: The entry in array that is closest to value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
