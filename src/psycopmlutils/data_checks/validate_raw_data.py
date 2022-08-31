import time
from typing import List, Optional

import numpy as np
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from wasabi import Printer

from psycopmlutils.data_checks.data_integrity import get_name_of_failed_checks
from psycopmlutils.feature_describer.feature_describer import create_unicode_hist
from psycopmlutils.utils import RAW_DATA_VALIDATION_PATH


def validate_raw_data(
    df: pd.DataFrame,
    feature_set_name: str,
    deviation_baseline_column: Optional[str] = "median",
    deviation_threshold: Optional[float] = 4.0,
    deviation_variation_column: Optional[str] = "median_absolute_deviation",
) -> None:
    """Validates raw data from SQL database (or any dataframe, really). Runs
    data integrity checks from deepchecks, and calculates summary statistics.
    Summary statistics are saved as a table with one row for each column. Rows
    are colored yellow if the 99th/1st percentile exceeds.

    `deviation_baseline_column'  +- `deviation_treshold` * `deviation_variation_column`.
    All files are saved to the `RAW_DATA_VALIDATION_PATH` directory in a subdirectory
    named `feature_set_name`.

    Args:
        df (pd.DataFrame): Dataframe to validate.
        feature_set_name (str): Name of the feature set.
        deviation_baseline_column (Optional[str], optional): _description_. Defaults to "mean".
        deviation_threshold (Optional[float], optional): _description_. Defaults to 3.0.
        deviation_variation_column (Optional[str], optional): _description_. Defaults to "std".
    """

    msg = Printer(timestamp=True)
    failed_checks = {}

    savepath = (
        RAW_DATA_VALIDATION_PATH / {feature_set_name} / time.strftime("%Y_%m_%d_%H_%M")
    )
    if not savepath.exists():
        savepath.mkdir(parents=True)

    # check if `timestamp` and `dw_ek_borger` columns exist
    timestamp_col_name = "timestamp" if "timestamp" in df.columns else None
    id_col_name = "dw_ek_borger" if "dw_ek_borger" in df.columns else None

    # Deepchecks
    ds = Dataset(df=df, index_name=id_col_name, datetime_name=timestamp_col_name)
    integ_suite = data_integrity(timeout=0)
    with msg.loading("Running data integrity checks..."):
        suite_results = integ_suite.run(ds)
        suite_results.save_as_html(str(savepath / "data_integrity.html"))
        failed_checks["data_integrity"] = get_name_of_failed_checks(suite_results)
    msg.good("Finished data integrity checks.")
    suite_results.save_as_html(str(savepath / "deepchecks.html"))

    # Data description
    data_columns = [
        col for col in df.columns if col not in [id_col_name, timestamp_col_name]
    ]
    with msg.loading("Generating data description..."):
        data_description = [
            generate_column_description(df[col]) for col in data_columns
        ]
    msg.good("Finished data description.")

    data_description = pd.DataFrame(data_description)
    data_description.to_csv(savepath / "data_description.csv", index=False)
    # Highlight rows with large deviations from the baseline
    data_description = data_description.style.apply(
        highlight_large_deviation,
        threshold=deviation_threshold,
        baseline_column=deviation_baseline_column,
        variation_column=deviation_variation_column,
        axis=1,
    )
    to_html_pretty(
        data_description,
        "data_description.html",
        title=f"Data description - {feature_set_name}",
        subtitle=f"Yellow rows indicate large deviations from the {deviation_baseline_column}\n(99th/1st percentile within +- {deviation_variation_column} * threshold={deviation_threshold}) from the baseline.)",
    )


def generate_column_description(series: pd.Series) -> dict:
    """Generates a dictionary with column description.

    Args:
        series (pd.Series): Series to describe.

    Returns:
        dict: Dictionary with column description.
    """

    d = {
        "col_name": series.name,
        "dtype": series.dtype,
        "nunique": series.nunique(),
        "nmissing": series.isna().sum(),
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "std": series.std(),
        "median": series.median(),
        "median_absolute_deviation": median_absolute_deviation(series),
    }
    d["histogram"] = create_unicode_hist(series)
    for percentile in [0.01, 0.25, 0.5, 0.75, 0.99]:
        d[f"{percentile}th_percentile"] = round(series.quantile(percentile), 1)

    return d


def median_absolute_deviation(series: pd.Series) -> np.array:
    """Calculates the median absolute deviation of a series.

    Args:
        series (pd.Series): Series to calculate the median absolute deviation of.

    Returns:
        np.array: Median absolute deviation of the series.
    """
    med = np.median(series)
    return np.median(np.abs(series - med))


def highlight_large_deviation(
    series: pd.Series,
    threshold: float,
    baseline_column: str,
    variation_column: str,
) -> List[str]:
    """Highlights rows where the 99th/1st percentile is x times the standard
    deviation larger/smaller than the column (probably mean or median).

    Args:
        series (pd.Series): Series to describe.
        threshold (float): Threshold for deviation. 3-4 might be a good value.
        baseline_column (str): Name of the column to use as baseline. Commonly 'mean' or 'median'.
        variation_column (str): Name of the column containing the variation.
        Commonly 'std' or 'mad' (mean aboslute deviation).

    Returns:
        List[str]: List of styles for each row.
    """
    above_threshold = pd.Series(data=False, index=series.index)
    lower_bound = series[baseline_column] - series[variation_column] * threshold
    upper_bound = series[baseline_column] + series[variation_column] * threshold

    above_threshold[baseline_column] = (
        series.loc["0.99th_percentile"] > upper_bound
        or series.loc["0.01th_percentile"] < lower_bound
    )
    return [
        "background-color: yellow" if above_threshold.any() else ""
        for v in above_threshold
    ]


def to_html_pretty(
    df: pd.DataFrame,
    filename: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> None:
    """Write dataframe to a HTML file with nice formatting. Stolen from
    stackoverflow: https://stackoverflow.com/a/52722850.

    Args:
        df (pd.DataFrame): Dataframe to write.
        filename (str): File name to write to.
        title (Optional[str], optional): Title for the table. Defaults to None.
        subtitle (Optional[str], optional): Subtitle for the table. Defaults to None.
    """

    ht = ""
    if title:
        ht += "<h2> %s </h2>\n" % title
    if subtitle:
        ht += "<h3> %s </h3>\n" % subtitle
    ht += df.to_html(classes="wide", escape=False)

    with open(filename, "w") as f:
        f.write(HTML_TEMPLATE1 + ht + HTML_TEMPLATE2)


# Templates for saving dataframes as pretty html tables
HTML_TEMPLATE1 = """
<html>
<head>
<style>
  h2 {
    text-align: center;
    font-family: Helvetica, Arial, sans-serif;
  }
  table { 
    margin-left: auto;
    margin-right: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
    text-align: center;
    font-family: Helvetica, Arial, sans-serif;
    font-size: 90%;
  }
  table tbody tr:hover {
    background-color: #dddddd;
  }
  .wide {
    width: 90%; 
  }
</style>
</head>
<body>
"""

HTML_TEMPLATE2 = """
</body>
</html>
"""
