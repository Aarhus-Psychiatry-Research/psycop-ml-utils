from pathlib import Path
from typing import Optional

import pandas as pd


def save_df_to_pretty_html(
    df: pd.DataFrame,
    filename: Path,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> None:
    """Write dataframe to a HTML file with nice formatting. Stolen from
    stackoverflow: https://stackoverflow.com/a/52722850.

    Args:
        df (pd.DataFrame): Dataframe to write.
        filename (Path): Path to save the HTML file to.
        title (Optional[str], optional): Title for the table. Defaults to None.
        subtitle (Optional[str], optional): Subtitle for the table. Defaults to None.
    """

    ht = ""
    if title:
        ht += "<h2> %s </h2>\n" % title
    if subtitle:
        ht += "<h3> %s </h3>\n" % subtitle
    ht += df.to_html(classes="wide", escape=False)

    with open(filename, "w", encoding="utf-8") as f:
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