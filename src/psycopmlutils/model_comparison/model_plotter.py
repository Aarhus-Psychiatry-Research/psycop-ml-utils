import seaborn as sns
from matplotlib import pyplot as plt

import pandas as pd
from typing import Optional


class ModelPlotter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.palette_two_cats = ["grey", "darkorange"]

    def plot_scatter_line(
        self, score_type: str, y: str, split, title: Optional[str] = None
    ):
        df = self.df[self.df["score_type"] == score_type]

        fix, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=df, y=y, x="value", palette=self.palette_two_cats)
        plt.legend(title=title)
        plt.xlabel(score_type)
        plt.ylabel("")


if __name__ == "__main__":
    from model_comparison import ModelComparison

    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "scores": [[0.8, 0.2], [0.5, 0.5], [0.6, 0.4], [0.9, 0.1]],
            "label": ["TD", "TD", "DEPR", "DEPR"],
            "optional_grouping1": ["grouping1"] * 4,
            "optional_grouping2": ["grouping2"] * 4,
        }
    )

    model_comparer = ModelComparison(
        id_col="id",
        metadata_cols=["optional_grouping1", "optional_grouping2"],
        score_mapping={0: "TD", 1: "DEPR"},
    )
    res = model_comparer.transform_data_from_dataframe(df)

    plotter = ModelPlotter(res)
    plotter.plot_scatter_line("f1_macro", "optional_grouping1", split="")
