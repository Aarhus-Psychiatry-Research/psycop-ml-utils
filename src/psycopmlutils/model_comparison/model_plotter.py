import seaborn as sns
from matplotlib import pyplot as plt

import pandas as pd
from typing import Optional, Union, List
from psycopmlutils.model_comparison.utils import string_to_list, subset_df_from_dict


class ModelPlotter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.palette_two_cats = ["grey", "darkorange"]
        self.fig_size = (20, 4)

    def plot_scatter(
        self,
        scores: str,
        y: str,
        groups: Union[str, List] = "overall",
        levels: Union[str, List] = "overall",
        grouping: Optional[str] = None,
        title: Optional[str] = None,
    ):

        scores = string_to_list(scores)
        groups = string_to_list(groups)
        levels = string_to_list(levels)

        df = self.df[
            (self.df["score_type"].isin(scores))
            & (self.df["class"].isin(groups))
            & (self.df["level"].isin(levels))
        ]

        fix, ax = plt.subplots(figsize=self.fig_size)
        sns.scatterplot(data=df, y=y, x="value", hue=grouping)
        title_s = scores[0].capitalize() if len(scores) == 1 else " ".join(scores)
        if not title:
            title_g = (
                "all classes" if "".join(groups) == "overall" else " ".join(groups)
            )
            title_l = "row" if "".join(levels) == "overall" else " ".join(levels)
            title = f"{title_s} for {title_g} on {title_l} level"
        plt.title(label=title)
        plt.xlabel(title_s)
        plt.ylabel("")

    def plot_swarm(
        self,
        scores: Union[List, str] = ["f1", "precision", "recall"],
        groups: Optional[Union[str, List]] = None,
        level: str = "overall",
        grouping: str = None,
        facet: str = "score_type",
        title: str = None,
        subset_by: dict = None,
    ):
        # if not specified, plot all classes/groups
        if not groups:
            groups = self.df["class"].unique().tolist()
        scores = string_to_list(scores)
        groups = string_to_list(groups)

        df = self.df.copy()
        if subset_by:
            df = subset_df_from_dict(df, subset_by)

        df = self.df[
            (self.df["score_type"].isin(scores))
            & (self.df["class"].isin(groups))
            & (self.df["level"] == level)
        ]

        p = sns.catplot(
            x="class",
            y="value",
            hue=grouping,
            col=facet,
            data=df,
            kind="swarm",
            height=4,
            aspect=0.7,
        )
        if not title:
            title_l = "row" if level == "overall" else level
            title = f"Performance on {title_l} level"
        p.fig.subplots_adjust(top=0.87)
        p.fig.suptitle(title)

    def plot_line(
        self,
        x,
        y,
        scores: Union[str, List],
        grouping: str,
        facet: str = "score_type",
        groups: Optional[Union[str, List]] = None,
        levels: Optional[Union[str, List]] = "overall",
        title: str = None,
        subset_by: dict = None,
    ):
        if not groups:
            groups = self.df["class"].unique().tolist()

        scores = string_to_list(scores)
        groups = string_to_list(groups)
        levels = string_to_list(levels)

        df = self.df.copy()
        if subset_by:
            df = subset_df_from_dict(df, subset_by)

        df = self.df[
            (self.df["score_type"].isin(scores))
            & (self.df["class"].isin(groups))
            & (self.df["level"].isin(levels))
        ]

        p = sns.catplot(
            x=x,
            y=y,
            hue=grouping,
            col=facet,
            data=df,
            kind="line",
            height=4,
            aspect=0.7,
        )


if __name__ == "__main__":
    from .model_comparison import ModelComparison

    model_comparer = ModelComparison(
        id_col="id",
        score_mapping={0: "ASD", 1: "TD"},
        metadata_cols=["type", "split", "model_name"],
    )
    res = model_comparer.transform_data_from_folder(
        "/Users/au554730/Desktop/Projects/psycop-ml-utils/tests/test_model_comparison/test_data"
    )

    plotter = ModelPlotter(res)
    plotter.plot_scatter(
        score_type="f1",
        y="model_name",
        group=["ASD", "TD"],
        level="id",
        grouping="class",
    )
    plotter.plot_swarm(level="id")
    plotter.plot_line(x="class", y="f1", grouping="class", facet="score_type")
    # df = pd.DataFrame(
    #     {
    #         "id": [1, 1, 2, 2],
    #         "scores": [[0.8, 0.2], [0.5, 0.5], [0.6, 0.4], [0.9, 0.1]],
    #         "label": ["TD", "TD", "DEPR", "DEPR"],
    #         "optional_grouping1": ["grouping1"] * 4,
    #         "optional_grouping2": ["grouping2"] * 4,
    #     }
    # )

    # model_comparer = ModelComparison(
    #     id_col="id",
    #     metadata_cols=["optional_grouping1", "optional_grouping2"],
    #     score_mapping={0: "TD", 1: "DEPR"},
    # )
    # res = model_comparer.transform_data_from_dataframe(df)

    # plotter = ModelPlotter(res)
    # plotter.plot_scatter_line("f1_macro", "optional_grouping1", split="")
