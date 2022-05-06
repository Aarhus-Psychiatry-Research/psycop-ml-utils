""""Tools for model comparison"""

import pandas as pd

from typing import Union, List, Optional
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


import numpy as np

from psycopmlutils.model_comparison.utils import (
    parse_column_as_list,
    aggregate_predictions,
    idx_to_class,
)

"""Expect input with columns
label,scores,model_name,[optional_grouping_columns]
"""


class ModelComparison:
    def __init__(
        self, id_col: Optional[str] = None, score_mapping: Optional[dict] = None
    ):
        """_summary_

        Args:
            id_col (Optional[str]): id column in case of multiple predictions.
            score_mapping (Optional[dict]): Mapping from scores index to group (should match label). E.g. if scores [0.3, 0.6, 0.1]
                score_mapping={0:"control", 1:"depression", 2:"schizophrenia}. Not needed for binary models.
        """

        self.score_mapping = score_mapping

    def load_data_from_csv(self, path: Union[str, Path]):
        df = pd.read_csv(path)

        performance = self._evaluate_single_model(df, aggregate_by_id=False)
        performance["level"] = "overall"

        if self.id_col:
            performance_by_id = self._evaluate_single_model(df, aggregate_by_id=True)
            performance_by_id["level"] = "id"
            performance = pd.concat([performance, performance_by_id])
        return performance

    def load_data_from_folder(self, path: Union[str, Path], pattern: str = "*.csv"):
        pass

    def load_data_from_dataframe(self, dfs: Union[pd.DataFrame, List[pd.DataFrame]]):
        pass

    def _evaluate_single_model(self, df: pd.DataFrame, aggregate_by_id: bool):
        # if there is a mapping and scores is not a float, parse the column as a lsit
        if self.score_mapping and df["scores"].dtype != "float":
            df["scores"] = parse_column_as_list(df, "scores")

        if aggregate_by_id:
            df = aggregate_predictions(df, self.id_col)

        # get predicted labels
        if df["scores"].dtype != "float":
            argmax_indices = df["scores"].apply(lambda x: np.argmax(x))
            predictions = idx_to_class(argmax_indices, self.score_mapping)
        else:
            predictions = np.round(df["scores"])
        return self.compute_metrics(df["label"], predictions)

    def evaluate_models(self):
        pass

    @staticmethod
    def compute_metrics(
        labels: Union[pd.Series, List],
        predicted: Union[pd.Series, List],
    ) -> pd.DataFrame:
        """Computes performance metrics for both binary and multiclass tasks

        Arguments:
            labels {Union[pd.Series, List]} -- true class
            predicted {Union[pd.Series, List]} -- predicted class

        Returns:
            pd.DataFrame -- Long format dataframe with performance metrics
        """
        classes = sorted(set(labels))
        performance = {}

        performance["acc-overall"] = accuracy_score(labels, predicted)
        performance["f1_macro-overall"] = f1_score(labels, predicted, average="macro")
        performance["f1_micro-overall"] = f1_score(labels, predicted, average="micro")
        performance["precision_macro-overall"] = precision_score(
            labels, predicted, average="macro"
        )
        performance["precision_micro-overall"] = precision_score(
            labels, predicted, average="micro"
        )
        performance["recall_macro-overall"] = recall_score(
            labels, predicted, average="macro"
        )
        performance["recall_micro-overall"] = recall_score(
            labels, predicted, average="micro"
        )

        if len(classes) == 2:
            performance["roc_auc-overall"] = roc_auc_score(labels, predicted)

        # calculate scores by class
        f1_by_class = f1_score(labels, predicted, average=None)
        precision_by_class = precision_score(labels, predicted, average=None)
        recall_by_class = recall_score(labels, predicted, average=None)

        for i, c in enumerate(classes):
            performance[f"f1-{c}"] = f1_by_class[i]
            performance[f"precision-{c}"] = precision_by_class[i]
            performance[f"recall-{c}"] = recall_by_class[i]

        # to df
        performance = pd.DataFrame.from_records([performance])
        # convert to long format
        performance = pd.melt(performance)
        # split score and class into two columns
        performance[["score_type", "class"]] = performance["variable"].str.split(
            "-", 1, expand=True
        )
        # drop unused columns and rearrange
        performance = performance.drop("variable", axis=1)
        performance = performance[["class", "score_type", "value"]]
        return performance

    def plot_f1(self):
        "lots of fun grouping options to handle"
        pass


if __name__ == "__main__":

    example_data = "/Users/au554730/Desktop/Projects/psycop-ml-utils/tests/test_model_comparison/agg_mfccs_eval.csv"
    df = pd.read_csv(example_data)

    df = df[["label", "scores", "id"]]
    scores_mapping = {0: "TD", 1: "DEPR", 2: "SCHZ", 3: "ASD"}

    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "scores": [
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.7, 0.1, 0.1],
            ],
            "label": ["ASD", "DEPR", "TD", "SCHZ"],
        }
    )

    model_comparer = ModelComparison(score_mapping=scores_mapping)

    model_comparer.load_data_from_csv(example_data)

    x = model_comparer.compute_metrics(df["label"], df["prediction"])
    print("2")
