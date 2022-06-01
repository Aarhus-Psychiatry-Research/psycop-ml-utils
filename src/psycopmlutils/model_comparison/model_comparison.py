""""Tools for model comparison"""

from unittest.util import unorderable_list_difference
import pandas as pd

from typing import Union, List, Optional
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


import numpy as np

from psycopmlutils.model_comparison.utils import (
    add_metadata_cols,
    aggregate_predictions,
    labels_to_int,
    idx_to_class,
    get_metadata_cols,
    add_metadata_cols,
    scores_to_probs,
)


class ModelComparison:
    def __init__(
        self,
        label_col: str = "label",
        scores_col: str = "scores",
        id_col: Optional[str] = None,
        id2label: Optional[dict] = None,
        metadata_cols: Optional[List[str]] = None,
    ):
        """Methods for loading and transforming dataframes with 1 row per prediction into aggregated results.
        Expects files/dataframes to have the following columns:
            label,scores,[id_col], [optional_grouping_columns]
        Where `label` is the true label for the row, `scores` is the list output
        of a softmax layer or a float. If data is grouped by an id, specifying
        an id_col will allow the class methods to also calculate performance by
        id.

        Args:
            id_col (Optional[str]): id column in case of multiple predictions.
            label_col (str): column containing ground truth label
            scores (str): column containing softmaxed output or float of probabilities
            id2label (Optional[dict]): Mapping from scores index to group (should match label). E.g. if scores [0.3, 0.6, 0.1]
                id2label={0:"control", 1:"depression", 2:"schizophrenia}. Not needed for binary models if labels are 0 and 1.
            metadata_cols (Optional[List[str]], optional): Column(s) containing metadata to add to the performance dataframe.
                Each column should only contain 1 unique value. E.g. model_name, modality.. Auto-detects columns with only 1
                unique value if not specified.

        Returns:
            pd.Dataframe: _description_
        """
        self.label_col = label_col
        self.scores_col = scores_col
        self.id_col = id_col
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()} if id2label else None
        if isinstance(metadata_cols, str):
            metadata_cols = [metadata_cols]
        self.metadata_cols = metadata_cols

    def transform_data_from_file(self, path: Union[str, Path]) -> pd.DataFrame:
        path = Path(path)
        if path.suffix != ".jsonl":
            raise ValueError(
                f"Only .jsonl files are supported for import, not {path.suffix}"
            )
        df = pd.read_json(path, orient="records", lines=True)
        return self.transform_data_from_dataframe(df)

    def transform_data_from_folder(
        self, path: Union[str, Path], pattern: str = "*.jsonl"
    ) -> pd.DataFrame:
        """Loads and transforms all files matching a pattern in a folder to the long result format.
        Only supports jsonl for now.

        Args:
            path (Union[str, Path]): Path to folder.
            pattern (str, optional): Pattern to match. Defaults to "*.jsonl".

        Returns:
            pd.Dataframe: Long format dataframe with aggreagted predictions.
        """
        path = Path(path)
        dfs = [self.transform_data_from_file(p) for p in path.glob(pattern)]
        return pd.concat(dfs)

    def transform_data_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe of individual predictions into long format with
        results and optionally adds metadata. Row-level performance is identified
        by the `level` column as 'overall'. If the class was instantiated with an id_col,
        id-level performance is added and identfied via the ´level´ column as 'id'.

        Args:
            df (pd.DataFrame): Dataframe with 1 row per prediction.

        Returns:
            pd.Dataframe: Long format dataframe with aggregated predictions.
        """
        performance = self._evaluate_single_model(df, aggregate_by_id=False)
        performance["level"] = "overall"

        if self.id_col:
            # Calculate performance id and add to the dataframe
            performance_by_id = self._evaluate_single_model(df, aggregate_by_id=True)
            performance_by_id["level"] = "id"
            performance = pd.concat([performance, performance_by_id])

        if self.metadata_cols:
            # Add metadata if specified
            metadata = get_metadata_cols(df, self.metadata_cols)
            performance = add_metadata_cols(performance, metadata)
        return performance

    def _evaluate_single_model(
        self, df: pd.DataFrame, aggregate_by_id: bool
    ) -> pd.DataFrame:
        """Transforms a dataframe of individual predictions into long format with columns
        ´class´, ´score_type`, and ´value´.

        Args:
            df (pd.DataFrame): Dataframe with one prediction per row
            aggregate_by_id (bool): Whether to calculate predictions on row level or aggregate by id

        Returns:
            pd.Dataframe: _description_
        """
        if aggregate_by_id:
            df = aggregate_predictions(df, self.id_col, self.scores_col, self.label_col)

        # get predicted labels
        if df[self.scores_col].dtype != "float":
            argmax_indices = df[self.scores_col].apply(lambda x: np.argmax(x))
            predictions = idx_to_class(argmax_indices, self.id2label)
        else:
            predictions = np.round(df[self.scores_col])

        metrics = self.compute_metrics(df[self.label_col], predictions)

        # calculate roc if binary model
        # convoluted way to take first element of scores column and test how how many items it contains 
        first_score = df[self.scores_col].take([0]).values[0]
        
        if isinstance(first_score, float) or len(first_score) <= 2:
            probs = scores_to_probs(df[self.scores_col])
            label_int = labels_to_int(df[self.label_col], self.label2id)
            roc_df = self._calculate_roc(label_int, probs)

            metrics = pd.concat([metrics, roc_df]).reset_index()

        return metrics

    @staticmethod
    def _calculate_roc(labels: Union[pd.Series, List], predicted: Union[pd.Series, List]):
        roc_auc = roc_auc_score(labels, predicted)
        return pd.DataFrame([{"class" : "overall", "score_type" : "auc", "value" : roc_auc}])

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
        performance["confusion_matrix-overall"] = confusion_matrix(labels, predicted)

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
        performance = performance[["class", "score_type", "value"]]
        return performance


if __name__ == "__main__":

    # example_data = "/Users/au554730/Desktop/Projects/psycop-ml-utils/tests/test_model_comparison/agg_mfccs_eval.jsonl"
    #    df = pd.read_csv(example_data)

    #   df = df[["label", "scores", "id"]]
    # scores_mapping = {0: "TD", 1: "DEPR", 2: "SCHZ", 3: "ASD"}

    multiclass_df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "scores": [
                # id 1
                [0.8, 0.1, 0.05, 0.05],
                [0.4, 0.7, 0.1, 0.1],
                # id 2
                [0.1, 0.05, 0.8, 0.05],
                [0.1, 0.7, 0.1, 0.1],
                # id 3
                [0.1, 0.1, 0.7, 0.1],
                [0.2, 0.5, 0.2, 0.1],
                # id 4
                [0.1, 0.1, 0.2, 0.6],
                [0.1, 0.2, 0.1, 0.6],
            ],
            "label": ["ASD", "ASD", "DEPR", "DEPR", "TD", "TD", "SCHZ", "SCHZ"],
            "model_name": ["test"] * 8,
        }
    )
    id2label = {0: "ASD", 1: "DEPR", 2: "TD", 3: "SCHZ"}

    model_comparer = ModelComparison(
        id2label=id2label, id_col="id", metadata_cols="model_name"
    )

    res = model_comparer.transform_data_from_dataframe(multiclass_df)


    binary_df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "scores": [[0.8, 0.2], [0.5, 0.5], [0.4, 0.6], [0.9, 0.1]],
            "label": ["TD", "TD", "DEPR", "DEPR"],
            "optional_grouping1": ["grouping1"] * 4,
            "optional_grouping2": ["grouping2"] * 4,
        }
    )

    model_comparer = ModelComparison(
        id_col="id",
        metadata_cols=["optional_grouping1", "optional_grouping2"],
        id2label={0: "TD", 1: "DEPR"},
    )
    res = model_comparer.transform_data_from_dataframe(binary_df)
