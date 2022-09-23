"""Code to generate data integrity and train/val/test drift reports."""
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from deepchecks.core.suite import SuiteResult
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    CategoryMismatchTrainTest,
    DataDuplicates,
    DatasetsSizeComparison,
    FeatureFeatureCorrelation,
    FeatureLabelCorrelation,
    FeatureLabelCorrelationChange,
    IdentifierLabelCorrelation,
    IndexTrainTestLeakage,
    IsSingleValue,
    MixedDataTypes,
    MixedNulls,
    NewLabelTrainTest,
    OutlierSampleDetection,
    StringLengthOutOfBounds,
    TrainTestLabelDrift,
)
from wasabi import Printer

from psycopmlutils.loaders.flattened import load_split_outcomes, load_split_predictors


def pruned_data_integrity_checks(**kwargs) -> Suite:
    """Deepchecks data integrity suite with only wanted checks.
    Disables: SpecialCharacters, StringMismatch, ConflictingLabels.

    Args:
        **kwargs: keyword arguments to be passed to all checks.

    Returns:
        Suite: a deepchecks Suite
    """
    return Suite(
        "Data Integrity Suite",
        IsSingleValue(**kwargs).add_condition_not_single_value(),
        MixedNulls(**kwargs).add_condition_different_nulls_less_equal_to(),
        MixedDataTypes(**kwargs).add_condition_rare_type_ratio_not_in_range(),
        DataDuplicates(**kwargs).add_condition_ratio_less_or_equal(),
        StringLengthOutOfBounds(
            **kwargs
        ).add_condition_ratio_of_outliers_less_or_equal(),
        OutlierSampleDetection(**kwargs),
        FeatureLabelCorrelation(**kwargs).add_condition_feature_pps_less_than(),
        FeatureFeatureCorrelation(
            **kwargs
        ).add_condition_max_number_of_pairs_above_threshold(),
        IdentifierLabelCorrelation(**kwargs).add_condition_pps_less_or_equal(),
    )


def label_integrity_checks() -> Suite:
    """Deepchecks data integrity suite for checks that require a label.

    Returns:
        Suite: A deepchecks Suite

    Example:
    >>> suite = label_integrity_checks()
    >>> result = suite.run(some_deepchecks_dataset)
    >>> result.show()
    """
    return Suite(
        "Data integrity checks requiring labels",
        IdentifierLabelCorrelation().add_condition_pps_less_or_equal(),
        FeatureLabelCorrelation().add_condition_feature_pps_less_than(),
    )


def custom_train_test_validation(**kwargs) -> Suite:
    """Deepchecks train/test validation suite for train/test checks which slow
    checks disabled.

    Args:
        **kwargs: Keyword arguments to pass to the Suite constructor.

    Returns:
        Suite: A deepchecks Suite
    """
    return Suite(
        "Train Test Validation Suite",
        DatasetsSizeComparison(
            **kwargs
        ).add_condition_test_train_size_ratio_greater_than(),
        NewLabelTrainTest(**kwargs).add_condition_new_labels_number_less_or_equal(),
        CategoryMismatchTrainTest(
            **kwargs
        ).add_condition_new_category_ratio_less_or_equal(),
        IndexTrainTestLeakage(**kwargs).add_condition_ratio_less_or_equal(),
    )


def label_split_checks() -> Suite:
    """Deepchecks train/test validation suite for checks that require a label.

    Returns:
        Suite: a deepchecks Suite
    """
    return Suite(
        "Split validation checks requiring labels",
        FeatureLabelCorrelationChange()
        .add_condition_feature_pps_difference_less_than()
        .add_condition_feature_pps_in_train_less_than(),
        TrainTestLabelDrift().add_condition_drift_score_less_than(),
    )


def get_failed_check_names(result: SuiteResult) -> list[str]:
    """Returns a list of names of failed checks.

    Args:
        result (SuiteResult): A deepchecks SuiteResult

    Returns:
        list[str]: list of names of failed checks
    """
    return [
        check_result.check.name() for check_result in result.get_not_passed_checks()
    ]


def check_train_data_integrity(
    feature_set_csv_dir: Path,
    out_dir: Path,
    train_outcomes_df: pd.DataFrame,
    outcome_checks_dir: Path,
    n_rows: Optional[int] = None,
):
    """Runs Deepcheck data integrity checks for the train split.

    Args:
        feature_set_csv_dir (Path): Path to a directory containing train/val/test files
        out_dir (Path): Path to the directory where the reports should be saved
        train_outcomes_df (pd.DataFrame): The train outcomes dataframe
        outcome_checks_dir (Path): Path to the directory where the outcome specific reports should be saved
        n_rows (Optional[int]): Whether to only load a subset of the data.
            Should only be used for debugging.

    Returns:
        failures (dict): A dictionary containing the failed checks
    """
    failures = {}

    msg = Printer(timestamp=True)

    msg.info("Running data integrity checks...")

    # Only running data integrity checks on the training set to reduce the
    # chance of any form of peaking at the test set
    train_predictors = load_split_predictors(
        feature_set_csv_dir=feature_set_csv_dir,
        split="train",
        include_id=True,
        nrows=n_rows,
    )

    data_s = Dataset(
        df=train_predictors,
        index_name="dw_ek_borger",
        datetime_name="timestamp",
    )

    # Running checks that do not require a label
    integ_suite = pruned_data_integrity_checks(
        timeout=0,
    )  # timeout=0 removes timeout

    suite_results = integ_suite.run(data_s)
    suite_results.save_as_html(str(out_dir / "data_integrity.html"))

    failures["data_integrity"] = get_failed_check_names(suite_results)

    # Running checks that require a label for each outcome
    label_checks = label_integrity_checks()

    for outcome_column in train_outcomes_df.columns:
        msg.info(f"Running data integrity for {outcome_column}")

        data_s = Dataset(
            df=train_predictors,
            index_name="dw_ek_borger",
            datetime_name="timestamp",
            label=train_outcomes_df[outcome_column],
        )

        suite_results = label_checks.run(data_s)

        suite_results.save_as_html(
            str(outcome_checks_dir / f"{outcome_column}_check.html"),
        )

        failures[f"{outcome_column}_check"] = get_failed_check_names(
            suite_results,
        )

    msg.good("Finshed data integrity checks!")

    return failures


def get_suite_results_for_split_pair_and_save_to_disk(
    out_dir: Path,
    deepchecks_suite: Any,
    split_dicts: dict[str, dict[str, Any]],
    split_pair: tuple[str, str],
    file_suffix: str = "",
) -> SuiteResult:
    """Runs a Deepchecks suite on a given split and saves the results to a
    file.

    Args:
        out_dir (Path): Path to the directory where the results should be saved
        deepchecks_suite (Any): Deepchecks suite to run
        split_dicts (dict[str, dict[str, Any]]): Dictionary containing the splits.
        split_pair (tuple[str, str]): Splits to run the suite on
        file_suffix (str, optional): Suffix to add to the file name. Defaults to "".

    Returns:
        SuiteResults: Results of the suite
    """

    suite_results = deepchecks_suite.run(
        split_dicts[split_pair[0]]["ds"],
        split_dicts[split_pair[1]]["ds"],
    )

    suite_results.save_as_html(
        str(out_dir / f"{split_pair[0]}_{split_pair[1]}_{file_suffix}.html"),
    )

    return suite_results


def get_split_as_ds_dict(
    feature_set_csv_dir: Path,
    n_rows: Optional[int],
    split: str,
) -> dict[str, Any]:
    """Loads a split as a Deepchecks Dataset dict.

    Args:
        feature_set_csv_dir (Path): Path to a directory containing train/val/test files
        n_rows (Optional[int]): Whether to only load a subset of the data.
            Should only be used for debugging.
        split (str): Which split to load

    Returns:
        dict: Deepchecks Dataset dict
    """

    predictors = load_split_predictors(
        feature_set_csv_dir=feature_set_csv_dir,
        split=split,
        include_id=True,
        nrows=n_rows,
    )
    outcomes = load_split_outcomes(
        feature_set_csv_dir=feature_set_csv_dir,
        split=split,
        nrows=n_rows,
    )

    data_s = Dataset(
        df=predictors,
        index_name="dw_ek_borger",
        datetime_name="timestamp",
    )

    return {
        "predictors": predictors,
        "outcomes": outcomes,
        "ds": data_s,
    }


def run_validation_requiring_split_comparison(
    feature_set_csv_dir: Path,
    split_names: list[str],
    out_dir: Path,
    train_outcome_df: pd.DataFrame,
    n_rows: Optional[int] = None,
):
    """Runs Deepcheck data validation checks for the train/val/test splits.

    Args:
        feature_set_csv_dir (Path): Path to a directory containing train/val/test files
        split_names (list[str]): list of splits to check (train, val, test)
        out_dir (Path): Path to the directory where the reports should be saved
        train_outcome_df (pd.DataFrame): The train outcomes.
        n_rows (int): Whether to only load a subset of the data.
            Should only be used for debugging.
    """
    msg = Printer(timestamp=True)

    failed_checks = {}

    validation_suite = custom_train_test_validation()

    split_dicts = {}

    for split_name in split_names:
        split_dicts[split_name] = get_split_as_ds_dict(
            feature_set_csv_dir=feature_set_csv_dir,
            n_rows=n_rows,
            split=split_name,
        )

    for split_pair in (("train", "val"), ("train", "test")):
        suite_results = get_suite_results_for_split_pair_and_save_to_disk(
            out_dir=out_dir,
            deepchecks_suite=validation_suite,
            split_dicts=split_dicts,
            split_pair=split_pair,
            file_suffix="integrity",
        )

        failed_checks[
            f"{split_pair[0]}_{split_pair[1]}_integrity"
        ] = get_failed_check_names(suite_results)

    for split_name, split_contents in split_dicts.items():
        # don't check train/train
        if split_name == "train":
            continue

        for outcome_col in train_outcome_df:
            msg.info(
                f"Running split validation for train/{split_name} and {outcome_col}",
            )

            deepchecks_ds_dict = {
                "train": Dataset(
                    df=split_dicts["train"]["predictors"],
                    index_name="dw_ek_borger",
                    datetime_name="timestamp",
                    label=split_dicts["train"]["outcomes"][outcome_col],
                ),
                split_name: Dataset(
                    df=split_contents["predictors"],
                    index_name="dw_ek_borger",
                    datetime_name="timestamp",
                    label=split_contents["outcomes"][outcome_col],
                ),
            }

            suite_results = get_suite_results_for_split_pair_and_save_to_disk(
                out_dir=out_dir,
                deepchecks_suite=label_split_checks(),
                split_dicts=deepchecks_ds_dict,
                split_pair=("train", split_name),
                file_suffix=f"{outcome_col}_check",
            )

            failed_checks[
                f"train_{split_name}_{outcome_col}_check"
            ] = get_failed_check_names(suite_results)

        msg.good(f"All data checks done! Saved to {out_dir}")

        if len(failed_checks.keys()) > 0:
            msg.warn(f"Failed checks: {failed_checks}")


def save_feature_set_integrity_from_dir(  # noqa pylint: disable=too-many-statements
    feature_set_csv_dir: Path,
    n_rows: Optional[int] = None,
    split_names: Optional[list[str]] = None,
    out_dir: Optional[Path] = None,
) -> None:
    """Runs Deepcheck data integrity and train/val/test checks for a given
    directory containing train/val/test files. Splits indicates which data.
    splits to check.

    The resulting reports are saved to a sub directory as .html files.

    Args:
        feature_set_csv_dir (Path): Path to a directory containing train/val/test files
        n_rows (Optional[int]): Whether to only load a subset of the data.
            Should only be used for debugging.
        split_names (list[str]): list of splits to check (train, val, test)
        out_dir (Optional[Path]): Path to the directory where the reports should be saved
    """
    if split_names is None:
        split_names = ["train", "val", "test"]

    if out_dir is None:
        out_dir = feature_set_csv_dir / "deepchecks"
    else:
        out_dir = out_dir / "deepchecks"

    if not out_dir.exists():
        out_dir.mkdir()

    train_outcomes_df = load_split_outcomes(
        feature_set_csv_dir=feature_set_csv_dir,
        split="train",
        nrows=n_rows,
    )

    failed_checks = (
        {}
    )  # Collect failed checks for error messages at the end of the function

    # Check if file splits exist before running checks
    for split_name in split_names:
        file = list(feature_set_csv_dir.glob(f"*{split_name}*.csv"))

        if not file:
            raise ValueError(f"{split_name} split not found in {feature_set_csv_dir}")
        if len(file) > 1:
            raise ValueError(
                f"Multiple {split_name} files found in {feature_set_csv_dir}",
            )

    # Create subfolder for outcome specific checks
    outcome_checks_dir = out_dir / "outcomes"
    if not outcome_checks_dir.exists():
        outcome_checks_dir.mkdir()

    # Check train data integrity
    if "train" in split_names:
        failures = check_train_data_integrity(
            feature_set_csv_dir=feature_set_csv_dir,
            n_rows=n_rows,
            out_dir=out_dir,
            outcome_checks_dir=outcome_checks_dir,
            train_outcomes_df=train_outcomes_df,
        )

        # Add all keys in failures to failed_checks
        for k, v in failures.items():
            failed_checks[k] = v

    # Running data validation checks on train/val and train/test splits that do not
    # require a label
    run_validation_requiring_split_comparison(
        feature_set_csv_dir=feature_set_csv_dir,
        split_names=split_names,
        n_rows=n_rows,
        out_dir=out_dir,
        train_outcome_df=train_outcomes_df,
    )
