"""Code to generate data integrity and train/val/test drift reports."""
from pathlib import Path
from typing import Tuple

import pandas as pd
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (FeatureLabelCorrelation,
                                       FeatureLabelCorrelationChange,
                                       IdentifierLabelCorrelation,
                                       TrainTestLabelDrift)
from deepchecks.tabular.suites import data_integrity, train_test_validation
from wasabi import Printer


def check_feature_sets_dir(path: Path) -> None:
    """Runs Deepcheck data integrity and train/val/test drift checks for a
    given directory containing train/val/test files. Expects the directory to
    contain 3 csv files, where the files have ["train", "val", "test"]
    somewhere in their name.

    The resulting reports are saved to a sub directory as .html files.

    Args:
        path (Path): Path to a directory containing train/val/test files
    """
    msg = Printer(timestamp=True)
    msg.info("Running data integrity checks...")

    out_dir = path / "deepchecks"
    if not out_dir.exists():
        out_dir.mkdir()
    # create subfolder for outcome specific checks
    outcome_checks_dir = out_dir / "outcomes"
    if not outcome_checks_dir.exists():
        outcome_checks_dir.mkdir()

    ###################
    #### DATA INTEGRITY
    ###################

    # Only running data integrity checks on the training set to reduce the
    # chance of any form of peaking at the test set
    train_predictors, train_outcomes = load_split_predictors_and_outcomes(
        path=path,
        split="train",
        include_id=False,
    )
    ds = Dataset(df=train_predictors, datetime_name="timestamp")

    # Running checks that do not require a label
    integ_suite = data_integrity()
    suite_results = integ_suite.run(ds)
    suite_results.save_as_html(str(out_dir / "data_integrity.html"))

    # Running checks that require a label for each outcome
    label_checks = label_integrity_checks()
    for outcome_column in train_outcomes.columns:
        msg.info(f"Running data integrity for {outcome_column}")
        ds = Dataset(
            df=train_predictors,
            datetime_name="timestamp",
            label=train_outcomes[outcome_column],
        )
        suite_results = label_checks.run(ds)
        suite_results.save_as_html(
            str(outcome_checks_dir / f"{outcome_column}_check.html")
        )

    msg.good("Finshed data integrity checks!")

    #####################
    #### SPLIT VALIDATION
    #####################

    msg.info("Running split validation...")
    # Running data validation checks on train/val and train/test splits that do not
    # require a label
    validation_suite = train_test_validation()

    train_predictors, train_outcomes = load_split_predictors_and_outcomes(
        path=path,
        split="train",
        include_id=True,
    )
    val_predictors, val_outcomes = load_split_predictors_and_outcomes(
        path=path,
        split="val",
        include_id=True,
    )
    test_predictors, test_outcomes = load_split_predictors_and_outcomes(
        path=path,
        split="test",
        include_id=True,
    )

    train_ds = Dataset(
        train_predictors,
        index_name="dw_ek_borger",
        datetime_name="timestamp",
    )
    val_ds = Dataset(
        val_predictors,
        index_name="dw_ek_borger",
        datetime_name="timestamp",
    )
    test_ds = Dataset(
        test_predictors,
        index_name="dw_ek_borger",
        datetime_name="timestamp",
    )
    suite_results = validation_suite.run(train_ds, val_ds)
    suite_results.save_as_html(str(out_dir / "train_val_integrity.html"))
    suite_results = validation_suite.run(train_ds, test_ds)
    suite_results.save_as_html(str(out_dir / "train_test_integrity.html"))

    # Running checks that require a label for each outcome
    label_split_check = label_split_checks()

    split_dict = {
        "val": {"predictors": val_predictors, "outcomes": val_outcomes},
        "test": {"predictors": test_predictors, "outcomes": test_outcomes},
    }

    for split in split_dict:
        for outcome_column in train_outcomes:
            msg.info(f"Running split validation for train/{split} and {outcome_column}")
            train_ds = Dataset(
                df=train_predictors,
                index_name="dw_ek_borger",
                datetime_name="timestamp",
                label=train_outcomes[outcome_column],
            )
            split_ds = Dataset(
                df=split["predictors"],
                index_name="dw_ek_borger",
                datetime_name="timestamp",
                label=split["outcomes"][outcome_column],
            )
            suite_results = label_split_check.run(train_ds, split_ds)
            suite_results.save_as_html(
                str(outcome_checks_dir / f"train_{split}_{outcome_column}_check.html")
            )

    msg.good(f"All data checks done! Saved to {out_dir}")


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


def load_split_predictors_and_outcomes(
    path: Path,
    split: str,
    include_id: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads a given data split from a directory and returns predictors and
    outcomes separately.

    Args:
        path (Path): Path to directory containing data files
        split (str): Which split to load
        include_id (bool): Whether to include 'dw_ek_borger' in predictor df

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple where first element is the
        predictors and second element the outcomes
    """
    split = load_split(path, split)
    predictors, outcomes = separate_predictors_and_outcome(split, include_id=include_id)
    return predictors, outcomes


def separate_predictors_and_outcome(
    df: pd.DataFrame,
    include_id: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split predictors and outcomes into two dataframes. Assumes predictors to
    be prefixed with 'pred', and outcomes to be prefixed with 'outc'. Timestamp
    is also returned for predictors, and optionally also dw_ek_borger.

    Args:
        df (pd.DataFrame): Dataframe containing generates features
        include_id (bool): Whether to include 'dw_ek_borger' in predictor df

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple where first element is the
        predictors and second element the outcomes
    """
    pred_regex = (
        "^pred|^timestamp" if not include_id else "^pred|^timestamp|dw_ek_borger"
    )
    predictors = df.filter(regex=pred_regex)
    outcomes = df.filter(regex="^outc")
    return predictors, outcomes


def load_split(path: Path, split: str) -> pd.DataFrame:
    """Loads a given data split as a dataframe.

    Args:
        path (Path): Path to directory containing data files
        split (str): Which string to look for (e.g. 'train', 'val', 'test')

    Returns:
        pd.DataFrame: The loaded dataframe
    """
    return pd.read_csv(list(path.glob(f"*{split}*"))[0])
