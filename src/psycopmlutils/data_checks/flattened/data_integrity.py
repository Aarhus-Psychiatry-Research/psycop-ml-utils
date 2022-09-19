"""Code to generate data integrity and train/val/test drift reports."""
from pathlib import Path
from typing import List, Optional

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


def get_name_of_failed_checks(result: SuiteResult) -> List[str]:
    """Returns a list of names of failed checks.

    Args:
        result (SuiteResult): A deepchecks SuiteResult

    Returns:
        List[str]: List of names of failed checks
    """
    return [
        check_result.check.name() for check_result in result.get_not_passed_checks()
    ]


def check_feature_set_integrity_from_dir(  # pylint: disable=too-many-statements
    path: Path,
    splits: Optional[List[str]] = None,
    nrows: Optional[int] = None,
) -> None:
    """Runs Deepcheck data integrity and train/val/test checks for a given
    directory containing train/val/test files. Splits indicates which data.

    splits to check. If nrows is provided, only the first nrows are loaded - should only be used for debugging.

    The resulting reports are saved to a sub directory as .html files.

    Args:
        path (Path): Path to a directory containing train/val/test files
        splits (List[str]): List of splits to check (train, val, test)
        nrows (Optional[int]): Whether to only load a subset of the data.
        Should only be used for debugging.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    msg = Printer(timestamp=True)
    failed_checks = (
        {}
    )  # Collect failed checks for error messages at the end of the function

    # Check if file splits exist before running checks
    for split in splits:
        file = list(path.glob(f"*{split}*.csv"))
        if not file or len(file) > 1:
            raise ValueError(f"{split} split not found in {path}")

    out_dir = path / "deepchecks"
    if not out_dir.exists():
        out_dir.mkdir()
    # create subfolder for outcome specific checks
    outcome_checks_dir = out_dir / "outcomes"
    if not outcome_checks_dir.exists():
        outcome_checks_dir.mkdir()

    ###################
    # DATA INTEGRITY
    ###################

    if "train" in splits:
        msg.info("Running data integrity checks...")
        # Only running data integrity checks on the training set to reduce the
        # chance of any form of peaking at the test set
        train_predictors = load_split_predictors(
            path=path,
            split="train",
            include_id=True,
            nrows=nrows,
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
        failed_checks["data_integrity"] = get_name_of_failed_checks(suite_results)

        train_outcomes = load_split_outcomes(path=path, split="train", nrows=nrows)
        # Running checks that require a label for each outcome
        label_checks = label_integrity_checks()
        for outcome_column in train_outcomes.columns:
            msg.info(f"Running data integrity for {outcome_column}")
            data_s = Dataset(
                df=train_predictors,
                index_name="dw_ek_borger",
                datetime_name="timestamp",
                label=train_outcomes[outcome_column],
            )
            suite_results = label_checks.run(data_s)
            suite_results.save_as_html(
                str(outcome_checks_dir / f"{outcome_column}_check.html"),
            )
            failed_checks[f"{outcome_column}_check"] = get_name_of_failed_checks(
                suite_results,
            )

        msg.good("Finshed data integrity checks!")

    #####################
    # SPLIT VALIDATION
    #####################
    msg.info("Running split validation...")
    # Running data validation checks on train/val and train/test splits that do not
    # require a label
    validation_suite = custom_train_test_validation()

    split_dict = {}
    for split in splits:
        predictors = load_split_predictors(
            path=path,
            split=split,
            include_id=True,
            nrows=nrows,
        )
        outcomes = load_split_outcomes(path=path, split=split, nrows=nrows)
        data_s = Dataset(
            df=predictors,
            index_name="dw_ek_borger",
            datetime_name="timestamp",
        )
        split_dict[split] = {
            "predictors": predictors,
            "outcomes": outcomes,
            "ds": data_s,
        }

    suite_results = validation_suite.run(
        split_dict["train"]["ds"],
        split_dict["val"]["ds"],
    )
    suite_results.save_as_html(str(out_dir / "train_val_integrity.html"))
    failed_checks["train_val_integrity"] = get_name_of_failed_checks(suite_results)

    suite_results = validation_suite.run(
        split_dict["train"]["ds"],
        split_dict["test"]["ds"],
    )
    suite_results.save_as_html(str(out_dir / "train_test_integrity.html"))
    failed_checks["train_test_integrity"] = get_name_of_failed_checks(suite_results)

    # Running checks that require a label for each outcome
    label_split_check = label_split_checks()

    for split, content in split_dict.items():
        # don't check train/train
        if split == "train":
            continue
        for outcome_column in train_outcomes:
            msg.info(f"Running split validation for train/{split} and {outcome_column}")
            train_ds = Dataset(
                df=split_dict["train"]["predictors"],
                index_name="dw_ek_borger",
                datetime_name="timestamp",
                label=split_dict["train"]["outcomes"][outcome_column],
            )
            split_ds = Dataset(
                df=content["predictors"],
                index_name="dw_ek_borger",
                datetime_name="timestamp",
                label=content["outcomes"][outcome_column],
            )
            suite_results = label_split_check.run(train_ds, split_ds)
            suite_results.save_as_html(
                str(outcome_checks_dir / f"train_{split}_{outcome_column}_check.html"),
            )
            failed_checks[
                f"train_{split}_{outcome_column}_check"
            ] = get_name_of_failed_checks(suite_results)

        msg.good(f"All data checks done! Saved to {out_dir}")
        if len(failed_checks.keys()) > 0:
            msg.warn(f"Failed checks: {failed_checks}")
