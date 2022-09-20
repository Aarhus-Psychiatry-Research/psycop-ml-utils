"""Main example on how to generate features.

Uses T2D-features. WIP, will be migrated to psycop-t2d when reaching
maturity.
"""

import random
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import psutil
import wandb
from wasabi import Printer

import psycopmlutils.loaders.raw  # noqa
from application.t2d.features_blood_samples import get_lab_feature_spec
from application.t2d.features_diagnoses import get_diagnosis_feature_spec
from application.t2d.features_medications import get_medication_feature_spec
from psycopmlutils.data_checks.flattened.data_integrity import (
    check_feature_set_integrity_from_dir,
)
from psycopmlutils.data_checks.flattened.feature_describer import (
    create_feature_description_from_dir,
)
from psycopmlutils.loaders.raw.pre_load_dfs import pre_load_unique_dfs
from psycopmlutils.timeseriesflattener import (
    FlattenedDataset,
    create_feature_combinations,
)
from psycopmlutils.utils import FEATURE_SETS_PATH


def log_to_wandb(predictor_combinations, save_dir):
    """Log poetry lock file and file prefix to WandB for reproducibility."""

    feature_settings = {
        "save_path": save_dir,
        "predictor_list": predictor_combinations,
    }

    run = wandb.init(project="psycop-feature-files", config=feature_settings)
    run.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")

    run.finish()


def save_feature_set_description_to_disk(
    predictor_combinations: list,
    flattened_csv_dir: Path,
):
    """Describe output.

    Args:
        predictor_combinations (list): List of predictor specs.
        flattened_csv_dir (Path): Path to flattened csv dir.
    """

    # Create data integrity report
    create_feature_description_from_dir(
        path=flattened_csv_dir,
        predictor_dicts=predictor_combinations,
        splits=["train"],
    )

    check_feature_set_integrity_from_dir(
        feature_set_csv_dir=flattened_csv_dir,
        split_names=["train", "val", "test"],
    )


def create_save_dir(
    proj_path: Path,
    feature_set_id: str,
) -> Path:
    """Create save directory.

    Args:
        proj_path (Path): Path to project.
        feature_set_id (str): Feature set id.

    Returns:
        Path: Path to sub directory.
    """

    # Split and save to disk
    # Create directory to store all files related to this run
    save_dir = proj_path / feature_set_id

    if not save_dir.exists():
        save_dir.mkdir()

    return save_dir


def split_and_save_to_disk(
    flattened_df: FlattenedDataset,
    output_dir: Path,
    file_prefix: str,
) -> tuple[Path, str]:
    """Split and save to disk.

    Args:
        flattened_df (FlattenedDataset): Flattened dataset.
        output_dir (Path): Path to output directory.
        file_prefix (str): File prefix.

    Returns:
        tuple[Path, str]: Path to sub directory and file prefix.
    """
    splits = ["test", "val", "train"]
    msg = Printer(timestamp=True)

    flattened_df_ids = flattened_df.df["dw_ek_borger"].unique()

    # Version table with current date and time
    # prefix with user name to avoid potential clashes

    # Create splits
    for dataset_name in splits:
        df_split_ids = psycopmlutils.loaders.raw.load_ids(split=dataset_name)

        # Find IDs which are in split_ids, but not in flattened_df
        split_ids = df_split_ids["dw_ek_borger"].unique()
        flattened_df_ids = flattened_df.df["dw_ek_borger"].unique()

        ids_in_split_but_not_in_flattened_df = split_ids[
            ~np.isin(split_ids, flattened_df_ids)
        ]

        msg.warn(
            f"{dataset_name}: There are {len(ids_in_split_but_not_in_flattened_df)} ({round(len(ids_in_split_but_not_in_flattened_df)/len(split_ids)*100, 2)}%) ids which are in {dataset_name}_ids but not in flattened_df_ids, will get dropped during merge. If examining patients based on physical visits, see 'OBS: Patients without physical visits' on the wiki for more info.",
        )

        split_df = pd.merge(flattened_df.df, df_split_ids, how="inner", validate="m:1")

        # Version table with current date and time
        filename = f"{file_prefix}_{dataset_name}.csv"
        msg.info(f"Saving {filename} to disk")

        file_path = output_dir / filename

        split_df.to_csv(file_path, index=False)

        msg.good(f"{dataset_name}: Succesfully saved to {file_path}")

    return file_prefix


def add_metadata(
    outcome_loader_str: str,
    pre_loaded_dfs: dict[str, pd.DataFrame],
    flattened_df: FlattenedDataset,
) -> FlattenedDataset:
    """Add metadata.

    Args:
        outcome_loader_str (str): String to lookup in catalogue to load outcome.
        pre_loaded_dfs (dict[str, pd.DataFrame]): Dictionary of pre-loaded dataframes.
        flattened_df (FlattenedDataset): Flattened dataset.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    # Add timestamp from outcomes
    flattened_df.add_static_info(
        info_df=pre_loaded_dfs[outcome_loader_str],
        prefix="",
        input_col_name="timestamp",
        output_col_name="timestamp_first_t2d",
    )

    return flattened_df


def add_outcomes(
    outcome_loader_str: str,
    pre_loaded_dfs: dict[str, pd.DataFrame],
    flattened_df: FlattenedDataset,
    lookahead_years: list[Union[int, float]],
) -> FlattenedDataset:
    """Add outcome.

    Args:
        outcome_loader_str (str): String to lookup in catalogue to load outcome.
        pre_loaded_dfs (dict[str, pd.DataFrame]): Dictionary of pre-loaded dataframes.
        flattened_df (FlattenedDataset): Flattened dataset.
        lookahead_years (list[Union[int, float]]): List of lookahead years.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    msg = Printer(timestamp=True)
    msg.info("Adding outcome")

    for i in lookahead_years:
        lookahead_days = int(i * 365)
        msg.info(f"Adding outcome with {lookahead_days} days of lookahead")
        flattened_df.add_temporal_outcome(
            outcome_df=pre_loaded_dfs[outcome_loader_str],
            lookahead_days=lookahead_days,
            resolve_multiple="max",
            fallback=0,
            new_col_name="t2d",
            incident=True,
            dichotomous=True,
        )

    msg.good("Finished adding outcome")

    return flattened_df


def add_predictors(pre_loaded_dfs, predictor_combinations, flattened_df):
    """Add predictors.

    Args:
        pre_loaded_dfs (dict[str, pd.DataFrame]): Dictionary of pre-loaded dataframes.
        predictor_combinations (list[dict[str, dict[str, Any]]]): List of predictor combinations.
        flattened_df (FlattenedDataset): Flattened dataset.
    """

    msg = Printer(timestamp=True)

    msg.info("Adding static predictors")
    flattened_df.add_static_info(pre_loaded_dfs["sex_female"])
    flattened_df.add_age(pre_loaded_dfs["birthdays"])

    start_time = time.time()

    msg.info("Adding temporal predictors")
    flattened_df.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictors=predictor_combinations,
        predictor_dfs=pre_loaded_dfs,
    )

    end_time = time.time()

    # Finish
    msg.good(
        f"Finished adding {len(predictor_combinations)} predictors, took {round((end_time - start_time)/60, 1)} minutes",
    )

    return flattened_df


def create_flattened_dataset(
    outcome_loader_str: str,
    prediction_time_loader_str: str,
    pre_loaded_dfs: dict[str, pd.DataFrame],
    predictor_combinations: list[dict[str, dict[str, Any]]],
    proj_path: Path,
    lookahead_years: list[Union[int, float]],
):
    """Create flattened dataset.

    Args:
        outcome_loader_str (str): String to lookup in catalogue to load outcome.
        prediction_time_loader_str (str): String to lookup in catalogue to load prediction time.
        pre_loaded_dfs (dict[str, pd.DataFrame]): Dictionary of pre-loaded dataframes.
        predictor_combinations (list[dict[str, dict[str, Any]]]): List of predictor combinations.
        proj_path (Path): Path to project directory.
        lookahead_years (list[Union[int,float]]): List of lookahead years.

    Returns:
        FlattenedDataset: Flattened dataset.
    """
    msg = Printer(timestamp=True)

    msg.info(f"Generating {len(predictor_combinations)} features")

    msg.info("Initialising flattened dataset")

    flattened_df = FlattenedDataset(
        prediction_times_df=pre_loaded_dfs[prediction_time_loader_str],
        n_workers=min(
            len(predictor_combinations),
            psutil.cpu_count(logical=False) * 3,
        ),  # * 3 since dataframe loading is IO intensive, cores are likely to wait for a lot of them.
        feature_cache_dir=proj_path / "feature_cache",
    )

    # Outcome
    flattened_df = add_outcomes(
        pre_loaded_dfs=pre_loaded_dfs,
        outcome_loader_str=outcome_loader_str,
        flattened_df=flattened_df,
        lookahead_years=lookahead_years,
    )

    flattened_df = add_predictors(
        pre_loaded_dfs=pre_loaded_dfs,
        predictor_combinations=predictor_combinations,
        flattened_df=flattened_df,
    )

    flattened_df = add_metadata(
        pre_loaded_dfs=pre_loaded_dfs,
        outcome_loader_str=outcome_loader_str,
        flattened_df=flattened_df,
    )

    msg.info(
        f"Dataframe size is {int(flattened_df.df.memory_usage(index=True, deep=True).sum() / 1024 / 102)} MiB",
    )

    return flattened_df


def setup_for_main(
    predictor_spec_list: list[dict[str, dict[str, Any]]],
    feature_sets_path: Path,
    proj_name: str,
) -> tuple[Path, list[dict[str, dict[str, Any]]]]:
    """Setup for main.

    Args:
        predictor_spec_list (list[dict[str, dict[str, Any]]]): List of predictor specifications.
        feature_sets_path (Path): Path to feature sets.
        proj_name (str): Name of project.

    Returns:
        tuple[list[dict[str, dict[str, Any]]], dict[str, pd.DataFrame], Path]: Tuple of predictor combinations, pre-loaded dataframes, and project path.
    """
    predictor_combinations = create_feature_combinations(predictor_spec_list)

    # Some predictors take way longer to complete. Shuffling ensures that e.g. the ones that take the longest aren't all
    # at the end of the list.
    random.shuffle(predictor_spec_list)
    random.shuffle(predictor_combinations)

    proj_path = feature_sets_path / proj_name

    if not proj_path.exists():
        proj_path.mkdir()

    current_user = Path().home().name
    feature_set_id = f"psycop_{proj_name}_{current_user}_{len(predictor_combinations)}_features_{time.strftime('%Y_%m_%d_%H_%M')}"

    return predictor_combinations, proj_path, feature_set_id


def pre_load_project_dfs(
    predictor_spec_list: list[dict[str, dict[str, Any]]],
    outcome_loader_str: str,
    prediction_time_loader_str: str,
) -> dict[str, pd.DataFrame]:
    """Pre-load dataframes for project.

    Args:
        predictor_spec_list (list[dict[str, dict[str, Any]]]): List of predictor specs.
        outcome_loader_str (str): Outcome loader string.
        prediction_time_loader_str (str): Prediction time loader string.

    Returns:
        dict[str, pd.DataFrame]: Dictionary of pre-loaded dataframes.
    """

    dfs_to_preload = (
        predictor_spec_list
        + {"predictor_df": outcome_loader_str}
        + {"predictor_df": prediction_time_loader_str}
        + {"predictor_df": "birthdays"}
        + {"predictor_df": "sex_female"}
    )

    # Many features will use the same dataframes, so we can load them once and reuse them.
    pre_loaded_dfs = pre_load_unique_dfs(
        unique_predictor_dict_list=dfs_to_preload,
    )

    return pre_loaded_dfs


def main(
    proj_name: str,
    feature_sets_path: Path,
    prediction_time_loader_str: str,
    outcome_loader_str: str,
    predictor_spec_list: list[dict[str, dict[str, Any]]],
    lookahead_years: list[Union[int, float]],
):
    """Main function for loading, generating and evaluating a flattened
    dataset.

    Args:
        proj_name (str): Name of project.
        feature_sets_path (Path): Path to where feature sets should be stored.
        prediction_time_loader_str (str): Key to lookup in data_loaders registry for prediction time dataframe.
        outcome_loader_str (str): Key to lookup in data_loaders registry for outcome dataframe.
        predictor_spec_list (list[dict[str, dict[str, Any]]]): List of predictor specs.
        lookahead_years (list[Union[int,float]]): List of lookahead years.
    """

    predictor_combinations, proj_path, feature_set_id = setup_for_main(
        predictor_spec_list=predictor_spec_list,
        feature_sets_path=feature_sets_path,
        proj_name=proj_name,
    )

    pre_loaded_dfs = pre_load_project_dfs(
        predictor_spec_list=predictor_spec_list,
        outcome_loader_str=outcome_loader_str,
        prediction_time_loader_str=prediction_time_loader_str,
    )

    flattened_df = create_flattened_dataset(
        outcome_loader_str=outcome_loader_str,
        prediction_time_loader_str=prediction_time_loader_str,
        pre_loaded_dfs=pre_loaded_dfs,
        predictor_combinations=predictor_combinations,
        proj_path=proj_path,
        lookahead_years=lookahead_years,
    )

    output_dir = create_save_dir(
        feature_set_id=feature_set_id,
        proj_path=proj_path,
    )

    split_and_save_to_disk(
        flattened_df=flattened_df,
        output_dir=output_dir,
        file_prefix=feature_set_id,
    )

    log_to_wandb(
        predictor_combinations=predictor_combinations,
        save_dir=output_dir,  # Save-dir as argument because we want to log the path
    )

    save_feature_set_description_to_disk(
        predictor_combinations=predictor_combinations,
        flattened_csv_dir=output_dir,
    )


if __name__ == "__main__":
    RESOLVE_MULTIPLE = ["max", "min", "mean", "latest"]
    LOOKBEHIND_DAYS = [365, 1825, 9999]

    LAB_PREDICTORS = get_lab_feature_spec(
        resolve_multiple=RESOLVE_MULTIPLE,
        lookbehind_days=LOOKBEHIND_DAYS,
        values_to_load="numerical_and_coerce",
    )

    DIAGNOSIS_PREDICTORS = get_diagnosis_feature_spec(
        resolve_multiple=RESOLVE_MULTIPLE,
        lookbehind_days=LOOKBEHIND_DAYS,
        fallback=0,
    )

    MEDICATION_PREDICTORS = get_medication_feature_spec(
        lookbehind_days=LOOKBEHIND_DAYS,
        resolve_multiple=["count"],
        fallback=0,
    )

    PREDICTOR_SPEC_LIST = DIAGNOSIS_PREDICTORS + LAB_PREDICTORS + MEDICATION_PREDICTORS

    main(
        feature_sets_path=FEATURE_SETS_PATH,
        predictor_spec_list=PREDICTOR_SPEC_LIST,
        proj_name="t2d",
        outcome_loader_str="t2d",
        prediction_time_loader_str="physical_visits_to_psychiatry",
        lookahead_years=[1, 3, 5],
    )
