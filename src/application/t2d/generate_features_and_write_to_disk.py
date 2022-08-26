import time
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from wasabi import msg

import psycopmlutils.loaders.raw  # noqa
from application.t2d.features_blood_samples import create_lab_feature_combinations
from application.t2d.features_diagnoses import create_diag_feature_combinations
from application.t2d.features_medications import create_medication_feature_combinations
from psycopmlutils.timeseriesflattener import FlattenedDataset
from psycopmlutils.timeseriesflattener.data_integrity import (
    check_feature_set_integrity_from_dir,
)
from psycopmlutils.utils import FEATURE_SETS_PATH

if __name__ == "__main__":
    # set path to save features to
    SAVE_PATH = FEATURE_SETS_PATH / "t2d"

    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir()

    RESOLVE_MULTIPLE = ["latest"]  # , "max", "min", "mean"]
    LOOKBEHIND_DAYS = [365]  # , 730, 1825, 9999]

    LAB_PREDICTORS = create_lab_feature_combinations(
        RESOLVE_MULTIPLE=RESOLVE_MULTIPLE,
        LOOKBEHIND_DAYS=LOOKBEHIND_DAYS,
    )

    DIAGNOSIS_PREDICTORS = create_diag_feature_combinations(
        resolve_multiple=RESOLVE_MULTIPLE,
        lookbehind_days=LOOKBEHIND_DAYS,
        fallback=0,
    )

    MEDICATION_PREDICTORS = create_medication_feature_combinations(
        LOOKBEHIND_DAYS=LOOKBEHIND_DAYS,
        RESOLVE_MULTIPLE=["count"],
        fallback=0,
    )

    PREDICTOR_LIST = MEDICATION_PREDICTORS + DIAGNOSIS_PREDICTORS + LAB_PREDICTORS

    event_times = psycopmlutils.loaders.LoadOutcome.t2d()

    msg.info(f"Generating {len(PREDICTOR_LIST)} features")

    msg.info("Loading prediction times")
    prediction_times = (
        psycopmlutils.loaders.raw.LoadVisits.physical_visits_to_psychiatry()
    )

    msg.info("Initialising flattened dataset")
    flattened_df = FlattenedDataset(prediction_times_df=prediction_times, n_workers=60)
    flattened_df.add_age(psycopmlutils.loaders.raw.LoadDemographic.birthdays())

    # Outcome
    msg.info("Adding outcome")
    for i in [0.5, 1, 2, 3, 4, 5]:
        lookahead_days = int(i * 365)
        msg.info(f"Adding outcome with {lookahead_days} days of lookahead")
        flattened_df.add_temporal_outcome(
            outcome_df=event_times,
            lookahead_days=lookahead_days,
            resolve_multiple="max",
            fallback=0,
            outcome_df_values_col_name="value",
            new_col_name="t2d",
            incident=True,
            dichotomous=True,
        )

    # Add timestamp from outcomes
    flattened_df.add_static_info(
        info_df=event_times,
        prefix="",
        input_col_name="timestamp",
        output_col_name="timestamp_first_t2d",
    )
    msg.good("Finished adding outcome")

    # Predictors
    msg.info("Adding static predictors")
    flattened_df.add_static_info(psycopmlutils.loaders.raw.LoadDemographic.sex_female())

    start_time = time.time()

    msg.info("Adding temporal predictors")
    flattened_df.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictors=PREDICTOR_LIST,
    )

    end_time = time.time()

    # Finish
    msg.good(
        f"Finished adding {len(PREDICTOR_LIST)} predictors, took {round((end_time - start_time)/60, 1)} minutes",
    )

    msg.info(
        f"Dataframe size is {int(flattened_df.df.memory_usage(index=True, deep=True).sum() / 1024 / 102)} MiB",
    )

    msg.good("Done!")

    # Split and save to disk
    splits = ["test", "val", "train"]

    flattened_df_ids = flattened_df.df["dw_ek_borger"].unique()

    # Version table with current date and time
    # prefix with user name to avoid potential clashes
    current_user = Path().home().name
    file_prefix = current_user + f"_{time.strftime('%Y_%m_%d_%H_%M')}"

    # Create directory to store all files related to this run
    sub_dir = SAVE_PATH / current_user + f"_{time.strftime('%Y_%m_%d_%H_%M')}"
    sub_dir.mkdir()

    # Log poetry lock file and file prefix to WandB for reproducibility
    feature_settings = {
        "file_prefix": file_prefix,
        "save_path": sub_dir / file_prefix,
        "predictor_list": PREDICTOR_LIST,
    }

    run = wandb.init(project="psycop-feature-files", config=feature_settings)
    wandb.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")

    # Create splits
    for dataset_name in splits:
        df_split_ids = psycopmlutils.loaders.raw.LoadIDs.load(split=dataset_name)

        # Find IDs which are in split_ids, but not in flattened_df
        split_ids = df_split_ids["dw_ek_borger"].unique()
        flattened_df_ids = flattened_df.df["dw_ek_borger"].unique()

        ids_in_split_but_not_in_flattened_df = split_ids[
            ~np.isin(split_ids, flattened_df_ids)
        ]

        msg.warn(
            f"{dataset_name}: There are {len(ids_in_split_but_not_in_flattened_df)} ({round(len(ids_in_split_but_not_in_flattened_df)/len(split_ids)*100, 2)}%) ids which are in {dataset_name}_ids but not in flattened_df_ids, will get dropped during merge",
        )

        split_df = pd.merge(flattened_df.df, df_split_ids, how="inner")

        # Version table with current date and time
        filename = f"psycop_t2d_{file_prefix}_{dataset_name}.csv"
        msg.info(f"Saving {filename} to disk")

        file_path = sub_dir / filename

        split_df.to_csv(file_path, index=False)

        msg.good(f"{dataset_name}: Succesfully saved to {file_path}")
    wandb.finish()

    ## Create data integrity report
    check_feature_set_integrity_from_dir(sub_dir, splits=["train", "val", "test"])
