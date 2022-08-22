import time
from pathlib import Path

import numpy as np
import pandas as pd
from wasabi import msg

import psycopmlutils.loaders  # noqa
import wandb
from psycopmlutils.timeseriesflattener import (
    FlattenedDataset,
    create_feature_combinations,
)
from psycopmlutils.utils import FEATURE_SETS_PATH

if __name__ == "__main__":
    # set path to save features to
    SAVE_PATH = FEATURE_SETS_PATH / "t2d"

    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir()

    RESOLVE_MULTIPLE = ["mean", "max", "min"]
    LOOKBEHIND_DAYS = [365, 730, 1825, 9999]

    PREDICTOR_LIST = create_feature_combinations(
        [
            {
                "predictor_df": "hba1c",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": ["mean", "max", "min", "count"],
                "fallback": np.nan,
            },
            {
                "predictor_df": "alat",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
            },
            {
                "predictor_df": "hdl",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
            },
            {
                "predictor_df": "ldl",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
            },
        ],
    )

    event_times = psycopmlutils.loaders.LoadOutcome.t2d()

    msg.info(f"Generating {len(PREDICTOR_LIST)} features")

    msg.info("Loading prediction times")
    prediction_times = psycopmlutils.loaders.LoadVisits.physical_visits_to_psychiatry()

    msg.info("Initialising flattened dataset")
    flattened_df = FlattenedDataset(prediction_times_df=prediction_times, n_workers=60)
    flattened_df.add_age(psycopmlutils.loaders.LoadDemographic.birthdays())

    # Outcome
    msg.info("Adding outcome")
    for i in [0.5, 1, 2, 3, 4, 5]:
        lookahead_days = i * 365.25
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

    end_time = time.time()

    # Predictors
    msg.info("Adding static predictors")
    flattened_df.add_static_info(psycopmlutils.loaders.LoadDemographic.sex_female())

    start_time = time.time()

    msg.info("Adding temporal predictors")
    flattened_df.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictors=PREDICTOR_LIST,
    )

    # Finish
    msg.good(
        f"Finished adding {len(PREDICTOR_LIST)} predictors, took {round((end_time - start_time)/60, 1)} minutes",
    )

    msg.info(
        f"Dataframe size is {flattened_df.df.memory_usage(index=True, deep=True).sum() / 1024 / 1024} MiB",
    )

    msg.good("Done!")

    # Split and save to disk
    splits = ["test", "val", "train"]

    flattened_df_ids = flattened_df.df["dw_ek_borger"].unique()

    # Version table with current date and time
    # prefix with user name to avoid potential clashes
    current_user = Path().home().name + "_"
    file_prefix = current_user + f"psycop_t2d_{time.strftime('%Y_%m_%d_%H_%M')}"

    # Log poetry lock file and file prefix to WandB for reproducibility
    feature_settings = {
        "filename": file_prefix,
        "save_path": SAVE_PATH,
        "predictor_list": PREDICTOR_LIST,
    }

    run = wandb.init(project="psycop-feature-files", config=feature_settings)
    wandb.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")

    for dataset_name in splits:
        df_split_ids = psycopmlutils.loaders.LoadIDs.load(split=dataset_name)

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
        filename = f"{file_prefix}_{dataset_name}.csv"
        msg.info(f"Saving {filename} to disk")

        file_path = SAVE_PATH / filename

        split_df.to_csv(file_path, index=False)

        msg.good(f"{dataset_name}: Succesfully saved to {file_path}")
    wandb.finish()
