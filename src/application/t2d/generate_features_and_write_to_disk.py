"""Main example on how to generate features.

Uses T2D-features. WIP, will be migrated to psycop-t2d when reaching
maturity.
"""

import random
import time
from pathlib import Path

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

if __name__ == "__main__":
    msg = Printer(timestamp=True)
    # set path to save features to
    PROJ_PATH = FEATURE_SETS_PATH / "t2d"

    if not PROJ_PATH.exists():
        PROJ_PATH.mkdir()

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
    PREDICTOR_COMBINATIONS = create_feature_combinations(PREDICTOR_SPEC_LIST)

    # Some predictors take way longer to complete. Shuffling ensures that e.g. the ones that take the longest aren't all
    # at the end of the list.
    random.shuffle(PREDICTOR_SPEC_LIST)
    random.shuffle(PREDICTOR_COMBINATIONS)

    # Many features will use the same dataframes, so we can load them once and reuse them.
    pre_loaded_dfs = pre_load_unique_dfs(
        unique_predictor_dict_list=PREDICTOR_SPEC_LIST,
    )

    event_times = psycopmlutils.loaders.raw.t2d()

    msg.info(f"Generating {len(PREDICTOR_COMBINATIONS)} features")

    msg.info("Loading prediction times")
    prediction_times = psycopmlutils.loaders.raw.physical_visits_to_psychiatry()

    msg.info("Initialising flattened dataset")
    flattened_df = FlattenedDataset(
        prediction_times_df=prediction_times,
        n_workers=min(
            len(PREDICTOR_COMBINATIONS),
            psutil.cpu_count(logical=False) * 3,
        ),  # * 3 since dataframe loading is IO intensive, cores are likely to wait for a lot of them.
        feature_cache_dir=PROJ_PATH / "feature_cache",
    )
    flattened_df.add_age(psycopmlutils.loaders.raw.birthdays())

    # Outcome
    msg.info("Adding outcome")
    for i in [1, 3, 5]:
        LOOKAHEAD_DAYS = int(i * 365)
        msg.info(f"Adding outcome with {LOOKAHEAD_DAYS} days of lookahead")
        flattened_df.add_temporal_outcome(
            outcome_df=event_times,
            lookahead_days=LOOKAHEAD_DAYS,
            resolve_multiple="max",
            fallback=0,
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
    flattened_df.add_static_info(psycopmlutils.loaders.raw.sex_female())

    start_time = time.time()

    msg.info("Adding temporal predictors")

    flattened_df.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictors=PREDICTOR_COMBINATIONS,
        predictor_dfs=pre_loaded_dfs,
    )

    end_time = time.time()

    # Finish
    msg.good(
        f"Finished adding {len(PREDICTOR_COMBINATIONS)} predictors, took {round((end_time - start_time)/60, 1)} minutes",
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

    # Create directory to store all files related to this run
    sub_dir = (
        PROJ_PATH
        / f"{current_user}_{len(PREDICTOR_COMBINATIONS)}_features_{time.strftime('%Y_%m_%d_%H_%M')}"
    )
    sub_dir.mkdir()

    file_prefix = f"psycop_t2d_{current_user}_{len(PREDICTOR_COMBINATIONS)}_predictors_{time.strftime('%Y_%m_%d_%H_%M')}"

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

        file_path = sub_dir / filename

        split_df.to_csv(file_path, index=False)

        msg.good(f"{dataset_name}: Succesfully saved to {file_path}")

    # Log poetry lock file and file prefix to WandB for reproducibility
    feature_settings = {
        "file_prefix": file_prefix,
        "save_path": sub_dir / file_prefix,
        "predictor_list": PREDICTOR_COMBINATIONS,
    }

    ## Create data integrity report
    create_feature_description_from_dir(
        path=sub_dir,
        predictor_dicts=PREDICTOR_COMBINATIONS,
        splits=["train"],
    )

    check_feature_set_integrity_from_dir(path=sub_dir, splits=["train", "val", "test"])

    run = wandb.init(project="psycop-feature-files", config=feature_settings)
    wandb.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")

    wandb.finish()
