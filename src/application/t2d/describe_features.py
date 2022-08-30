from pathlib import Path

import pandas as pd
from wasabi import Printer

from application.t2d.features_blood_samples import create_lab_feature_combinations
from application.t2d.features_diagnoses import create_diag_feature_combinations
from application.t2d.features_medications import create_medication_feature_combinations
from psycopmlutils.feature_describer.feature_describer import (
    generate_feature_description_df,
)

if __name__ == "__main__":
    msg = Printer(timestamp=True)

    feature_set_dir = Path(
        "C:/shared_resources/feature_sets/t2d/adminmanber_260_features_2022_08_26_14_10/",
    )

    feature_set_path = (
        feature_set_dir
        / "adminmanber_psycop_t2d_260_features_2022_08_26_14_10_train.csv"
    )
    out_dir = feature_set_dir / "feature_description"

    RESOLVE_MULTIPLE = ["latest", "max", "min", "mean"]
    LOOKBEHIND_DAYS = [365, 730, 1825, 9999]

    LAB_PREDICTORS = create_lab_feature_combinations(
        RESOLVE_MULTIPLE=RESOLVE_MULTIPLE,
        LOOKBEHIND_DAYS=LOOKBEHIND_DAYS,
    )

    DIAGNOSIS_PREDICTORS = create_diag_feature_combinations(
        RESOLVE_MULTIPLE=RESOLVE_MULTIPLE,
        LOOKBEHIND_DAYS=LOOKBEHIND_DAYS,
        fallback=0,
    )

    MEDICATION_PREDICTORS = create_medication_feature_combinations(
        LOOKBEHIND_DAYS=LOOKBEHIND_DAYS,
        RESOLVE_MULTIPLE=["count"],
        fallback=0,
    )

    PREDICTOR_LIST = MEDICATION_PREDICTORS + DIAGNOSIS_PREDICTORS + LAB_PREDICTORS

    msg.info("Loading features")
    features = pd.read_csv(feature_set_path, nrows=10_000)

    msg.info("Generating feature description df")
    feature_description_df = generate_feature_description_df(
        df=features,
        predictor_dicts=PREDICTOR_LIST,
    )

    # Output dataframe as word document
    msg.info("Outputting feature description df")
    feature_description_df.to_csv(out_dir / "train_description.csv")
