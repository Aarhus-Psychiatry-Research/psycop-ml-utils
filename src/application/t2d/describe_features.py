from pathlib import Path

from psycopmlutils.data_checks.feature_describer import (
    create_feature_description_from_dir,
)
from src.application.t2d.features_blood_samples import create_lab_feature_combinations
from src.application.t2d.features_diagnoses import create_diag_feature_combinations
from src.application.t2d.features_medications import (
    create_medication_feature_combinations,
)

if __name__ == "__main__":
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

    create_feature_description_from_dir(
        path=feature_set_dir,
        predictor_dicts=PREDICTOR_LIST,
    )
