import numpy as np

from psycopmlutils.timeseriesflattener import create_feature_combinations


def create_lab_feature_combinations(
    LOOKBEHIND_DAYS=[365, 730, 1825, 9999],
    RESOLVE_MULTIPLE=["mean", "max", "min"],
):
    return create_feature_combinations(
        [
            {
                "predictor_df": "hba1c",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "alat",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "hdl",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "ldl",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "scheduled_glc",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "unscheduled_p_glc",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "triglycerides",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "fasting_ldl",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "alat",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "crp",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "egfr",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
            {
                "predictor_df": "albumine_creatinine_ratio",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
                "allowed_nan_value_prop": 0.0,
            },
        ],
    )
