import numpy as np


def create_lab_feature_combinations(
    LOOKBEHIND_DAYS=[365, 730, 1825, 9999],
    RESOLVE_MULTIPLE=["mean", "max", "min"],
):
    return [
        {
            "predictor_df": df,
            "lookbehind_days": LOOKBEHIND_DAYS,
            "resolve_multiple": RESOLVE_MULTIPLE,
            "fallback": np.nan,
            "allowed_nan_value_prop": 0.0,
        }
        for df in [
            "hba1c",
            "alat",
            "hdl",
            "ldl",
            "scheduled_glc",
            "unscheduled_p_glc",
            "triglycerides",
            "fasting_ldl",
            "alat",
            "crp",
            "egfr",
            "albumine_creatinine_ratio",
        ]
    ]
