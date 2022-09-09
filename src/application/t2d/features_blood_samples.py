import numpy as np


def get_lab_feature_spec(
    lookbehind_days=[365, 730, 1825, 9999],
    resolve_Multiple=["mean", "max", "min"],
    values_to_load="all",
):
    return [
        {
            "predictor_df": df,
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_Multiple,
            "fallback": np.nan,
            "allowed_nan_value_prop": 0.0,
            "values_to_load": values_to_load,
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
