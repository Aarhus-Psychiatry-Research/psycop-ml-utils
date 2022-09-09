def get_medication_feature_spec(
    LOOKBEHIND_DAYS=[365, 730, 1825, 9999],
    RESOLVE_MULTIPLE=["mean", "max", "min"],
):
    return [
        {
            "predictor_df": "antipsychotics",
            "lookbehind_days": LOOKBEHIND_DAYS,
            "resolve_multiple": RESOLVE_MULTIPLE,
            "fallback": 0,
            "allowed_nan_value_prop": 0.0,
        },
    ]
