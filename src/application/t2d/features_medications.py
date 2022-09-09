def get_medication_feature_spec(
    lookbehind_days=[365, 730, 1825, 9999],
    resolve_multiple=["mean", "max", "min"],
    fallback=0,
):
    return [
        {
            "predictor_df": "antipsychotics",
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_multiple,
            "fallback": fallback,
            "allowed_nan_value_prop": 0.0,
        },
    ]
